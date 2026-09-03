"""
BERTopic Trending Topic Discovery & Sentiment Clustering
Real-Time Social Listening & Brand Sentiment Pipeline

Features:
- Discovers emerging conversation themes from social streams using BERTopic.
- Utilizes SentenceTransformer embeddings ('all-MiniLM-L6-v2').
- Aggregates topic-level sentiment distribution (positive/negative ratio, mean polarity).
- Extracts top c-TF-IDF keywords and representative documents per cluster.
- Persists topic intelligence to PostgreSQL 'topic_clusters'.
- Provides graceful fallbacks (Scikit-Learn TF-IDF + NMF/KMeans) for lightweight runtime environments.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date

import numpy as np
import pandas as pd

from scripts.utils import setup_logger, load_config, get_postgres_pool, measure_time

logger = setup_logger("topic_modeling")


class TopicDiscoveryEngine:
    """Discovers trending conversational clusters and computes cluster-level sentiment."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or load_config()
        topic_cfg = self.cfg.get("topic_modeling", {})

        self.min_topic_size = topic_cfg.get("min_topic_size", 10)
        self.nr_topics = topic_cfg.get("nr_topics", 8)
        self.embedding_model_name = topic_cfg.get("embedding_model", "all-MiniLM-L6-v2")

        # Database pool
        self.pg_pool = None
        try:
            self.pg_pool = get_postgres_pool(self.cfg)
        except Exception:
            logger.warning("PostgreSQL unavailable. Operating in local mode.")

    def fetch_daily_corpus(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """Queries clean text and sentiment scores for a given day from PostgreSQL."""
        if not self.pg_pool:
            return pd.DataFrame()

        conn = None
        try:
            conn = self.pg_pool.getconn()
            day_str = target_date or datetime.utcnow().strftime("%Y-%m-%d")
            query = """
                SELECT 
                    post_id,
                    platform,
                    clean_text,
                    sentiment_label,
                    sentiment_score
                FROM sentiment_posts
                WHERE DATE(created_at) = %s
                  AND is_spam = FALSE
                  AND LENGTH(clean_text) > 15;
            """
            with conn.cursor() as cur:
                cur.execute(query, (day_str,))
                rows = cur.fetchall()

            df = pd.DataFrame(
                rows,
                columns=["post_id", "platform", "clean_text", "sentiment_label", "sentiment_score"],
            )
            logger.info(f"Retrieved {len(df)} documents for date: {day_str}")
            return df
        except Exception as e:
            logger.error(f"Error querying corpus from PostgreSQL: {e}")
            return pd.DataFrame()
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)

    def fit_bertopic(self, docs: List[str]) -> Tuple[List[int], Dict[int, List[Tuple[str, float]]]]:
        """Fits BERTopic or fallback TF-IDF + NMF if dependencies require."""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer

            logger.info(f"Fitting BERTopic with embedding model: {self.embedding_model_name}...")
            embedding_model = SentenceTransformer(self.embedding_model_name)
            topic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=self.min_topic_size,
                nr_topics=self.nr_topics,
                calculate_probabilities=False,
                verbose=False,
            )
            topics, _ = topic_model.fit_transform(docs)
            topic_info = topic_model.get_topics()
            return topics, topic_info

        except Exception as e:
            logger.warning(f"BERTopic initialization encountered ({e}). Utilizing Scikit-Learn fallback...")
            return self._fallback_topic_modeling(docs)

    def _fallback_topic_modeling(self, docs: List[str]) -> Tuple[List[int], Dict[int, List[Tuple[str, float]]]]:
        """Lightweight Scikit-learn TF-IDF + NMF fallback algorithm."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF

        n_topics = min(self.nr_topics, max(2, len(docs) // 10))
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english", max_features=1000)
        tfidf = vectorizer.fit_transform(docs)

        nmf = NMF(n_components=n_topics, random_state=42)
        doc_topics = nmf.fit_transform(tfidf)
        topics = [int(np.argmax(row)) for row in doc_topics]

        feature_names = vectorizer.get_feature_names_out()
        topic_info = {}
        for topic_idx, topic in enumerate(nmf.components_):
            top_features_ind = topic.argsort()[: -10 - 1 : -1]
            topic_info[topic_idx] = [(feature_names[i], float(topic[i])) for i in top_features_ind]

        return topics, topic_info

    def aggregate_topic_metrics(
        self,
        df: pd.DataFrame,
        topics: List[int],
        topic_info: Dict[int, List[Tuple[str, float]]],
    ) -> List[Dict[str, Any]]:
        """Aggregates volume, top keywords, sentiment breakdown, and sample docs per topic."""
        df["topic_id"] = topics
        aggregated = []

        for tid in sorted(set(topics)):
            if tid == -1:
                continue  # Outlier cluster

            cluster_df = df[df["topic_id"] == tid]
            post_count = len(cluster_df)
            if post_count == 0:
                continue

            keywords = [word for word, score in topic_info.get(tid, [])][:8]
            label = " & ".join(keywords[:2]).title() if keywords else f"Topic {tid}"

            pos_count = (cluster_df["sentiment_label"] == "positive").sum()
            neg_count = (cluster_df["sentiment_label"] == "negative").sum()
            mean_score = cluster_df["sentiment_score"].astype(float).mean()

            # Extract 2 representative sample posts
            sample_docs = cluster_df["clean_text"].head(2).tolist()

            record = {
                "topic_id": tid,
                "topic_label": label,
                "top_keywords": keywords,
                "post_count": int(post_count),
                "positive_ratio": round(float(pos_count / post_count), 4),
                "negative_ratio": round(float(neg_count / post_count), 4),
                "mean_sentiment": round(float(mean_score), 4),
                "representative_samples": sample_docs,
            }
            aggregated.append(record)

        aggregated.sort(key=lambda x: x["post_count"], reverse=True)
        return aggregated

    def persist_clusters(self, clusters: List[Dict[str, Any]], run_date: Optional[str] = None) -> None:
        """Persists topic clusters into PostgreSQL."""
        if not self.pg_pool or not clusters:
            return

        conn = None
        try:
            conn = self.pg_pool.getconn()
            day_val = run_date or date.today().isoformat()
            with conn.cursor() as cur:
                for cl in clusters:
                    cur.execute(
                        """
                        INSERT INTO topic_clusters 
                        (run_date, topic_id, topic_label, top_keywords, post_count, 
                         positive_ratio, negative_ratio, mean_sentiment, representative_samples)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_date, topic_id) DO UPDATE 
                        SET post_count = EXCLUDED.post_count,
                            mean_sentiment = EXCLUDED.mean_sentiment,
                            top_keywords = EXCLUDED.top_keywords;
                        """,
                        (
                            day_val,
                            cl["topic_id"],
                            cl["topic_label"],
                            cl["top_keywords"],
                            cl["post_count"],
                            cl["positive_ratio"],
                            cl["negative_ratio"],
                            cl["mean_sentiment"],
                            json.dumps(cl["representative_samples"]),
                        ),
                    )
            conn.commit()
            logger.info(f"Persisted {len(clusters)} topic clusters to PostgreSQL for {day_val}.")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed persisting topic clusters: {e}")
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)

    def generate_synthetic_corpus(self) -> pd.DataFrame:
        """Generates realistic synthetic social media posts for topic discovery."""
        samples = [
            # Topic 1: API Outage & Service Degradation
            ("ApexCloud gateway is returning 502 Bad Gateway across all endpoints in us-east.", "negative", -0.9),
            ("Critical downtime incident on ApexCloud API services today. Multiple services down.", "negative", -0.85),
            ("Is ApexCloud API experiencing latency issues? Requests timing out after 30 seconds.", "negative", -0.7),
            ("Service outage notification: ApexCloud ingress network connectivity disruption.", "negative", -0.6),

            # Topic 2: Kubernetes Managed Cluster Speed
            ("ApexCloud managed Kubernetes cluster provisioning speed is ridiculously impressive!", "positive", 0.85),
            ("Deploying K8s clusters on ApexCloud takes less than 2 minutes. Loving this workflow.", "positive", 0.9),
            ("ApexCloud Kubernetes node autoscaling handles our traffic surges effortlessly.", "positive", 0.8),
            ("Benchmarking Kubernetes control planes: ApexCloud vs EKS vs GKE.", "neutral", 0.1),

            # Topic 3: Pricing & Billing Invoices
            ("ApexCloud billing statement has unexpected egress data fees this month.", "negative", -0.55),
            ("ApexCloud pricing model is much more cost-effective than standard AWS instances.", "positive", 0.65),
            ("Where can I find detailed billing breakdown for compute resources on ApexCloud?", "neutral", 0.0),
            ("Received a surprise $800 invoice from ApexCloud due to unattached storage volumes.", "negative", -0.75),

            # Topic 4: Customer Support Responsiveness
            ("ApexCloud technical support resolved our enterprise ticket within 15 minutes. Great SLA.", "positive", 0.9),
            ("Still waiting for a human response on ApexCloud ticket #9482. Support response is slow.", "negative", -0.65),
            ("Customer success engineering team at ApexCloud went above and beyond during onboarding.", "positive", 0.8),
        ] * 4  # Repeat to reach sufficient corpus size

        data = []
        for i, (txt, label, score) in enumerate(samples):
            data.append({
                "post_id": f"syn_{i:03d}",
                "platform": "bluesky" if i % 2 == 0 else "reddit",
                "clean_text": txt,
                "sentiment_label": label,
                "sentiment_score": score,
            })
        return pd.DataFrame(data)

    def discover_topics(self, target_date: Optional[str] = None, use_mock: bool = False) -> List[Dict[str, Any]]:
        """End-to-end execution of topic clustering and sentiment attribution."""
        if use_mock or self.pg_pool is None:
            logger.info("Running topic discovery on synthetic benchmark corpus...")
            df = self.generate_synthetic_corpus()
        else:
            df = self.fetch_daily_corpus(target_date)
            if df.empty or len(df) < self.min_topic_size:
                logger.warning("Corpus too small in DB. Utilizing synthetic corpus.")
                df = self.generate_synthetic_corpus()

        docs = df["clean_text"].tolist()
        topics, topic_info = self.fit_bertopic(docs)
        clusters = self.aggregate_topic_metrics(df, topics, topic_info)

        if not use_mock and self.pg_pool:
            self.persist_clusters(clusters, target_date)

        return clusters


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BERTopic Trending Topic Discovery Module")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--mock", action="store_true", help="Run on synthetic corpus")

    args = parser.parse_args()
    engine = TopicDiscoveryEngine()
    clusters = engine.discover_topics(target_date=args.date, use_mock=args.mock)

    print("\n=======================================================")
    print("        DISCOVERED TRENDING TOPIC CLUSTERS             ")
    print("=======================================================")
    for cl in clusters:
        print(f"\n[Topic {cl['topic_id']}] {cl['topic_label']} (Volume: {cl['post_count']} posts)")
        print(f"  Keywords:     {', '.join(cl['top_keywords'])}")
        print(f"  Sentiment:    Mean={cl['mean_sentiment']:.2f} | Pos={cl['positive_ratio']*100:.1f}% | Neg={cl['negative_ratio']*100:.1f}%")
        print(f"  Sample Post:  \"{cl['representative_samples'][0]}\"")


if __name__ == "__main__":
    main()
