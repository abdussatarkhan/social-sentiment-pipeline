"""
Social Graph Influence & Detractor Amplification Analysis
Real-Time Social Listening & Brand Sentiment Pipeline

Features:
- Constructs directed interaction graphs (replies, reposts, mentions) using NetworkX.
- Calculates PageRank influence scores to quantify structural authority.
- Measures sentiment contagion and identifies 'Top Negative Amplifiers'
  (high-centrality nodes actively propagating brand detractors).
- Persists computed influence metrics to PostgreSQL 'user_influence_scores'.
- Supports GraphML/JSON export for interactive graph visualization.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

from scripts.utils import setup_logger, load_config, get_postgres_pool, measure_time

logger = setup_logger("influence_analysis")


class SocialInfluenceAnalyzer:
    """Builds social interaction topology and identifies influential sentiment amplifiers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or load_config()
        inf_cfg = self.cfg.get("influence_analysis", {})

        self.damping = inf_cfg.get("damping_factor", 0.85)
        self.max_iter = inf_cfg.get("max_iterations", 100)
        self.tol = inf_cfg.get("convergence_tol", 1e-6)
        self.top_k = inf_cfg.get("top_amplifiers_count", 20)

        self.graph = nx.DiGraph()

        # Database pool
        self.pg_pool = None
        try:
            self.pg_pool = get_postgres_pool(self.cfg)
        except Exception:
            logger.warning("PostgreSQL unavailable. Running influence analyzer in standalone mode.")

    def fetch_interaction_data(self, lookback_days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Queries interaction edges and node sentiment aggregates from PostgreSQL."""
        if not self.pg_pool:
            return pd.DataFrame(), pd.DataFrame()

        conn = None
        try:
            conn = self.pg_pool.getconn()
            # 1. Edge query (replies and conversation linkages)
            edges_query = """
                SELECT 
                    r1.author_id AS source_user,
                    r1.author_handle AS source_handle,
                    r2.author_id AS target_user,
                    r2.author_handle AS target_handle,
                    COUNT(*) AS interaction_count
                FROM raw_social_posts r1
                JOIN raw_social_posts r2 ON r1.reply_to_id = r2.post_id
                WHERE r1.created_at >= NOW() - INTERVAL '%s days'
                  AND r1.author_id != r2.author_id
                GROUP BY 1, 2, 3, 4;
            """

            # 2. Node sentiment query
            nodes_query = """
                SELECT 
                    author_id,
                    author_handle,
                    platform,
                    COUNT(*) AS total_posts,
                    COUNT(*) FILTER (WHERE sentiment_label = 'negative') AS negative_posts,
                    COUNT(*) FILTER (WHERE sentiment_label = 'positive') AS positive_posts,
                    AVG(sentiment_score) AS mean_sentiment
                FROM sentiment_posts
                WHERE created_at >= NOW() - INTERVAL '%s days'
                  AND is_spam = FALSE
                GROUP BY 1, 2, 3;
            """

            with conn.cursor() as cur:
                cur.execute(edges_query, (lookback_days,))
                edge_rows = cur.fetchall()
                edges_df = pd.DataFrame(
                    edge_rows,
                    columns=["source_user", "source_handle", "target_user", "target_handle", "interaction_count"],
                )

                cur.execute(nodes_query, (lookback_days,))
                node_rows = cur.fetchall()
                nodes_df = pd.DataFrame(
                    node_rows,
                    columns=["author_id", "author_handle", "platform", "total_posts", "negative_posts", "positive_posts", "mean_sentiment"],
                )

            return edges_df, nodes_df
        except Exception as e:
            logger.error(f"Error fetching interaction data from DB: {e}")
            return pd.DataFrame(), pd.DataFrame()
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)

    def build_network_graph(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> nx.DiGraph:
        """Constructs NetworkX directed graph with node attributes and weighted edges."""
        self.graph.clear()

        # Add nodes with sentiment attributes
        for _, row in nodes_df.iterrows():
            self.graph.add_node(
                row["author_id"],
                handle=row["author_handle"] or row["author_id"],
                platform=row.get("platform", "unknown"),
                total_posts=int(row["total_posts"]),
                negative_posts=int(row["negative_posts"]),
                positive_posts=int(row["positive_posts"]),
                mean_sentiment=float(row["mean_sentiment"]) if pd.notnull(row["mean_sentiment"]) else 0.0,
            )

        # Add directed edges (Source replied to Target)
        for _, row in edges_df.iterrows():
            u = row["source_user"]
            v = row["target_user"]
            weight = float(row["interaction_count"])

            if not self.graph.has_node(u):
                self.graph.add_node(u, handle=row.get("source_handle", u), total_posts=1, negative_posts=0)
            if not self.graph.has_node(v):
                self.graph.add_node(v, handle=row.get("target_handle", v), total_posts=1, negative_posts=0)

            self.graph.add_edge(u, v, weight=weight)

        logger.info(
            f"Social interaction graph constructed: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges."
        )
        return self.graph

    def compute_influence_metrics(self) -> pd.DataFrame:
        """Calculates PageRank, degree centralities, and Amplification Risk Scores."""
        if self.graph.number_of_nodes() == 0:
            logger.warning("Empty network graph. Cannot compute influence scores.")
            return pd.DataFrame()

        # 1. Directed PageRank
        try:
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.damping,
                max_iter=self.max_iter,
                tol=self.tol,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank power iteration did not converge. Falling back to unweighted.")
            pagerank_scores = nx.pagerank(self.graph, alpha=self.damping, max_iter=200)

        # 2. In-Degree & Out-Degree
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())

        records = []
        for node, pr in pagerank_scores.items():
            attrs = self.graph.nodes[node]
            tot_posts = attrs.get("total_posts", 0)
            neg_posts = attrs.get("negative_posts", 0)
            neg_ratio = (neg_posts / tot_posts) if tot_posts > 0 else 0.0

            # Amplification Risk Index:
            # High PageRank + High Negative Post Proportion + Non-trivial post volume
            risk_index = pr * (1.0 + np.log1p(neg_posts)) * (neg_ratio ** 1.2) * 1000.0

            records.append({
                "author_id": node,
                "author_handle": attrs.get("handle", node),
                "platform": attrs.get("platform", "bluesky"),
                "pagerank_score": float(pr),
                "in_degree": int(in_degrees.get(node, 0)),
                "out_degree": int(out_degrees.get(node, 0)),
                "total_posts": tot_posts,
                "negative_posts": neg_posts,
                "negative_ratio": round(neg_ratio, 3),
                "amplification_risk_score": float(risk_index),
            })

        df = pd.DataFrame(records)
        df.sort_values(by="amplification_risk_score", ascending=False, inplace=True)
        return df

    def persist_influence_scores(self, df: pd.DataFrame) -> None:
        """Persists calculated influence and risk scores to PostgreSQL."""
        if not self.pg_pool or df.empty:
            return

        conn = None
        try:
            conn = self.pg_pool.getconn()
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    cur.execute(
                        """
                        INSERT INTO user_influence_scores 
                        (author_id, platform, author_handle, pagerank_score, in_degree, out_degree, 
                         total_posts, negative_posts, amplification_risk_score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (platform, author_id, calculated_at) DO UPDATE 
                        SET pagerank_score = EXCLUDED.pagerank_score,
                            amplification_risk_score = EXCLUDED.amplification_risk_score;
                        """,
                        (
                            row["author_id"],
                            row["platform"],
                            row["author_handle"],
                            row["pagerank_score"],
                            row["in_degree"],
                            row["out_degree"],
                            row["total_posts"],
                            row["negative_posts"],
                            row["amplification_risk_score"],
                        ),
                    )
            conn.commit()
            logger.info(f"Persisted {len(df)} user influence scores to PostgreSQL.")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to persist influence scores: {e}")
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)

    def generate_synthetic_graph(self, num_nodes: int = 60, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates a synthetic scale-free network modeling viral social threads."""
        np.random.seed(seed)
        G = nx.scale_free_graph(num_nodes, seed=seed)
        G = nx.DiGraph(G)  # Remove multi-edges

        node_records = []
        for i in range(num_nodes):
            is_detractor = (i < 8)  # Key early negative agitators
            tot = np.random.randint(5, 50)
            neg = int(tot * np.random.uniform(0.7, 0.95)) if is_detractor else int(tot * np.random.uniform(0.05, 0.25))
            pos = tot - neg

            node_records.append({
                "author_id": f"did:plc:sim_user_{i:03d}",
                "author_handle": f"@tech_influencer_{i}" if i < 10 else f"@dev_user_{i}",
                "platform": "bluesky" if i % 2 == 0 else "reddit",
                "total_posts": tot,
                "negative_posts": neg,
                "positive_posts": pos,
                "mean_sentiment": -0.65 if is_detractor else 0.40,
            })

        edge_records = []
        for u, v in G.edges():
            if u != v:
                edge_records.append({
                    "source_user": f"did:plc:sim_user_{u:03d}",
                    "source_handle": f"@user_{u}",
                    "target_user": f"did:plc:sim_user_{v:03d}",
                    "target_handle": f"@user_{v}",
                    "interaction_count": np.random.randint(1, 12),
                })

        return pd.DataFrame(edge_records), pd.DataFrame(node_records)

    def run_analysis(self, lookback_days: int = 7, use_mock: bool = False) -> pd.DataFrame:
        """End-to-end execution of graph influence modeling."""
        if use_mock or self.pg_pool is None:
            logger.info("Running influence pipeline on synthetic social network graph...")
            edges_df, nodes_df = self.generate_synthetic_graph()
        else:
            edges_df, nodes_df = self.fetch_interaction_data(lookback_days=lookback_days)
            if edges_df.empty:
                logger.warning("No interactions found in DB. Falling back to synthetic graph.")
                edges_df, nodes_df = self.generate_synthetic_graph()

        self.build_network_graph(edges_df, nodes_df)
        df_scores = self.compute_influence_metrics()

        if not use_mock and self.pg_pool:
            self.persist_influence_scores(df_scores)

        return df_scores


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Social Network Influence & Detractor Analysis")
    parser.add_argument("--mock", action="store_true", help="Run on synthetic network data")
    parser.add_argument("--top", type=int, default=10, help="Display top K amplifiers")
    parser.add_argument("--export-graph", type=str, help="Path to export GraphML network file")

    args = parser.parse_args()
    analyzer = SocialInfluenceAnalyzer()
    df = analyzer.run_analysis(use_mock=args.mock)

    print("\n=======================================================")
    print("      TOP DETRACTOR AMPLIFIERS (BY RISK INDEX)         ")
    print("=======================================================")
    top_df = df.head(args.top)[
        ["author_handle", "platform", "pagerank_score", "negative_posts", "total_posts", "amplification_risk_score"]
    ]
    print(top_df.to_string(index=False))

    if args.export_graph:
        nx.write_graphml(analyzer.graph, args.export_graph)
        print(f"\nSaved graph topology to {args.export_graph}")


if __name__ == "__main__":
    main()
