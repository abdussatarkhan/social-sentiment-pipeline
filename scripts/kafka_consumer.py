"""
Kafka Stream Consumer & NLP Inference Worker
Real-Time Social Listening & Brand Sentiment Pipeline

Workflow:
1. Consumes raw JSON events from Kafka topic 'social-raw-stream'
2. Cleans text, strips noise, demojizes, detects language, and checks spam
3. Runs batched DistilBERT ABSA inference (calculates polarity & aspect scores)
4. Persists records to PostgreSQL (raw_posts, sentiment_posts, post_aspect_sentiment)
5. Indexes structured document into Elasticsearch for real-time Kibana/Grafana search
6. Handles dead-letter queue routing upon unrecoverable parse or inference errors
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

from scripts.utils import (
    setup_logger,
    load_config,
    get_kafka_consumer,
    get_kafka_producer,
    get_postgres_pool,
    get_elasticsearch_client,
    ensure_elasticsearch_index,
    measure_time,
)
from scripts.preprocessing import SocialTextPreprocessor, RawSocialPost, CleanedSocialPost
from scripts.sentiment_model import SentimentInferenceEngine

logger = setup_logger("kafka_consumer")


class SentimentStreamProcessor:
    """Stream processor consuming social events and running sentiment NLP enrichment."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
        self.cfg = config or load_config()
        self.dry_run = dry_run

        self.raw_topic = self.cfg["kafka"]["topics"]["raw_stream"]
        self.enriched_topic = self.cfg["kafka"]["topics"]["enriched_stream"]
        self.dlq_topic = self.cfg["kafka"]["topics"]["dead_letter"]
        self.batch_size = self.cfg["kafka"].get("max_poll_records", 50)
        self.es_index = f"{self.cfg['elasticsearch']['index_prefix']}_stream"

        # Initialize Preprocessor & DistilBERT ABSA Engine
        self.preprocessor = SocialTextPreprocessor(replace_emoji=True)
        self.sentiment_engine = SentimentInferenceEngine()

        # Database pool & Elasticsearch client
        self.pg_pool = None
        self.es_client = None
        self.producer = None
        self.consumer = None

        if not self.dry_run:
            try:
                self.pg_pool = get_postgres_pool(self.cfg)
            except Exception as e:
                logger.warning(f"PostgreSQL pool unavailable: {e}. DB persistence disabled.")

            try:
                self.es_client = get_elasticsearch_client(self.cfg)
                ensure_elasticsearch_index(self.es_client, self.es_index)
            except Exception as e:
                logger.warning(f"Elasticsearch unavailable: {e}. ES indexing disabled.")

            try:
                self.producer = get_kafka_producer(self.cfg)
            except Exception as e:
                logger.warning(f"Kafka Producer unavailable: {e}. Outbound publishing disabled.")

    def process_raw_batch(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans text, runs DistilBERT inference, and packages unified payload."""
        if not raw_records:
            return []

        cleaned_posts: List[CleanedSocialPost] = []
        texts_to_infer: List[str] = []

        # Step 1: Preprocessing & Spam filtering
        for record in raw_records:
            try:
                raw_post = RawSocialPost(**record)
                cleaned = self.preprocessor.process(raw_post)
                cleaned_posts.append(cleaned)
                texts_to_infer.append(cleaned.clean_text)
            except Exception as e:
                logger.error(f"Error preprocessing record {record.get('post_id')}: {e}")
                self.route_to_dlq(record, str(e))

        if not texts_to_infer:
            return []

        # Step 2: Batched DistilBERT Inference
        inference_results = self.sentiment_engine.predict_batch(texts_to_infer)

        # Step 3: Combine into enriched output objects
        enriched_batch = []
        for post, inf in zip(cleaned_posts, inference_results):
            enriched_item = {
                "post_id": post.post_id,
                "platform": post.platform,
                "author_id": post.author_id,
                "author_handle": post.author_handle,
                "raw_text": post.raw_text,
                "clean_text": post.clean_text,
                "language": post.language,
                "is_spam": post.is_spam,
                "spam_reason": post.spam_reason,
                "sentiment_label": inf["sentiment_label"],
                "sentiment_score": inf["sentiment_score"],
                "confidence": inf["confidence"],
                "aspects": inf["aspects"],
                "created_at": post.created_at,
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "url": post.url,
                "reply_to_id": post.reply_to_id,
                "reply_to_user": post.reply_to_user,
                "engagement": post.engagement,
            }
            enriched_batch.append(enriched_item)

        return enriched_batch

    def persist_to_postgres(self, enriched_batch: List[Dict[str, Any]]) -> None:
        """Writes batch to PostgreSQL with transaction isolation."""
        if not self.pg_pool or not enriched_batch:
            return

        conn = None
        try:
            conn = self.pg_pool.getconn()
            with conn.cursor() as cur:
                for item in enriched_batch:
                    # 1. Insert into raw_social_posts
                    cur.execute(
                        """
                        INSERT INTO raw_social_posts 
                        (post_id, platform, author_id, author_handle, raw_text, url, created_at, reply_to_id, reply_to_user, engagement)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (platform, post_id) DO NOTHING;
                        """,
                        (
                            item["post_id"],
                            item["platform"],
                            item["author_id"],
                            item["author_handle"],
                            item["raw_text"],
                            item["url"],
                            item["created_at"],
                            item["reply_to_id"],
                            item["reply_to_user"],
                            json.dumps(item["engagement"]),
                        ),
                    )

                    # 2. Insert into sentiment_posts
                    cur.execute(
                        """
                        INSERT INTO sentiment_posts 
                        (post_id, platform, author_id, author_handle, clean_text, language, 
                         sentiment_label, sentiment_score, confidence, is_spam, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (platform, post_id) DO UPDATE 
                        SET sentiment_score = EXCLUDED.sentiment_score,
                            sentiment_label = EXCLUDED.sentiment_label,
                            confidence = EXCLUDED.confidence;
                        """,
                        (
                            item["post_id"],
                            item["platform"],
                            item["author_id"],
                            item["author_handle"],
                            item["clean_text"],
                            item["language"],
                            item["sentiment_label"],
                            item["sentiment_score"],
                            item["confidence"],
                            item["is_spam"],
                            item["created_at"],
                        ),
                    )

                    # 3. Insert aspect scores
                    for aspect_name, asp in item["aspects"].items():
                        cur.execute(
                            """
                            INSERT INTO post_aspect_sentiment 
                            (post_id, platform, aspect, sentiment_label, sentiment_score, confidence)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (platform, post_id, aspect) DO NOTHING;
                            """,
                            (
                                item["post_id"],
                                item["platform"],
                                aspect_name,
                                asp["label"],
                                asp["score"],
                                asp["confidence"],
                            ),
                        )

            conn.commit()
            logger.debug(f"Persisted {len(enriched_batch)} posts to PostgreSQL.")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"PostgreSQL batch insert error: {e}")
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)

    def persist_to_elasticsearch(self, enriched_batch: List[Dict[str, Any]]) -> None:
        """Indexes batch documents into Elasticsearch."""
        if not self.es_client or not enriched_batch:
            return

        for item in enriched_batch:
            try:
                # Prepare nested aspect format for Elasticsearch
                es_aspects = [
                    {"aspect": k, "label": v["label"], "score": v["score"]}
                    for k, v in item["aspects"].items()
                ]
                doc = dict(item)
                doc["aspects"] = es_aspects

                self.es_client.index(
                    index=self.es_index,
                    id=f"{item['platform']}_{item['post_id']}",
                    document=doc,
                )
            except Exception as e:
                logger.error(f"Failed to index post {item['post_id']} to Elasticsearch: {e}")

    def route_to_dlq(self, payload: Dict[str, Any], reason: str) -> None:
        """Routes malformed or failing records to Dead-Letter Queue."""
        dlq_entry = {
            "failed_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "original_payload": payload,
        }
        if self.producer:
            try:
                self.producer.send(self.dlq_topic, value=dlq_entry)
            except Exception as e:
                logger.error(f"Failed sending message to DLQ: {e}")
        else:
            logger.warning(f"[DLQ DROP] {reason} -> {payload}")

    def run_consumer_loop(self):
        """Continuously polls Kafka, executes enrichment, and writes to sinks."""
        logger.info(f"Starting consumer loop on topic '{self.raw_topic}'...")
        try:
            self.consumer = get_kafka_consumer(self.raw_topic, config=self.cfg)
        except Exception as e:
            logger.error(f"Unable to start Kafka consumer: {e}")
            return

        batch: List[Dict[str, Any]] = []
        last_flush = time.time()

        for message in self.consumer:
            raw_data = message.value
            batch.append(raw_data)

            if len(batch) >= self.batch_size or (time.time() - last_flush > 2.0 and batch):
                enriched = self.process_raw_batch(batch)
                self.persist_to_postgres(enriched)
                self.persist_to_elasticsearch(enriched)

                # Re-publish enriched records to downstream Kafka topic
                if self.producer:
                    for item in enriched:
                        self.producer.send(self.enriched_topic, key=item["post_id"], value=item)
                    self.producer.flush()

                self.consumer.commit()
                batch.clear()
                last_flush = time.time()

    def run_mock_simulation(self, count: int = 15):
        """Processes a sequence of simulated posts locally for verification."""
        logger.info(f"Running standalone simulation on {count} mock posts...")
        mock_posts = [
            {
                "post_id": f"sim_{idx}",
                "platform": "bluesky" if idx % 2 == 0 else "reddit",
                "author_id": f"usr_{idx}",
                "author_handle": f"@user_{idx}",
                "text": text,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "url": f"https://example.com/post/{idx}",
                "reply_to_id": None,
                "reply_to_user": None,
                "engagement": {"likes": idx * 3, "reposts": idx},
            }
            for idx, text in enumerate([
                "ApexCloud reliability has degraded terribly! 3 service interruptions today.",
                "Incredible customer support experience with ApexCloud. Immediate response.",
                "ApexCloud pricing is way too expensive for early-stage startups.",
                "The ApexCloud web console UI is snappy, modern, and very well designed.",
                "Just regular routine database maintenance on ApexCloud tonight.",
                "FREE CRYPTO AIRDROP AIRDROP CLICK HERE TO CLAIM 1000 BTC NOW!!!",
                "ApexCloud API latency is at an all-time low. Great work engineering team.",
            ] * 3)
        ][:count]

        enriched = self.process_raw_batch(mock_posts)
        logger.info(f"Processed {len(enriched)} posts successfully.")
        for item in enriched[:3]:
            print(f"\n[ID: {item['post_id']}] Platform: {item['platform']}")
            print(f"  Clean Text: {item['clean_text']}")
            print(f"  Sentiment:  {item['sentiment_label']} (Score: {item['sentiment_score']:.2f}, Conf: {item['confidence']:.2f})")
            print(f"  Aspects:    {item['aspects']}")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kafka Consumer & DistilBERT Sentiment Worker")
    parser.add_argument("--mock", action="store_true", help="Run local simulation without Kafka")
    parser.add_argument("--count", type=int, default=10, help="Number of records to simulate")
    parser.add_argument("--dry-run", action="store_true", help="Disable persistent DB/ES writes")

    args = parser.parse_args()
    processor = SentimentStreamProcessor(dry_run=args.dry_run or args.mock)

    if args.mock:
        processor.run_mock_simulation(count=args.count)
    else:
        processor.run_consumer_loop()


if __name__ == "__main__":
    main()
