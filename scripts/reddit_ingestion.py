"""
Reddit Ingestion Service using PRAW
Real-Time Social Listening & Brand Sentiment Pipeline

Streams submissions and comments across technology and enterprise IT subreddits:
  - r/technology, r/sysadmin, r/cloudcomputing, r/aws, r/devops, r/stocks
Filters for brand keywords and enterprise cloud outage indicators.
Normalizes metadata and publishes directly to Kafka topic 'social-raw-stream'.
Includes mock replay mode for local and automated testing.
"""

import os
import sys
import time
import argparse
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import praw
from prawcore.exceptions import ResponseException, RequestException

from scripts.utils import setup_logger, load_config, get_kafka_producer

logger = setup_logger("reddit_ingestion")


class RedditIngestionWorker:
    """Streams Reddit posts and comments via PRAW into Apache Kafka."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, mock_mode: bool = False):
        self.cfg = config or load_config()
        reddit_cfg = self.cfg.get("reddit", {})
        self.subreddits = reddit_cfg.get("subreddits", ["technology", "sysadmin", "cloudcomputing", "aws"])
        self.keywords = [k.lower() for k in reddit_cfg.get("keywords", ["apexcloud", "cloud outage", "sla breach"])]
        self.target_topic = self.cfg["kafka"]["topics"]["raw_stream"]
        self.mock_mode = mock_mode

        # Initialize Kafka Producer
        try:
            self.producer = get_kafka_producer(self.cfg)
            logger.info(f"Connected Kafka producer for topic '{self.target_topic}'")
        except Exception as e:
            logger.warning(f"Kafka connection error ({e}). Operating in dry-run/logging mode.")
            self.producer = None

        # PRAW Client Setup
        self.client_id = reddit_cfg.get("client_id")
        self.client_secret = reddit_cfg.get("client_secret")
        self.user_agent = reddit_cfg.get("user_agent", "SocialSentimentPipeline/1.0")

        self.reddit = None
        if not self.mock_mode and self.client_id and not self.client_id.startswith("MOCK_"):
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
                logger.info(f"Authenticated with Reddit API (PRAW) read-only client.")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Reddit PRAW: {e}. Defaulting to mock mode.")
                self.mock_mode = True
        else:
            self.mock_mode = True

    def matches_keywords(self, text: str) -> bool:
        """Determines if text contains any tracked brand keywords."""
        if not text:
            return False
        lower_text = text.lower()
        return any(kw in lower_text for kw in self.keywords)

    def format_item(
        self,
        item_id: str,
        author: str,
        text: str,
        created_utc: float,
        url: str,
        score: int,
        parent_id: Optional[str] = None,
        subreddit_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalizes Reddit post or comment into common schema."""
        created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
        return {
            "post_id": f"reddit_{item_id}",
            "platform": "reddit",
            "author_id": f"reddit_u_{author}",
            "author_handle": f"u/{author}",
            "text": text,
            "created_at": created_dt,
            "url": url,
            "reply_to_id": f"reddit_{parent_id}" if parent_id else None,
            "reply_to_user": None,
            "engagement": {
                "score": score,
                "likes": max(0, score),
                "reposts": 0,
                "replies": 0,
            },
            "raw_metadata": {
                "subreddit": subreddit_name,
                "source": "praw_stream",
            },
        }

    def publish_to_kafka(self, payload: Dict[str, Any]) -> None:
        """Publishes post payload to Kafka."""
        if self.producer:
            try:
                self.producer.send(
                    topic=self.target_topic,
                    key=payload["post_id"],
                    value=payload,
                )
            except Exception as e:
                logger.error(f"Failed to produce Reddit payload to Kafka: {e}")
        else:
            logger.info(f"[REDDIT OUT] {payload['post_id']} -> {payload['text'][:70]}...")

    def run_live_stream(self, max_items: Optional[int] = None):
        """Continuously streams submissions and comments from target subreddits."""
        if not self.reddit:
            logger.error("Reddit client not configured. Use --mock flag.")
            return

        sub_string = "+".join(self.subreddits)
        logger.info(f"Opening PRAW stream on subreddits: r/{sub_string}")
        subreddit = self.reddit.subreddit(sub_string)
        count = 0

        while True:
            try:
                # Stream comments
                for comment in subreddit.stream.comments(skip_existing=True, pause_after=2):
                    if comment is None:
                        break
                    body = comment.body
                    if self.matches_keywords(body):
                        author_name = str(comment.author) if comment.author else "[deleted]"
                        parent = comment.parent_id if hasattr(comment, "parent_id") else None
                        item = self.format_item(
                            item_id=comment.id,
                            author=author_name,
                            text=body,
                            created_utc=comment.created_utc,
                            url=f"https://reddit.com{comment.permalink}",
                            score=comment.score,
                            parent_id=parent,
                            subreddit_name=str(comment.subreddit),
                        )
                        self.publish_to_kafka(item)
                        count += 1
                        if max_items and count >= max_items:
                            return

                # Stream submissions
                for submission in subreddit.stream.submissions(skip_existing=True, pause_after=2):
                    if submission is None:
                        break
                    content = f"{submission.title}\n{submission.selftext}"
                    if self.matches_keywords(content):
                        author_name = str(submission.author) if submission.author else "[deleted]"
                        item = self.format_item(
                            item_id=submission.id,
                            author=author_name,
                            text=content,
                            created_utc=submission.created_utc,
                            url=submission.url,
                            score=submission.score,
                            parent_id=None,
                            subreddit_name=str(submission.subreddit),
                        )
                        self.publish_to_kafka(item)
                        count += 1
                        if max_items and count >= max_items:
                            return

            except (ResponseException, RequestException) as api_err:
                logger.warning(f"Reddit API network error: {api_err}. Pausing 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Unexpected Reddit ingestion exception: {e}. Pausing 5s...")
                time.sleep(5)

    def run_mock_stream(self, max_items: int = 40, interval_sec: float = 0.5):
        """Simulates realistic Reddit community thread activity for testing."""
        logger.info(f"Commencing Mock Reddit Ingestion Generator ({max_items} events)...")
        synthetic_posts = [
            ("PSA: ApexCloud control plane is completely unresponsive in eu-central. Anyone else?", "sysadmin_lead", "sysadmin", -5),
            ("ApexCloud's Kubernetes managed service just halved their node pricing! Incredible move against AWS EKS.", "cloud_architect", "cloudcomputing", 42),
            ("Avoid ApexCloud if you rely on immediate 24/7 phone support. Waited 6 hours during a severity-1 incident.", "devops_guru", "devops", -12),
            ("Just benchmarked ApexCloud object storage vs S3. 15% throughput improvement on large uploads.", "storage_nerd", "technology", 18),
            ("Our company incurred unexpected ApexCloud egress charges of $4,500 due to ambiguous docs.", "finops_expert", "sysadmin", -8),
            ("ApexCloud just released their automated zero-trust networking mesh. The architecture is brilliant.", "security_eng", "technology", 35),
            ("ApexCloud database cluster failover failed miserably today. Total data loss scare.", "db_admin_bob", "sysadmin", -25),
        ]

        for i in range(max_items):
            text, author, sub, score = synthetic_posts[i % len(synthetic_posts)]
            item_id = f"mock_{int(time.time())}_{i:03d}"
            payload = self.format_item(
                item_id=item_id,
                author=author,
                text=f"{text} (Event ID: #{i+1})",
                created_utc=time.time(),
                url=f"https://reddit.com/r/{sub}/comments/{item_id}",
                score=score,
                parent_id=None if i % 2 == 0 else f"parent_{i-1}",
                subreddit_name=sub,
            )
            self.publish_to_kafka(payload)
            time.sleep(interval_sec)

        logger.info(f"Finished emitting {max_items} mock Reddit records.")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reddit PRAW Ingestion Service")
    parser.add_argument("--mock", action="store_true", help="Run with simulated mock stream")
    parser.add_argument("--limit", type=int, default=None, help="Max records to emit")
    parser.add_argument("--interval", type=float, default=0.4, help="Mock stream emit interval")

    args = parser.parse_args()
    worker = RedditIngestionWorker(mock_mode=args.mock)

    if worker.mock_mode or args.mock:
        worker.run_mock_stream(max_items=args.limit or 30, interval_sec=args.interval)
    else:
        worker.run_live_stream(max_items=args.limit)


if __name__ == "__main__":
    main()
