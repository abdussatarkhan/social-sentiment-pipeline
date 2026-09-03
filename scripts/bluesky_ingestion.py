"""
Bluesky AT Protocol Firehose WebSocket Ingestion
Real-Time Social Listening & Brand Sentiment Pipeline

Connects to the Bluesky firehose relay endpoint:
  wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos
Parses incoming AT Protocol commit events, extracts post records, applies
brand-keyword filtering, normalizes the schema, and streams records into Kafka.
Includes exponential backoff reconnection logic and mock replay capability.
"""

import os
import sys
import json
import time
import asyncio
import argparse
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import websockets
import cbor2

from scripts.utils import setup_logger, load_config, get_kafka_producer, CustomJSONEncoder

logger = setup_logger("bluesky_ingestion")


class BlueskyFirehoseConsumer:
    """Consumes real-time Bluesky posts from the AT Protocol firehose."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, mock_mode: bool = False):
        self.cfg = config or load_config()
        bsky_cfg = self.cfg.get("bluesky", {})
        self.firehose_url = bsky_cfg.get(
            "firehose_url", "wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos"
        )
        self.keywords = [k.lower() for k in bsky_cfg.get("keywords", ["apexcloud", "cloud outage", "downtime"])]
        self.reconnect_delay = bsky_cfg.get("reconnect_delay_sec", 5)
        self.max_reconnect_delay = bsky_cfg.get("max_reconnect_delay_sec", 60)
        self.backoff_factor = bsky_cfg.get("backoff_factor", 2.0)
        self.target_topic = self.cfg["kafka"]["topics"]["raw_stream"]
        self.mock_mode = mock_mode

        # Initialize Kafka Producer
        try:
            self.producer = get_kafka_producer(self.cfg)
            logger.info(f"Connected Kafka producer for topic '{self.target_topic}'")
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}). Running with local stream logging.")
            self.producer = None

    def matches_filter(self, text: str) -> bool:
        """Checks if post contains any target brand keywords or aliases."""
        if not text:
            return False
        lower_text = text.lower()
        return any(kw in lower_text for kw in self.keywords)

    def format_post(
        self,
        post_id: str,
        author_did: str,
        text: str,
        created_at: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Maps Bluesky record to standard pipeline schema."""
        return {
            "post_id": f"bsky_{post_id}",
            "platform": "bluesky",
            "author_id": author_did,
            "author_handle": f"{author_did[:16]}.bsky.social",
            "text": text,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "url": f"https://bsky.app/profile/{author_did}/post/{post_id}",
            "reply_to_id": reply_to,
            "reply_to_user": None,
            "engagement": {"likes": 0, "reposts": 0, "replies": 0},
            "raw_metadata": {"source": "firehose_commit"},
        }

    def publish_to_kafka(self, post_payload: Dict[str, Any]) -> None:
        """Emits normalized record to Kafka."""
        if self.producer:
            try:
                self.producer.send(
                    topic=self.target_topic,
                    key=post_payload["post_id"],
                    value=post_payload,
                )
            except Exception as e:
                logger.error(f"Failed to publish Bluesky message to Kafka: {e}")
        else:
            logger.info(f"[MOCK KAFKA OUT] {post_payload['post_id']} -> {post_payload['text'][:60]}...")

    async def run_firehose_stream(self, max_messages: Optional[int] = None):
        """Connects to the firehose WebSocket and processes commit records."""
        current_delay = self.reconnect_delay
        message_count = 0

        while True:
            try:
                logger.info(f"Connecting to Bluesky Firehose: {self.firehose_url}")
                async with websockets.connect(
                    self.firehose_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    logger.info("Successfully connected to Bluesky firehose. Streaming records...")
                    current_delay = self.reconnect_delay  # Reset backoff upon successful connection

                    while True:
                        raw_bytes = await ws.recv()
                        if not isinstance(raw_bytes, bytes):
                            continue

                        # Firehose frame parsing: AT Protocol sends header CBOR + payload CBOR
                        try:
                            # Parse header
                            header, offset = cbor2.loads(raw_bytes), len(cbor2.dumps(cbor2.loads(raw_bytes)))
                            # Parse body
                            body = cbor2.loads(raw_bytes[offset:])
                        except Exception:
                            # Fallback single decode
                            try:
                                body = cbor2.loads(raw_bytes)
                            except Exception:
                                continue

                        if not isinstance(body, dict):
                            continue

                        # Check if message is a repo commit with ops
                        ops = body.get("ops", [])
                        repo_did = body.get("repo", "unknown_did")

                        for op in ops:
                            action = op.get("action")
                            path = op.get("path", "")
                            # Only capture newly created posts
                            if action == "create" and path.startswith("app.bsky.feed.post"):
                                rkey = path.split("/")[-1]
                                # Look for record data in body
                                record = op.get("record", {})
                                text = record.get("text", "")

                                if self.matches_filter(text):
                                    reply_obj = record.get("reply", {})
                                    reply_parent_uri = reply_obj.get("parent", {}).get("uri")
                                    reply_id = reply_parent_uri.split("/")[-1] if reply_parent_uri else None

                                    formatted = self.format_post(
                                        post_id=rkey,
                                        author_did=repo_did,
                                        text=text,
                                        created_at=record.get("createdAt"),
                                        reply_to=reply_id,
                                    )
                                    self.publish_to_kafka(formatted)
                                    message_count += 1

                                    if max_messages and message_count >= max_messages:
                                        logger.info(f"Reached message limit ({max_messages}). Halting.")
                                        return

            except (websockets.ConnectionClosed, websockets.WebSocketException, OSError) as conn_err:
                logger.warning(
                    f"Bluesky WebSocket disconnected: {conn_err}. Reconnecting in {current_delay}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * self.backoff_factor, self.max_reconnect_delay)
            except Exception as e:
                logger.error(f"Unexpected error in Bluesky consumer: {e}. Retrying in {current_delay}s...")
                await asyncio.sleep(current_delay)

    async def run_mock_stream(self, max_messages: int = 50, interval_sec: float = 0.5):
        """Simulates realistic firehose streaming for testing and environments without public internet."""
        logger.info(f"Commencing Mock Bluesky Firehose generator ({max_messages} events)...")
        synthetic_stream = [
            ("ApexCloud infrastructure in us-east-1 is throwing 504 errors on all ingress controllers.", "did:plc:mock_dev_01"),
            ("Loving the new developer CLI released by @ApexCloud. Deployment time slashed by half!", "did:plc:mock_dev_02"),
            ("Can customer support at ApexCloud please review ticket #8942? Our production DB is locked.", "did:plc:mock_dev_03"),
            ("Thinking about moving our workloads to ApexCloud. How is the cost comparison with AWS ECS?", "did:plc:mock_dev_04"),
            ("Another cloud outage reported for ApexCloud. SLA breach compensation needed ASAP.", "did:plc:mock_dev_05"),
            ("ApexCloud database snapshots restore feature is blazingly fast. Very impressed!", "did:plc:mock_dev_06"),
            ("ApexCloud billing dashboard is double counting compute credits. Massive discrepancy.", "did:plc:mock_dev_07"),
            ("Our team just finished migrating 40 microservices to ApexCloud without a hitch. 10/10 experience.", "did:plc:mock_dev_08"),
        ]

        for i in range(max_messages):
            text_template, author_did = synthetic_stream[i % len(synthetic_stream)]
            post_id = f"bsky_sim_{int(time.time())}_{i:04d}"
            formatted = self.format_post(
                post_id=post_id,
                author_did=author_did,
                text=f"{text_template} [Seq: {i}]",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self.publish_to_kafka(formatted)
            await asyncio.sleep(interval_sec)

        logger.info(f"Mock Bluesky stream completed. Emitted {max_messages} messages.")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bluesky Firehose Ingestion Service")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulation mode")
    parser.add_argument("--limit", type=int, default=None, help="Maximum messages to ingest")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval for mock streaming")

    args = parser.parse_args()
    consumer = BlueskyFirehoseConsumer(mock_mode=args.mock)

    if args.mock:
        asyncio.run(consumer.run_mock_stream(max_messages=args.limit or 30, interval_sec=args.interval))
    else:
        asyncio.run(consumer.run_firehose_stream(max_messages=args.limit))


if __name__ == "__main__":
    main()
