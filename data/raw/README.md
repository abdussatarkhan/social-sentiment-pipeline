# Raw Data Sources & Ingestion Guide

This project ingests and analyzes real-time streaming data from decentralized and traditional social networks alongside historical archive benchmarks.

---

## 1. Bluesky Firehose API (AT Protocol)

Bluesky provides an unauthenticated, public real-time firehose stream via WebSockets over the AT Protocol.

### Connection Details
- **Endpoint**: `wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos`
- **Protocol**: WebSocket streaming CBOR-encoded CAR (Content Addressable aRchive) blocks containing `app.bsky.feed.post` commit records.
- **Authentication**: None required for public firehose ingestion.
- **Script**: `scripts/bluesky_ingestion.py`

### Ingestion Pipeline
1. Connects to the Bluesky relay endpoint with automatic exponential backoff reconnection.
2. Unpacks the CBOR binary stream and filters for `app.bsky.feed.post` records.
3. Filters for brand-relevant keywords, stock tickers, or product names defined in `config/config.yaml`.
4. Enriches post payloads with post ID, author DID/handle, timestamp, text, language metadata, and reply references.
5. Emits JSON records to the Apache Kafka topic `social-raw-stream`.

---

## 2. Reddit API (PRAW - Python Reddit API Wrapper)

The pipeline streams discussions, thread submissions, and comments from high-traffic, relevant subreddits (e.g., `r/technology`, `r/gadgets`, `r/apple`, `r/stocks`, `r/investing`, `r/SysAdmin`).

### Prerequisites & Credentials
1. Register a Reddit developer application at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
2. Select **"script"** application type.
3. Populate credentials in `.env` or `config/config.yaml`:
   ```bash
   REDDIT_CLIENT_ID="your_client_id"
   REDDIT_CLIENT_SECRET="your_client_secret"
   REDDIT_USER_AGENT="SocialSentimentPipeline/1.0 by your_username"
   ```

### Script Execution
- **Script**: `scripts/reddit_ingestion.py`
- Streams comments and submissions in near real-time via `subreddit.stream.comments(skip_existing=True)` and `subreddit.stream.submissions(skip_existing=True)`.
- Normalizes posts into the unified schema (`platform="reddit"`, `post_id`, `author`, `text`, `score`, `url`, `created_at`).
- Publishes to Kafka topic `social-raw-stream`.

---

## 3. Archive.org Twitter Benchmark Dataset (100M+ Tweets)

To benchmark CUSUM crisis detection and train aspect-based sentiment models, the pipeline utilizes the public Twitter Stream Grab archive hosted on Archive.org.

### Dataset Overview
- **Source**: [Archive.org Twitter Stream Election / Crisis Archives](https://archive.org/details/twitterstream)
- **Format**: JSON / JSON Lines compressed in `.tar.gz` and `.bz2`.
- **Benchmark Size**: 100M+ historical tweets spanning viral product launches, service outages, and PR crises.

### Download & Extraction Commands

To download a benchmark chunk (e.g., a specific high-volume week):

```bash
# Example: Download sample day archive
curl -O https://archive.org/download/archiveteam-twitter-stream-2021-01/archiveteam-twitter-stream-2021-01.tar

# Extract specific json.bz2 chunks
tar -xvf archiveteam-twitter-stream-2021-01.tar --wildcards --no-anchored '*.json.bz2'
```

### Ingestion to Kafka / Spark
Use `scripts/utils.py` or the provided data loading helpers to replay historical Twitter archive chunks into the Kafka topic `social-historical-stream` at a calibrated rate (e.g., 500 records/sec) to test CUSUM change-point detection algorithms under realistic burst conditions.

---

## Unified Data Schema

All data streams are transformed into a canonical JSON schema before publishing to Kafka:

```json
{
  "post_id": "bsky_3lbk7q4vksk2b",
  "platform": "bluesky",
  "author_id": "did:plc:ragtjsm2j2vknwk2uhuvrhbu",
  "author_handle": "tech_insider.bsky.social",
  "text": "The latest enterprise cloud update is completely broken, service degraded for 3 hours!",
  "created_at": "2026-09-04T01:45:00Z",
  "reply_to_id": null,
  "reply_to_user": null,
  "engagement": {
    "likes": 42,
    "reposts": 15,
    "replies": 8
  },
  "raw_metadata": {}
}
```
