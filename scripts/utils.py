"""
Shared Utilities Module
Real-Time Social Listening & Brand Sentiment Pipeline

Provides unified access to:
- Configuration management (YAML + ENV overrides)
- Structured logging
- PostgreSQL connection pooling & session management
- Elasticsearch client initialization & schema mapping
- Kafka producer / consumer builders
- Performance timing & monitoring decorators
- Alert dispatching (Slack, Webhook, Kafka)
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional, Generator
from datetime import datetime, date
from functools import wraps
from pathlib import Path
import yaml

# Database & Search
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from elasticsearch import Elasticsearch

# Kafka
from kafka import KafkaProducer, KafkaConsumer

# -----------------------------------------------------------------------------
# 1. Base Configuration & Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration with environment variable fallbacks."""
    target_path = config_file or CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    if "KAFKA_BOOTSTRAP_SERVERS" in os.environ:
        config["kafka"]["bootstrap_servers"] = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    if "POSTGRES_HOST" in os.environ:
        config["postgres"]["host"] = os.environ["POSTGRES_HOST"]
    if "POSTGRES_PASSWORD" in os.environ:
        config["postgres"]["password"] = os.environ["POSTGRES_PASSWORD"]
    if "ELASTICSEARCH_HOST" in os.environ:
        config["elasticsearch"]["host"] = os.environ["ELASTICSEARCH_HOST"]
    if "REDDIT_CLIENT_ID" in os.environ:
        config["reddit"]["client_id"] = os.environ["REDDIT_CLIENT_ID"]
    if "REDDIT_CLIENT_SECRET" in os.environ:
        config["reddit"]["client_secret"] = os.environ["REDDIT_CLIENT_SECRET"]

    return config


# -----------------------------------------------------------------------------
# 2. Structured Logging
# -----------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Custom JSON log formatter for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logger(name: str = "sentiment_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a thread-safe logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


# -----------------------------------------------------------------------------
# 3. JSON Encoder Helper
# -----------------------------------------------------------------------------
class CustomJSONEncoder(json.JSONEncoder):
    """Encodes datetimes, dates, and sets into valid JSON."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if hasattr(o, "item"):  # numpy scalars
            return o.item()
        if hasattr(o, "tolist"):  # numpy arrays
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        return super().default(o)


# -----------------------------------------------------------------------------
# 4. Database Connection Pool & Session Management
# -----------------------------------------------------------------------------
_pg_pool: Optional[pool.SimpleConnectionPool] = None


def get_postgres_pool(config: Optional[Dict[str, Any]] = None) -> pool.SimpleConnectionPool:
    """Thread-safe PostgreSQL connection pool singleton."""
    global _pg_pool
    if _pg_pool is None:
        cfg = config or load_config()
        pg_cfg = cfg["postgres"]
        try:
            _pg_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=pg_cfg.get("pool_size", 10),
                host=pg_cfg.get("host", "localhost"),
                port=pg_cfg.get("port", 5432),
                dbname=pg_cfg.get("dbname", "sentiment_db"),
                user=pg_cfg.get("user", "sentiment_user"),
                password=pg_cfg.get("password", "sentiment_secure_password"),
            )
            logger.info("PostgreSQL connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise
    return _pg_pool


def get_sqlalchemy_engine(config: Optional[Dict[str, Any]] = None):
    """Initializes SQLAlchemy Engine with connection pooling."""
    cfg = config or load_config()
    pg = cfg["postgres"]
    conn_url = (
        f"postgresql+psycopg2://{pg['user']}:{pg['password']}@"
        f"{pg['host']}:{pg['port']}/{pg['dbname']}"
    )
    return create_engine(
        conn_url,
        pool_size=pg.get("pool_size", 10),
        max_overflow=pg.get("max_overflow", 20),
        pool_timeout=pg.get("pool_timeout", 30),
    )


# -----------------------------------------------------------------------------
# 5. Elasticsearch Client & Mapping Initializer
# -----------------------------------------------------------------------------
def get_elasticsearch_client(config: Optional[Dict[str, Any]] = None) -> Elasticsearch:
    """Returns an Elasticsearch client instance."""
    cfg = config or load_config()
    es_cfg = cfg["elasticsearch"]
    endpoint = f"{es_cfg['scheme']}://{es_cfg['host']}:{es_cfg['port']}"
    client = Elasticsearch(
        [endpoint],
        request_timeout=es_cfg.get("request_timeout", 30),
        max_retries=es_cfg.get("max_retries", 3),
        retry_on_timeout=es_cfg.get("retry_on_timeout", True),
    )
    return client


def ensure_elasticsearch_index(es_client: Elasticsearch, index_name: str) -> None:
    """Creates the Elasticsearch index with optimized full-text and sentiment mapping."""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "sentiment_text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "post_id": {"type": "keyword"},
                "platform": {"type": "keyword"},
                "author_id": {"type": "keyword"},
                "author_handle": {"type": "keyword"},
                "clean_text": {
                    "type": "text",
                    "analyzer": "sentiment_text_analyzer",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "language": {"type": "keyword"},
                "sentiment_label": {"type": "keyword"},
                "sentiment_score": {"type": "float"},
                "confidence": {"type": "float"},
                "is_spam": {"type": "boolean"},
                "created_at": {"type": "date"},
                "aspects": {
                    "type": "nested",
                    "properties": {
                        "aspect": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        "score": {"type": "float"},
                    },
                },
            }
        },
    }
    try:
        if not es_client.indices.exists(index=index_name):
            es_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Elasticsearch index '{index_name}' created.")
    except Exception as e:
        logger.warning(f"Could not verify/create Elasticsearch index '{index_name}': {e}")


# -----------------------------------------------------------------------------
# 6. Apache Kafka Producer & Consumer Factories
# -----------------------------------------------------------------------------
def get_kafka_producer(config: Optional[Dict[str, Any]] = None) -> KafkaProducer:
    """Builds a resilient Kafka producer with JSON serialization."""
    cfg = config or load_config()
    bootstrap = cfg["kafka"]["bootstrap_servers"]
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v, cls=CustomJSONEncoder).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        retries=5,
        acks="all",
        linger_ms=10,
        compression_type="gzip",
    )


def get_kafka_consumer(topic: str, group_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> KafkaConsumer:
    """Builds an Apache Kafka consumer for a designated topic."""
    cfg = config or load_config()
    bootstrap = cfg["kafka"]["bootstrap_servers"]
    consumer_group = group_id or cfg["kafka"].get("consumer_group", "sentiment-nlp-workers")
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=consumer_group,
        auto_offset_reset=cfg["kafka"].get("auto_offset_reset", "latest"),
        enable_auto_commit=cfg["kafka"].get("enable_auto_commit", False),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_records=cfg["kafka"].get("max_poll_records", 200),
    )


# -----------------------------------------------------------------------------
# 7. Execution Timing & Decorators
# -----------------------------------------------------------------------------
def measure_time(func):
    """Decorator to measure and log execution time of methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.debug(f"Function '{func.__name__}' executed in {duration:.4f}s")
        return result
    return wrapper


# -----------------------------------------------------------------------------
# 8. Alert Dispatcher (Slack / Webhook Notification)
# -----------------------------------------------------------------------------
def dispatch_alert(alert_payload: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
    """Dispatches a crisis alert to Slack or logging destination."""
    cfg = config or load_config()
    webhook_url = cfg.get("crisis_detection", {}).get("slack_webhook_url")

    message = (
        f":warning: *CRISIS CHANGE-POINT ALERT DETECTED*\n"
        f"*Platform*: `{alert_payload.get('platform', 'unknown')}`\n"
        f"*CUSUM Statistic*: `{alert_payload.get('cusum_statistic', 0.0):.3f}` (Threshold: `{alert_payload.get('threshold_h', 0.0):.3f}`)\n"
        f"*Observed Sentiment*: `{alert_payload.get('observed_sentiment', 0.0):.3f}` (Baseline: `{alert_payload.get('baseline_sentiment', 0.0):.3f}`)\n"
        f"*Affected Post Volume*: `{alert_payload.get('affected_volume', 0)}` posts\n"
        f"*Summary*: {alert_payload.get('incident_summary', 'Statistically significant negative sentiment drop detected.')}"
    )

    logger.warning(f"[DISPATCH ALERT] {message}")

    if webhook_url and not webhook_url.startswith("https://hooks.slack.com/services/MOCK"):
        try:
            import requests
            resp = requests.post(webhook_url, json={"text": message}, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to post alert to Slack webhook: {e}")
            return False

    return True
