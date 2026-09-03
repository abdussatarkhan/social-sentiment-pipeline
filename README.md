# Real-Time Social Listening & Brand Sentiment Pipeline

[![Architecture](https://img.shields.io/badge/Architecture-Kafka%20%7C%20Postgres%20%7C%20Elasticsearch%20%7C%20Airflow-blue.svg)](file:///docker-compose.yml)
[![NLP](https://img.shields.io/badge/NLP-DistilBERT%20ABSA%20%7C%20BERTopic-green.svg)](file:///scripts/sentiment_model.py)
[![Statistical-Detection](https://img.shields.io/badge/Change--Point-CUSUM%20Algorithm-orange.svg)](file:///scripts/crisis_detection.py)
[![Graph-Analysis](https://img.shields.io/badge/Graph-PageRank%20Influence-purple.svg)](file:///scripts/influence_analysis.py)
[![Python-Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](file:///requirements.txt)

A production-grade, enterprise real-time social listening and brand reputation intelligence platform. The pipeline streams conversations from decentralized protocols (**Bluesky AT Protocol Firehose**) and community forums (**Reddit API / PRAW**), runs high-throughput batched **Aspect-Based Sentiment Analysis (ABSA)** using a fine-tuned **DistilBERT** model, detects PR crises via **CUSUM sequential change-point detection**, models social network influence using **PageRank**, extracts emerging themes via **BERTopic**, and automates daily executive PDF briefings via **Apache Airflow**.

---

## Architecture Overview

```
                          ┌────────────────────────────┐
                          │    Bluesky AT Firehose     │ (WebSocket / CBOR)
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │       Reddit (PRAW)        │ (Submissions & Comments)
                          └─────────────┬──────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Apache Kafka (Raw Stream)   │
                        │      'social-raw-stream'      │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                        STREAM PROCESSING WORKER                        │
    │  1. Preprocessing (Emoji demojization, regex cleaning, spam filtering) │
    │  2. DistilBERT ABSA Inference (Multi-head: Quality, Support, Pricing)  │
    │  3. Sentiment Polarity & Confidence Scoring ([-1.0, +1.0])             │
    └───────┬───────────────────────────┬────────────────────────────┬───────┘
            │                           │                            │
            ▼                           ▼                            ▼
┌───────────────────────┐   ┌───────────────────────┐    ┌───────────────────────┐
│  PostgreSQL (Bronze,  │   │  Elasticsearch        │    │  Kafka Outbound       │
│  Silver, Gold Layers) │   │  (Faceted Search)     │    │  'social-enriched'    │
└───────────┬───────────┘   └───────────┬───────────┘    └───────────────────────┘
            │                           │
            ├───────────────────────────┤
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│ CUSUM Change-Point    │   │  Grafana Command      │
│ Crisis Detector       │   │  Center Dashboard     │
│ (15-min rolling S_t)  │   │  (Gauges, Trends)     │
└───────────┬───────────┘   └───────────────────────┘
            │
            ▼ (Triggers Alert)
┌───────────────────────┐
│ Slack / Kafka / Webhook│
└───────────────────────┘

                 ORCHESTRATION & EXECUTIVE INTELLIGENCE (AIRFLOW)
┌────────────────────────────────────────────────────────────────────────────┐
│ • Daily Sentiment Rollup: 24h Aggregates -> BERTopic Clustering ->         │
│   NetworkX PageRank Influence -> Jinja2/WeasyPrint Executive PDF Digest    │
│ • Weekly Model Retraining: Human-in-the-loop auditing -> DistilBERT ABSA   │
│   Fine-Tuning -> Macro F1 Gating -> Checkpoint Promotion                   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 1. Multi-Source Streaming Ingestion
- **Bluesky AT Protocol Firehose (`wss://bsky.network/xrpc/...`)**: Unauthenticated, real-time WebSocket client unpacking CBOR-encoded commit records (`app.bsky.feed.post`) with exponential backoff and automatic session resumption.
- **Reddit Ingestion (PRAW)**: Live stream consumer monitoring high-impact developer and technology communities (`r/technology`, `r/sysadmin`, `r/aws`, `r/cloudcomputing`, `r/stocks`).
- **Archive.org Historical Benchmark**: Support for replaying 100M+ tweet archive datasets to benchmark model latency, throughput, and crisis recall.

### 2. High-Performance Text Normalization & Spam Defense
- **Demojization**: Semantic translation of emoji signals (e.g. `:fire:` $\rightarrow$ `"fire"`, `:angry_face:` $\rightarrow$ `"angry face"`), capturing critical emotional tone often missed by raw text tokenizers.
- **Bot & Spam Filtering**: Heuristics detecting character flooding, hyper-dense URL spam, shouting uppercase ratios, and crypto airdrop patterns.
- **Language Detection**: High-speed filtering targeting brand-relevant languages.

### 3. Aspect-Based Sentiment Analysis (DistilBERT ABSA)
- **Multi-Task Architecture**: Built on `distilbert-base-uncased` with parallel task-specific classification heads:
  1. `general`
  2. `product_quality`
  3. `customer_support`
  4. `pricing_value`
  5. `ui_performance`
- **Granular Polarity**: Prevents sentiment bleed when users review multiple features with opposing polarities (e.g., praising infrastructure performance while criticizing support ticket SLA).
- **Inference Optimization**: Batched tensor evaluations with CPU/CUDA hardware auto-discovery.

### 4. CUSUM Change-Point Statistical Crisis Detection
- **Sequential Change-Point Detection**: Implements the Cumulative Sum (CUSUM) control chart on 15-minute rolling sentiment averages:
  $$S_t^- = \max(0, S_{t-1}^- + (\mu_0 - X_t - k))$$
- **Statistical Calibration**: Calibrated slack parameter ($k = 0.5\sigma$) and decision boundary ($h = 4.5\sigma$) to guarantee an in-control Average Run Length ($ARL_0 \approx 500$ intervals) while detecting true viral outages within **15–30 minutes**.
- **Automated Incident Dispatch**: Automatically dispatches alerts to Slack webhooks, logging targets, and Kafka topic `social-crisis-alerts`.

### 5. Network Influence & Detractor Amplification
- **Directed Social Graph**: Built using NetworkX from user-to-user interactions (replies, mentions, quotes, and parent-child thread relationships).
- **PageRank Centrality**: Evaluates structural authority to differentiate high-impact industry leaders from low-influence social accounts.
- **Amplification Risk Index**:
  $$\text{Risk}(u) = \text{PageRank}(u) \cdot \left(1 + \log(1 + \text{NegPosts}(u))\right) \cdot \left(\frac{\text{NegPosts}(u)}{\text{TotalPosts}(u)}\right)^{1.2} \times 1000$$
  Identifies the top 10 detractor nodes driving negative sentiment contagion.

### 6. BERTopic Trending Topic Discovery
- Extracts dynamic conversational themes from daily post corpora using SentenceTransformer embeddings (`all-MiniLM-L6-v2`), dimensionality reduction, and class-based TF-IDF (c-TF-IDF).
- Attributes cluster-level sentiment (positive ratio, negative ratio, mean score) to identify emerging technical issues and feature praise.

### 7. Executive Automated Reporting & Dashboards
- **Daily Executive PDF Reports**: Formatted using Jinja2 templates and compiled via WeasyPrint, featuring high-resolution embedded Matplotlib KPI charts, Net Sentiment Score (NSS), and influencer risk tables.
- **Real-Time Grafana Command Center**: Ready-to-import Grafana dashboard with real-time sentiment gauges, volume time series, platform comparisons, and live detractor tables.

---

## Directory Structure

```
social-sentiment-pipeline/
├── .gitignore                          # Standard gitignore for data science and secrets
├── README.md                           # Comprehensive architecture and operation guide
├── requirements.txt                    # Project Python dependencies
├── docker-compose.yml                  # Full-stack: Kafka, Postgres, Elasticsearch, Airflow, Grafana
├── config/
│   └── config.yaml                     # Centralized configuration (Kafka, DB, models, thresholds)
├── data/
│   ├── raw/
│   │   └── README.md                   # Data source guide (Bluesky, Reddit, Twitter Archive)
│   └── processed/
│       └── .gitkeep                    # Cleaned and partitioned parquet datasets
├── scripts/
│   ├── utils.py                        # DB pool, ES client, Kafka helpers, structured logging
│   ├── preprocessing.py                # Regex cleaner, emoji translator, spam heuristic filter
│   ├── sentiment_model.py              # DistilBERT ABSA PyTorch model, trainer, inference engine
│   ├── bluesky_ingestion.py            # Bluesky firehose WebSocket consumer with auto-reconnect
│   ├── reddit_ingestion.py             # Reddit PRAW stream consumer for targeted subreddits
│   ├── kafka_consumer.py               # Kafka consumer -> DistilBERT -> Postgres + Elasticsearch
│   ├── crisis_detection.py             # CUSUM change-point detector & threshold calibration
│   ├── influence_analysis.py           # NetworkX interaction graph, PageRank & amplifier risk
│   ├── topic_modeling.py               # BERTopic cluster discovery & c-TF-IDF keyword extraction
│   └── report_generator.py             # Jinja2 + WeasyPrint executive PDF briefing generator
├── airflow/
│   └── dags/
│       ├── daily_sentiment_summary.py  # Daily Airflow DAG: Aggregates -> BERTopic -> Graph -> PDF
│       └── model_retraining.py         # Weekly Airflow DAG: Audit check -> Fine-tune -> F1 Gate
├── templates/
│   └── daily_report.html               # Professional Jinja2 HTML template for executive PDF
├── notebooks/
│   ├── 01_data_exploration.py          # EDA on Bluesky, Reddit, and Twitter streams (# %% format)
│   ├── 02_model_training.py            # DistilBERT ABSA training walkthrough & confusion matrix
│   ├── 03_crisis_detection.py          # CUSUM change-point simulation & ARL parameter calibration
│   └── 04_influence_analysis.py        # Social interaction network, PageRank & community detection
├── grafana/
│   └── dashboards/
│       └── sentiment_dashboard.json    # Complete Grafana dashboard configuration
├── sql/
│   ├── schema.sql                      # PostgreSQL schema: bronze/silver/gold tables, views, indexes
│   └── queries.sql                     # Analytical queries: NSS, CUSUM inputs, amplifier ranking
├── models/
│   └── .gitkeep                        # Checkpoints for fine-tuned DistilBERT models
├── images/
│   └── .gitkeep                        # Embedded diagrams, charts, and visual assets
└── reports/
    ├── README.md                       # Documentation for executive briefing distribution
    └── .gitkeep                        # Destination for compiled PDF reports
```

---

## Quickstart & Deployment

### 1. Prerequisites
- Docker & Docker Compose (v2.0+)
- Python 3.10+
- (Optional) NVIDIA GPU for accelerated transformer inference

### 2. Environment Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/satarabdus692-bot/social-sentiment-pipeline.git
cd social-sentiment-pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Full Infrastructure Stack via Docker Compose
Spin up the Kafka cluster, PostgreSQL, Elasticsearch, Apache Airflow, and Grafana:

```bash
docker-compose up -d
```

Verify service health:
- **Apache Airflow**: [http://localhost:8080](http://localhost:8080) (Credentials: `admin` / `admin_password`)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `admin`)
- **Elasticsearch**: [http://localhost:9200](http://localhost:9200)
- **PostgreSQL**: `localhost:5432` (`sentiment_db`)

Database schemas and views are automatically initialized from [sql/schema.sql](file:///sql/schema.sql).

---

## Running Pipeline Components

### 1. Ingestion Streams
Stream real-time posts into Kafka:

```bash
# Live Bluesky Firehose (or --mock for simulation)
python scripts/bluesky_ingestion.py --mock --limit 100

# Live Reddit Subreddit Stream (or --mock for simulation)
python scripts/reddit_ingestion.py --mock --limit 100
```

### 2. Stream Consumer & DistilBERT NLP Worker
Consume raw events from Kafka, execute preprocessing, run DistilBERT aspect inference, and persist to PostgreSQL and Elasticsearch:

```bash
# Run streaming consumer
python scripts/kafka_consumer.py

# Or run standalone dry-run simulation
python scripts/kafka_consumer.py --mock --count 20
```

### 3. Statistical Crisis Change-Point Detection (CUSUM)
Evaluate the 15-minute rolling sentiment average against calibrated threshold $h$:

```bash
# Evaluate latest rolling window
python scripts/crisis_detection.py --run-once

# Calibrate h and k thresholds via Monte Carlo simulation
python scripts/crisis_detection.py --calibrate

# Simulate a full crisis shock lifecycle
python scripts/crisis_detection.py --simulate
```

### 4. Network Influence & Amplifier Analysis
Construct the interaction graph and rank detractor accounts:

```bash
# Calculate PageRank and Amplification Risk Scores
python scripts/influence_analysis.py --mock --top 10 --export-graph reports/network_graph.graphml
```

### 5. BERTopic Trending Topic Discovery
Identify emerging topics from today's discussion corpus:

```bash
python scripts/topic_modeling.py --mock
```

### 6. Executive Report Generation
Compile publication-ready daily executive briefing (PDF & HTML):

```bash
# Generate report for today
python scripts/report_generator.py --mock --output reports/executive_sentiment_briefing.pdf
```

---

## Interactive Notebooks

All notebooks are structured as clean, reproducible Python scripts utilizing the `# %%` cell standard (compatible with VS Code Interactive Window, JupyterLab, and PyCharm):

1. [notebooks/01_data_exploration.py](file:///notebooks/01_data_exploration.py): Exploratory data analysis across Bluesky, Reddit, and Twitter benchmark datasets.
2. [notebooks/02_model_training.py](file:///notebooks/02_model_training.py): DistilBERT ABSA fine-tuning, loss curves, confusion matrices, and sarcasm/negation stress tests.
3. [notebooks/03_crisis_detection.py](file:///notebooks/03_crisis_detection.py): Sequential CUSUM change-point modeling, Average Run Length ($ARL$) calibration, and false alarm rate control.
4. [notebooks/04_influence_analysis.py](file:///notebooks/04_influence_analysis.py): Directed social graph topology, community detection, PageRank vs In-Degree centrality, and detractor risk ranking.

---

## Apache Airflow Workflows

1. **`daily_sentiment_summary` DAG**:
   - **Schedule**: `0 6 * * *` (Daily at 06:00 UTC)
   - **Pipeline**: Database readiness $\rightarrow$ 24h KPI aggregation $\rightarrow$ BERTopic clustering $\rightarrow$ PageRank influence calculation $\rightarrow$ Jinja2/WeasyPrint PDF report compilation $\rightarrow$ Slack executive digest dispatch.

2. **`model_retraining` DAG**:
   - **Schedule**: `0 2 * * 0` (Weekly on Sunday at 02:00 UTC)
   - **Pipeline**: New labeled sample audit $\rightarrow$ Data validation $\rightarrow$ DistilBERT fine-tuning $\rightarrow$ Macro F1 evaluation on gold benchmark $\rightarrow$ Automated model promotion gating.

---

## Database & Analytics Reference

Key PostgreSQL analytical queries are documented in [sql/queries.sql](file:///sql/queries.sql):
- **Net Sentiment Score (NSS)**:
  $$\text{NSS} = \frac{\text{Positive Posts} - \text{Negative Posts}}{\text{Total Posts}} \times 100 \quad \in [-100, +100]$$
- **15-Minute CUSUM Rolling Inputs**: High-performance window calculations tracking mean and standard deviation.
- **Detractor Risk Index**: Identifies accounts combining high network authority with viral negative brand sentiment.

---

## License & Contributing
Licensed under the [Apache-2.0 License](LICENSE). Built for enterprise brand analytics and real-time NLP stream processing.
