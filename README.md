# Real-Time Social Listening & Crisis Detection Pipeline

[![DistilBERT](https://img.shields.io/badge/NLP-DistilBERT-FFA800?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/) [![Apache Airflow](https://img.shields.io/badge/Airflow-ETL_DAGs-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)](https://airflow.apache.org/) [![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **An asynchronous stream processing pipeline capturing Bluesky and Reddit firehoses, running DistilBERT transformer sentiment inference, and detecting brand perception anomalies via Cumulative Sum (CUSUM) statistical control charts.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Firehose[Bluesky & Reddit Social Firehose] --> Ingest[FastAPI Ingestion Worker]
    Ingest --> Transformer[HuggingFace DistilBERT Sentiment Classifier]
    Transformer --> CUSUM[Statistical CUSUM Crisis Drift Detector]
    CUSUM --> Grafana[Grafana Real-Time Alert Dashboard]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `DistilBERT` `Apache Airflow` `Grafana` `Docker` `NLP`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![social-sentiment-pipeline preview](images/sentiment_timeseries_cusum.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/social-sentiment-pipeline.git
cd social-sentiment-pipeline
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.
