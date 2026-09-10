# Real-Time Social Listening & Crisis Detection Pipeline

[![CI](https://github.com/abdussatarkhan/social-sentiment-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/social-sentiment-pipeline/actions)
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

<div align="center">

[![Daily Streak](https://img.shields.io/badge/Daily%20Streak-Active%20%F0%9F%94%A5-brightgreen?style=flat-square&logo=github)](https://github.com/abdussatarkhan)
[![Master Portfolio](https://img.shields.io/badge/Portfolio-50%2B%20Enterprise%20Projects-0e75b6?style=flat-square&logo=github)](https://github.com/abdussatarkhan/abdussatarkhan)
[![Author: Abdussatar](https://img.shields.io/badge/Author-Abdussatar-24292e?style=flat-square&logo=github)](https://github.com/abdussatarkhan)

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

## 🗺️ Roadmap & Upcoming Features

- [x] DistilBERT transformer sentiment classification
- [x] Cumulative Sum (CUSUM) crisis anomaly detection
- [ ] Multi-lingual sentiment inference via XLM-RoBERTa (Urdu, Arabic, Spanish)
- [ ] Apache Airflow production DAG orchestration
- [ ] Grafana live dashboard export with Telegram webhook alerts

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.


---

<div align="center">

### 👨‍💻 Maintained by [Abdussatar (@abdussatarkhan)](https://github.com/abdussatarkhan)
Part of the **[Master Enterprise Data Analytics & AI Portfolio](https://github.com/abdussatarkhan/abdussatarkhan)**.

⭐ If you find this repository valuable, consider dropping a star! ⭐

</div>
