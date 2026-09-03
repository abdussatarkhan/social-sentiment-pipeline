# Automated Executive Sentiment Reports

This directory stores the automatically generated daily and weekly executive brand intelligence reports produced by `scripts/report_generator.py` and scheduled through Apache Airflow (`airflow/dags/daily_sentiment_summary.py`).

## Report Structure

Each generated executive report is compiled into a standalone PDF document using Jinja2 and WeasyPrint. The report covers:

1. **Executive Summary & High-Level KPIs**:
   - **Net Sentiment Score (NSS)**: Calculated as `% Positive - % Negative`, bounded between `[-100, +100]`.
   - **Total Monitored Volume**: Total social posts ingested across Bluesky, Reddit, and historical Twitter benchmarks.
   - **Crisis Incident Frequency**: Number of CUSUM-detected statistical anomalies.
   - **Negative Sentiment Amplifier Count**: High-PageRank accounts actively propagating negative brand sentiment.

2. **Aspect-Level Breakdown**:
   - Fine-grained sentiment scores across key enterprise brand dimensions:
     - **Product Quality & Reliability**
     - **Customer Support & SLA**
     - **Pricing, Billing & Value**
     - **Corporate Ethics & Governance**
     - **UI / UX & Performance**

3. **Platform Comparison**:
   - Cross-platform volume share, sentiment variance, and engagement velocity comparing Bluesky vs. Reddit.

4. **Trending Topic Clusters**:
   - Extracted through BERTopic embeddings and c-TF-IDF keyword extraction, highlighting emerging conversation clusters.

5. **CUSUM Change-Point Incident Log**:
   - Detailed event logs of any 15-minute rolling sentiment deviations triggering alert thresholds ($h = 4.5, k = 0.5$).

6. **Influencer Risk Analysis**:
   - Top 10 social nodes ranked by directed PageRank centrality with high negative sentiment volume.

## Generating Reports Manually

To trigger an on-demand report for a specific date:

```bash
python scripts/report_generator.py --date 2026-09-03 --output reports/executive_sentiment_2026-09-03.pdf
```

To run with dry-run mock data (useful when running without live database connectivity):

```bash
python scripts/report_generator.py --mock --output reports/sample_daily_report.pdf
```
