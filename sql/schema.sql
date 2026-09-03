-- =============================================================================
-- Database Schema: Real-Time Social Listening & Brand Sentiment Pipeline
-- PostgreSQL 15+ compatible
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. Raw Social Posts (Bronze Layer)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_social_posts (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(128) NOT NULL,
    platform VARCHAR(32) NOT NULL CHECK (platform IN ('bluesky', 'reddit', 'twitter_archive')),
    author_id VARCHAR(255) NOT NULL,
    author_handle VARCHAR(255),
    raw_text TEXT NOT NULL,
    url TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    reply_to_id VARCHAR(128),
    reply_to_user VARCHAR(255),
    engagement JSONB DEFAULT '{}'::jsonb,
    raw_metadata JSONB DEFAULT '{}'::jsonb,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_raw_platform_post UNIQUE (platform, post_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_platform_created ON raw_social_posts(platform, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_author ON raw_social_posts(author_id);
CREATE INDEX IF NOT EXISTS idx_raw_reply ON raw_social_posts(reply_to_id) WHERE reply_to_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 2. Enriched Sentiment Posts (Silver Layer)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_posts (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(128) NOT NULL,
    platform VARCHAR(32) NOT NULL CHECK (platform IN ('bluesky', 'reddit', 'twitter_archive')),
    author_id VARCHAR(255) NOT NULL,
    author_handle VARCHAR(255),
    clean_text TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    sentiment_label VARCHAR(16) NOT NULL CHECK (sentiment_label IN ('negative', 'neutral', 'positive')),
    sentiment_score NUMERIC(5, 4) NOT NULL CHECK (sentiment_score >= -1.0000 AND sentiment_score <= 1.0000),
    confidence NUMERIC(5, 4) NOT NULL CHECK (confidence >= 0.0000 AND confidence <= 1.0000),
    is_spam BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_sentiment_platform_post UNIQUE (platform, post_id)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_created ON sentiment_posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_platform_created ON sentiment_posts(platform, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_posts(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_sentiment_score ON sentiment_posts(sentiment_score);
CREATE INDEX IF NOT EXISTS idx_sentiment_author ON sentiment_posts(author_id);

-- -----------------------------------------------------------------------------
-- 3. Aspect-Based Sentiment Scores (Fine-grained ABSA)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS post_aspect_sentiment (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(128) NOT NULL,
    platform VARCHAR(32) NOT NULL,
    aspect VARCHAR(64) NOT NULL CHECK (aspect IN ('general', 'product_quality', 'customer_support', 'pricing_value', 'ui_performance')),
    sentiment_label VARCHAR(16) NOT NULL CHECK (sentiment_label IN ('negative', 'neutral', 'positive')),
    sentiment_score NUMERIC(5, 4) NOT NULL,
    confidence NUMERIC(5, 4) NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_aspect_sentiment_post FOREIGN KEY (platform, post_id) 
        REFERENCES sentiment_posts(platform, post_id) ON DELETE CASCADE,
    CONSTRAINT uq_post_aspect UNIQUE (platform, post_id, aspect)
);

CREATE INDEX IF NOT EXISTS idx_aspect_name_score ON post_aspect_sentiment(aspect, sentiment_score);

-- -----------------------------------------------------------------------------
-- 4. Rolling Time-Series Metrics & Net Sentiment (Gold Layer)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_metrics_timeseries (
    id BIGSERIAL PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    granularity VARCHAR(16) NOT NULL CHECK (granularity IN ('15min', '1hour', '1day')),
    platform VARCHAR(32) NOT NULL,
    total_volume INTEGER NOT NULL DEFAULT 0,
    positive_count INTEGER NOT NULL DEFAULT 0,
    neutral_count INTEGER NOT NULL DEFAULT 0,
    negative_count INTEGER NOT NULL DEFAULT 0,
    mean_sentiment NUMERIC(6, 4) NOT NULL DEFAULT 0.0000,
    sentiment_std_dev NUMERIC(6, 4) NOT NULL DEFAULT 0.0000,
    net_sentiment_score NUMERIC(6, 2) NOT NULL DEFAULT 0.00, -- (Pos - Neg) / Total * 100
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_metrics_window_plat UNIQUE (granularity, platform, window_start)
);

CREATE INDEX IF NOT EXISTS idx_metrics_window ON sentiment_metrics_timeseries(window_start DESC);

-- -----------------------------------------------------------------------------
-- 5. CUSUM Change-Point Crisis Alerts
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crisis_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    platform VARCHAR(32) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    cusum_statistic NUMERIC(8, 4) NOT NULL,
    threshold_h NUMERIC(6, 3) NOT NULL,
    baseline_sentiment NUMERIC(6, 4) NOT NULL,
    observed_sentiment NUMERIC(6, 4) NOT NULL,
    deviation_magnitude NUMERIC(6, 4) NOT NULL,
    affected_volume INTEGER NOT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'triggered' CHECK (status IN ('triggered', 'acknowledged', 'investigating', 'resolved', 'false_positive')),
    incident_summary TEXT,
    alert_payload JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_crisis_detected ON crisis_alerts(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_crisis_status ON crisis_alerts(status);

-- -----------------------------------------------------------------------------
-- 6. Social Graph & PageRank Influence Scores
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_influence_scores (
    id BIGSERIAL PRIMARY KEY,
    author_id VARCHAR(255) NOT NULL,
    platform VARCHAR(32) NOT NULL,
    author_handle VARCHAR(255),
    pagerank_score NUMERIC(10, 8) NOT NULL DEFAULT 0.00000000,
    in_degree INTEGER NOT NULL DEFAULT 0,
    out_degree INTEGER NOT NULL DEFAULT 0,
    total_posts INTEGER NOT NULL DEFAULT 0,
    negative_posts INTEGER NOT NULL DEFAULT 0,
    amplification_risk_score NUMERIC(8, 4) NOT NULL DEFAULT 0.0000, -- PageRank * Negative Volume Ratio
    network_cluster_id INTEGER,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_influence UNIQUE (platform, author_id, calculated_at)
);

CREATE INDEX IF NOT EXISTS idx_influence_risk ON user_influence_scores(amplification_risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_influence_pagerank ON user_influence_scores(pagerank_score DESC);

-- -----------------------------------------------------------------------------
-- 7. BERTopic Discovery Clusters
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topic_clusters (
    id BIGSERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    topic_id INTEGER NOT NULL,
    topic_label VARCHAR(255) NOT NULL,
    top_keywords TEXT[] NOT NULL,
    coherence_score NUMERIC(5, 4),
    post_count INTEGER NOT NULL,
    positive_ratio NUMERIC(5, 4) NOT NULL,
    negative_ratio NUMERIC(5, 4) NOT NULL,
    mean_sentiment NUMERIC(5, 4) NOT NULL,
    representative_samples JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_topic_run UNIQUE (run_date, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_topics_date ON topic_clusters(run_date DESC);

-- -----------------------------------------------------------------------------
-- 8. Real-Time Analytics Views
-- -----------------------------------------------------------------------------

-- View: 24-Hour Net Sentiment KPIs
CREATE OR REPLACE VIEW v_realtime_sentiment_kpis AS
SELECT 
    platform,
    COUNT(*) AS total_posts_24h,
    COUNT(*) FILTER (WHERE sentiment_label = 'positive') AS positive_posts,
    COUNT(*) FILTER (WHERE sentiment_label = 'neutral') AS neutral_posts,
    COUNT(*) FILTER (WHERE sentiment_label = 'negative') AS negative_posts,
    ROUND(AVG(sentiment_score), 4) AS avg_sentiment_score,
    ROUND(
        (COUNT(*) FILTER (WHERE sentiment_label = 'positive') - COUNT(*) FILTER (WHERE sentiment_label = 'negative'))::numeric 
        / NULLIF(COUNT(*), 0) * 100, 
        2
    ) AS net_sentiment_score_nss
FROM sentiment_posts
WHERE created_at >= NOW() - INTERVAL '24 hours' AND is_spam = FALSE
GROUP BY platform;

-- View: Aspect-level Sentiment Summary (24h)
CREATE OR REPLACE VIEW v_aspect_summary AS
SELECT 
    pas.aspect,
    COUNT(*) AS mention_count,
    ROUND(AVG(pas.sentiment_score), 4) AS avg_score,
    ROUND(COUNT(*) FILTER (WHERE pas.sentiment_label = 'positive')::numeric / COUNT(*) * 100, 1) AS pct_positive,
    ROUND(COUNT(*) FILTER (WHERE pas.sentiment_label = 'neutral')::numeric / COUNT(*) * 100, 1) AS pct_neutral,
    ROUND(COUNT(*) FILTER (WHERE pas.sentiment_label = 'negative')::numeric / COUNT(*) * 100, 1) AS pct_negative
FROM post_aspect_sentiment pas
JOIN sentiment_posts sp ON pas.platform = sp.platform AND pas.post_id = sp.post_id
WHERE sp.created_at >= NOW() - INTERVAL '24 hours' AND sp.is_spam = FALSE
GROUP BY pas.aspect
ORDER BY mention_count DESC;

-- View: Top Negative Influencers / Crisis Risk
CREATE OR REPLACE VIEW v_top_negative_amplifiers AS
SELECT 
    author_id,
    author_handle,
    platform,
    pagerank_score,
    total_posts,
    negative_posts,
    amplification_risk_score,
    calculated_at
FROM user_influence_scores
WHERE calculated_at >= NOW() - INTERVAL '48 hours'
ORDER BY amplification_risk_score DESC
LIMIT 50;
