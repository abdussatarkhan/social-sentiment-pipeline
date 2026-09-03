-- =============================================================================
-- Analytical Queries: Real-Time Social Listening & Brand Sentiment Pipeline
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Hourly Net Sentiment Score (NSS) & Volume Trend (Last 7 Days)
-- -----------------------------------------------------------------------------
SELECT 
    DATE_TRUNC('hour', created_at) AS hour_bucket,
    platform,
    COUNT(*) AS total_posts,
    COUNT(*) FILTER (WHERE sentiment_label = 'positive') AS pos_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'neutral') AS neu_count,
    COUNT(*) FILTER (WHERE sentiment_label = 'negative') AS neg_count,
    ROUND(AVG(sentiment_score), 4) AS mean_sentiment,
    ROUND(
        (COUNT(*) FILTER (WHERE sentiment_label = 'positive') - COUNT(*) FILTER (WHERE sentiment_label = 'negative'))::numeric 
        / NULLIF(COUNT(*), 0) * 100, 
        2
    ) AS net_sentiment_score
FROM sentiment_posts
WHERE created_at >= NOW() - INTERVAL '7 days'
  AND is_spam = FALSE
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

-- -----------------------------------------------------------------------------
-- 2. 15-Minute Rolling Moving Average & CUSUM Input Computation
-- -----------------------------------------------------------------------------
WITH windows AS (
    SELECT 
        DATE_TRUNC('hour', created_at) + 
        INTERVAL '15 minute' * FLOOR(DATE_PART('minute', created_at) / 15.0) AS interval_15min,
        platform,
        COUNT(*) AS post_count,
        AVG(sentiment_score) AS avg_sentiment,
        STDDEV(sentiment_score) AS stddev_sentiment
    FROM sentiment_posts
    WHERE created_at >= NOW() - INTERVAL '24 hours'
      AND is_spam = FALSE
    GROUP BY 1, 2
)
SELECT 
    interval_15min,
    platform,
    post_count,
    ROUND(avg_sentiment, 4) AS current_mean,
    ROUND(COALESCE(stddev_sentiment, 0.0), 4) AS current_stddev,
    ROUND(
        AVG(avg_sentiment) OVER (
            PARTITION BY platform 
            ORDER BY interval_15min 
            ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
        ), 
        4
    ) AS rolling_2hr_baseline
FROM windows
ORDER BY interval_15min DESC;

-- -----------------------------------------------------------------------------
-- 3. Aspect-Based Sentiment Deep-Dive: Pain Points Ranking
-- -----------------------------------------------------------------------------
SELECT 
    pas.aspect,
    COUNT(*) AS total_mentions,
    ROUND(AVG(pas.sentiment_score), 4) AS aspect_sentiment_score,
    ROUND(COUNT(*) FILTER (WHERE pas.sentiment_label = 'negative')::numeric / COUNT(*) * 100, 2) AS negative_pct,
    ROUND(COUNT(*) FILTER (WHERE pas.sentiment_label = 'positive')::numeric / COUNT(*) * 100, 2) AS positive_pct,
    ROUND(
        (COUNT(*) FILTER (WHERE pas.sentiment_label = 'positive') - COUNT(*) FILTER (WHERE pas.sentiment_label = 'negative'))::numeric 
        / COUNT(*) * 100, 
        2
    ) AS aspect_nss
FROM post_aspect_sentiment pas
JOIN sentiment_posts sp ON pas.platform = sp.platform AND pas.post_id = sp.post_id
WHERE sp.created_at >= NOW() - INTERVAL '48 hours'
GROUP BY pas.aspect
ORDER BY aspect_sentiment_score ASC; -- Lowest sentiment score first (critical pain points)

-- -----------------------------------------------------------------------------
-- 4. Cross-Platform Comparison: Bluesky vs Reddit Engagement & Polarity
-- -----------------------------------------------------------------------------
SELECT 
    sp.platform,
    COUNT(DISTINCT sp.author_id) AS unique_active_users,
    COUNT(sp.id) AS total_posts,
    ROUND(AVG(sp.sentiment_score), 4) AS avg_sentiment,
    ROUND(
        (COUNT(*) FILTER (WHERE sp.sentiment_label = 'positive') - COUNT(*) FILTER (WHERE sp.sentiment_label = 'negative'))::numeric 
        / COUNT(*) * 100, 
        2
    ) AS nss,
    ROUND(AVG((rsp.engagement->>'likes')::numeric), 1) AS avg_likes,
    ROUND(AVG((rsp.engagement->>'reposts')::numeric), 1) AS avg_reposts,
    ROUND(AVG((rsp.engagement->>'replies')::numeric), 1) AS avg_replies
FROM sentiment_posts sp
JOIN raw_social_posts rsp ON sp.platform = rsp.platform AND sp.post_id = rsp.post_id
WHERE sp.created_at >= NOW() - INTERVAL '24 hours'
GROUP BY sp.platform;

-- -----------------------------------------------------------------------------
-- 5. Top 10 Detractor Amplifiers (High PageRank + Negative Sentiment Contagion)
-- -----------------------------------------------------------------------------
SELECT 
    author_id,
    author_handle,
    platform,
    ROUND(pagerank_score, 6) AS pagerank,
    in_degree AS incoming_citations,
    total_posts,
    negative_posts,
    ROUND((negative_posts::numeric / NULLIF(total_posts, 0)) * 100, 1) AS neg_post_pct,
    ROUND(amplification_risk_score, 4) AS risk_index
FROM user_influence_scores
WHERE calculated_at = (SELECT MAX(calculated_at) FROM user_influence_scores)
  AND negative_posts > 2
ORDER BY amplification_risk_score DESC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- 6. Viral High-Engagement Negative Posts Requiring Immediate PR Action
-- -----------------------------------------------------------------------------
SELECT 
    sp.post_id,
    sp.platform,
    sp.author_handle,
    LEFT(sp.clean_text, 120) AS snippet,
    sp.sentiment_score,
    sp.confidence,
    COALESCE((rsp.engagement->>'likes')::int, 0) AS likes,
    COALESCE((rsp.engagement->>'reposts')::int, 0) AS reposts,
    COALESCE((rsp.engagement->>'replies')::int, 0) AS replies,
    (COALESCE((rsp.engagement->>'likes')::int, 0) + 
     COALESCE((rsp.engagement->>'reposts')::int, 0) * 2 + 
     COALESCE((rsp.engagement->>'replies')::int, 0) * 3) AS impact_weight,
    sp.created_at
FROM sentiment_posts sp
JOIN raw_social_posts rsp ON sp.platform = rsp.platform AND sp.post_id = rsp.post_id
WHERE sp.sentiment_label = 'negative'
  AND sp.created_at >= NOW() - INTERVAL '24 hours'
ORDER BY impact_weight DESC
LIMIT 15;

-- -----------------------------------------------------------------------------
-- 7. Active Topic Clusters & Dominant Discussion Categories
-- -----------------------------------------------------------------------------
SELECT 
    run_date,
    topic_id,
    topic_label,
    array_to_string(top_keywords[1:5], ', ') AS primary_keywords,
    post_count,
    ROUND(mean_sentiment, 3) AS cluster_sentiment,
    ROUND(positive_ratio * 100, 1) AS pct_pos,
    ROUND(negative_ratio * 100, 1) AS pct_neg
FROM topic_clusters
WHERE run_date = (SELECT MAX(run_date) FROM topic_clusters)
ORDER BY post_count DESC;

-- -----------------------------------------------------------------------------
-- 8. CUSUM Change-Point Incident History & Duration
-- -----------------------------------------------------------------------------
SELECT 
    alert_id,
    detected_at,
    platform,
    ROUND(cusum_statistic, 3) AS cusum_val,
    ROUND(threshold_h, 3) AS threshold,
    ROUND(baseline_sentiment, 3) AS baseline,
    ROUND(observed_sentiment, 3) AS observed,
    affected_volume,
    status,
    incident_summary
FROM crisis_alerts
ORDER BY detected_at DESC
LIMIT 20;

-- -----------------------------------------------------------------------------
-- 9. Ingestion Pipeline Health & Spam Rejection Ratio
-- -----------------------------------------------------------------------------
SELECT 
    DATE_TRUNC('day', created_at) AS date_day,
    platform,
    COUNT(*) AS total_evaluated,
    COUNT(*) FILTER (WHERE is_spam = TRUE) AS spam_rejected,
    ROUND(COUNT(*) FILTER (WHERE is_spam = TRUE)::numeric / COUNT(*) * 100, 2) AS spam_rejection_pct,
    COUNT(*) FILTER (WHERE confidence < 0.60) AS low_confidence_count
FROM sentiment_posts
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1, 2
ORDER BY 1 DESC, 2;
