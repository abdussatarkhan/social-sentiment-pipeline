"""
CUSUM Change-Point Detection for Real-Time Social Sentiment Crises
Real-Time Social Listening & Brand Sentiment Pipeline

Statistical Foundations:
- Cumulative Sum (CUSUM) control chart detects abrupt negative shifts in brand sentiment.
- Computes rolling 15-minute sentiment average: X_t
- In-control baseline mean: mu_0, baseline std: sigma
- Slack parameter: k = delta / 2 (typically 0.5 * sigma)
- Decision threshold: h (calibrated for desired Average Run Length ARL_0)
- Downward deviation statistic:
    S_t^- = max(0, S_{t-1}^- + (mu_0 - X_t - k))
- Triggers alert when S_t^- > h, dispatching incident events to PostgreSQL and Slack.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from scripts.utils import (
    setup_logger,
    load_config,
    get_postgres_pool,
    dispatch_alert,
    get_kafka_producer,
    measure_time,
)

logger = setup_logger("crisis_detection")


class CUSUMCrisisDetector:
    """Detects statistically significant negative sentiment change-points."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or load_config()
        crisis_cfg = self.cfg.get("crisis_detection", {})

        self.window_minutes = crisis_cfg.get("window_minutes", 15)
        self.min_posts_threshold = crisis_cfg.get("min_posts_threshold", 10)
        self.baseline_mean = crisis_cfg.get("baseline_sentiment_target", 0.35)
        self.baseline_std = crisis_cfg.get("baseline_std_dev", 0.20)
        self.k = crisis_cfg.get("slack_allowance_k", 0.50) * self.baseline_std
        self.h = crisis_cfg.get("decision_threshold_h", 4.50) * self.baseline_std
        self.alert_cooldown_min = crisis_cfg.get("alert_cooldown_minutes", 60)

        self.s_negative = 0.0  # Cumulative negative statistic
        self.last_alert_time: Optional[datetime] = None

        # DB Pool & Kafka Producer
        self.pg_pool = None
        self.producer = None
        try:
            self.pg_pool = get_postgres_pool(self.cfg)
        except Exception:
            logger.warning("PostgreSQL unavailable. CUSUM detector will operate in standalone mode.")

        try:
            self.producer = get_kafka_producer(self.cfg)
        except Exception:
            pass

    def update_cusum(self, observed_mean: float) -> Tuple[float, bool]:
        """
        Updates the downward CUSUM statistic:
        S_t^- = max(0, S_{t-1}^- + (mu_0 - X_t - k))
        Returns (S_t^-, is_anomaly_detected)
        """
        deviation = self.baseline_mean - observed_mean - self.k
        self.s_negative = max(0.0, self.s_negative + deviation)
        is_alarm = self.s_negative > self.h

        if is_alarm:
            logger.warning(
                f"[CRISIS ALARM] S_t^- = {self.s_negative:.4f} exceeded threshold h = {self.h:.4f} "
                f"(Observed: {observed_mean:.3f}, Baseline: {self.baseline_mean:.3f})"
            )
        return self.s_negative, is_alarm

    def reset_statistic(self) -> None:
        """Resets the accumulated CUSUM score following crisis resolution or cooldown."""
        self.s_negative = 0.0

    def query_recent_window_metrics(self, platform: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Queries the latest rolling window of sentiment posts from PostgreSQL."""
        if not self.pg_pool:
            return None

        conn = None
        try:
            conn = self.pg_pool.getconn()
            with conn.cursor() as cur:
                where_clause = "WHERE created_at >= NOW() - INTERVAL '%s minutes' AND is_spam = FALSE"
                params = [self.window_minutes]
                if platform:
                    where_clause += " AND platform = %s"
                    params.append(platform)

                query = f"""
                    SELECT 
                        COUNT(*) AS post_count,
                        COALESCE(AVG(sentiment_score), 0.0) AS mean_sentiment,
                        COALESCE(STDDEV(sentiment_score), 0.0) AS std_sentiment,
                        MIN(created_at) AS window_start,
                        MAX(created_at) AS window_end
                    FROM sentiment_posts
                    {where_clause};
                """
                cur.execute(query, tuple(params))
                row = cur.fetchone()

                if row and row[0] >= self.min_posts_threshold:
                    return {
                        "post_count": row[0],
                        "mean_sentiment": float(row[1]),
                        "std_sentiment": float(row[2]),
                        "window_start": row[3],
                        "window_end": row[4],
                        "platform": platform or "all_platforms",
                    }
        except Exception as e:
            logger.error(f"Error querying rolling metrics for CUSUM: {e}")
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)
        return None

    def record_crisis_event(self, alert_payload: Dict[str, Any]) -> None:
        """Persists the detected crisis event to PostgreSQL."""
        if not self.pg_pool:
            return

        conn = None
        try:
            conn = self.pg_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO crisis_alerts 
                    (platform, window_start, window_end, cusum_statistic, threshold_h, 
                     baseline_sentiment, observed_sentiment, deviation_magnitude, affected_volume, 
                     status, incident_summary, alert_payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        alert_payload["platform"],
                        alert_payload["window_start"],
                        alert_payload["window_end"],
                        alert_payload["cusum_statistic"],
                        alert_payload["threshold_h"],
                        alert_payload["baseline_sentiment"],
                        alert_payload["observed_sentiment"],
                        alert_payload["deviation_magnitude"],
                        alert_payload["affected_volume"],
                        "triggered",
                        alert_payload["incident_summary"],
                        json.dumps(alert_payload),
                    ),
                )
            conn.commit()
            logger.info("Successfully registered crisis alert in PostgreSQL.")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to record crisis event in DB: {e}")
        finally:
            if conn and self.pg_pool:
                self.pg_pool.putconn(conn)

    def check_and_evaluate(self, platform: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Executes a single check on the latest rolling window."""
        metrics = self.query_recent_window_metrics(platform)
        if not metrics:
            logger.debug("Insufficient volume in current window for CUSUM evaluation.")
            return None

        stat, is_alarm = self.update_cusum(metrics["mean_sentiment"])

        if is_alarm:
            now = datetime.utcnow()
            if self.last_alert_time and (now - self.last_alert_time) < timedelta(minutes=self.alert_cooldown_min):
                logger.info(f"Crisis alarm active but suppressed by cooldown ({self.alert_cooldown_min}m).")
                return None

            self.last_alert_time = now
            alert_payload = {
                "detected_at": now.isoformat() + "Z",
                "platform": metrics["platform"],
                "window_start": metrics["window_start"].isoformat() if hasattr(metrics["window_start"], "isoformat") else str(metrics["window_start"]),
                "window_end": metrics["window_end"].isoformat() if hasattr(metrics["window_end"], "isoformat") else str(metrics["window_end"]),
                "cusum_statistic": round(stat, 4),
                "threshold_h": round(self.h, 4),
                "baseline_sentiment": round(self.baseline_mean, 4),
                "observed_sentiment": round(metrics["mean_sentiment"], 4),
                "deviation_magnitude": round(self.baseline_mean - metrics["mean_sentiment"], 4),
                "affected_volume": metrics["post_count"],
                "incident_summary": (
                    f"Significant sentiment crash detected on {metrics['platform']}. "
                    f"Rolling sentiment plunged to {metrics['mean_sentiment']:.3f} "
                    f"across {metrics['post_count']} posts."
                ),
            }

            self.record_crisis_event(alert_payload)
            dispatch_alert(alert_payload, self.cfg)

            if self.producer:
                try:
                    self.producer.send(
                        self.cfg["kafka"]["topics"]["crisis_alerts"],
                        value=alert_payload,
                    )
                except Exception as e:
                    logger.error(f"Failed sending alert to Kafka: {e}")

            # Partial reset to prevent endless re-triggers
            self.s_negative = self.h * 0.7
            return alert_payload

        return None

    @staticmethod
    def calibrate_thresholds(
        desired_in_control_arl: float = 500.0,
        baseline_mean: float = 0.35,
        baseline_std: float = 0.20,
        expected_shift_size: float = 1.0,  # in units of sigma
        n_simulations: int = 2000,
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation to calibrate decision interval h and slack k
        to achieve desired Average Run Length (ARL_0) under the null hypothesis.
        """
        k = 0.5 * expected_shift_size * baseline_std
        h_candidates = np.linspace(3.0 * baseline_std, 6.0 * baseline_std, 10)
        results = {}

        logger.info(f"Calibrating CUSUM parameters for target ARL_0 = {desired_in_control_arl}...")

        for h_val in h_candidates:
            run_lengths = []
            for _ in range(n_simulations):
                s = 0.0
                step = 0
                while s <= h_val and step < desired_in_control_arl * 3:
                    step += 1
                    # Sample in-control observation from Gaussian(baseline_mean, baseline_std)
                    x_t = np.random.normal(loc=baseline_mean, scale=baseline_std)
                    dev = baseline_mean - x_t - k
                    s = max(0.0, s + dev)
                run_lengths.append(step)

            mean_arl = float(np.mean(run_lengths))
            results[round(float(h_val / baseline_std), 2)] = round(mean_arl, 1)

        # Select closest h
        best_h_factor = min(results.keys(), key=lambda factor: abs(results[factor] - desired_in_control_arl))

        calibration_report = {
            "target_ARL_0": desired_in_control_arl,
            "recommended_h_multiplier": best_h_factor,
            "recommended_h_absolute": round(best_h_factor * baseline_std, 4),
            "recommended_k_absolute": round(k, 4),
            "curve_h_multiplier_to_ARL": results,
        }
        return calibration_report

    def simulate_crisis_sequence(
        self,
        normal_windows: int = 20,
        crisis_windows: int = 10,
        recovery_windows: int = 15,
    ) -> pd.DataFrame:
        """Simulates a complete brand crisis lifecycle and tracks CUSUM behavior."""
        records = []
        self.reset_statistic()

        # 1. Normal state
        for t in range(normal_windows):
            val = float(np.random.normal(self.baseline_mean, self.baseline_std))
            stat, alarm = self.update_cusum(val)
            records.append({
                "window": t + 1,
                "phase": "Normal Baseline",
                "sentiment": val,
                "cusum_s_minus": stat,
                "threshold_h": self.h,
                "alarm_triggered": alarm,
            })

        # 2. Crisis shock state (e.g. Major server outage / PR disaster)
        crisis_mean = self.baseline_mean - 3.0 * self.baseline_std  # Drop by 3 sigma (-0.25)
        for t in range(crisis_windows):
            val = float(np.random.normal(crisis_mean, self.baseline_std))
            stat, alarm = self.update_cusum(val)
            records.append({
                "window": normal_windows + t + 1,
                "phase": "Crisis Shock",
                "sentiment": val,
                "cusum_s_minus": stat,
                "threshold_h": self.h,
                "alarm_triggered": alarm,
            })

        # 3. Post-crisis stabilization
        for t in range(recovery_windows):
            val = float(np.random.normal(self.baseline_mean * 0.8, self.baseline_std))
            stat, alarm = self.update_cusum(val)
            records.append({
                "window": normal_windows + crisis_windows + t + 1,
                "phase": "Post-Crisis Recovery",
                "sentiment": val,
                "cusum_s_minus": stat,
                "threshold_h": self.h,
                "alarm_triggered": alarm,
            })

        df = pd.DataFrame(records)
        return df


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CUSUM Change-Point Crisis Detector")
    parser.add_argument("--run-once", action="store_true", help="Evaluate current rolling window")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate h and k parameters via Monte Carlo")
    parser.add_argument("--simulate", action="store_true", help="Simulate a crisis lifecycle and print metrics")
    parser.add_argument("--loop", action="store_true", help="Run polling loop every 60 seconds")

    args = parser.parse_args()
    detector = CUSUMCrisisDetector()

    if args.calibrate:
        report = detector.calibrate_thresholds(desired_in_control_arl=500.0)
        print("\n=== CUSUM THRESHOLD CALIBRATION REPORT ===")
        print(json.dumps(report, indent=2))

    elif args.simulate:
        df = detector.simulate_crisis_sequence()
        first_alarm = df[df["alarm_triggered"]]["window"].min()
        print("\n=== CUSUM CRISIS SIMULATION RESULTS ===")
        print(f"Total windows simulated: {len(df)}")
        print(f"First alarm triggered at window: {first_alarm} (Phase: {df.loc[df['window'] == first_alarm, 'phase'].values[0]})")
        print("\nSample Progression:")
        print(df[["window", "phase", "sentiment", "cusum_s_minus", "threshold_h", "alarm_triggered"]].iloc[18:25].to_string(index=False))

    elif args.loop:
        logger.info("Entering CUSUM detection polling daemon...")
        while True:
            detector.check_and_evaluate()
            time.sleep(60)

    else:
        result = detector.check_and_evaluate()
        if result:
            print(f"ALERT: {json.dumps(result, indent=2)}")
        else:
            print("Status: No crisis condition detected in current window.")


if __name__ == "__main__":
    main()
