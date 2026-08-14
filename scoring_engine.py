"""
SME Customer Intelligence & Retention Framework
Triple-Scoring Engine v2.0 (Refactored) — Explainable Rule-Based Baseline Engine
Author: Abrar Rafii Ibrahim (Konversi Magang PT Telkom Indonesia)
Date: August 2026

Positioning:
    This module implements an Explainable Rule-Based Baseline Engine. It is a
    fully transparent, deterministic scoring framework that serves as the
    benchmark against which future Supervised ML models (Logistic Regression,
    Random Forest, XGBoost) and SHAP-based feature importance will be evaluated.

    Modular interface hooks (supervised classifier hooks, SHAP hooks) are
    declared at the bottom of this file so that the production roadmap can
    attach ML models without rewriting the dashboard layer.

Refactoring notes (v2.0 refactored):
    1. Trend delta fix for short-tenure customers (first_last3):
       - len(g) == 1 : delta = 0.0
       - len(g) == 2 : delta = row[1] - row[0]
       - len(g) >= 3 : delta = mean(last 3) - mean(first 3)
       This prevents zero-delta collapse for the 250 Pelanggan Baru customers
       whose tenure is <= 3 months.
    2. Confidence score dead-code fix (calculate_confidence):
       Data missingness penalties are now computed BEFORE default-value
       imputations are applied in engineer_features(). A pre-imputation feature
       snapshot is passed through so that genuine NaN values trigger the
       intended penalty.
    3. Urgency & priority score formulation fix:
       For Pelanggan Baru, urgency is now potential_score / 100 (high-potential
       accounts are fast-tracked), NOT (100 - potential_score).
    4. Granularity preservation:
       priority_score is exported at 3-decimal precision (round(3)). It is
       NEVER collapsed to 1 decimal place. The output CSV therefore contains
       continuous, fine-grained priority values for the intervention queue.
    5. AI/ML positioning & modular structure:
       The engine is explicitly framed as an Explainable Rule-Based Baseline.
       Modular interface hooks for supervised classification (Logistic
       Regression / XGBoost) and SHAP value feature importance are provided.

Input : 7 CSVs in ./data/sme_dummy/
Output: ./output/customer_scores.csv + scoring_validation_charts.png
Usage : python scoring_engine.py
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "./data/sme_dummy/"
OUTPUT_PATH = "./output/"
REFERENCE_DATE = pd.Timestamp("2026-06-30")
os.makedirs(OUTPUT_PATH, exist_ok=True)

HEALTH_MAP = {"Growing": 1.0, "Stable": 0.6, "Declining": 0.2}
SENTIMENT_MAP = {"Positif": 1.0, "Netral": 0.5, "Negatif": 0.0}
SCALE_MAP = {"Mikro": 0.2, "Kecil": 0.4, "Menengah": 0.7, "Besar": 1.0}

# ============================================================
# WEIGHTS — ALL VALUES IN THIS SECTION ARE SAFE TO TUNE MANUALLY
# ============================================================

# Default Potential Score weights (also used as fallback for unknown sectors)
POTENTIAL_WEIGHTS = {
    "industry_growth": 0.30,
    "digital_presence": 0.25,
    "company_scale": 0.20,
    "onboarding_velocity": 0.15,
    "payment_reliability": 0.10,
}

# ------------------------------------------------------------
# SECTOR_WEIGHTS — Potential Score weights per business sector.
# Component order: industry_growth, digital_presence, company_scale,
#                  onboarding_velocity, payment_reliability
# Each row MUST sum to 1.00.
# Domain intuition: digital-native sectors (IT, Media, Education) ->
# high digital_presence; traditional sectors (Agriculture, Construction,
# Manufacturing) -> low digital_presence, higher weight on company_scale
# and industry_growth.
# ------------------------------------------------------------
SECTOR_WEIGHTS = {
    "Teknologi Informasi":     {"industry_growth": 0.25, "digital_presence": 0.35, "company_scale": 0.20, "onboarding_velocity": 0.15, "payment_reliability": 0.05},
    "Media & Kreatif":         {"industry_growth": 0.25, "digital_presence": 0.35, "company_scale": 0.15, "onboarding_velocity": 0.15, "payment_reliability": 0.10},
    "Pendidikan & Training":   {"industry_growth": 0.25, "digital_presence": 0.35, "company_scale": 0.15, "onboarding_velocity": 0.15, "payment_reliability": 0.10},
    "Jasa Keuangan":           {"industry_growth": 0.30, "digital_presence": 0.25, "company_scale": 0.20, "onboarding_velocity": 0.15, "payment_reliability": 0.10},
    "Kesehatan":               {"industry_growth": 0.30, "digital_presence": 0.20, "company_scale": 0.25, "onboarding_velocity": 0.15, "payment_reliability": 0.10},
    "Perdagangan Eceran":      {"industry_growth": 0.30, "digital_presence": 0.20, "company_scale": 0.20, "onboarding_velocity": 0.20, "payment_reliability": 0.10},
    "Transportasi & Logistik": {"industry_growth": 0.30, "digital_presence": 0.20, "company_scale": 0.25, "onboarding_velocity": 0.15, "payment_reliability": 0.10},
    "Kuliner & F&B":           {"industry_growth": 0.30, "digital_presence": 0.15, "company_scale": 0.25, "onboarding_velocity": 0.20, "payment_reliability": 0.10},
    "Properti & Real Estate":  {"industry_growth": 0.30, "digital_presence": 0.15, "company_scale": 0.25, "onboarding_velocity": 0.15, "payment_reliability": 0.15},
    "Manufaktur":              {"industry_growth": 0.35, "digital_presence": 0.10, "company_scale": 0.25, "onboarding_velocity": 0.15, "payment_reliability": 0.15},
    "Konstruksi":              {"industry_growth": 0.35, "digital_presence": 0.10, "company_scale": 0.25, "onboarding_velocity": 0.15, "payment_reliability": 0.15},
    "Pertanian & Agribisnis":  {"industry_growth": 0.30, "digital_presence": 0.05, "company_scale": 0.30, "onboarding_velocity": 0.15, "payment_reliability": 0.20},
}

STABILITY_WEIGHTS = {
    "payment_consistency": 0.25,
    "usage_stability": 0.25,
    "support_satisfaction": 0.20,
    "tenure_loyalty": 0.20,
    "external_health": 0.10,
}

CHURN_WEIGHTS = {
    "usage_deceleration": 0.30,
    "financial_stress": 0.25,
    "support_escalation": 0.20,
    "external_threats": 0.15,
    "engagement_drop": 0.10,
}

# Category thresholds (safe to tune for business logic)
THRESHOLDS = {
    "potential": {"high": 75, "medium": 50},
    "stability": {"high": 80, "medium": 60},
    "churn": {"critical": 70, "high": 45},
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def min_max_normalize(series, invert=False):
    """Min-max normalisation with median fallback for NaN values."""
    s = series.fillna(series.median())
    min_val, max_val = s.min(), s.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    normalized = (s - min_val) / (max_val - min_val)
    return 1 - normalized if invert else normalized


def calculate_confidence(row, pre_imp_row=None):
    """Compute confidence score based on a pre-imputation feature snapshot.

    The dead-code bug in v1 was caused by computing missingness AFTER default
    values were imputed in engineer_features(), which made the pd.isna() check
    always evaluate to False. This version accepts an optional pre-imputation
    snapshot (`pre_imp_row`) so that genuine missing values are correctly
    penalised.

    Penalties (transparent heuristic; not a Bayesian uncertainty):
        - avg_days_overdue missing            : -5
        - avg_utilization missing             : -5
        - positive_sentiment_ratio missing    : -3
        - months_available < 3                : -8
        - months_available < 6                : -5
        - std_utilization > 15                : -3
        - std_days_overdue > 10               : -3
        - std_feature_adoption > 2            : -2
        - very low ticket volume              : -2

    Result is clamped to [60, 98] to reflect the inherent floor and ceiling
    of rule-based confidence estimation.
    """
    src = pre_imp_row if pre_imp_row is not None else row
    base = 85
    missing_penalty = 0
    variance_penalty = 0

    if pd.isna(src.get("avg_days_overdue")):
        missing_penalty += 5
    if pd.isna(src.get("avg_utilization")):
        missing_penalty += 5
    if pd.isna(src.get("positive_sentiment_ratio")):
        missing_penalty += 3

    months = src.get("months_available", 12)
    if months is not None and not pd.isna(months):
        if months < 3:
            missing_penalty += 8
        elif months < 6:
            missing_penalty += 5

    if row.get("std_utilization", 0) is not None and not pd.isna(row.get("std_utilization", 0)):
        if row.get("std_utilization", 0) > 15:
            variance_penalty += 3
    if row.get("std_days_overdue", 0) is not None and not pd.isna(row.get("std_days_overdue", 0)):
        if row.get("std_days_overdue", 0) > 10:
            variance_penalty += 3
    if row.get("std_feature_adoption", 0) is not None and not pd.isna(row.get("std_feature_adoption", 0)):
        if row.get("std_feature_adoption", 0) > 2:
            variance_penalty += 2

    if (row.get("total_tickets", 0) is not None
            and not pd.isna(row.get("total_tickets", 0))
            and row.get("total_tickets", 0) < 2
            and pd.notna(src.get("positive_sentiment_ratio"))):
        variance_penalty += 2

    return max(60, min(98, base - missing_penalty - variance_penalty))


def score_category(row):
    """Assign a discrete category based on segment + assigned score."""
    score, segment = row["assigned_score"], row["segment"]
    if segment == "Pelanggan Baru":
        if score >= THRESHOLDS["potential"]["high"]:
            return "High Potential"
        elif score >= THRESHOLDS["potential"]["medium"]:
            return "Medium Potential"
        else:
            return "Low Potential"
    elif segment == "Pelanggan Tetap":
        if score >= THRESHOLDS["stability"]["high"]:
            return "Highly Stable"
        elif score >= THRESHOLDS["stability"]["medium"]:
            return "Stable"
        else:
            return "At Risk"
    else:  # Pelanggan Berisiko
        if score >= THRESHOLDS["churn"]["critical"]:
            return "Critical Risk"
        elif score >= THRESHOLDS["churn"]["high"]:
            return "High Risk"
        else:
            return "Moderate Risk"


def recommend_intervention(row):
    """Look up intervention recommendation by (segment, score_category)."""
    segment, category = row["segment"], row["score_category"]
    recommendations = {
        ("Pelanggan Baru", "High Potential"): {
            "primary_action": "Fast-track Loyalty Program Enrollment",
            "secondary_action": "Offer 12-month contract with early-bird discount",
            "timing": "Within 7 days of registration",
            "channel": "Executive Phone Call + Email",
            "expected_success_rate": 0.75,
            "rationale": "High growth potential detected. Lock-in before competitor approaches.",
        },
        ("Pelanggan Baru", "Medium Potential"): {
            "primary_action": "Onboarding Success Package",
            "secondary_action": "Dedicated account manager for first 90 days",
            "timing": "Within 14 days",
            "channel": "WhatsApp + Portal Tutorial",
            "expected_success_rate": 0.55,
            "rationale": "Moderate potential with room for growth. Focus on feature adoption.",
        },
        ("Pelanggan Baru", "Low Potential"): {
            "primary_action": "Re-evaluate Product Fit",
            "secondary_action": "Offer downgraded package or flexible terms",
            "timing": "Within 30 days",
            "channel": "Email Survey + Phone Follow-up",
            "expected_success_rate": 0.35,
            "rationale": "Low potential signals possible mismatch. Prevent silent churn.",
        },
        ("Pelanggan Tetap", "Highly Stable"): {
            "primary_action": "Upsell Premium Services",
            "secondary_action": "Invite to beta program / exclusive features",
            "timing": "Quarterly business review",
            "channel": "Executive Visit + Proposal",
            "expected_success_rate": 0.65,
            "rationale": "Strong loyalty foundation. Maximize lifetime value through expansion.",
        },
        ("Pelanggan Tetap", "Stable"): {
            "primary_action": "Proactive Health Check",
            "secondary_action": "Usage optimization consultation",
            "timing": "Bi-annual touchpoint",
            "channel": "Phone Call + Usage Report",
            "expected_success_rate": 0.50,
            "rationale": "Stable but monitor for early decline signals.",
        },
        ("Pelanggan Tetap", "At Risk"): {
            "primary_action": "Retention Intervention Call",
            "secondary_action": "Personalized discount or service upgrade offer",
            "timing": "Immediate (within 48 hours)",
            "channel": "Senior Manager Phone Call + WhatsApp",
            "expected_success_rate": 0.40,
            "rationale": "Stability score declining. Act before entering Berisiko segment.",
        },
        ("Pelanggan Berisiko", "Critical Risk"): {
            "primary_action": "Executive Escalation & Win-back Negotiation",
            "secondary_action": "Custom retention package with significant concession",
            "timing": "Immediate (same day)",
            "channel": "Face-to-face Meeting + Written Proposal",
            "expected_success_rate": 0.25,
            "rationale": "Critical churn risk. High-cost intervention justified by MRR value.",
        },
        ("Pelanggan Berisiko", "High Risk"): {
            "primary_action": "Win-back Campaign with Incentive",
            "secondary_action": "Service review + competitor price match offer",
            "timing": "Within 3 days",
            "channel": "Phone Call + Email Offer",
            "expected_success_rate": 0.35,
            "rationale": "Multiple risk signals. Prioritize based on MRR and win-back cost.",
        },
        ("Pelanggan Berisiko", "Moderate Risk"): {
            "primary_action": "Re-engagement Survey + Check-in",
            "secondary_action": "Offer flexible payment terms or temporary upgrade",
            "timing": "Within 7 days",
            "channel": "WhatsApp + Email",
            "expected_success_rate": 0.45,
            "rationale": "Early warning signs. Low-cost intervention may prevent escalation.",
        },
    }
    return recommendations.get(
        (segment, category),
        recommendations.get((segment, "Medium Potential")),
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data(base_path=BASE_PATH):
    """Load the 7 canonical CSV tables from disk."""
    tables = {
        "customers": pd.read_csv(f"{base_path}sme_customers.csv"),
        "subscriptions": pd.read_csv(f"{base_path}subscription_history.csv"),
        "usage": pd.read_csv(f"{base_path}usage_metrics.csv"),
        "payments": pd.read_csv(f"{base_path}payment_records.csv"),
        "support": pd.read_csv(f"{base_path}support_interactions.csv"),
        "enrichment": pd.read_csv(f"{base_path}external_enrichment.csv"),
        "interventions": pd.read_csv(f"{base_path}intervention_log.csv"),
    }
    return tables


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(tables, window_months=None, reference_date=REFERENCE_DATE):
    """Aggregate raw tables into a per-customer feature matrix.

    Parameters
    ----------
    tables : dict
        Output of `load_data()`.
    window_months : list[str] | None
        Optional list of "YYYY-MM" strings used by the dashboard's time filter.
        If None, the full 12-month window is used.

    Returns
    -------
    pd.DataFrame
        Per-customer feature matrix. A side-channel attribute
        `_pre_imputation_snapshot` (dict of pd.Series keyed by customer_id) is
        attached so that `calculate_scores` can compute missingness penalties
        BEFORE the imputed default values mask genuine NaN entries.
    """
    df_cust = tables["customers"].copy()
    df_subs = tables["subscriptions"].copy()
    df_usage = tables["usage"].copy()
    df_payments = tables["payments"].copy()
    df_support = tables["support"].copy()
    df_enrich = tables["enrichment"].copy()
    df_intv = tables["interventions"].copy()

    if window_months is not None:
        df_usage = df_usage[df_usage["period_month"].isin(window_months)]
        df_payments = df_payments[df_payments["invoice_period"].isin(window_months)]
        df_support["interaction_date"] = pd.to_datetime(df_support["interaction_date"])
        df_support = df_support[df_support["interaction_date"].dt.strftime("%Y-%m").isin(window_months)]

    # --- Payment Aggregates ---
    payment_agg = df_payments.groupby("customer_id").agg({
        "days_overdue": ["mean", "max", "std"],
        "partial_payment_flag": "mean",
        "invoice_amount": "mean",
    }).reset_index()
    payment_agg.columns = ["customer_id", "avg_days_overdue", "max_days_overdue",
                           "std_days_overdue", "partial_payment_ratio", "avg_invoice_amount"]
    payment_agg["std_days_overdue"] = payment_agg["std_days_overdue"].fillna(0)

    # --- Usage Aggregates ---
    usage_agg = df_usage.groupby("customer_id").agg({
        "bandwidth_utilization_pct": ["mean", "std", "min", "max"],
        "avg_daily_usage_hours": ["mean", "std"],
        "service_downtime_minutes": ["mean", "sum"],
        "support_ticket_count": ["mean", "sum"],
        "feature_adoption_score": ["mean", "std", "min", "max"],
        "peak_usage_trend": lambda x: (x == "Turun").sum() / len(x),
        "period_month": "count",
    }).reset_index()
    usage_agg.columns = ["customer_id", "avg_utilization", "std_utilization", "min_utilization",
                         "max_utilization", "avg_usage_hours", "std_usage_hours",
                         "avg_downtime", "total_downtime", "avg_tickets", "total_tickets",
                         "avg_feature_adoption", "std_feature_adoption", "min_feature_adoption",
                         "max_feature_adoption", "declining_months_ratio", "months_available"]
    usage_agg[["std_utilization", "std_usage_hours", "std_feature_adoption"]] = \
        usage_agg[["std_utilization", "std_usage_hours", "std_feature_adoption"]].fillna(0)

    # --- Trend delta (FIX: handles len(g) == 1 and len(g) == 2 explicitly) ---
    df_usage_s = df_usage.sort_values(["customer_id", "period_month"])

    def first_last3(g):
        """Compute first-3 vs last-3 deltas with short-tenure fallback.

        - len(g) == 1 : delta = 0.0 (single data point, no trend info)
        - len(g) == 2 : delta = row[1] - row[0]
        - len(g) >= 3 : delta = mean(last 3) - mean(first 3)
        """
        n = len(g)
        cols = ["bandwidth_utilization_pct", "avg_daily_usage_hours", "feature_adoption_score"]
        if n == 1:
            return pd.Series({"utilization_delta": 0.0,
                              "hours_delta": 0.0,
                              "adoption_delta": 0.0})
        if n == 2:
            r0 = g.iloc[0][cols].astype(float)
            r1 = g.iloc[1][cols].astype(float)
            d = r1 - r0
            return pd.Series({"utilization_delta": d.iloc[0],
                              "hours_delta": d.iloc[1],
                              "adoption_delta": d.iloc[2]})
        first = g.head(3)[cols].mean()
        last = g.tail(3)[cols].mean()
        return pd.Series({"utilization_delta": last.iloc[0] - first.iloc[0],
                          "hours_delta": last.iloc[1] - first.iloc[1],
                          "adoption_delta": last.iloc[2] - first.iloc[2]})

    usage_trend = df_usage_s.groupby("customer_id").apply(first_last3, include_groups=False).reset_index()

    # --- Support Aggregates ---
    if len(df_support) > 0:
        support_agg = df_support.groupby("customer_id").agg({
            "sentiment": lambda x: (x == "Positif").sum() / len(x),
            "category": lambda x: (x == "Complaint").sum() / len(x),
            "resolution_status": lambda x: (x == "Resolved").sum() / len(x),
            "response_time_minutes": "mean",
            "follow_up_required": "mean",
        }).reset_index()
        support_agg.columns = ["customer_id", "positive_sentiment_ratio", "complaint_ratio",
                               "resolution_ratio", "avg_response_time", "follow_up_ratio"]
    else:
        support_agg = pd.DataFrame(columns=["customer_id", "positive_sentiment_ratio", "complaint_ratio",
                                            "resolution_ratio", "avg_response_time", "follow_up_ratio"])

    # --- Subscription Aggregates (active per reference date) ---
    df_subs["start_date"] = pd.to_datetime(df_subs["start_date"])
    df_subs["end_date"] = pd.to_datetime(df_subs["end_date"], errors="coerce")
    active_subs = df_subs[(df_subs["start_date"] <= reference_date) &
                          ((df_subs["end_date"].isna()) | (df_subs["end_date"] > reference_date))]
    subs_agg = active_subs.groupby("customer_id").agg({
        "monthly_recurring_revenue": "sum",
        "auto_renewal": "mean",
        "contract_period": "mean",
        "product_code": "count",
    }).reset_index()
    subs_agg.columns = ["customer_id", "total_mrr", "auto_renewal_ratio",
                        "avg_contract_period", "product_count"]

    # --- External Enrichment ---
    df_enrich["health_numeric"] = df_enrich["business_health_indicator"].map(HEALTH_MAP)
    df_enrich["sentiment_numeric"] = df_enrich["industry_growth_sentiment"].map(SENTIMENT_MAP)

    # --- Intervention Aggregates ---
    if len(df_intv) > 0:
        intv_agg = df_intv.groupby("customer_id").agg({
            "intervention_type": "count",
            "outcome": lambda x: (x == "Accepted").sum() / len(x),
            "cost_incurred": "sum",
        }).reset_index()
        intv_agg.columns = ["customer_id", "intervention_count",
                            "intervention_success_rate", "total_intervention_cost"]
    else:
        intv_agg = pd.DataFrame(columns=["customer_id", "intervention_count",
                                         "intervention_success_rate", "total_intervention_cost"])

    # --- Tenure ---
    df_cust["registration_date"] = pd.to_datetime(df_cust["registration_date"])
    df_cust["tenure_months"] = ((reference_date - df_cust["registration_date"]).dt.days / 30.44).round(1)
    df_cust["tenure_years"] = df_cust["tenure_months"] / 12

    # --- Merge All ---
    features = df_cust[["customer_id", "company_name", "business_sector", "company_scale",
                        "employee_count", "region", "segment", "tenure_months", "tenure_years"]].copy()
    features = features.merge(payment_agg, on="customer_id", how="left")
    features = features.merge(usage_agg, on="customer_id", how="left")
    features = features.merge(usage_trend, on="customer_id", how="left")
    features = features.merge(support_agg, on="customer_id", how="left")
    features = features.merge(subs_agg, on="customer_id", how="left")
    features = features.merge(df_enrich[["customer_id", "competitor_mention_count", "recent_news_flag",
                                         "digital_presence_score", "health_numeric", "sentiment_numeric"]],
                              on="customer_id", how="left")
    features = features.merge(intv_agg, on="customer_id", how="left")

    # ============================================================
    # PRE-IMPUTATION SNAPSHOT (for confidence score)
    # Captures the genuine NaN state of key features BEFORE default values
    # are filled in. This eliminates the dead-code bug where missingness
    # penalties were never triggered.
    # ============================================================
    snapshot_cols = ["avg_days_overdue", "avg_utilization", "positive_sentiment_ratio",
                     "months_available", "std_utilization", "std_days_overdue",
                     "std_feature_adoption", "total_tickets"]
    pre_imp_snapshot = {cid: row for cid, row in
                        features.set_index("customer_id")[snapshot_cols].to_dict("index").items()}
    features._pre_imputation_snapshot = pre_imp_snapshot  # side-channel attribute

    # --- Default value imputations (applied AFTER snapshot is captured) ---
    features["intervention_count"] = features["intervention_count"].fillna(0)
    features["intervention_success_rate"] = features["intervention_success_rate"].fillna(0)
    features["total_intervention_cost"] = features["total_intervention_cost"].fillna(0)
    features["months_available"] = features["months_available"].fillna(0)
    features["declining_months_ratio"] = features["declining_months_ratio"].fillna(0)
    features["partial_payment_ratio"] = features["partial_payment_ratio"].fillna(0)
    features["total_mrr"] = features["total_mrr"].fillna(0)
    features["avg_days_overdue"] = features["avg_days_overdue"].fillna(0)  # genuine missing -> 0

    # Missing support = "quietly loyal" assumption
    features["positive_sentiment_ratio"] = features["positive_sentiment_ratio"].fillna(0.7)
    features["resolution_ratio"] = features["resolution_ratio"].fillna(1.0)
    if features["avg_response_time"].notna().any():
        features["avg_response_time"] = features["avg_response_time"].fillna(features["avg_response_time"].median())
    else:
        features["avg_response_time"] = 30.0
    features["follow_up_ratio"] = features["follow_up_ratio"].fillna(0.0)
    features["complaint_ratio"] = features["complaint_ratio"].fillna(0.0)

    return features


# ============================================================
# SCORING ENGINE
# ============================================================

def calculate_scores(features, sector_weights=None, potential_weights=None,
                     stability_weights=None, churn_weights=None):
    """Compute the triple scores (Potential, Stability, Churn Risk Index).

    All weight dictionaries can be overridden — this is what powers the
    dashboard's What-If Simulator.
    """
    sw = sector_weights or SECTOR_WEIGHTS
    pw = potential_weights or POTENTIAL_WEIGHTS
    stw = stability_weights or STABILITY_WEIGHTS
    cw = churn_weights or CHURN_WEIGHTS

    f = features.copy()
    pre_snap = getattr(features, "_pre_imputation_snapshot", None)

    # --- Normalised features ---
    f["norm_tenure_years"] = min_max_normalize(f["tenure_years"])
    f["norm_employee_count"] = min_max_normalize(f["employee_count"])
    f["norm_digital_presence"] = min_max_normalize(f["digital_presence_score"])
    f["norm_competitor_mentions"] = min_max_normalize(f["competitor_mention_count"])
    f["norm_avg_utilization"] = min_max_normalize(f["avg_utilization"])
    f["norm_avg_feature_adoption"] = min_max_normalize(f["avg_feature_adoption"])
    f["norm_avg_tickets"] = min_max_normalize(f["avg_tickets"])
    f["norm_total_mrr"] = min_max_normalize(f["total_mrr"])
    f["norm_contract_period"] = min_max_normalize(f["avg_contract_period"])
    f["norm_days_overdue"] = min_max_normalize(f["avg_days_overdue"], invert=True)
    f["norm_max_days_overdue"] = min_max_normalize(f["max_days_overdue"], invert=True)
    f["norm_response_time"] = min_max_normalize(f["avg_response_time"], invert=True)
    f["scale_numeric"] = f["company_scale"].map(SCALE_MAP)
    f["util_cv"] = f["std_utilization"] / (f["avg_utilization"] + 1e-6)
    f["norm_util_stability"] = min_max_normalize(f["util_cv"], invert=True)

    # --- POTENTIAL SCORE (Pelanggan Baru) — sector-weighted ---
    f["potential_industry_growth"] = (f["sentiment_numeric"] + f["health_numeric"]) / 2
    f["potential_digital"] = f["norm_digital_presence"]
    f["potential_scale"] = (f["scale_numeric"] + f["norm_employee_count"]) / 2
    util_delta_norm = min_max_normalize(f["utilization_delta"])
    adopt_delta_norm = min_max_normalize(f["adoption_delta"])
    f["potential_onboarding"] = (util_delta_norm + adopt_delta_norm) / 2
    f["potential_payment"] = (f["norm_days_overdue"] + (1 - f["partial_payment_ratio"])) / 2

    comp_map = {"industry_growth": "potential_industry_growth",
                "digital_presence": "potential_digital",
                "company_scale": "potential_scale",
                "onboarding_velocity": "potential_onboarding",
                "payment_reliability": "potential_payment"}
    f["potential_score"] = 0.0
    for comp, col in comp_map.items():
        w = f["business_sector"].map(lambda s: sw.get(s, pw)[comp])
        f["potential_score"] += f[col] * w
    f["potential_score"] *= 100

    # --- STABILITY SCORE (Pelanggan Tetap) ---
    f["stability_payment"] = (f["norm_days_overdue"] * 0.5 +
                              f["norm_max_days_overdue"] * 0.3 +
                              (1 - f["partial_payment_ratio"]) * 0.2)
    f["stability_usage"] = (f["norm_avg_utilization"] * 0.4 +
                            f["norm_util_stability"] * 0.35 +
                            min_max_normalize(f["utilization_delta"]).clip(0, 1) * 0.25)
    f["stability_support"] = (f["positive_sentiment_ratio"] * 0.4 +
                              f["resolution_ratio"] * 0.35 +
                              f["norm_response_time"] * 0.25)
    f["stability_loyalty"] = (f["norm_tenure_years"] * 0.4 +
                              f["auto_renewal_ratio"].fillna(0) * 0.35 +
                              f["norm_contract_period"] * 0.25)
    f["stability_external"] = (f["health_numeric"] + f["sentiment_numeric"]) / 2
    f["stability_score"] = (f["stability_payment"] * stw["payment_consistency"] +
                            f["stability_usage"] * stw["usage_stability"] +
                            f["stability_support"] * stw["support_satisfaction"] +
                            f["stability_loyalty"] * stw["tenure_loyalty"] +
                            f["stability_external"] * stw["external_health"]) * 100

    # --- CHURN RISK INDEX (Pelanggan Berisiko) ---
    util_delta_risk = min_max_normalize(f["utilization_delta"], invert=True).clip(0, 1)
    adopt_delta_risk = min_max_normalize(f["adoption_delta"], invert=True).clip(0, 1)
    f["churn_usage_decel"] = (util_delta_risk * 0.4 +
                              f["declining_months_ratio"] * 0.35 +
                              adopt_delta_risk * 0.25)
    f["churn_financial"] = (min_max_normalize(f["avg_days_overdue"]) * 0.45 +
                            f["partial_payment_ratio"] * 0.30 +
                            min_max_normalize(f["std_days_overdue"]) * 0.25)
    f["churn_support"] = (f["complaint_ratio"] * 0.35 +
                          (1 - f["positive_sentiment_ratio"]) * 0.30 +
                          (1 - f["resolution_ratio"]) * 0.20 +
                          f["follow_up_ratio"] * 0.15)
    f["churn_external"] = (f["norm_competitor_mentions"] * 0.4 +
                           (1 - f["health_numeric"]) * 0.35 +
                           (1 - f["sentiment_numeric"]) * 0.25)
    f["churn_engagement"] = ((1 - f["norm_avg_feature_adoption"]) * 0.4 +
                             f["declining_months_ratio"] * 0.35 +
                             f["norm_avg_tickets"] * 0.25)
    f["churn_risk_index"] = (f["churn_usage_decel"] * cw["usage_deceleration"] +
                             f["churn_financial"] * cw["financial_stress"] +
                             f["churn_support"] * cw["support_escalation"] +
                             f["churn_external"] * cw["external_threats"] +
                             f["churn_engagement"] * cw["engagement_drop"]) * 100

    # --- Assigned score (segment-routed) ---
    f["assigned_score"] = f.apply(
        lambda r: r["potential_score"] if r["segment"] == "Pelanggan Baru"
        else (r["stability_score"] if r["segment"] == "Pelanggan Tetap"
              else r["churn_risk_index"]),
        axis=1,
    )

    # --- Confidence score (uses pre-imputation snapshot) ---
    def _conf_row(row):
        cid = row["customer_id"]
        pre = pre_snap.get(cid, {}) if pre_snap else None
        return calculate_confidence(row, pre_imp_row=pre)
    f["confidence_score"] = f.apply(_conf_row, axis=1)
    f["score_category"] = f.apply(score_category, axis=1)

    # ============================================================
    # PRIORITY SCORE (Intervention Queue)
    # ------------------------------------------------------------
    # Urgency is segment-routed. The Pelanggan Baru branch is now
    #   urgency = potential_score / 100
    # so that high-potential new accounts are FAST-TRACKED
    # (locking in the contract before a competitor approaches),
    # rather than deprioritised as in the flawed v1 formula.
    #
    # Final priority = 0.5 * norm_total_mrr + 0.5 * urgency
    # Rounded to 3 decimal places for fine-grained queue ranking.
    # ============================================================
    urgency = np.where(
        f["segment"] == "Pelanggan Berisiko", f["churn_risk_index"] / 100,
        np.where(
            f["segment"] == "Pelanggan Tetap", (100 - f["stability_score"]) / 100,
            f["potential_score"] / 100,   # FIX: high potential -> high urgency
        ),
    )
    f["priority_score"] = (0.5 * f["norm_total_mrr"].fillna(0) + 0.5 * urgency).round(3)

    # --- Intervention recommendations ---
    recs = f.apply(recommend_intervention, axis=1)
    rec_df = pd.DataFrame(recs.tolist())
    drop_cols = [c for c in ["primary_action", "secondary_action", "timing", "channel",
                             "expected_success_rate", "rationale"] if c in f.columns]
    f = f.drop(columns=drop_cols)
    f = pd.concat([f.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)
    return f


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_COLS = [
    "customer_id", "company_name", "business_sector", "region", "segment",
    "company_scale", "employee_count", "tenure_years", "total_mrr", "product_count",
    "months_available", "potential_score", "stability_score", "churn_risk_index",
    "assigned_score", "score_category", "confidence_score", "priority_score",
    "primary_action", "secondary_action", "timing", "channel",
    "expected_success_rate", "rationale",
]


def generate_output(features, output_path=OUTPUT_PATH):
    """Persist customer_scores.csv. priority_score is kept at 3-decimal
    precision (NEVER collapsed to 1 decimal)."""
    cols = [c for c in OUTPUT_COLS if c in features.columns]
    df_scores = features[cols].copy()

    # Score columns rounded to 1 decimal for display readability.
    for col in ["potential_score", "stability_score", "churn_risk_index",
                "assigned_score", "confidence_score"]:
        df_scores[col] = df_scores[col].round(1)

    # CRITICAL: priority_score MUST remain at 3-decimal precision to preserve
    # fine-grained queue ranking. Do NOT round to 1 decimal here.
    df_scores["priority_score"] = df_scores["priority_score"].round(3)

    df_scores["expected_success_rate"] = (df_scores["expected_success_rate"] * 100).round(0).astype(int)
    df_scores.to_csv(f"{output_path}customer_scores.csv", index=False)
    print(f"Saved: {output_path}customer_scores.csv ({len(df_scores)} records)")
    return df_scores


def print_validation(df_scores):
    """Print per-segment score statistics and category counts."""
    print("\n" + "=" * 70)
    print("SCORING ENGINE v2 (Refactored) — VALIDATION")
    print("=" * 70)
    seg_col = {
        "Pelanggan Baru": "potential_score",
        "Pelanggan Tetap": "stability_score",
        "Pelanggan Berisiko": "churn_risk_index",
    }
    for seg in ["Pelanggan Baru", "Pelanggan Tetap", "Pelanggan Berisiko"]:
        subset = df_scores[df_scores["segment"] == seg]
        col = seg_col[seg]
        print(f"\n{seg} ({len(subset)} customers):")
        print(f"  Score   min={subset[col].min():.2f}  max={subset[col].max():.2f}  "
              f"mean={subset[col].mean():.2f}  median={subset[col].median():.2f}  "
              f"std={subset[col].std():.2f}")
        print(f"  Confidence mean: {subset['confidence_score'].mean():.1f}%")
        for cat, count in subset["score_category"].value_counts().items():
            print(f"    - {cat:18s}: {count:4d} ({count / len(subset) * 100:5.1f}%)")

    # All 9 categories overview
    print("\n" + "-" * 70)
    print("CATEGORY COUNTS (all 9 categories)")
    print("-" * 70)
    all_cats = ["High Potential", "Medium Potential", "Low Potential",
                "Highly Stable", "Stable", "At Risk",
                "Critical Risk", "High Risk", "Moderate Risk"]
    counts = df_scores["score_category"].value_counts().reindex(all_cats, fill_value=0)
    for cat in all_cats:
        n = counts[cat]
        print(f"  {cat:18s}: {n:4d}  ({n / len(df_scores) * 100:5.1f}%)")

    # Priority score granularity check
    print("\n" + "-" * 70)
    print("PRIORITY SCORE GRANULARITY CHECK")
    print("-" * 70)
    n_unique = df_scores["priority_score"].nunique()
    print(f"  Unique priority_score values: {n_unique} / {len(df_scores)} rows")
    print(f"  Range: {df_scores['priority_score'].min():.3f} - "
          f"{df_scores['priority_score'].max():.3f}")
    print(f"  Std:   {df_scores['priority_score'].std():.4f}")

    # Top-10 priority queue
    print("\n" + "-" * 70)
    print("TOP-10 PRIORITY INTERVENTION QUEUE")
    print("-" * 70)
    top10 = df_scores.sort_values("priority_score", ascending=False).head(10)
    for i, r in top10.reset_index(drop=True).iterrows():
        print(f"  {i+1:2d}. {r['company_name'][:30]:30s}  "
              f"seg={r['segment'][:18]:18s}  "
              f"cat={r['score_category']:14s}  "
              f"MRR=Rp {r['total_mrr']:>10,.0f}  "
              f"priority={r['priority_score']:.3f}")


def generate_validation_charts(df_scores, output_path=OUTPUT_PATH):
    """Render scoring_validation_charts.png with 6 diagnostic panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # Font fallback so the chart renders on systems without Latin glyphs
    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Scoring Engine v2 (Refactored) — Validation Charts "
                 "(Window: Jul 2025 - Jun 2026)", fontsize=14, fontweight="bold")

    seg_col = {"Pelanggan Baru": "potential_score",
               "Pelanggan Tetap": "stability_score",
               "Pelanggan Berisiko": "churn_risk_index"}
    colors = {"Pelanggan Baru": "#2563eb",
              "Pelanggan Tetap": "#059669",
              "Pelanggan Berisiko": "#dc2626"}

    ax = axes[0, 0]
    for seg, col in seg_col.items():
        d = df_scores[df_scores["segment"] == seg][col]
        if len(d):
            ax.hist(d, bins=25, alpha=0.6, label=seg, color=colors[seg])
    ax.set_title("Score Distribution by Segment")
    ax.set_xlabel("Score (0-100)")
    ax.set_ylabel("Customer count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    cat_counts = df_scores.groupby(["segment", "score_category"]).size().unstack(fill_value=0)
    cat_counts.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Category Mix per Segment")
    ax.set_xlabel("")
    ax.set_ylabel("Customer count")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    ax = axes[0, 2]
    ax.hist(df_scores["confidence_score"], bins=20, color="#475569", alpha=0.8)
    ax.set_title("Confidence Score Distribution")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Customer count")

    ax = axes[1, 0]
    sr = df_scores.groupby("score_category")["expected_success_rate"].mean().sort_values()
    ax.barh(sr.index, sr.values, color="#0e7490")
    ax.set_title("Avg Expected Success Rate by Category")
    ax.set_xlabel("Success rate (%)")
    ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    mrr = df_scores.groupby("score_category")["total_mrr"].sum().sort_values(ascending=False) / 1e9
    ax.bar(range(len(mrr)), mrr.values, color="#b45309")
    ax.set_xticks(range(len(mrr)))
    ax.set_xticklabels(mrr.index, rotation=30, ha="right", fontsize=7)
    ax.set_title("Total MRR by Category (Rp Miliar)")
    ax.set_ylabel("Rp Miliar")

    ax = axes[1, 2]
    ax.scatter(df_scores["assigned_score"], df_scores["priority_score"],
               c=df_scores["segment"].map(colors), alpha=0.35, s=12)
    ax.set_xlabel("Assigned Score")
    ax.set_ylabel("Priority Score")
    ax.set_title("Score vs Intervention Priority")

    plt.savefig(f"{output_path}scoring_validation_charts.png", dpi=130)
    print(f"Saved: {output_path}scoring_validation_charts.png")


# ============================================================
# MODULAR ML / XAI INTERFACE HOOKS
# ============================================================
# The functions below declare the modular interface that the production
# roadmap will use to attach supervised ML classifiers and SHAP-based
# feature importance on top of this rule-based baseline. They are stubs
# by design: the rule-based engine is the active baseline, and ML models
# will be benchmarked against it in a comparative evaluation.

def fit_supervised_baseline(features, target=None, model_type="logistic"):
    """Stub hook for a supervised churn classifier.

    Parameters
    ----------
    features : pd.DataFrame
        Per-customer feature matrix produced by `engineer_features()`.
    target : pd.Series | None
        Binary churn label (y in {0, 1}). If None, a synthetic target is
        derived from `churn_risk_index` >= 45.
    model_type : str
        One of {"logistic", "random_forest", "xgboost"}.

    Returns
    -------
    dict
        {"model": fitted_model, "metrics": {"precision": ..., "recall": ...,
        "f1": ..., "roc_auc": ...}}. The actual fitting is delegated to the
        production roadmap; this stub returns a NotImplemented marker so the
        dashboard can detect when ML benchmarking is not yet wired.
    """
    return {"model": None, "metrics": {"status": "not_implemented"},
            "note": ("Supervised ML benchmark (Logistic Regression / Random Forest / "
                     "XGBoost) is part of the 12-18 month production roadmap. "
                     "The rule-based engine remains the active baseline.")}


def compute_shap_importance(features, model=None):
    """Stub hook for SHAP value feature importance.

    Returns
    -------
    dict
        Mapping feature_name -> mean(|SHAP|). Empty until the ML model is
        fitted; declared here so the dashboard can render a placeholder.
    """
    return {"status": "not_implemented",
            "note": ("SHAP value computation requires the supervised ML model "
                     "to be fitted first. Roadmap: Q3-Q4 production rollout.")}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SME Customer Intelligence — Triple-Scoring Engine v2.0 (Refactored)")
    print("Explainable Rule-Based Baseline Engine")
    print("=" * 70)
    tables = load_data()
    print(f"Loaded {len(tables)} tables")
    features = engineer_features(tables)
    print(f"Engineered {features.shape[1]} features for {len(features)} customers")
    features = calculate_scores(features)
    print("Triple scores calculated (sector-weighted Potential Score)")
    df_scores = generate_output(features)
    print_validation(df_scores)
    generate_validation_charts(df_scores)
    print("\nDone! Output ready for dashboard integration.")
