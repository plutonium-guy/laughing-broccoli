# Databricks notebook source
# MAGIC %md
# MAGIC # Business-Friendly Feature Explainer
# MAGIC
# MAGIC Drop-in replacement for `build_business_explanation` and
# MAGIC `build_business_explanations` in the original notebook.
# MAGIC
# MAGIC Problem solved: SHAP output exposes raw feature names
# MAGIC ("max_dpd", "pct_90_plus", "oldest_invoice_age") that a
# MAGIC business user cannot parse. This module maps every feature
# MAGIC in the model to plain English with proper formatting and
# MAGIC unit hints.

# COMMAND ----------

import pandas as pd


# =========================================================
# FEATURE DICTIONARY
# =========================================================
# Each entry:
#   label  — short human name
#   format — value formatter (function)
#   unit   — appended after value (e.g. "days", "%")
#   detail — long-form explanation for tooltip / docs
# =========================================================

def _pct(v):
    """0.65 -> '65%'"""
    try:
        return f"{float(v) * 100:.0f}%"
    except Exception:
        return str(v)


def _days(v):
    try:
        return f"{int(round(float(v)))}"
    except Exception:
        return str(v)


def _money(v):
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


def _count(v):
    try:
        return f"{int(round(float(v)))}"
    except Exception:
        return str(v)


def _ratio(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _level(v):
    try:
        return f"level {int(round(float(v)))}"
    except Exception:
        return str(v)


FEATURE_DICT = {

    # ---------- EXPOSURE ----------
    "total_outstanding": {
        "label":  "Total unpaid balance",
        "format": _money,
        "unit":   "EUR",
        "detail": "Sum of all open invoice amounts at snapshot.",
    },
    "num_open_invoices": {
        "label":  "Number of open invoices",
        "format": _count,
        "unit":   "invoices",
        "detail": "Count of unpaid invoices.",
    },
    "avg_invoice_size": {
        "label":  "Average invoice size",
        "format": _money,
        "unit":   "EUR",
        "detail": "Total outstanding divided by open invoice count.",
    },
    "max_dpd": {
        "label":  "Worst overdue invoice",
        "format": _days,
        "unit":   "days late",
        "detail": "Days past due of the oldest unpaid invoice.",
    },
    "avg_dpd": {
        "label":  "Average overdue",
        "format": _days,
        "unit":   "days late",
        "detail": "Average days past due across all open invoices.",
    },
    "amt_30_plus": {
        "label":  "Amount overdue more than 30 days",
        "format": _money,
        "unit":   "EUR",
        "detail": "Sum of outstanding amounts with DPD > 30.",
    },
    "amt_60_plus": {
        "label":  "Amount overdue more than 60 days",
        "format": _money,
        "unit":   "EUR",
        "detail": "Sum of outstanding amounts with DPD > 60.",
    },
    "amt_90_plus": {
        "label":  "Amount overdue more than 90 days",
        "format": _money,
        "unit":   "EUR",
        "detail": "Sum of outstanding amounts with DPD > 90.",
    },
    "pct_30_plus": {
        "label":  "Share of debt 30+ days overdue",
        "format": _pct,
        "unit":   "",
        "detail": "Fraction of total outstanding more than 30 days past due.",
    },
    "pct_60_plus": {
        "label":  "Share of debt 60+ days overdue",
        "format": _pct,
        "unit":   "",
        "detail": "Fraction of total outstanding more than 60 days past due.",
    },
    "pct_90_plus": {
        "label":  "Share of debt 90+ days overdue",
        "format": _pct,
        "unit":   "",
        "detail": "Fraction of total outstanding more than 90 days past due.",
    },
    "oldest_invoice_age": {
        "label":  "Age of oldest open invoice",
        "format": _days,
        "unit":   "days old",
        "detail": "Days since the oldest unpaid invoice was issued.",
    },
    "avg_invoice_age": {
        "label":  "Average open invoice age",
        "format": _days,
        "unit":   "days old",
        "detail": "Average days since invoice issuance for open invoices.",
    },

    # ---------- BEHAVIOR (overall) ----------
    "avg_days_to_pay": {
        "label":  "Average payment delay (lifetime)",
        "format": _days,
        "unit":   "days",
        "detail": "Average days from invoice date to clearance date, all history.",
    },
    "max_days_to_pay": {
        "label":  "Worst payment delay (lifetime)",
        "format": _days,
        "unit":   "days",
        "detail": "Longest payment delay ever recorded.",
    },
    "on_time_ratio": {
        "label":  "On-time payment rate (lifetime)",
        "format": _pct,
        "unit":   "",
        "detail": "Share of invoices paid on or before due date.",
    },
    "total_payments": {
        "label":  "Total invoices paid (lifetime)",
        "format": _count,
        "unit":   "payments",
        "detail": "Count of cleared invoices across history.",
    },
    "days_since_last_payment": {
        "label":  "Time since last payment",
        "format": _days,
        "unit":   "days",
        "detail": "Days between last cleared invoice and snapshot.",
    },

    # ---------- BEHAVIOR (rolling) ----------
    "avg_days_to_pay_60d":  {"label": "Average payment delay (last 60 days)",  "format": _days, "unit": "days", "detail": "Recent 60-day average payment delay."},
    "max_days_to_pay_60d":  {"label": "Worst payment delay (last 60 days)",    "format": _days, "unit": "days", "detail": "Recent 60-day worst payment delay."},
    "on_time_ratio_60d":    {"label": "On-time rate (last 60 days)",           "format": _pct,  "unit": "",     "detail": "Recent 60-day share of on-time payments."},
    "num_payments_60d":     {"label": "Payments made (last 60 days)",          "format": _count,"unit": "payments", "detail": "Cleared invoices in the last 60 days."},

    "avg_days_to_pay_90d":  {"label": "Average payment delay (last 90 days)",  "format": _days, "unit": "days", "detail": "Recent 90-day average payment delay."},
    "max_days_to_pay_90d":  {"label": "Worst payment delay (last 90 days)",    "format": _days, "unit": "days", "detail": "Recent 90-day worst payment delay."},
    "on_time_ratio_90d":    {"label": "On-time rate (last 90 days)",           "format": _pct,  "unit": "",     "detail": "Recent 90-day share of on-time payments."},
    "num_payments_90d":     {"label": "Payments made (last 90 days)",          "format": _count,"unit": "payments", "detail": "Cleared invoices in the last 90 days."},

    "avg_days_to_pay_180d": {"label": "Average payment delay (last 180 days)", "format": _days, "unit": "days", "detail": "Recent 180-day average payment delay."},
    "max_days_to_pay_180d": {"label": "Worst payment delay (last 180 days)",   "format": _days, "unit": "days", "detail": "Recent 180-day worst payment delay."},
    "on_time_ratio_180d":   {"label": "On-time rate (last 180 days)",          "format": _pct,  "unit": "",     "detail": "Recent 180-day share of on-time payments."},
    "num_payments_180d":    {"label": "Payments made (last 180 days)",         "format": _count,"unit": "payments", "detail": "Cleared invoices in the last 180 days."},

    # ---------- DUNNING ----------
    "max_dunning_level": {
        "label":  "Highest dunning severity reached",
        "format": _level,
        "unit":   "",
        "detail": "Worst dunning escalation level ever sent (higher = more serious).",
    },
    "total_dunning_events": {
        "label":  "Total dunning letters sent",
        "format": _count,
        "unit":   "letters",
        "detail": "Number of dunning letters ever issued.",
    },
    "avg_dunning_level": {
        "label":  "Average dunning severity",
        "format": _ratio,
        "unit":   "",
        "detail": "Average severity level across all dunning letters.",
    },
    "high_severity_dunning": {
        "label":  "Severe dunning letters",
        "format": _count,
        "unit":   "letters",
        "detail": "Count of dunning letters at level 3 or higher.",
    },
    "high_dunning_ratio": {
        "label":  "Share of severe dunning letters",
        "format": _pct,
        "unit":   "",
        "detail": "Fraction of all dunning letters that were high severity.",
    },

    "dunning_events_60d":          {"label": "Dunning letters (last 60 days)",        "format": _count, "unit": "letters", "detail": "Dunning letters in last 60 days."},
    "high_severity_dunning_60d":   {"label": "Severe dunning (last 60 days)",         "format": _count, "unit": "letters", "detail": "High-severity dunning in last 60 days."},
    "dunning_events_90d":          {"label": "Dunning letters (last 90 days)",        "format": _count, "unit": "letters", "detail": "Dunning letters in last 90 days."},
    "high_severity_dunning_90d":   {"label": "Severe dunning (last 90 days)",         "format": _count, "unit": "letters", "detail": "High-severity dunning in last 90 days."},
    "dunning_events_180d":         {"label": "Dunning letters (last 180 days)",       "format": _count, "unit": "letters", "detail": "Dunning letters in last 180 days."},
    "high_severity_dunning_180d":  {"label": "Severe dunning (last 180 days)",        "format": _count, "unit": "letters", "detail": "High-severity dunning in last 180 days."},

    # ---------- PROMISE-TO-PAY ----------
    "total_promises": {
        "label":  "Total payment promises made",
        "format": _count,
        "unit":   "promises",
        "detail": "All promise-to-pay agreements ever made.",
    },
    "broken_promises": {
        "label":  "Promises broken",
        "format": _count,
        "unit":   "promises",
        "detail": "Count of promise-to-pay agreements not honored.",
    },
    "kept_promises": {
        "label":  "Promises kept",
        "format": _count,
        "unit":   "promises",
        "detail": "Count of promise-to-pay agreements honored.",
    },
    "total_promised_amount": {
        "label":  "Total amount promised",
        "format": _money,
        "unit":   "EUR",
        "detail": "Sum of all promised payment amounts.",
    },
    "broken_ratio": {
        "label":  "Promise-break rate",
        "format": _pct,
        "unit":   "",
        "detail": "Fraction of promises that were broken.",
    },
    "kept_ratio": {
        "label":  "Promise-keep rate",
        "format": _pct,
        "unit":   "",
        "detail": "Fraction of promises that were kept.",
    },
    "avg_promised_amount": {
        "label":  "Average promise size",
        "format": _money,
        "unit":   "EUR",
        "detail": "Average amount per promise-to-pay.",
    },
    "promise_activity_flag": {
        "label":  "Has used promise-to-pay before",
        "format": lambda v: "yes" if float(v) > 0 else "no",
        "unit":   "",
        "detail": "Whether the customer has ever made a payment promise.",
    },

    "promises_60d":     {"label": "Promises made (last 60 days)",  "format": _count, "unit": "promises", "detail": "Promises in last 60 days."},
    "broken_60d":       {"label": "Promises broken (last 60 days)","format": _count, "unit": "promises", "detail": "Broken promises in last 60 days."},
    "kept_60d":         {"label": "Promises kept (last 60 days)",  "format": _count, "unit": "promises", "detail": "Kept promises in last 60 days."},
    "promised_amt_60d": {"label": "Amount promised (last 60 days)","format": _money, "unit": "EUR",      "detail": "Total promised value in last 60 days."},

    "promises_90d":     {"label": "Promises made (last 90 days)",  "format": _count, "unit": "promises", "detail": "Promises in last 90 days."},
    "broken_90d":       {"label": "Promises broken (last 90 days)","format": _count, "unit": "promises", "detail": "Broken promises in last 90 days."},
    "kept_90d":         {"label": "Promises kept (last 90 days)",  "format": _count, "unit": "promises", "detail": "Kept promises in last 90 days."},
    "promised_amt_90d": {"label": "Amount promised (last 90 days)","format": _money, "unit": "EUR",      "detail": "Total promised value in last 90 days."},

    "promises_180d":     {"label": "Promises made (last 180 days)",  "format": _count, "unit": "promises", "detail": "Promises in last 180 days."},
    "broken_180d":       {"label": "Promises broken (last 180 days)","format": _count, "unit": "promises", "detail": "Broken promises in last 180 days."},
    "kept_180d":         {"label": "Promises kept (last 180 days)",  "format": _count, "unit": "promises", "detail": "Kept promises in last 180 days."},
    "promised_amt_180d": {"label": "Amount promised (last 180 days)","format": _money, "unit": "EUR",      "detail": "Total promised value in last 180 days."},
}


# =========================================================
# CORE FORMATTER
# =========================================================

def explain_feature(feature, value, shap_value):
    """
    Convert a (feature, value, shap_value) tuple into one
    business-friendly sentence.
    """
    meta = FEATURE_DICT.get(feature)

    impact = "increased risk" if shap_value > 0 else "reduced risk"

    if meta is None:
        # graceful fallback
        return f"{feature.replace('_', ' ').capitalize()} ({value}) {impact}."

    label = meta["label"]
    formatted = meta["format"](value)
    unit = meta["unit"]

    if unit:
        return f"{label}: {formatted} {unit} — {impact}."
    return f"{label}: {formatted} — {impact}."


# =========================================================
# DROP-IN REPLACEMENTS
# =========================================================

# --- replaces build_business_explanation in EXPLAINABILITY cell ---
def build_business_explanation(contribution_df):
    """
    Takes a DataFrame with columns: feature, feature_value, shap_value.
    Returns list of business-friendly sentences.
    """
    out = []
    for _, row in contribution_df.iterrows():
        out.append(
            explain_feature(
                feature=row["feature"],
                value=row["feature_value"],
                shap_value=row["shap_value"],
            )
        )
    return out


# --- replaces build_business_explanations in PREDICTION cell ---
def build_business_explanations(shap_row, feature_row, feature_cols, top_n=5):
    """
    Takes raw SHAP row + raw feature value row + model feature list,
    returns top N business-friendly explanations sorted by abs(shap).

    feature_cols is REQUIRED — must match the exact order of training
    features (model.get_booster().feature_names). Defaulting to
    FEATURE_DICT.keys() would silently misalign SHAP values.
    """
    if feature_cols is None or len(feature_cols) != len(shap_row):
        raise ValueError(
            "feature_cols must be provided and match shap_row length. "
            "Use model.get_booster().feature_names from the trained model."
        )

    contribution_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": shap_row,
        "feature_value": feature_row,
    })

    contribution_df["abs_shap"] = contribution_df["shap_value"].abs()
    contribution_df = contribution_df.sort_values(
        by="abs_shap", ascending=False
    ).head(top_n)

    return build_business_explanation(contribution_df)


# =========================================================
# QUICK SANITY DEMO
# =========================================================

if __name__ == "__main__":

    sample_drivers = [
        ("pct_90_plus",            0.62,    0.42),
        ("max_dpd",                145,     0.31),
        ("broken_ratio",           0.55,    0.27),
        ("avg_days_to_pay",        58.3,    0.18),
        ("max_dunning_level",      4,       0.12),
        ("on_time_ratio",          0.18,   -0.09),
        ("oldest_invoice_age",     220,     0.07),
        ("total_outstanding",      48750,   0.05),
        ("days_since_last_payment",95,      0.04),
        ("kept_ratio",             0.10,   -0.03),
    ]

    print("BUSINESS-FRIENDLY OUTPUT:\n")
    for feat, val, shap_val in sample_drivers:
        print("•", explain_feature(feat, val, shap_val))
