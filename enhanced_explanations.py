# Databricks notebook source
# MAGIC %md
# MAGIC # Enhanced Business Explanations
# MAGIC
# MAGIC Drop-in replacement for `build_business_explanation` and
# MAGIC `build_business_explanations` from the original notebook.
# MAGIC
# MAGIC Keeps the **same IF/ELIF style** as the original — just exhaustive,
# MAGIC properly worded, and with correct units. Every feature in the
# MAGIC model now produces a complete business sentence instead of leaking
# MAGIC raw column names like `max_dpd` or `pct_30_plus`.
# MAGIC
# MAGIC Each branch:
# MAGIC   - reads like a stakeholder note (no jargon)
# MAGIC   - formats the value with the correct unit (%, days, EUR, count)
# MAGIC   - keeps the same "increased risk" / "reduced risk" tail so the
# MAGIC     downstream direction filter in the LLM prompt still works

# COMMAND ----------

import pandas as pd


# =========================================================
# CORE EXPLAINER — one line per feature
# =========================================================
def explain_one(feature, value, shap_value):
    """
    Convert a single (feature, value, shap_value) into a
    business-friendly explanation sentence.
    """

    impact = "increased risk" if shap_value > 0 else "reduced risk"

    # ----- EXPOSURE -----
    if feature == "total_outstanding":
        return f"Total unpaid balance of {value:,.0f} EUR {impact}."

    elif feature == "num_open_invoices":
        return f"Customer has {int(value)} open invoices, which {impact}."

    elif feature == "avg_invoice_size":
        return f"Average invoice size of {value:,.0f} EUR {impact}."

    elif feature == "max_dpd":
        return f"Worst invoice is {int(value)} days overdue, which {impact}."

    elif feature == "avg_dpd":
        return f"On average invoices are {int(value)} days overdue, which {impact}."

    elif feature == "amt_30_plus":
        return f"{value:,.0f} EUR is more than 30 days overdue, which {impact}."

    elif feature == "amt_60_plus":
        return f"{value:,.0f} EUR is more than 60 days overdue, which {impact}."

    elif feature == "amt_90_plus":
        return f"{value:,.0f} EUR is more than 90 days overdue, which {impact}."

    elif feature == "pct_30_plus":
        return f"{value*100:.0f}% of the unpaid balance is 30+ days overdue, which {impact}."

    elif feature == "pct_60_plus":
        return f"{value*100:.0f}% of the unpaid balance is 60+ days overdue, which {impact}."

    elif feature == "pct_90_plus":
        return f"{value*100:.0f}% of the unpaid balance is 90+ days overdue, which {impact}."

    elif feature == "oldest_invoice_age":
        return f"Oldest open invoice is {int(value)} days old, which {impact}."

    elif feature == "avg_invoice_age":
        return f"Open invoices are on average {int(value)} days old, which {impact}."

    # ----- BEHAVIOR (overall) -----
    elif feature == "avg_days_to_pay":
        return f"Customer historically pays {int(value)} days after invoice date, which {impact}."

    elif feature == "max_days_to_pay":
        return f"Worst historical payment delay is {int(value)} days, which {impact}."

    elif feature == "on_time_ratio":
        return f"Customer pays on time {value*100:.0f}% of the time, which {impact}."

    elif feature == "total_payments":
        return f"Customer has cleared {int(value)} invoices to date, which {impact}."

    elif feature == "days_since_last_payment":
        return f"Customer has not paid for {int(value)} days, which {impact}."

    # ----- BEHAVIOR (rolling windows) -----
    elif feature == "avg_days_to_pay_60d":
        return f"In the last 60 days customer paid {int(value)} days late on average, which {impact}."
    elif feature == "max_days_to_pay_60d":
        return f"In the last 60 days the worst payment delay was {int(value)} days, which {impact}."
    elif feature == "on_time_ratio_60d":
        return f"In the last 60 days {value*100:.0f}% of payments were on time, which {impact}."
    elif feature == "num_payments_60d":
        return f"Customer made {int(value)} payments in the last 60 days, which {impact}."

    elif feature == "avg_days_to_pay_90d":
        return f"In the last 90 days customer paid {int(value)} days late on average, which {impact}."
    elif feature == "max_days_to_pay_90d":
        return f"In the last 90 days the worst payment delay was {int(value)} days, which {impact}."
    elif feature == "on_time_ratio_90d":
        return f"In the last 90 days {value*100:.0f}% of payments were on time, which {impact}."
    elif feature == "num_payments_90d":
        return f"Customer made {int(value)} payments in the last 90 days, which {impact}."

    elif feature == "avg_days_to_pay_180d":
        return f"In the last 180 days customer paid {int(value)} days late on average, which {impact}."
    elif feature == "max_days_to_pay_180d":
        return f"In the last 180 days the worst payment delay was {int(value)} days, which {impact}."
    elif feature == "on_time_ratio_180d":
        return f"In the last 180 days {value*100:.0f}% of payments were on time, which {impact}."
    elif feature == "num_payments_180d":
        return f"Customer made {int(value)} payments in the last 180 days, which {impact}."

    # ----- DUNNING -----
    elif feature == "max_dunning_level":
        return f"Customer reached dunning level {int(value)} (out of 4), which {impact}."

    elif feature == "total_dunning_events":
        return f"Customer received {int(value)} dunning letters in total, which {impact}."

    elif feature == "avg_dunning_level":
        return f"Average dunning severity is {value:.1f}, which {impact}."

    elif feature == "high_severity_dunning":
        return f"Customer received {int(value)} severe dunning letters (level 3+), which {impact}."

    elif feature == "high_dunning_ratio":
        return f"{value*100:.0f}% of dunning letters were high severity, which {impact}."

    elif feature == "dunning_events_60d":
        return f"{int(value)} dunning letters sent in the last 60 days, which {impact}."
    elif feature == "high_severity_dunning_60d":
        return f"{int(value)} severe dunning letters sent in the last 60 days, which {impact}."
    elif feature == "dunning_events_90d":
        return f"{int(value)} dunning letters sent in the last 90 days, which {impact}."
    elif feature == "high_severity_dunning_90d":
        return f"{int(value)} severe dunning letters sent in the last 90 days, which {impact}."
    elif feature == "dunning_events_180d":
        return f"{int(value)} dunning letters sent in the last 180 days, which {impact}."
    elif feature == "high_severity_dunning_180d":
        return f"{int(value)} severe dunning letters sent in the last 180 days, which {impact}."

    # ----- PROMISE-TO-PAY -----
    elif feature == "total_promises":
        return f"Customer has made {int(value)} payment promises, which {impact}."

    elif feature == "broken_promises":
        return f"Customer has broken {int(value)} payment promises, which {impact}."

    elif feature == "kept_promises":
        return f"Customer has kept {int(value)} payment promises, which {impact}."

    elif feature == "total_promised_amount":
        return f"Customer has promised {value:,.0f} EUR in total, which {impact}."

    elif feature == "broken_ratio":
        return f"Customer breaks {value*100:.0f}% of payment promises, which {impact}."

    elif feature == "kept_ratio":
        return f"Customer keeps {value*100:.0f}% of payment promises, which {impact}."

    elif feature == "avg_promised_amount":
        return f"Average promise amount is {value:,.0f} EUR, which {impact}."

    elif feature == "promise_activity_flag":
        flag = "uses" if value > 0 else "does not use"
        return f"Customer {flag} payment promises, which {impact}."

    elif feature == "promises_60d":
        return f"{int(value)} payment promises in the last 60 days, which {impact}."
    elif feature == "broken_60d":
        return f"{int(value)} broken promises in the last 60 days, which {impact}."
    elif feature == "kept_60d":
        return f"{int(value)} promises kept in the last 60 days, which {impact}."
    elif feature == "promised_amt_60d":
        return f"{value:,.0f} EUR promised in the last 60 days, which {impact}."

    elif feature == "promises_90d":
        return f"{int(value)} payment promises in the last 90 days, which {impact}."
    elif feature == "broken_90d":
        return f"{int(value)} broken promises in the last 90 days, which {impact}."
    elif feature == "kept_90d":
        return f"{int(value)} promises kept in the last 90 days, which {impact}."
    elif feature == "promised_amt_90d":
        return f"{value:,.0f} EUR promised in the last 90 days, which {impact}."

    elif feature == "promises_180d":
        return f"{int(value)} payment promises in the last 180 days, which {impact}."
    elif feature == "broken_180d":
        return f"{int(value)} broken promises in the last 180 days, which {impact}."
    elif feature == "kept_180d":
        return f"{int(value)} promises kept in the last 180 days, which {impact}."
    elif feature == "promised_amt_180d":
        return f"{value:,.0f} EUR promised in the last 180 days, which {impact}."

    # ----- FALLBACK (should never be hit if dict above is complete) -----
    else:
        readable = feature.replace("_", " ")
        return f"{readable.capitalize()} value of {value} {impact}."


# =========================================================
# REPLACES `build_business_explanation` in EXPLAINABILITY cell
# =========================================================
def build_business_explanation(contribution_df):
    """
    Takes DataFrame with columns: feature, feature_value, shap_value.
    Returns list of business-friendly sentences.
    """
    return [
        explain_one(
            feature=row["feature"],
            value=row["feature_value"],
            shap_value=row["shap_value"],
        )
        for _, row in contribution_df.iterrows()
    ]


# =========================================================
# REPLACES `build_business_explanations` in PREDICTION cell
# =========================================================
def build_business_explanations(shap_row, feature_row, top_n=5, FEATURE_COLS=None):
    """
    Same signature as the original (keeps drop-in compatibility).
    FEATURE_COLS must match training feature order — typically
    `model.get_booster().feature_names`.
    """
    if FEATURE_COLS is None:
        raise ValueError(
            "FEATURE_COLS must be provided (use model.get_booster().feature_names)."
        )

    contribution_df = pd.DataFrame({
        "feature": FEATURE_COLS,
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

    sample = [
        ("pct_90_plus",            0.62,    0.42),
        ("max_dpd",                145,     0.31),
        ("broken_ratio",           0.55,    0.27),
        ("avg_days_to_pay",        58.3,    0.18),
        ("max_dunning_level",      4,       0.12),
        ("on_time_ratio",          0.18,   -0.09),
        ("oldest_invoice_age",     220,     0.07),
        ("total_outstanding",      48750,   0.05),
        ("days_since_last_payment",95,      0.04),
        ("amt_60_plus",            12500,   0.03),
    ]

    print("ENHANCED BUSINESS-FRIENDLY OUTPUT:\n")
    for feat, val, shap_val in sample:
        print("•", explain_one(feat, val, shap_val))
