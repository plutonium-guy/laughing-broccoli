# Databricks notebook source
# MAGIC %md
# MAGIC # SME-Friendly LLM Summary
# MAGIC
# MAGIC Drop-in replacement for the LLM Summary cell in `collection_risk_model.py`.
# MAGIC
# MAGIC Improvements:
# MAGIC 1. **SME-friendly output** — uses `FEATURE_DICT` to translate jargon
# MAGIC    (max_dpd, pct_90_plus, oldest_invoice_age) into plain English
# MAGIC    BEFORE the prompt reaches the LLM.
# MAGIC 2. **Class-aligned filtering** — only direction-matching drivers
# MAGIC    are sent so the LLM cannot contradict the predicted class.
# MAGIC 3. **Class semantics** — High/Medium/Low/DNT meaning injected.
# MAGIC 4. **Strict business voice** — no jargon, no model talk, no SHAP.
# MAGIC 5. **DNT short-circuit** — saves LLM cost on hardcoded customers.

# COMMAND ----------

import asyncio

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

# Original notebook (collection_risk_model.py) must define:
#   create_llm_client()  — Azure OpenAI client via Kong
# feature_explainer.py must define:
#   FEATURE_DICT, explain_feature
from feature_explainer import FEATURE_DICT, explain_feature


# =========================================================
# CLASS SEMANTICS
# =========================================================
CLASS_DEFINITIONS = {
    "High":   "customer is unlikely to pay — less than 40% of outstanding expected to be collected in the next 30 days",
    "Medium": "customer is a moderate collection risk — roughly 40% to 70% expected to be collected in 30 days",
    "Low":    "customer is reliable — more than 70% expected to be collected in 30 days",
    "DNT":    "customer is on the do-not-target list — excluded from collections activity by business rule",
}


# =========================================================
# JARGON-FREE DRIVER REWRITER
# =========================================================
def rewrite_drivers_for_sme(collector_explanations, predicted_class):
    """
    Handles BOTH input formats produced by the original notebook:

    1. Raw feature name (unmapped in original IF/ELIF chain):
       "max_dpd (145.0) increased risk."
       "pct_30_plus (0.45) increased risk."

    2. Pre-mapped business text (in original IF/ELIF chain):
       "High percentage of invoices over 90 days past due (0.62) increased risk."
       "Severe dunning activity (level 4) increased risk."

    Output for case 1: looks up FEATURE_DICT → SME label + formatted value.
    Output for case 2: kept as-is (already business-friendly).

    Also filters phrases by direction to prevent LLM contradicting the class.
    Strips commas from money values before float parsing (handles "48,750.00").
    """
    if not collector_explanations:
        return ""

    direction = None
    if predicted_class in ("High", "Medium"):
        direction = "increased risk"
    elif predicted_class == "Low":
        direction = "reduced risk"

    rewritten = []
    for raw in collector_explanations.split("|"):
        raw = raw.strip().rstrip(".")
        if not raw:
            continue

        # Filter wrong-direction phrases
        if direction and direction not in raw:
            continue

        # Try to parse "<head> (<value>) <impact>"
        try:
            head, _ = raw.rsplit(")", 1)
            feature_part, value_part = head.split("(", 1)
            feature_token = feature_part.strip()
            value_token = value_part.strip()

            # Case 1: raw feature name we recognize → look up FEATURE_DICT
            if feature_token in FEATURE_DICT:
                try:
                    value_clean = value_token.replace(",", "")
                    value_float = float(value_clean)
                    sign = 1.0 if "increased" in raw else -1.0
                    pretty = explain_feature(feature_token, value_float, sign)
                    rewritten.append(pretty.rstrip("."))
                    continue
                except Exception:
                    pass

            # Case 2: pre-mapped business text → keep as-is
            rewritten.append(raw)

        except Exception:
            rewritten.append(raw)

    return " | ".join(rewritten) if rewritten else collector_explanations


# =========================================================
# PROMPT BUILDER
# =========================================================
def build_prompt(collector_explanations, predicted_class):

    class_meaning = CLASS_DEFINITIONS.get(str(predicted_class), "unknown risk class")

    sme_drivers = rewrite_drivers_for_sme(
        collector_explanations, predicted_class
    )

    system_msg = (
        "You are writing a one-sentence collection-risk note for a "
        "business stakeholder (sales manager or credit controller) who is NOT a data scientist. "
        "Rules you MUST follow:\n"
        "  - Plain business English. Zero jargon.\n"
        "  - Never mention: model, SHAP, score, probability, feature, prediction.\n"
        "  - Never contradict the predicted class.\n"
        "  - Reuse the driver wording given (it is already SME-friendly).\n"
        "  - Output exactly ONE sentence, around 10–15 words.\n"
        "  - Active voice. Past or present tense.\n"
        "  - No bullets, no quotes, no preamble, no trailing notes."
    )

    user_msg = f"""
PREDICTED RISK CATEGORY: {predicted_class}
WHAT THAT MEANS: {class_meaning}

KEY DRIVERS (already class-aligned and jargon-free):
{sme_drivers}

TASK: Write ONE plain-English sentence explaining to a non-technical business
stakeholder why this customer was placed in the "{predicted_class}" risk category.

Use the drivers above. Do not introduce new facts. Do not soften the message
if the class is High. Do not exaggerate the message if the class is Low.

OUTPUT (one sentence, 10–15 words):
""".strip()

    return system_msg, user_msg


# =========================================================
# GENERATE
# =========================================================
async def generate_summary(collector_explanations, predicted_class):

    llm = create_llm_client()

    system_msg, user_msg = build_prompt(collector_explanations, predicted_class)

    response = await llm.ainvoke([
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ])

    return response.content.strip()


# =========================================================
# SPARK UDF WRAPPER
# =========================================================
def summarize_explanation(collector_explanations, predicted_class):

    if collector_explanations is None or predicted_class is None:
        return None

    # Skip LLM call for DNT — fixed business label
    if str(predicted_class) == "DNT":
        return (
            "Customer is on the do-not-target list — "
            "no collections action required per business policy."
        )

    try:
        return asyncio.run(
            generate_summary(collector_explanations, predicted_class)
        )
    except Exception as e:
        return f"ERROR: {str(e)}"


# =========================================================
# REGISTER UDF + APPLY
# =========================================================
summary_udf = udf(summarize_explanation, StringType())

df = spark.table("f_erp_glide_o2c_12.collection_ml_customer")

df_final = df.withColumn(
    "llm_summary",
    summary_udf(col("collector_explanations"), col("predicted_class"))
)

display(
    df_final.select(
        "customer_id",
        "predicted_class",
        "collector_explanations",
        "llm_summary",
    )
)
