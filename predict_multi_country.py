# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Country Prediction Pipeline
# MAGIC
# MAGIC Inputs:
# MAGIC   - `MODEL_REGISTRY` dict: country code -> MLflow model path
# MAGIC   - Unified view with all SAP data
# MAGIC
# MAGIC For each country:
# MAGIC   1. Build inference features from unified view
# MAGIC   2. Load country-specific model
# MAGIC   3. Score with country-specific threshold + bands
# MAGIC   4. Compute SHAP top driver per customer
# MAGIC   5. Union into combined prediction table

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType


# =========================================================
# CONFIG — country -> model path + threshold + bands
# =========================================================
MODEL_REGISTRY = {
    "FR": {
        "model_path": "models:/collection_risk_model_customer_v1_FR/latest",
        "threshold":  0.40,
        "bands":      [0, 0.4, 0.7, 1.0],
        "labels":     ["Low", "Medium", "High"],
    },
    "AT": {
        "model_path": "models:/collection_risk_model_customer_v1_AT/latest",
        "threshold":  0.35,
        "bands":      [0, 0.35, 0.65, 1.0],
        "labels":     ["Low", "Medium", "High"],
    },
    "DE": {
        "model_path": "models:/collection_risk_model_customer_v1_DE/latest",
        "threshold":  0.40,
        "bands":      [0, 0.4, 0.7, 1.0],
        "labels":     ["Low", "Medium", "High"],
    },
    # Add more countries as models get registered
}

UNIFIED_VIEW = "f_erp_glide_o2c_12.vw_invoice_unified"
OUTPUT_TABLE = "f_erp_glide_o2c_12.collection_ml_customer"
WINDOWS = [60, 90, 180]


# =========================================================
# MODEL CACHE — load each model once
# =========================================================
_MODEL_CACHE = {}


def load_model(country):
    if country in _MODEL_CACHE:
        return _MODEL_CACHE[country]

    cfg = MODEL_REGISTRY[country]
    model = mlflow.sklearn.load_model(cfg["model_path"])
    feature_cols = model.get_booster().feature_names

    bundle = {
        "model":        model,
        "feature_cols": feature_cols,
        "threshold":    cfg["threshold"],
        "bands":        cfg["bands"],
        "labels":       cfg["labels"],
        "model_path":   cfg["model_path"],
    }
    _MODEL_CACHE[country] = bundle
    print(f"[{country}] loaded {len(feature_cols)} features, threshold={cfg['threshold']}")
    return bundle


# =========================================================
# FEATURE ENGINEERING — unified view, single snapshot = TODAY
# =========================================================
def load_base(country):
    """Load unified view for one country. Source-aware (BSID + BSAD)."""
    return spark.table(UNIFIED_VIEW).select(
        F.col("customer_id"),
        F.col("invoice_id"),
        F.col("line_item"),
        F.col("invoice_amount").cast("double"),
        F.col("open_amount").cast("double"),
        F.col("baseline_date").cast("date"),
        F.col("clearing_date").cast("date"),
        F.col("cash_discount_days_1").cast("int"),
        F.col("cash_discount_days_2").cast("int"),
        F.col("net_payment_days").cast("int"),
        F.col("due_date").cast("date"),
        F.col("country"),
        F.col("region"),
        F.col("customer_tenure_days").cast("int"),
        F.col("dunning_level").cast("int"),
        F.col("last_dunned_date").cast("date"),
        F.col("dunning_count").cast("int"),
        F.col("fin_promised_amt").cast("double"),
        F.col("fin_p2p_state").cast("int"),
        F.col("promise_dt").cast("date"),
        F.col("risk_class"),
        F.col("credit_group"),
        F.col("credit_limit").cast("double"),
        F.col("number_of_disputes").cast("int"),
        F.col("open_dispute_amount").cast("double"),
        F.upper(F.col("source")).alias("source"),
    ).filter(
        F.col("baseline_date").isNotNull()
        & F.col("source").isin(["BSID", "BSAD"])
        & (F.col("country") == country)
    )


def ensure_due_date(df):
    return df.withColumn(
        "payment_terms_days",
        F.when(F.col("net_payment_days").isNotNull() & (F.col("net_payment_days") != 0),
               F.col("net_payment_days"))
         .when(F.col("cash_discount_days_2").isNotNull() & (F.col("cash_discount_days_2") != 0),
               F.col("cash_discount_days_2"))
         .when(F.col("cash_discount_days_1").isNotNull() & (F.col("cash_discount_days_1") != 0),
               F.col("cash_discount_days_1"))
         .otherwise(0)
         .cast(IntegerType())
    ).withColumn(
        "due_date",
        F.coalesce(F.col("due_date"),
                   F.date_add("baseline_date", F.col("payment_terms_days")))
    )


def build_invoice_level(df):
    return df.groupBy("customer_id", "invoice_id").agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.sum("open_amount").alias("open_amount"),
        F.min("baseline_date").alias("baseline_date"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date"),
        F.max(F.when(F.col("source") == "BSAD", 1).otherwise(0)).alias("is_cleared"),
        F.max("dunning_level").alias("dunning_level"),
        F.max("last_dunned_date").alias("last_dunned_date"),
        F.max("dunning_count").alias("dunning_count"),
        F.max("fin_promised_amt").alias("fin_promised_amt"),
        F.first("fin_p2p_state", ignorenulls=True).alias("fin_p2p_state"),
        F.max("promise_dt").alias("promise_dt"),
        F.first("risk_class", ignorenulls=True).alias("risk_class"),
        F.first("credit_group", ignorenulls=True).alias("credit_group"),
        F.first("credit_limit", ignorenulls=True).alias("credit_limit"),
        F.max("number_of_disputes").alias("number_of_disputes"),
        F.max("open_dispute_amount").alias("open_dispute_amount"),
        F.max("customer_tenure_days").alias("customer_tenure_days"),
    )


def make_spine(inv_df):
    return inv_df.select("customer_id").distinct() \
        .withColumn("snapshot_date", F.current_date())


def exposure_features(inv_df, spine):
    inv = inv_df.alias("i")
    s = spine.alias("s")

    open_inv = inv.join(
        s,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner",
    ).filter(
        (F.col("i.is_cleared") == 0)
        | (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).withColumn(
        "dpd",
        F.when(F.col("i.due_date") <= F.col("s.snapshot_date"),
               F.datediff("s.snapshot_date", "i.due_date")).otherwise(0)
    ).withColumn(
        "invoice_age", F.datediff("s.snapshot_date", "i.baseline_date")
    )

    return open_inv.groupBy("s.customer_id", "s.snapshot_date").agg(
        F.sum("invoice_amount").alias("total_outstanding"),
        F.sum("open_amount").alias("total_open_amount"),
        F.countDistinct("invoice_id").alias("num_open_invoices"),
        F.max("dpd").alias("max_dpd"),
        F.avg("dpd").alias("avg_dpd"),
        F.sum(F.when(F.col("dpd") > 30, F.col("invoice_amount")).otherwise(0)).alias("amt_30_plus"),
        F.sum(F.when(F.col("dpd") > 60, F.col("invoice_amount")).otherwise(0)).alias("amt_60_plus"),
        F.sum(F.when(F.col("dpd") > 90, F.col("invoice_amount")).otherwise(0)).alias("amt_90_plus"),
        F.max("invoice_age").alias("oldest_invoice_age"),
        F.avg("invoice_age").alias("avg_invoice_age"),
        F.max("credit_limit").alias("credit_limit"),
        F.max("number_of_disputes").alias("number_of_disputes"),
        F.sum("open_dispute_amount").alias("open_dispute_amount"),
        F.max("customer_tenure_days").alias("customer_tenure_days"),
    ).withColumn(
        "avg_invoice_size", F.col("total_outstanding") / F.col("num_open_invoices")
    ).withColumn(
        "pct_30_plus", F.col("amt_30_plus") / F.col("total_outstanding")
    ).withColumn(
        "pct_60_plus", F.col("amt_60_plus") / F.col("total_outstanding")
    ).withColumn(
        "pct_90_plus", F.col("amt_90_plus") / F.col("total_outstanding")
    ).withColumn(
        "credit_utilization",
        F.when(F.col("credit_limit") > 0, F.col("total_outstanding") / F.col("credit_limit"))
         .otherwise(0)
    )


def behavior_features(inv_df, spine):
    b = inv_df.alias("b")
    s = spine.alias("s")

    hist = b.join(
        s,
        (F.col("b.customer_id") == F.col("s.customer_id"))
        & (F.col("b.is_cleared") == 1)
        & (F.col("b.clearing_date").isNotNull())
        & (F.col("b.clearing_date") <= F.col("s.snapshot_date")),
        "inner",
    ).withColumn(
        "days_to_pay", F.datediff("b.clearing_date", "b.baseline_date")
    )

    aggs = [
        F.avg("days_to_pay").alias("avg_days_to_pay"),
        F.max("days_to_pay").alias("max_days_to_pay"),
        F.avg(F.when(F.col("days_to_pay") <= 0, 1).otherwise(0)).alias("on_time_ratio"),
        F.count("*").alias("total_payments"),
        F.datediff(F.col("s.snapshot_date"), F.max("b.clearing_date")).alias("days_since_last_payment"),
    ]
    for w in WINDOWS:
        cond = F.col("b.clearing_date") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.avg(F.when(cond, F.col("days_to_pay"))).alias(f"avg_days_to_pay_{w}d"),
            F.max(F.when(cond, F.col("days_to_pay"))).alias(f"max_days_to_pay_{w}d"),
            F.avg(F.when(cond & (F.col("days_to_pay") <= 0), 1).otherwise(0)).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"num_payments_{w}d"),
        ])

    return hist.groupBy("b.customer_id", "s.snapshot_date").agg(*aggs)


def dunning_features(inv_df, spine):
    i = inv_df.alias("i")
    s = spine.alias("s")

    hist = i.join(
        s,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.last_dunned_date").isNotNull())
        & (F.col("i.last_dunned_date") <= F.col("s.snapshot_date")),
        "inner",
    )

    aggs = [
        F.max("dunning_level").alias("max_dunning_level"),
        F.sum("dunning_count").alias("total_dunning_events"),
        F.avg("dunning_level").alias("avg_dunning_level"),
        F.sum(F.when(F.col("dunning_level") >= 3, F.col("dunning_count")).otherwise(0))
         .alias("high_severity_dunning"),
    ]
    for w in WINDOWS:
        cond = F.col("i.last_dunned_date") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.sum(F.when(cond, F.col("dunning_count")).otherwise(0)).alias(f"dunning_events_{w}d"),
            F.sum(F.when(cond & (F.col("dunning_level") >= 3), F.col("dunning_count")).otherwise(0))
             .alias(f"high_severity_dunning_{w}d"),
        ])

    return hist.groupBy("i.customer_id", "s.snapshot_date").agg(*aggs).withColumn(
        "high_dunning_ratio",
        F.when(F.col("total_dunning_events") > 0,
               F.col("high_severity_dunning") / F.col("total_dunning_events")).otherwise(0)
    )


def p2p_features(inv_df, spine):
    i = inv_df.alias("i")
    s = spine.alias("s")

    hist = i.join(
        s,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.promise_dt").isNotNull())
        & (F.col("i.promise_dt") <= F.col("s.snapshot_date")),
        "inner",
    )

    aggs = [
        F.count("*").alias("total_promises"),
        F.sum(F.when(F.col("fin_p2p_state") == 1, 1).otherwise(0)).alias("broken_promises"),
        F.sum(F.when(F.col("fin_p2p_state") == 3, 1).otherwise(0)).alias("kept_promises"),
        F.sum("fin_promised_amt").alias("total_promised_amount"),
    ]
    for w in WINDOWS:
        cond = F.col("i.promise_dt") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"promises_{w}d"),
            F.sum(F.when(cond & (F.col("fin_p2p_state") == 1), 1).otherwise(0)).alias(f"broken_{w}d"),
            F.sum(F.when(cond & (F.col("fin_p2p_state") == 3), 1).otherwise(0)).alias(f"kept_{w}d"),
            F.sum(F.when(cond, F.col("fin_promised_amt"))).alias(f"promised_amt_{w}d"),
        ])

    return hist.groupBy("i.customer_id", "s.snapshot_date").agg(*aggs).withColumn(
        "broken_ratio",
        F.when(F.col("total_promises") > 0,
               F.col("broken_promises") / F.col("total_promises")).otherwise(0)
    ).withColumn(
        "kept_ratio",
        F.when(F.col("total_promises") > 0,
               F.col("kept_promises") / F.col("total_promises")).otherwise(0)
    ).withColumn(
        "avg_promised_amount",
        F.when(F.col("total_promises") > 0,
               F.col("total_promised_amount") / F.col("total_promises")).otherwise(0)
    ).withColumn(
        "promise_activity_flag",
        F.when(F.col("total_promises") > 0, 1).otherwise(0)
    )


def build_inference_features(country):
    """Full feature engineering for one country, single snapshot = TODAY."""
    base = ensure_due_date(load_base(country))
    inv = build_invoice_level(base)
    spine = make_spine(inv)

    feats = (
        spine
        .join(exposure_features(inv, spine), ["customer_id", "snapshot_date"], "left")
        .join(behavior_features(inv, spine), ["customer_id", "snapshot_date"], "left")
        .join(dunning_features(inv, spine),  ["customer_id", "snapshot_date"], "left")
        .join(p2p_features(inv, spine),      ["customer_id", "snapshot_date"], "left")
        .fillna(0)
    )

    return feats


# =========================================================
# SPARK -> PANDAS SAFE
# =========================================================
def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(
                field.name, spark_df[field.name].cast(DoubleType())
            )
    return spark_df.fillna(0).toPandas()


# =========================================================
# SCORE ONE COUNTRY
# =========================================================
def score_country(country):
    print(f"\n========== Scoring {country} ==========")
    bundle = load_model(country)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]
    bands = bundle["bands"]
    labels = bundle["labels"]

    # Build features for this country
    feats = build_inference_features(country)

    # Convert to pandas — only columns this model expects
    needed = ["customer_id", "snapshot_date"] + feature_cols
    pdf = spark_to_pandas_safe(feats.select(*needed)).fillna(0)

    if len(pdf) == 0:
        print(f"[{country}] no customers to score, skipping")
        return None

    # Coerce to numeric float for XGBoost + SHAP
    X = pdf[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0).astype("float64")

    # Predict
    probs = model.predict_proba(X)[:, 1]
    pdf["risk_score"] = probs
    pdf["binary_pred"] = (probs >= threshold).astype(int)
    pdf["predicted_class"] = pd.cut(
        probs, bins=bands, labels=labels, include_lowest=True
    ).astype(str)
    pdf["country"] = country
    pdf["threshold_used"] = threshold
    pdf["model_path"] = bundle["model_path"]

    # SHAP top driver per customer — native XGBoost (works with XGBoost 2.x)
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    top_idx = np.argmax(np.abs(shap_values), axis=1)
    pdf["top_risk_driver"] = [feature_cols[i] for i in top_idx]

    out_cols = [
        "customer_id", "country", "snapshot_date",
        "risk_score", "binary_pred", "predicted_class",
        "threshold_used", "top_risk_driver", "model_path",
    ]
    print(f"[{country}] scored {len(pdf):,} customers")
    return spark.createDataFrame(pdf[out_cols])


# =========================================================
# MULTI-COUNTRY RUNNER
# =========================================================
def run_all_countries():
    combined = None
    for country in MODEL_REGISTRY.keys():
        try:
            sdf = score_country(country)
            if sdf is None:
                continue
            combined = sdf if combined is None else combined.unionByName(sdf)
        except Exception as e:
            print(f"[{country}] FAILED: {e}")
            continue

    if combined is None:
        raise RuntimeError("No country produced results.")
    return combined


# =========================================================
# WRITE TO DELTA TABLE
# =========================================================
def write_output(predictions_sdf):
    predictions_sdf = predictions_sdf.withColumn("refresh_date", F.current_date())

    predictions_sdf.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(OUTPUT_TABLE)

    print(f"\nWrote {predictions_sdf.count():,} rows to {OUTPUT_TABLE}")


# =========================================================
# RUN
# =========================================================
predictions = run_all_countries()
display(predictions)

write_output(predictions)
