# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster Model Inference Pipeline (V2)
# MAGIC
# MAGIC Scores ALL countries using the cluster models trained by
# MAGIC `train_all_clusters_v2.py`.
# MAGIC
# MAGIC Flow:
# MAGIC   1. Read routing map (country -> cluster model + per-country threshold)
# MAGIC      written by the trainer
# MAGIC   2. Build inference features @ TODAY — V2-fixed feature engineering,
# MAGIC      ONE Spark pass for all routed countries
# MAGIC   3. Per cluster: load registered model, rebuild the country one-hots
# MAGIC      it was trained with, align columns to the booster, score
# MAGIC   4. Apply per-country threshold (from routing map)
# MAGIC      -> binary_pred (1 = HIGH RISK) + Low/Medium/High band
# MAGIC   5. SHAP per customer (native XGBoost pred_contribs):
# MAGIC      top-3 risk drivers + business-English collector_explanations
# MAGIC      (same explain_one wording as inference_original.py)
# MAGIC   6. Union all clusters -> write Delta output table
# MAGIC
# MAGIC Conventions (same as training):
# MAGIC   - risk_score = P(high risk); higher = worse
# MAGIC   - binary_pred = 1 = HIGH RISK (score >= country threshold)

# COMMAND ----------

!pip install -U xgboost mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
UNIFIED_VIEW = "f_erp_glide_o2c_12.table_invoice_unified_master"
ROUTING_TABLE = "f_erp_glide_o2c_12.collection_ml_country_model_map"
OUTPUT_TABLE = "f_erp_glide_o2c_12.collection_ml_customer_cluster_v2"

LOOKBACK_DAYS = 730
WINDOWS = [60, 90, 180]
TOP_K_DRIVERS = 3

# ---- CUSTOMER-UNIVERSE PARITY with inference_original.py ----
# The original scores ~10k FR customers; without these flags this
# pipeline scored ~1.6k. Three universe gates differ:
#
# COUNTRY_FROM:
#   "bukrs"  = first 2 chars of company_code (original derives country
#              from BSID.BUKRS -> book-of-business; an FR-billed Belgian
#              customer IS in scope)                      <- original
#   "master" = unified view country column (customer master geography)
COUNTRY_FROM = "bukrs"
#
# Original applies NO lookback to OPEN invoices — a 3-year-old unpaid
# invoice is live exposure (aged debt = the high-risk book). Lookback
# stays on CLEARED history only, capped by clearing_date like the
# original's behavior join.
OPEN_ITEMS_IGNORE_LOOKBACK = True
#
# Original spine = EVERY distinct customer in BSID, scored even with
# zero net outstanding (credit memos, negative netting). False = only
# customers with total_outstanding > 0.
INCLUDE_ZERO_EXPOSURE = True

# Set True to run the universe funnel diagnostic cell at the bottom
# (counts customers at each gate vs raw BSID — explains any drift).
RUN_DRIFT_DIAGNOSTIC = True
DIAG_COUNTRY = "FR"

# Band construction around the per-country threshold t:
#   Low    : score <  t
#   Medium : t <= score < t + MEDIUM_BAND_WIDTH * (1 - t)
#   High   : above that
# Threshold is the operating point (recall >= 0.80 by construction),
# so everything >= t is actionable; Medium/High split orders the queue.
MEDIUM_BAND_WIDTH = 0.5

# COMMAND ----------

# =========================================================
# ROUTING MAP — written by train_all_clusters_v2.py
# Carries country -> cluster, model_name, per-country threshold.
# =========================================================
routing = spark.table(ROUTING_TABLE).toPandas()
assert len(routing), f"{ROUTING_TABLE} is empty — run train_all_clusters_v2.py first"

routing["country"] = routing["country"].str.upper()
ALL_COUNTRIES = sorted(routing["country"].unique().tolist())
CLUSTER_OF = dict(zip(routing["country"], routing["cluster"]))
MODEL_OF = {cl: f"models:/{name}/latest"
            for cl, name in zip(routing["cluster"], routing["model_name"])}
THRESHOLD_OF = dict(zip(routing["country"], routing["threshold"]))   # per-country
COUNTRIES_OF = routing.groupby("cluster")["country"].apply(sorted).to_dict()

print(f"Routing: {len(ALL_COUNTRIES)} countries -> {routing['cluster'].nunique()} clusters")
print(routing[["country", "cluster", "model_name", "threshold"]]
      .sort_values(["cluster", "country"]).to_string(index=False))

# COMMAND ----------

# =========================================================
# PART A — V2-FIXED FEATURE ENGINEERING (inference @ TODAY)
# =========================================================

def to_date_any(col):
    c = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.to_date(c, "yyyyMMdd"), F.to_date(c))


def load_base_data(countries):
    df = spark.table(UNIFIED_VIEW).select(
        F.col("customer_id"),
        F.col("invoice_id"),
        F.col("line_item"),
        F.col("invoice_amount").cast("double"),
        F.col("open_amount").cast("double"),
        F.to_date(F.col("baseline_date"), "yyyyMMdd").alias("baseline_date"),
        F.to_date(F.col("clearing_date"), "yyyyMMdd").alias("clearing_date"),
        F.col("cash_discount_days_1").cast("int"),
        F.col("cash_discount_days_2").cast("int"),
        F.col("net_payment_days").cast("int"),
        to_date_any("due_date").alias("due_date"),                    # V2 FIX
        # country per COUNTRY_FROM: BUKRS prefix = original semantics
        (F.upper(F.substring(F.col("company_code"), 1, 2))
         if COUNTRY_FROM == "bukrs" else F.upper(F.col("country"))
         ).alias("country"),
        F.col("customer_tenure_days").cast("int"),
        F.col("dunning_level").cast("int"),
        to_date_any("last_dunned_date").alias("last_dunned_date"),    # V2 FIX
        F.col("dunning_count").cast("int"),
        F.col("fin_promised_amt").cast("double"),
        F.col("fin_p2p_state").cast("int"),
        to_date_any("promise_dt").alias("promise_dt"),                # V2 FIX
        F.col("credit_limit").cast("double"),
        F.col("number_of_disputes").cast("int"),
        F.col("open_dispute_amount").cast("double"),
        F.upper(F.col("source")).alias("source"),
    )

    cutoff = F.date_sub(F.current_date(), LOOKBACK_DAYS)
    if OPEN_ITEMS_IGNORE_LOOKBACK:
        # PARITY: open items (BSID) always in scope — aged debt is live
        # exposure. Cleared history capped by CLEARING date (matches
        # the original's behavior-join condition), not baseline date.
        keep = (F.col("source") == "BSID") | (
            (F.col("source") == "BSAD")
            & (F.coalesce(F.col("clearing_date"), F.col("baseline_date")) >= cutoff)
        )
    else:
        keep = F.col("baseline_date") >= cutoff

    return df.filter(
        F.col("baseline_date").isNotNull()
        & keep
        & F.col("source").isin(["BSID", "BSAD"])
        & F.col("country").isin(countries)
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
        F.first("country", ignorenulls=True).alias("country"),
        F.first("customer_tenure_days", ignorenulls=True).alias("customer_tenure_days"),
        F.max("dunning_level").alias("dunning_level"),
        F.max("last_dunned_date").alias("last_dunned_date"),
        F.max("dunning_count").alias("dunning_count"),
        F.max("fin_promised_amt").alias("fin_promised_amt"),
        F.first("fin_p2p_state", ignorenulls=True).alias("fin_p2p_state"),
        F.max("promise_dt").alias("promise_dt"),
        F.first("credit_limit", ignorenulls=True).alias("credit_limit"),
        F.max("number_of_disputes").alias("number_of_disputes"),
        F.max("open_dispute_amount").alias("open_dispute_amount"),
    )


def make_spine(base_df):
    """
    One row per customer, snapshot = TODAY, country carried for routing.
    PARITY: original spine = every distinct customer in BSID (open items
    table), regardless of net exposure. With INCLUDE_ZERO_EXPOSURE the
    spine mirrors that; otherwise any customer with rows in scope.
    """
    src = base_df.filter(F.col("source") == "BSID") \
        if INCLUDE_ZERO_EXPOSURE else base_df
    return src.groupBy("customer_id").agg(
        F.first("country", ignorenulls=True).alias("country")
    ).withColumn("snapshot_date", F.current_date())


def compute_exposure_features(invoice_df, snapshots):
    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")
    joined = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner",
    ).filter(
        (F.col("i.is_cleared") == 0)
        | (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).withColumn(
        "dpd",
        F.when(F.col("i.due_date") <= F.col("s.snapshot_date"),
               F.datediff("s.snapshot_date", "i.due_date")).otherwise(0),
    ).withColumn(
        "invoice_age", F.datediff("s.snapshot_date", "i.baseline_date")
    )

    return joined.groupBy("s.customer_id", "s.snapshot_date").agg(
        # country lives on the spine — not aggregated here, so
        # zero-exposure customers keep their routing country
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
        # At inference the snapshot IS today, so current tenure is correct
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


def compute_behavior_features(invoice_df, snapshots):
    b = invoice_df.alias("b")
    s = snapshots.alias("s")

    hist = b.join(
        s,
        (F.col("b.customer_id") == F.col("s.customer_id"))
        & (F.col("b.is_cleared") == 1)
        & (F.col("b.clearing_date").isNotNull())
        & (F.col("b.clearing_date") <= F.col("s.snapshot_date")),
        "inner",
    ).withColumn(
        "days_to_pay", F.datediff("b.clearing_date", "b.baseline_date")
    ).withColumn(
        "days_late", F.datediff("b.clearing_date", "b.due_date")   # V2 FIX
    )

    aggs = [
        F.avg("days_to_pay").alias("avg_days_to_pay"),
        F.max("days_to_pay").alias("max_days_to_pay"),
        F.avg(F.when(F.col("days_late") <= 0, 1).otherwise(0)).alias("on_time_ratio"),
        F.count("*").alias("total_payments"),
        F.datediff(F.col("s.snapshot_date"), F.max("b.clearing_date")).alias("days_since_last_payment"),
    ]
    for w in WINDOWS:
        cond = F.col("b.clearing_date") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.avg(F.when(cond, F.col("days_to_pay"))).alias(f"avg_days_to_pay_{w}d"),
            F.max(F.when(cond, F.col("days_to_pay"))).alias(f"max_days_to_pay_{w}d"),
            F.avg(F.when(cond, F.when(F.col("days_late") <= 0, 1).otherwise(0))).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"num_payments_{w}d"),
        ])

    return hist.groupBy("b.customer_id", "s.snapshot_date").agg(*aggs)


def compute_dunning_features(invoice_df, snapshots):
    i = invoice_df.alias("i")
    s = snapshots.alias("s")

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


def compute_p2p_features(invoice_df, snapshots):
    i = invoice_df.alias("i")
    s = snapshots.alias("s")

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


def build_inference_features(countries):
    """Snapshot = TODAY, one Spark pass for all routed countries."""
    print(f"Building inference features for {len(countries)} countries...")
    base = ensure_due_date(load_base_data(countries))
    inv = build_invoice_level(base)
    spine = make_spine(base)

    df = (
        spine
        .join(compute_exposure_features(inv, spine), ["customer_id", "snapshot_date"], "left")
        .join(compute_behavior_features(inv, spine), ["customer_id", "snapshot_date"], "left")
        .join(compute_dunning_features(inv, spine),  ["customer_id", "snapshot_date"], "left")
        .join(compute_p2p_features(inv, spine),      ["customer_id", "snapshot_date"], "left")
        .fillna(0)
    )

    if not INCLUDE_ZERO_EXPOSURE:
        # Optional gate — original scores zero-exposure customers too
        df = df.filter(F.col("total_outstanding") > 0)
    print(f"Inference rows: {df.count():,} customers")
    return df

# COMMAND ----------

# =========================================================
# PART B — SCORING
# =========================================================

def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


_MODEL_CACHE = {}


def load_cluster_model(cluster):
    if cluster in _MODEL_CACHE:
        return _MODEL_CACHE[cluster]
    model_path = MODEL_OF[cluster]          # full URI, e.g. models:/name/latest
    model = mlflow.sklearn.load_model(model_path)
    feature_cols = list(model.get_booster().feature_names)
    bundle = {"model": model, "feature_cols": feature_cols, "model_name": model_path}
    _MODEL_CACHE[cluster] = bundle
    print(f"[{cluster}] loaded {model_path} ({len(feature_cols)} features)")
    return bundle


def align_features(pdf, cluster, feature_cols):
    """
    Rebuild what training did:
      - country one-hots for the cluster's member countries
      - any feature the booster expects but the data lacks -> 0
      - exact column order of the booster
    """
    out = pdf.copy()
    for c in COUNTRIES_OF[cluster]:
        out[f"country_{c}"] = (out["country"] == c).astype(int)
    missing = [f for f in feature_cols if f not in out.columns]
    for f in missing:
        out[f] = 0.0
    if missing:
        print(f"[{cluster}] filled {len(missing)} missing features with 0: {missing[:5]}...")

    X = out[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0).astype("float64")


def band_of(prob, t):
    """Low < t <= Medium < t + MEDIUM_BAND_WIDTH*(1-t) <= High."""
    mid = t + MEDIUM_BAND_WIDTH * (1.0 - t)
    return np.where(prob < t, "Low", np.where(prob < mid, "Medium", "High"))


def shap_matrix(model, X):
    """Native XGBoost SHAP — one (n_rows x n_features) contribution matrix."""
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    return contribs[:, :-1]                            # drop bias term


def top_drivers(shap_vals, cols, k=TOP_K_DRIVERS):
    """Top-k |contribution| feature names per row, with +/- direction."""
    order = np.argsort(-np.abs(shap_vals), axis=1)[:, :k]
    cols = np.array(cols)
    signs = np.take_along_axis(shap_vals, order, axis=1) >= 0
    return [
        "; ".join(f"{cols[j]}({'+' if s else '-'})"
                  for j, s in zip(row, sign_row))
        for row, sign_row in zip(order, signs)
    ], [cols[row[0]] for row in order]


# =========================================================
# BUSINESS EXPLANATIONS — same wording as inference_original.py
# =========================================================

def explain_one(feature, value, shap_value):
    impact = "increased risk" if shap_value > 0 else "reduced risk"

    if feature == "total_outstanding":
        return f"Total unpaid balance of {value:,.0f} EUR {impact}."
    elif feature == "total_open_amount":
        return f"Open amount of {value:,.0f} EUR {impact}."
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
    # ---- V2-only features (not in inference_original.py) ----
    elif feature == "credit_limit":
        return f"Credit limit of {value:,.0f} EUR {impact}."
    elif feature == "credit_utilization":
        return f"{value*100:.0f}% of the credit limit is utilized, which {impact}."
    elif feature == "number_of_disputes":
        return f"Customer has {int(value)} disputes, which {impact}."
    elif feature == "open_dispute_amount":
        return f"{value:,.0f} EUR is under open dispute, which {impact}."
    elif feature == "customer_tenure_days":
        return f"Customer relationship is {int(value)} days old, which {impact}."
    else:
        readable = feature.replace("_", " ")
        return f"{readable.capitalize()} value of {value} {impact}."


def build_business_explanations(shap_row, feature_row, feature_cols, top_n=5):
    """Top-n |SHAP| drivers -> business sentences. Country one-hots are
    routing artifacts, not behavior — excluded from the narrative."""
    contribution_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": shap_row,
        "feature_value": feature_row,
    })
    contribution_df = contribution_df[
        ~contribution_df["feature"].str.startswith("country_")
    ]
    contribution_df["abs_shap"] = contribution_df["shap_value"].abs()
    contribution_df = contribution_df.sort_values("abs_shap", ascending=False).head(top_n)

    return [
        explain_one(row["feature"], row["feature_value"], row["shap_value"])
        for _, row in contribution_df.iterrows()
    ]


def score_cluster(cluster, feats_pdf):
    """Score every customer routed to this cluster's model."""
    members = COUNTRIES_OF[cluster]
    pdf = feats_pdf[feats_pdf["country"].isin(members)].copy()
    if len(pdf) == 0:
        print(f"[{cluster}] no customers, skipping")
        return None

    bundle = load_cluster_model(cluster)
    X = align_features(pdf, cluster, bundle["feature_cols"])

    pdf["risk_score"] = bundle["model"].predict_proba(X)[:, 1]

    # Per-country threshold from routing map. binary_pred=1 = HIGH RISK.
    pdf["threshold_used"] = pdf["country"].map(THRESHOLD_OF)
    pdf["binary_pred"] = (pdf["risk_score"] >= pdf["threshold_used"]).astype(int)
    pdf["predicted_class"] = [
        band_of(np.array([p]), t)[0]
        for p, t in zip(pdf["risk_score"], pdf["threshold_used"])
    ]

    # SHAP once -> drivers + business explanations
    shap_vals = shap_matrix(bundle["model"], X)
    drivers, top1 = top_drivers(shap_vals, X.columns)
    pdf["top_risk_drivers"] = drivers
    pdf["top_risk_driver"] = top1
    pdf["collector_explanations"] = [
        " | ".join(build_business_explanations(
            shap_row=shap_vals[i],
            feature_row=X.iloc[i].values,
            feature_cols=list(X.columns),
            top_n=5,
        ))
        for i in range(len(pdf))
    ]

    pdf["cluster"] = cluster
    pdf["model_name"] = bundle["model_name"]
    # Booster feature list as string — matches the existing
    # collection_ml_customer table's model_features column
    pdf["model_features"] = ", ".join(bundle["feature_cols"])

    n_high = int(pdf["binary_pred"].sum())
    print(f"[{cluster}] scored {len(pdf):,} customers, "
          f"{n_high:,} flagged HIGH RISK ({n_high/len(pdf)*100:.1f}%)")
    return pdf

# COMMAND ----------

# =========================================================
# RUN — features once, score per cluster, union, write
# =========================================================

feats = build_inference_features(ALL_COUNTRIES)
feats_pdf = spark_to_pandas_safe(feats)

# Output = ALL feature columns (inference_original.py writes every model
# feature alongside the prediction — downstream apps read them) + preds.
ID_COLS = ["customer_id", "country", "cluster", "snapshot_date"]
PRED_COLS = [
    "risk_score", "binary_pred", "predicted_class", "risk_band",
    "threshold_used", "top_risk_driver", "top_risk_drivers",
    "collector_explanations", "model_features", "model_name",
]
FEATURE_OUT_COLS = [c for c in feats_pdf.columns if c not in ID_COLS]

scored = []
for cluster in sorted(COUNTRIES_OF):
    try:
        out = score_cluster(cluster, feats_pdf)
        if out is not None:
            scored.append(out)
    except Exception as e:
        print(f"[{cluster}] FAILED: {e}")

assert scored, "No cluster produced scores"
predictions = pd.concat(scored, ignore_index=True)


# risk_band — relative tertiles of risk_score within each country
# (inference_original.py: pd.qcut per per-country run). Differs from
# predicted_class, which is the absolute threshold-based band.
def _risk_band(scores):
    try:
        q = pd.qcut(scores, q=3, duplicates="drop")
        labels = ["Low", "Medium", "High"][:len(q.cat.categories)]
        return pd.qcut(scores, q=len(q.cat.categories), labels=labels,
                       duplicates="drop").astype(str)
    except ValueError:
        # fewer distinct scores than bins (tiny country) — single band
        return pd.Series("Low", index=scores.index)


predictions["risk_band"] = (
    predictions.groupby("country")["risk_score"].transform(_risk_band)
)

predictions = predictions[ID_COLS + FEATURE_OUT_COLS + PRED_COLS]

print(f"\nTotal scored: {len(predictions):,} customers "
      f"({len(predictions.columns)} columns)")
print(predictions.groupby(["cluster", "predicted_class"]).size())

# COMMAND ----------

# Sanity views before write
display(spark.createDataFrame(predictions).orderBy(F.desc("risk_score")).limit(50))

summary = predictions.groupby("country").agg(
    customers=("customer_id", "size"),
    high_risk=("binary_pred", "sum"),
    avg_score=("risk_score", "mean"),
    threshold=("threshold_used", "first"),
)
summary["high_risk_pct"] = (summary["high_risk"] / summary["customers"] * 100).round(1)
display(summary.reset_index())

# COMMAND ----------

# =========================================================
# UNIVERSE DRIFT DIAGNOSTIC — where do customers drop vs
# inference_original.py? (original FR spine ~10,183)
# =========================================================
if RUN_DRIFT_DIAGNOSTIC:
    c = DIAG_COUNTRY.upper()

    # Original's spine: distinct KUNNR in raw BSID, country = BUKRS prefix
    raw_bsid_cust = spark.table(
        "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid"
    ).select(
        F.col("KUNNR").alias("customer_id"),
        F.upper(F.substring(F.col("BUKRS"), 1, 2)).alias("ctry"),
    ).filter(F.col("ctry") == c).select("customer_id").distinct()
    n_raw = raw_bsid_cust.count()

    uv_bsid = spark.table(UNIFIED_VIEW) \
        .filter(F.upper(F.col("source")) == "BSID")
    n_uv_bukrs = uv_bsid.filter(
        F.upper(F.substring(F.col("company_code"), 1, 2)) == c
    ).select("customer_id").distinct().count()
    n_uv_master = uv_bsid.filter(
        F.upper(F.col("country")) == c
    ).select("customer_id").distinct().count()

    n_absent_from_view = raw_bsid_cust.join(
        uv_bsid.select("customer_id").distinct(), "customer_id", "left_anti"
    ).count()

    n_loaded = load_base_data([c]).filter(F.col("source") == "BSID") \
        .select("customer_id").distinct().count()
    n_scored = int((predictions["country"] == c).sum())

    print(f"=== {c} universe funnel ===")
    print(f"raw BSID (BUKRS={c}) distinct customers : {n_raw:>8,}   <- original spine")
    print(f"unified view BSID, BUKRS country        : {n_uv_bukrs:>8,}   gap = view coverage")
    print(f"unified view BSID, master-data country  : {n_uv_master:>8,}   gap = country semantics")
    print(f"raw BSID customers ABSENT from view     : {n_absent_from_view:>8,}")
    print(f"after load_base_data (parity filters)   : {n_loaded:>8,}")
    print(f"scored in this run                      : {n_scored:>8,}")
    print("\nWith parity flags on, 'scored' should ~= 'unified view BUKRS'.")
    print("Any gap to raw BSID = rows missing from the unified view itself —")
    print("fix belongs in the view, not this pipeline.")

# COMMAND ----------

# =========================================================
# WRITE OUTPUT
# =========================================================
out_sdf = spark.createDataFrame(predictions) \
    .withColumn("refresh_date", F.current_date())

out_sdf.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(OUTPUT_TABLE)

print(f"Wrote {out_sdf.count():,} rows to {OUTPUT_TABLE}")
