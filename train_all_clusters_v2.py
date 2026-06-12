# Databricks notebook source
# MAGIC %md
# MAGIC # Train ALL Cluster Models (V2)
# MAGIC
# MAGIC **Input:** CLUSTERS dict below — you already know the clusters,
# MAGIC paste them in. No clustering runs here, training only.
# MAGIC
# MAGIC **Output:** one balanced, tuned XGBoost per cluster, all registered
# MAGIC to MLflow, plus a country -> model routing table for inference.
# MAGIC
# MAGIC Per cluster (same recipe as `train_cluster_model_v2.py`):
# MAGIC   - pooled rows, country one-hots
# MAGIC   - walk-forward CV, 30d leakage gap
# MAGIC   - Optuna (50 trials, macro per-country PR-AUC, stability-penalized)
# MAGIC     IF the pool passes the size gate, else conservative fixed params
# MAGIC   - scale_pos_weight = natural x tuned multiplier (class balance)
# MAGIC   - threshold = max precision @ recall >= 0.80, tuned on clean val
# MAGIC   - production refit on full dev, registered per cluster
# MAGIC
# MAGIC Data is loaded ONCE for all countries (single Spark pass), then
# MAGIC sliced per cluster in pandas.
# MAGIC
# MAGIC **V2.2 universe parity (matches predict_all_clusters_v2.py):**
# MAGIC   - COUNTRY_FROM="bukrs": country = company_code prefix (book of
# MAGIC     business). One-hots, cluster slicing, per-country thresholds and
# MAGIC     the routing table are ALL keyed on the same country definition
# MAGIC     inference routes by — no train/serve country skew.
# MAGIC   - OPEN_ITEMS_IGNORE_LOOKBACK=True: open (BSID) invoices always in
# MAGIC     scope (aged debt = live exposure, model must train on dpd>700);
# MAGIC     cleared (BSAD) history capped by CLEARING date, like inference.
# MAGIC
# MAGIC **V2.1 optimizations:**
# MAGIC   - feature selection: near-zero-importance features dropped after a
# MAGIC     first fit, model refit on survivors (less noise on small pools)
# MAGIC   - recency weighting: sample_weight halves every 180d of snapshot
# MAGIC     age — model tracks current payment behavior, not 2-year-old habits
# MAGIC   - per-country thresholds: each member country gets its own operating
# MAGIC     threshold (tuned on its val slice when it has enough positives,
# MAGIC     cluster threshold otherwise) — different base rates need different
# MAGIC     cut points
# MAGIC   - production refit on FULL data (dev+test) after honest evaluation —
# MAGIC     registered model sees the freshest snapshots; reported test
# MAGIC     metrics still come from the dev-trained model
# MAGIC   - multivariate TPE sampler (models hyperparam interactions)
# MAGIC   - vectorized threshold search via precision_recall_curve

# COMMAND ----------

!pip install -U optuna xgboost mlflow seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# =========================================================
# INPUT — paste your clusters here
# =========================================================
CLUSTERS = {
    "apac_big":   ["CN", "AU", "JP", "NZ", "TW"],
    "kr_my":      ["KR", "MY"],
    "sea":        ["SG", "HK", "PH", "TH", "ID", "VN"],
}

ALL_COUNTRIES = sorted({c.upper() for cs in CLUSTERS.values() for c in cs})
assert len(ALL_COUNTRIES) == sum(len(v) for v in CLUSTERS.values()), \
    "Same country appears in two clusters — fix CLUSTERS"
print(f"{len(CLUSTERS)} clusters, {len(ALL_COUNTRIES)} countries: {ALL_COUNTRIES}")

# COMMAND ----------

import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
UNIFIED_VIEW = "f_erp_glide_o2c_12.table_invoice_unified_master"

# Data-as-of anchor — update to the view's refresh date for each run
# (was stale at 2026-03-25, silently dropping the freshest months).
TODAY = "2026-06-07"
LOOKBACK_DAYS = 730
FUTURE_WINDOW_DAYS = 30
SNAPSHOT_STEP_DAYS = 7
HIGH_RISK_THRESHOLD = 0.4
WINDOWS = [60, 90, 180]

# ---- UNIVERSE PARITY with predict_all_clusters_v2.py ----
# Inference scores the original inference_original.py universe; the
# trainer must learn on the SAME universe or the model extrapolates
# at serve time.
#
# COUNTRY_FROM:
#   "bukrs"  = first 2 chars of company_code (book of business) —
#              matches inference routing, one-hots and thresholds
#   "master" = unified view country column (customer master geography)
COUNTRY_FROM = "bukrs"
#
# Open (BSID) invoices always in scope — a 3-year-old unpaid invoice is
# live exposure the model must see in training (dpd > 700). Cleared
# (BSAD) history capped by CLEARING date, matching inference.
OPEN_ITEMS_IGNORE_LOOKBACK = True
#
# NOTE: deliberately NO INCLUDE_ZERO_EXPOSURE flag here — training must
# keep the total_outstanding > 0 filter because the label
# (collection_ratio = collected_30d / total_outstanding) is undefined at
# zero exposure. Inference still scores those customers: their all-zero
# exposure features land them at low risk, which is the right call.

TEST_FRACTION = 0.15
CV_SPLITS = 4
LEAKAGE_GAP_DAYS = 30

MIN_ROWS_TO_TUNE = 500
MIN_POSITIVES_TO_TUNE = 100
N_TRIALS = 50
STABILITY_PENALTY = 0.5

EARLY_STOPPING_ROUNDS = 30
TARGET_RECALL = 0.80
MIN_TEST_POSITIVES = 20

# V2.1 optimizations
IMPORTANCE_CUTOFF = 0.003            # drop features below this importance
MIN_FEATURES_AFTER_SELECT = 10       # never select below this many
MIN_VAL_POSITIVES_PER_COUNTRY = 10   # per-country threshold needs this many
USE_RECENCY_WEIGHTS = True           # weight recent snapshots higher
RECENCY_HALF_LIFE_DAYS = 180         # weight halves every N days of age
REFIT_ON_FULL_DATA = True            # final registered model sees dev+test

MODEL_NAME_PREFIX = "collection_risk_model_cluster_v2"
EXPERIMENT_PATH = ("/Workspace/Users/amiya.x.mandal@gsk.com/APEC/exp/"
                   "collection_risk_all_clusters_v2")

WRITE_ROUTING_TABLE = True
ROUTING_TABLE = "f_erp_glide_o2c_12.collection_ml_country_model_map"

CONSERVATIVE_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "max_delta_step": 3,
}

FIXED_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_estimators": 500,
}

# COMMAND ----------

# =========================================================
# PART A — V2-FIXED FEATURE ENGINEERING (one pass, all countries)
# =========================================================

def to_date_any(col):
    c = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.to_date(c, "yyyyMMdd"), F.to_date(c))


def load_base_data(date_val, countries):
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
        # V2.2 PARITY: country per COUNTRY_FROM — must match inference
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

    cutoff = F.date_sub(F.lit(date_val), LOOKBACK_DAYS)
    if OPEN_ITEMS_IGNORE_LOOKBACK:
        # V2.2 PARITY: open items always in scope; cleared history capped
        # by CLEARING date — identical rule to predict_all_clusters_v2.py
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


def create_snapshots(invoice_df, date_val):
    customers = invoice_df.select("customer_id").distinct()
    # V2 FIX: censoring cap.
    # V2.2: aged open invoices can push min(baseline_date) back years —
    # floor the calendar at the lookback cutoff so snapshots stay inside
    # the period with full cleared-payment history.
    bounds = invoice_df.select(
        F.greatest(
            F.min("baseline_date"),
            F.date_sub(F.lit(date_val), LOOKBACK_DAYS),
        ).alias("min_date"),
        F.date_sub(F.lit(date_val), FUTURE_WINDOW_DAYS).alias("max_date"),
    ).collect()[0]

    calendar = spark.sql(f"""
        SELECT explode(
            sequence(
                to_date('{bounds["min_date"]}'),
                to_date('{bounds["max_date"]}'),
                interval {SNAPSHOT_STEP_DAYS} days
            )
        ) AS snapshot_date
    """)

    return customers.crossJoin(calendar) \
        .withColumn("month_bucket", F.date_format("snapshot_date", "yyyy-MM")) \
        .dropDuplicates(["customer_id", "month_bucket"]) \
        .drop("month_bucket")


def compute_exposure_features(invoice_df, snapshots, as_of_date):
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
        F.first("country", ignorenulls=True).alias("country"),
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
    ).withColumn(
        # V2 FIX: tenure as of snapshot
        "customer_tenure_days",
        F.greatest(
            F.col("customer_tenure_days")
            - F.datediff(F.to_date(F.lit(as_of_date)), F.col("snapshot_date")),
            F.lit(0),
        ),
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


def create_target(invoice_df, snapshots):
    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")

    future = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner",
    ).filter(
        (F.col("i.is_cleared") == 0)
        | (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).filter(
        (F.col("i.is_cleared") == 1)
        & (F.col("i.clearing_date") >= F.col("s.snapshot_date"))
        & (F.col("i.clearing_date") <= F.date_add(F.col("s.snapshot_date"), FUTURE_WINDOW_DAYS))
    )

    return future.groupBy("s.customer_id", "s.snapshot_date").agg(
        F.sum("i.invoice_amount").alias("collected_30d")
    )


def build_dataset(countries):
    print(f"Loading master table for {len(countries)} countries (single pass)...")
    base = ensure_due_date(load_base_data(TODAY, countries))

    inv = build_invoice_level(base)
    snaps = create_snapshots(inv, TODAY)

    exposure = compute_exposure_features(inv, snaps, TODAY)
    behavior = compute_behavior_features(inv, snaps)
    dunning = compute_dunning_features(inv, snaps)
    p2p = compute_p2p_features(inv, snaps)
    target = create_target(inv, snaps)

    df = (
        exposure
        .join(behavior, ["customer_id", "snapshot_date"], "left")
        .join(dunning,  ["customer_id", "snapshot_date"], "left")
        .join(p2p,      ["customer_id", "snapshot_date"], "left")
        .join(target,   ["customer_id", "snapshot_date"], "left")
    )

    df = df.fillna(0).filter(F.col("total_outstanding") > 0)

    df = df.withColumn(
        "collection_ratio", F.col("collected_30d") / F.col("total_outstanding")
    ).withColumn(
        "target",
        F.when(F.col("collection_ratio") < HIGH_RISK_THRESHOLD, 1).otherwise(0),
    )

    print(f"Dataset: {df.count():,} rows")
    return df

# COMMAND ----------

# =========================================================
# PART B — TRAINING MACHINERY (per cluster)
# =========================================================

DROP_COLS = ["customer_id", "snapshot_date", "target",
             "collected_30d", "collection_ratio", "country"]


def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


def build_time_series_splits(pdf, n_splits=CV_SPLITS, gap_days=LEAKAGE_GAP_DAYS):
    dates = np.sort(pdf["snapshot_date"].unique())
    n = len(dates)
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        return []

    splits = []
    for i in range(1, n_splits + 1):
        train_end = dates[fold_size * i - 1]
        val_start = train_end + pd.Timedelta(days=gap_days)
        val_end = dates[n - 1] if i == n_splits \
            else dates[min(fold_size * i + fold_size, n - 1)]

        train_mask = pdf["snapshot_date"] <= train_end
        val_mask = (pdf["snapshot_date"] >= val_start) & (pdf["snapshot_date"] <= val_end)
        if val_mask.sum() == 0 or pdf.loc[train_mask, "target"].nunique() < 2:
            continue
        splits.append((pdf[train_mask].index, pdf[val_mask].index))
    return splits


def recency_weights(dates, as_of):
    """
    V2.1: exponential time-decay sample weights — weight halves every
    RECENCY_HALF_LIFE_DAYS of snapshot age. Payment behavior drifts;
    a 2-year-old snapshot should not pull the trees as hard as last
    month's. Returns None when disabled (XGBoost treats None = equal).
    """
    if not USE_RECENCY_WEIGHTS:
        return None
    age = (as_of - dates).dt.days.clip(lower=0)
    return np.power(0.5, age / RECENCY_HALF_LIFE_DAYS).values


def macro_country_score(model, val_pdf, feature_cols):
    scores = []
    for c, grp in val_pdf.groupby("country"):
        if grp["target"].nunique() < 2:
            continue
        prob = model.predict_proba(grp[feature_cols])[:, 1]
        scores.append(average_precision_score(grp["target"], prob))
    if not scores:
        prob = model.predict_proba(val_pdf[feature_cols])[:, 1]
        return average_precision_score(val_pdf["target"], prob)
    return float(np.mean(scores))


def make_objective(pdf, feature_cols, splits):
    as_of = pdf["snapshot_date"].max()

    def objective(trial):
        params = {
            **FIXED_PARAMS,
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 1, 5),
        }
        spw_mult = trial.suggest_float("scale_pos_weight_mult", 0.5, 1.2)

        fold_scores = []
        for step, (tr_idx, va_idx) in enumerate(splits):
            tr, va = pdf.loc[tr_idx], pdf.loc[va_idx]
            spw = ((tr["target"] == 0).sum() / max((tr["target"] == 1).sum(), 1)) * spw_mult

            model = XGBClassifier(**params, scale_pos_weight=spw,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(tr[feature_cols], tr["target"],
                      sample_weight=recency_weights(tr["snapshot_date"], as_of),
                      eval_set=[(va[feature_cols], va["target"])], verbose=False)

            fold_scores.append(macro_country_score(model, va, feature_cols))
            trial.report(float(np.mean(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores) - STABILITY_PENALTY * np.std(fold_scores))

    return objective


def tune_or_fallback(dev_pdf, feature_cols, splits, cluster_name):
    n_pos = int(dev_pdf["target"].sum())
    if len(dev_pdf) < MIN_ROWS_TO_TUNE or n_pos < MIN_POSITIVES_TO_TUNE or len(splits) < 2:
        print(f"  [{cluster_name}] GATE FAILED ({len(dev_pdf)} rows, {n_pos} pos, "
              f"{len(splits)} folds) — conservative params, no Optuna")
        return dict(CONSERVATIVE_PARAMS), 1.0, None

    print(f"  [{cluster_name}] tuning: {N_TRIALS} trials on {len(dev_pdf):,} rows...")
    study = optuna.create_study(
        direction="maximize",
        # V2.1: multivariate TPE models hyperparam interactions
        # (depth x learning_rate x regularization move together)
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(make_objective(dev_pdf, feature_cols, splits),
                   n_trials=N_TRIALS, gc_after_trial=True)

    best = dict(study.best_params)
    spw_mult = best.pop("scale_pos_weight_mult", 1.0)
    print(f"  [{cluster_name}] best objective={study.best_value:.4f} | {best}")
    return best, spw_mult, study


def threshold_for_precision_at_recall(y_true, y_prob, min_recall=TARGET_RECALL):
    """V2.1: vectorized via precision_recall_curve (exact PR points,
    no 500-step grid). Max precision subject to recall >= floor."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]   # last point has no threshold
    ok = rec >= min_recall
    if not ok.any():
        return float(thr[int(np.argmax(rec))])   # floor unreachable -> max recall
    return float(thr[ok][int(np.argmax(prec[ok]))])


def eval_slice(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
    }


def train_one_cluster(cluster_name, countries, all_pdf):
    """Full tuned training for one cluster. Returns result bundle."""
    print(f"\n================ {cluster_name}: {countries} ================")
    pdf = all_pdf[all_pdf["country"].isin(countries)].copy()
    pdf = pdf.sort_values("snapshot_date").reset_index(drop=True)
    for c in sorted(countries):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
    feature_cols = [c for c in pdf.columns if c not in DROP_COLS]

    n_pos = int(pdf["target"].sum())
    print(f"  rows={len(pdf):,} positives={n_pos} ({pdf['target'].mean()*100:.1f}%)")
    if len(pdf) < 100 or n_pos < 10:
        print(f"  SKIPPED — too small to train anything honest "
              f"(need >=100 rows, >=10 positives). Route these countries "
              f"to a neighbor cluster model instead.")
        return None

    cut_test = pdf["snapshot_date"].quantile(1 - TEST_FRACTION, interpolation="lower")
    dev = pdf[pdf["snapshot_date"] <= cut_test].reset_index(drop=True)
    test = pdf[pdf["snapshot_date"] > cut_test].reset_index(drop=True)

    splits = build_time_series_splits(dev)
    best_params, spw_mult, study = tune_or_fallback(dev, feature_cols, splits, cluster_name)

    # inner train/val for early stopping + threshold
    cut_val = dev["snapshot_date"].quantile(0.85, interpolation="lower")
    train = dev[dev["snapshot_date"] <= cut_val]
    val = dev[dev["snapshot_date"] > cut_val]
    if val["target"].nunique() < 2:
        # tiny pools: fall back to last 15% rows regardless of date
        n_val = max(int(len(dev) * 0.15), 20)
        train, val = dev.iloc[:-n_val], dev.iloc[-n_val:]

    spw = ((train["target"] == 0).sum() / max((train["target"] == 1).sum(), 1)) * spw_mult
    as_of = pdf["snapshot_date"].max()
    w_train = recency_weights(train["snapshot_date"], as_of)

    model = XGBClassifier(**FIXED_PARAMS, **best_params, scale_pos_weight=spw,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(train[feature_cols], train["target"], sample_weight=w_train,
              eval_set=[(val[feature_cols], val["target"])], verbose=False)

    # V2.1: feature selection — drop near-zero-importance noise, refit.
    # Small pools + ~100 features = trees splitting on noise.
    keep = [f for f, imp in zip(feature_cols, model.feature_importances_)
            if imp >= IMPORTANCE_CUTOFF]
    if len(keep) >= MIN_FEATURES_AFTER_SELECT and len(keep) < len(feature_cols):
        print(f"  feature selection: {len(feature_cols)} -> {len(keep)}")
        feature_cols = keep
        model = XGBClassifier(**FIXED_PARAMS, **best_params, scale_pos_weight=spw,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(train[feature_cols], train["target"], sample_weight=w_train,
                  eval_set=[(val[feature_cols], val["target"])], verbose=False)

    # Cluster-level threshold on val (model never trained on it)
    val_prob = model.predict_proba(val[feature_cols])[:, 1]
    threshold = threshold_for_precision_at_recall(val["target"], val_prob)

    # V2.1: per-country thresholds — different base rates need different
    # cut points. Tuned on the country's val slice when it has enough
    # positives; cluster threshold otherwise.
    country_thresholds = {}
    for c in countries:
        vc = val[val["country"] == c]
        if (vc["target"].sum() >= MIN_VAL_POSITIVES_PER_COUNTRY
                and vc["target"].nunique() > 1):
            pcv = model.predict_proba(vc[feature_cols])[:, 1]
            country_thresholds[c] = threshold_for_precision_at_recall(vc["target"], pcv)
        else:
            country_thresholds[c] = threshold

    # test eval: pooled @ cluster threshold, per country @ own threshold
    test_prob = model.predict_proba(test[feature_cols])[:, 1]
    rows = [{"cluster": cluster_name, "slice": "POOLED", "threshold": threshold,
             **eval_slice(test["target"], pd.Series(test_prob, index=test.index), threshold)}]
    for c in countries:
        tc = test[test["country"] == c]
        if len(tc) == 0:
            continue
        pc = model.predict_proba(tc[feature_cols])[:, 1]
        rows.append({"cluster": cluster_name, "slice": c,
                     "threshold": country_thresholds[c],
                     **eval_slice(tc["target"], pd.Series(pc, index=tc.index),
                                  country_thresholds[c])})
    report = pd.DataFrame(rows)

    pooled = report[report["slice"] == "POOLED"].iloc[0]
    if pooled["n_pos"] < MIN_TEST_POSITIVES:
        print(f"  WARN: {int(pooled['n_pos'])} test positives — metrics directional only")
    print(f"  test: recall={pooled['recall']:.2f} precision={pooled['precision']:.2f} "
          f"pr_auc={pooled['pr_auc']:.3f} @ t={threshold:.2f}")

    # V2.1: production refit on FULL data (dev+test) — registered model
    # sees the freshest snapshots. Honest test metrics above come from
    # the dev-trained model; this refit is for deployment only.
    best_n = int(model.best_iteration) + 1 if model.best_iteration is not None \
        else FIXED_PARAMS["n_estimators"]
    prod_fixed = {k: v for k, v in FIXED_PARAMS.items() if k != "n_estimators"}
    fit_pdf = pdf if REFIT_ON_FULL_DATA else dev
    production_model = XGBClassifier(**prod_fixed, **best_params,
                                     n_estimators=best_n, scale_pos_weight=spw)
    production_model.fit(fit_pdf[feature_cols], fit_pdf["target"],
                         sample_weight=recency_weights(fit_pdf["snapshot_date"], as_of),
                         verbose=False)

    return {
        "cluster": cluster_name,
        "countries": countries,
        "model": production_model,
        "eval_model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "country_thresholds": country_thresholds,
        "best_params": best_params,
        "spw_mult": spw_mult,
        "tuned": study is not None,
        "n_estimators": best_n,
        "report": report,
        "test": test,
        "test_prob": test_prob,
        "n_rows": len(pdf),
        "production_rows": len(fit_pdf),
    }

# COMMAND ----------

# =========================================================
# RUN — load once, train every cluster
# =========================================================

training_df = build_dataset(ALL_COUNTRIES)

all_pdf = spark_to_pandas_safe(training_df)
all_pdf["snapshot_date"] = pd.to_datetime(all_pdf["snapshot_date"])
print(all_pdf.groupby("country")["target"].agg(rows="size", positives="sum"))

# COMMAND ----------

mlflow.set_registry_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)

results = []
with mlflow.start_run(run_name="train_all_clusters") as parent:
    mlflow.log_param("clusters", {k: ",".join(v) for k, v in CLUSTERS.items()})

    for cluster_name, countries in CLUSTERS.items():
        countries = [c.upper() for c in countries]
        with mlflow.start_run(run_name=f"cluster_{cluster_name}", nested=True) as child:
            res = train_one_cluster(cluster_name, countries, all_pdf)
            if res is None:
                mlflow.log_param("skipped", True)
                continue

            mlflow.log_param("countries", ",".join(countries))
            mlflow.log_param("tuned", res["tuned"])
            mlflow.log_params({f"hp_{k}": v for k, v in res["best_params"].items()})
            mlflow.log_param("hp_scale_pos_weight_mult", res["spw_mult"])
            mlflow.log_param("chosen_threshold", res["threshold"])
            mlflow.log_param("country_thresholds", res["country_thresholds"])
            mlflow.log_param("n_selected_features", len(res["feature_cols"]))
            mlflow.log_param("production_n_estimators", res["n_estimators"])
            mlflow.log_param("production_rows", res["production_rows"])
            mlflow.log_param("recency_weights", USE_RECENCY_WEIGHTS)
            pooled = res["report"][res["report"]["slice"] == "POOLED"].iloc[0]
            for m in ["recall", "precision", "f2", "pr_auc", "roc_auc"]:
                if not np.isnan(pooled[m]):
                    mlflow.log_metric(f"test_{m}", float(pooled[m]))

            model_name = f"{MODEL_NAME_PREFIX}_{cluster_name}"
            mlflow.sklearn.log_model(
                sk_model=res["model"],
                artifact_path="model",
                input_example=res["test"][res["feature_cols"]].head(5),
            )
            mlflow.register_model(model_uri=f"runs:/{child.info.run_id}/model",
                                  name=model_name)
            res["model_name"] = model_name
            print(f"  registered: {model_name}")
            results.append(res)

# COMMAND ----------

# =========================================================
# COMBINED REPORT + ROUTING MAP
# =========================================================

full_report = pd.concat([r["report"] for r in results], ignore_index=True)
display(full_report.round(3))

routing = pd.DataFrame([
    {"country": c, "cluster": r["cluster"], "model_name": r["model_name"],
     "threshold": r["country_thresholds"][c],            # V2.1: per-country
     "cluster_threshold": r["threshold"],
     "trained_rows": r["n_rows"],
     "tuned": r["tuned"], "refresh_date": TODAY}
    for r in results for c in r["countries"]
])
display(routing)

if WRITE_ROUTING_TABLE and len(routing):
    spark.createDataFrame(routing).write.format("delta") \
        .mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(ROUTING_TABLE)
    print(f"Routing map written to {ROUTING_TABLE}")

# COMMAND ----------

# =========================================================
# SUMMARY PLOTS
# =========================================================

def plot_summary(full_report):
    d = full_report[full_report["slice"] != "POOLED"].copy()
    pooled = full_report[full_report["slice"] == "POOLED"].set_index("cluster")
    clusters = list(pooled.index)
    cluster_color = {c: plt.cm.tab10(i % 10) for i, c in enumerate(clusters)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m in zip(axes, ["recall", "precision", "pr_auc"]):
        colors = [cluster_color[c] for c in d["cluster"]]
        ax.bar(d["slice"], d[m], color=colors)
        for cl in clusters:
            ax.axhline(pooled.loc[cl, m], color=cluster_color[cl],
                       linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(f"test {m} per country (dashes = cluster pooled)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        for i, (v, p) in enumerate(zip(d[m], d["n_pos"])):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}\n(p={int(p)})", ha="center", va="bottom", fontsize=7)
    plt.suptitle("All Cluster Models — Per-Country Test Performance",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_all_confusions(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        y_pred = (r["test_prob"] >= r["threshold"]).astype(int)
        cm = confusion_matrix(r["test"]["target"], y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"])
        ax.set_title(f"{r['cluster']}\nt={r['threshold']:.2f} "
                     f"({'tuned' if r['tuned'] else 'fixed'})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


plot_summary(full_report)
plot_all_confusions(results)

# COMMAND ----------

print("\n=== DONE ===")
for r in results:
    pooled = r["report"][r["report"]["slice"] == "POOLED"].iloc[0]
    print(f"{r['cluster']:>12} {str(r['countries']):<40} "
          f"tuned={r['tuned']} t={r['threshold']:.2f} "
          f"recall={pooled['recall']:.2f} precision={pooled['precision']:.2f} "
          f"pr_auc={pooled['pr_auc']:.3f} -> {r['model_name']}")
skipped = set(CLUSTERS) - {r["cluster"] for r in results}
if skipped:
    print(f"\nSKIPPED (too small): {sorted(skipped)} — route to neighbor cluster")
