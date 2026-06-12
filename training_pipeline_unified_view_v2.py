# Databricks notebook source
# =====================================================================
# V2 — bugfix revision of training_pipeline_unified_view_original.py
#
# Fixes vs original:
#   1. CRITICAL: last_dunned_date / promise_create_dt / promise_dt /
#      dispute_create_date were date_format()'d into 'yyyyMMdd' STRINGS,
#      which silently fail string-vs-date comparisons in the dunning/P2P
#      joins -> all dunning + P2P features were 0. Now parsed with
#      to_date_any() into proper DATE columns.
#   2. Label censoring: snapshot calendar now capped at
#      date_val - FUTURE_WINDOW_DAYS so every snapshot has a full 30-day
#      target window (was generating snapshots up to TODAY with
#      truncated/empty windows -> spurious high-risk labels in test set).
#   3. on_time_ratio now measured against due_date (clearing <= due_date)
#      instead of baseline_date (which made the feature ~always 0).
#      Windowed on_time_ratio_{w}d now averages over window payments
#      only (was diluted by all-history denominator).
#   4. Threshold tuning now uses the inner holdout val set (X_val) that
#      final_model never trained on (was tuned on splits[-1] val fold,
#      which overlaps final_model's training rows).
#   5. customer_tenure_days adjusted to tenure AS OF snapshot
#      (was current tenure leaked into historical snapshots).
#   6. Last CV fold now includes tail dates; snapshot_date coerced to
#      datetime64 before quantile(); removed bare `print` no-op.
#
# Known unified-view limitation (NOT fixable here): dunning_level /
# dunning_count / number_of_disputes / open_dispute_amount are
# current-state denormalized values, not as-of-snapshot. The
# last_dunned_date <= snapshot filter mitigates but does not eliminate
# this. For fully leakage-free dunning/P2P history use the multi-table
# pipeline in collection_risk_model.py (MHND / UDM_P2P_ATTR events).
# =====================================================================
!pip install -U optuna xgboost mlflow shap category-encoders seaborn plotly

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# COMMAND ----------

country_code = "CN"

# COMMAND ----------

import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

import optuna.visualization.matplotlib as ovm

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

UNIFIED_VIEW = "f_erp_glide_o2c_12.table_invoice_unified_master"


def to_date_any(col):
    """
    Parse a column to DATE regardless of source format.
    Handles 'yyyyMMdd' strings (SAP style), ISO strings, and native
    DATE/TIMESTAMP columns. Returns null only if nothing parses.

    V2 FIX: original used date_format() here, which produces a STRING —
    comparing that string against DATE columns in the dunning/P2P joins
    evaluated to null and silently dropped every row.
    """
    c = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.to_date(c, "yyyyMMdd"), F.to_date(c))


def load_base_data(date_val):
    df = spark.table(UNIFIED_VIEW).select(
        F.col("customer_id"),
        F.col("invoice_id"),
        F.col("line_item"),
        F.col("company_code"),
        F.col("currency"),
        F.col("invoice_amount").cast("double"),
        F.col("open_amount").cast("double"),
        F.to_date(F.col("baseline_date"), "yyyyMMdd").alias("baseline_date"),
        F.to_date(F.col("document_entry_date"), "yyyyMMdd").alias("document_entry_date"),
        F.to_date(F.col("clearing_date"), "yyyyMMdd").alias("clearing_date"),
        F.col("cash_discount_days_1").cast("int"),
        F.col("cash_discount_days_2").cast("int"),
        F.col("net_payment_days").cast("int"),
        F.col("payment_terms"),
        to_date_any("due_date").alias("due_date"),  # V2 FIX: DATE, not string
        F.col("days_past_due").cast("int"),
        F.col("invoice_status"),
        F.col("country"),
        F.col("region"),
        F.col("customer_tenure_days").cast("int"),
        # dunning (denormalized)
        F.col("dunning_level").cast("int"),
        to_date_any("last_dunned_date").alias("last_dunned_date"),  # V2 FIX
        F.col("dunning_count").cast("int"),
        # P2P (denormalized)
        F.col("fin_promised_amt").cast("double"),
        F.col("fin_p2p_state").cast("int"),
        to_date_any("promise_create_dt").alias("promise_create_dt"),  # V2 FIX
        to_date_any("promise_dt").alias("promise_dt"),  # V2 FIX
        # credit
        F.col("risk_class"),
        F.col("credit_group"),
        F.col("credit_limit").cast("double"),
        # disputes
        to_date_any("dispute_create_date").alias("dispute_create_date"),  # V2 FIX
        F.col("number_of_disputes").cast("int"),
        F.col("open_dispute_amount").cast("double"),
        # ops
        F.col("payment_method"),
        F.upper(F.col("source")).alias("source"),   # 'BSID' = open, 'BSAD' = cleared
    ).filter(
        F.col("baseline_date").isNotNull()
        & (F.col("baseline_date") >= F.date_sub(F.lit(date_val), LOOKBACK_DAYS))
        & F.col("source").isin(["BSID", "BSAD"])
    )

    return df

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
        F.coalesce(
            F.col("due_date"),
            F.date_add("baseline_date", F.col("payment_terms_days"))
        )
    )


def filter_by_countries(df, countries):
    if not countries:
        return df
    return df.filter(F.col("country").isin([c.upper() for c in countries]))


def build_invoice_level(df):
    return df.groupBy("customer_id", "invoice_id").agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.sum("open_amount").alias("open_amount"),
        F.min("baseline_date").alias("baseline_date"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date"),
        # Invoice cleared if any line item came from BSAD
        F.max(F.when(F.col("source") == "BSAD", 1).otherwise(0)).alias("is_cleared"),
        F.first("invoice_status", ignorenulls=True).alias("invoice_status"),
        F.first("country", ignorenulls=True).alias("country"),
        F.first("region", ignorenulls=True).alias("region"),
        F.first("customer_tenure_days", ignorenulls=True).alias("customer_tenure_days"),

        # dunning (per invoice)
        F.max("dunning_level").alias("dunning_level"),
        F.max("last_dunned_date").alias("last_dunned_date"),
        F.max("dunning_count").alias("dunning_count"),

        # p2p (per invoice — take latest promise)
        F.max("fin_promised_amt").alias("fin_promised_amt"),
        F.first("fin_p2p_state", ignorenulls=True).alias("fin_p2p_state"),
        F.max("promise_dt").alias("promise_dt"),

        # credit (per customer — same across invoices)
        F.first("risk_class", ignorenulls=True).alias("risk_class"),
        F.first("credit_group", ignorenulls=True).alias("credit_group"),
        F.first("credit_limit", ignorenulls=True).alias("credit_limit"),

        # disputes
        F.max("number_of_disputes").alias("number_of_disputes"),
        F.max("open_dispute_amount").alias("open_dispute_amount"),
    )

def create_snapshots(invoice_df, date_val):
    customers = invoice_df.select("customer_id").distinct()

    # V2 FIX: cap snapshots at date_val - FUTURE_WINDOW_DAYS so every
    # snapshot has a FULL 30-day target window. Original generated
    # snapshots up to date_val itself -> truncated/empty windows ->
    # collected_30d underestimated -> spurious high-risk labels
    # concentrated in the (most recent) test split.
    bounds = invoice_df.select(
        F.min("baseline_date").alias("min_date"),
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

    snaps = customers.crossJoin(calendar) \
        .withColumn("month_bucket", F.date_format("snapshot_date", "yyyy-MM")) \
        .dropDuplicates(["customer_id", "month_bucket"]) \
        .drop("month_bucket")

    return snaps

def compute_exposure_features(invoice_df, snapshots, as_of_date):
    # as_of_date: the date the unified view reflects (data load date).
    # Needed to back-date customer_tenure_days to each snapshot.
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
        F.when(
            F.col("i.due_date") <= F.col("s.snapshot_date"),
            F.datediff("s.snapshot_date", "i.due_date"),
        ).otherwise(0),
    ).withColumn(
        "invoice_age", F.datediff("s.snapshot_date", "i.baseline_date")
    )

    return joined.groupBy("s.customer_id", "s.snapshot_date").agg(
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
        # credit + dispute features (per snapshot, taken from open invoices)
        F.max("credit_limit").alias("credit_limit"),
        F.max("number_of_disputes").alias("number_of_disputes"),
        F.sum("open_dispute_amount").alias("open_dispute_amount"),
        F.max("customer_tenure_days").alias("customer_tenure_days"),
        F.first("risk_class", ignorenulls=True).alias("risk_class"),
        F.first("credit_group", ignorenulls=True).alias("credit_group"),
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
        # V2 FIX: tenure as of the SNAPSHOT, not as of the data load.
        # Original leaked current tenure into historical snapshots.
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
        # V2 FIX: on-time means cleared by DUE DATE. Original compared
        # against baseline_date (days_to_pay <= 0), which is only true
        # if paid the day the invoice was issued -> feature ~always 0.
        "days_late", F.datediff("b.clearing_date", "b.due_date")
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
            # V2 FIX: average over window payments only (when() without
            # otherwise() -> nulls outside window are ignored by avg).
            # Original divided by ALL historical payments, diluting the
            # windowed on-time rate.
            F.avg(F.when(cond, F.when(F.col("days_late") <= 0, 1).otherwise(0))).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"num_payments_{w}d"),
        ])

    return hist.groupBy("b.customer_id", "s.snapshot_date").agg(*aggs)



def compute_dunning_features(invoice_df, snapshots):
    # KNOWN LIMITATION (unified view): dunning_level / dunning_count are
    # the CURRENT denormalized values, not as-of-snapshot. The
    # last_dunned_date <= snapshot filter mitigates this but a customer
    # dunned after the snapshot still carries inflated counts. For
    # event-accurate dunning history use MHND (collection_risk_model.py).
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
            F.sum(F.when(cond & (F.col("dunning_level") >= 3),
                         F.col("dunning_count")).otherwise(0))
             .alias(f"high_severity_dunning_{w}d"),
        ])

    return hist.groupBy("i.customer_id", "s.snapshot_date").agg(*aggs).withColumn(
        "high_dunning_ratio",
        F.when(F.col("total_dunning_events") > 0,
               F.col("high_severity_dunning") / F.col("total_dunning_events"))
         .otherwise(0)
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

    # TARGET = invoices open at snapshot AND cleared within next 30 days
    future = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner",
    ).filter(
        # Must have been open at snapshot
        (F.col("i.is_cleared") == 0)
        | (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).filter(
        # AND must clear within snapshot..snapshot+30d
        (F.col("i.is_cleared") == 1)
        & (F.col("i.clearing_date") >= F.col("s.snapshot_date"))
        & (F.col("i.clearing_date") <= F.date_add(F.col("s.snapshot_date"), FUTURE_WINDOW_DAYS))
    )

    return future.groupBy("s.customer_id", "s.snapshot_date").agg(
        F.sum("i.invoice_amount").alias("collected_30d")
    )



def build_training_dataset():
    print("Loading master table...")
    base = load_base_data(TODAY)
    base = ensure_due_date(base)
    base = filter_by_countries(base, COUNTRIES)

    print("Rolling up to invoice level...")
    inv = build_invoice_level(base)

    print("Building snapshots...")
    snaps = create_snapshots(inv, TODAY)

    print("Computing feature layers...")
    exposure = compute_exposure_features(inv, snaps, TODAY)
    behavior = compute_behavior_features(inv, snaps)
    dunning = compute_dunning_features(inv, snaps)
    p2p = compute_p2p_features(inv, snaps)
    target = create_target(inv, snaps)

    print("Joining features...")
    df = (
        exposure
        .join(behavior, ["customer_id", "snapshot_date"], "left")
        .join(dunning,  ["customer_id", "snapshot_date"], "left")
        .join(p2p,      ["customer_id", "snapshot_date"], "left")
        .join(target,   ["customer_id", "snapshot_date"], "left")
    )

    df = df.fillna(0)
    df = df.filter(F.col("total_outstanding") > 0)

    df = df.withColumn(
        "collection_ratio", F.col("collected_30d") / F.col("total_outstanding")
    ).withColumn(
        "target",
        F.when(F.col("collection_ratio") < HIGH_RISK_THRESHOLD, 1).otherwise(0),
    )

    print(f"Training dataset built: {df.count():,} rows")
    return df


def build_inference_dataset(snapshot_date):

    print(f"Loading master table for inference @ {snapshot_date}...")
    base = load_base_data(snapshot_date)
    base = ensure_due_date(base)
    base = filter_by_countries(base, COUNTRIES)

    inv = build_invoice_level(base)

    # Spine = distinct customers with any invoice activity, snapshot
    spine = inv.select("customer_id").distinct() \
        .withColumn("snapshot_date", F.lit(snapshot_date).cast("date"))

    exposure = compute_exposure_features(inv, spine, snapshot_date)
    behavior = compute_behavior_features(inv, spine)
    dunning = compute_dunning_features(inv, spine)
    p2p = compute_p2p_features(inv, spine)

    df = (
        spine
        .join(exposure, ["customer_id", "snapshot_date"], "left")
        .join(behavior, ["customer_id", "snapshot_date"], "left")
        .join(dunning,  ["customer_id", "snapshot_date"], "left")
        .join(p2p,      ["customer_id", "snapshot_date"], "left")
        .fillna(0)
    )

    # Only score customers with live exposure today
    df = df.filter(F.col("total_outstanding") > 0)

    print(f"Inference dataset built: {df.count():,} customers")
    return df


# COMMAND ----------

TODAY = "2026-03-25"
LOOKBACK_DAYS = 730
FUTURE_WINDOW_DAYS = 30
SNAPSHOT_STEP_DAYS = 7
HIGH_RISK_THRESHOLD = 0.4
WINDOWS = [60, 90, 180]
COUNTRIES = [country_code]

# COMMAND ----------

# V2: bumped registry name + experiment so fixed models don't overwrite
# v1 versions trained with the buggy features.
MODEL_NAME = f"collection_risk_model_customer_v_2_{country_code}_optuna"
EXPERIMENT_PATH = f"/Workspace/Users/amiya.x.mandal@gsk.com/APEC/exp/collection_risk_model_customer_v_2_{country_code}"
N_TRIALS = 300
CV_SPLITS = 4
LEAKAGE_GAP_DAYS = 30
EARLY_STOPPING_ROUNDS = 30
IMPORTANCE_CUTOFF = 0.005
TARGET_RECALL = 0.80

# COMMAND ----------

def prepare_ml_data(df):
    drop_cols = [
        "customer_id", "snapshot_date", "target",
        "collected_30d", "collection_ratio", "last_payment_date",
        "country", "region", 'risk_class', 'credit_group'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = df.fillna(0)
    return df, feature_cols


def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(
                field.name, spark_df[field.name].cast(DoubleType())
            )
    return spark_df.fillna(0).toPandas()

# COMMAND ----------

def build_time_series_splits(pdf, n_splits=4, gap_days=30):
    pdf = pdf.sort_values("snapshot_date").reset_index(drop=True)
    dates = np.sort(pd.to_datetime(pdf["snapshot_date"]).unique())
    n = len(dates)
    fold_size = n // (n_splits + 1)

    splits = []
    for i in range(1, n_splits + 1):
        train_end = dates[fold_size * i - 1]
        val_start = train_end + pd.Timedelta(days=gap_days)
        # V2 FIX: last fold extends to the final date so tail snapshots
        # (when n is not divisible by n_splits+1) are not silently dropped.
        val_end = dates[n - 1] if i == n_splits \
            else dates[min(fold_size * i + fold_size, n - 1)]

        train_mask = pd.to_datetime(pdf["snapshot_date"]) <= train_end
        val_mask = (
            (pd.to_datetime(pdf["snapshot_date"]) >= val_start) &
            (pd.to_datetime(pdf["snapshot_date"]) <= val_end)
        )
        if val_mask.sum() == 0:
            continue
        splits.append((pdf[train_mask].index, pdf[val_mask].index))

    print(f"Built {len(splits)} CV folds (gap={gap_days}d)")

    # Imbalance sanity check — warn if any fold has too few positives.
    # For 7:1 data, want at least 100 positives per val fold for stable PR-AUC.
    for i, (tr_idx, val_idx) in enumerate(splits):
        n_pos_tr = int(pdf.loc[tr_idx, "target"].sum())
        n_pos_val = int(pdf.loc[val_idx, "target"].sum())
        ratio_tr = n_pos_tr / max(len(tr_idx), 1)
        ratio_val = n_pos_val / max(len(val_idx), 1)
        flag = "  WARN-sparse-positives" if n_pos_val < 100 else ""
        print(f"  fold {i+1}: train n={len(tr_idx):,} pos={n_pos_tr:,} ({ratio_tr*100:.1f}%) | "
              f"val n={len(val_idx):,} pos={n_pos_val:,} ({ratio_val*100:.1f}%){flag}")

    return splits


def make_objective(pdf, feature_cols, splits):
    def objective(trial):
        # Imbalance-aware hyperparameter search.
        # Tunable knobs added for ~12.5% positive class (7:1 ratio):
        #   - scale_pos_weight_mult: lets Optuna tune around natural ratio
        #     (fixed at n_neg/n_pos can over-correct, hurting precision)
        #   - max_delta_step: XGBoost docs recommend 1-10 for imbalanced
        #     logistic regression — stabilizes gradient updates
        #   - min_child_weight up to 20: prevents tiny-positive leaves
        #     from overfitting on rare class
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        }

        # Scale pos weight = (natural ratio) x (Optuna multiplier).
        # Multiplier 0.3–1.5 lets Optuna pick less-aggressive than natural
        # (improves precision) or more-aggressive (boosts recall).
        spw_mult = trial.suggest_float("scale_pos_weight_mult", 0.3, 1.5)

        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr = pdf.loc[train_idx, feature_cols]
            y_tr = pdf.loc[train_idx, "target"]
            X_val = pdf.loc[val_idx, feature_cols]
            y_val = pdf.loc[val_idx, "target"]

            n_neg = (y_tr == 0).sum()
            n_pos = max((y_tr == 1).sum(), 1)
            spw = (n_neg / n_pos) * spw_mult

            model = XGBClassifier(
                **params,
                scale_pos_weight=spw,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            y_prob = model.predict_proba(X_val)[:, 1]

            # Balanced objective — still recall-leaning but precision counts:
            #   0.5 * PR-AUC          (ranking quality across all thresholds)
            #   0.3 * F2 @ chosen     (recall x4 over precision at op-threshold)
            #   0.2 * recall@top-20%  (capacity-bounded recall)
            #
            # F2 keeps precision in the loss — without it Optuna picks models
            # that flag everyone. Recall@top-20% prevents PR-AUC games where
            # ranking is good overall but worst customers slip through.
            pr_auc = average_precision_score(y_val, y_prob)
            y_pred_at_thresh = (y_prob >= 0.5).astype(int)
            f2 = fbeta_score(y_val, y_pred_at_thresh, beta=2, zero_division=0)
            r_at_20 = recall_at_top_k(y_val, y_prob, k_pct=0.20)

            score = 0.5 * pr_auc + 0.3 * f2 + 0.2 * r_at_20

            fold_scores.append(score)

            trial.report(float(np.mean(fold_scores)), step=len(fold_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def select_features(model, feature_cols, cutoff=IMPORTANCE_CUTOFF):
    importance = model.feature_importances_
    keep = [f for f, imp in zip(feature_cols, importance) if imp >= cutoff]
    dropped = [f for f in feature_cols if f not in keep]
    print(f"Kept {len(keep)} features. Dropped {len(dropped)}: {dropped}")
    return keep


def optimal_threshold(y_true, y_prob, metric="f2"):
    """
    F2 metric heavily weights recall (recall counts 4x more than precision).
    Use 'f3' to weight recall even more aggressively if collections team
    cannot afford to miss high-risk customers.
    """
    thresholds = np.linspace(0.05, 0.7, 200)
    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f2":
            scores.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        elif metric == "f3":
            scores.append(fbeta_score(y_true, y_pred, beta=3, zero_division=0))
        else:
            scores.append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def threshold_for_recall(y_true, y_prob, target_recall=TARGET_RECALL):
    """
    Find the HIGHEST threshold whose recall is >= target_recall.
    Why highest: as threshold decreases, recall increases (more positives
    admitted). The highest threshold still meeting the floor gives the
    smallest positive set that satisfies recall — maximizes precision
    under the recall constraint.
    """
    thresholds = np.linspace(0.001, 0.9, 500)
    best_t = None
    # Iterate high -> low; first hit is the highest threshold meeting floor
    for t in thresholds[::-1]:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        recall = tp / (tp + fn + 1e-9)
        if recall >= target_recall:
            best_t = t
            break
    if best_t is None:
        # No threshold met the floor — fall back to lowest threshold (max recall)
        best_t = thresholds.min()
    return float(best_t)


def threshold_for_precision_at_recall(y_true, y_prob, min_recall=TARGET_RECALL):
    """
    Pick threshold that maximizes precision while keeping recall >= floor.
    Best operational trade-off — recall guaranteed, precision optimized.
    """
    thresholds = np.linspace(0.001, 0.9, 500)
    best_t, best_prec = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        recall = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        if recall >= min_recall and precision > best_prec:
            best_prec = precision
            best_t = t
    return float(best_t), float(best_prec)


def evaluate(model, X_test, y_test, threshold):
    """
    Recall-focused evaluation. Key metrics for collections:
      - recall_high_risk: % of true high-risk customers caught
      - fn_count: how many high-risk customers we missed
      - precision: false-alarm cost (collector workload)
      - pr_auc: ranking quality across thresholds
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tn = int(((y_test == 0) & (y_pred == 0)).sum())

    metrics = {
        # PRIMARY (recall-focused)
        "recall":           tp / (tp + fn + 1e-9),
        "fn_rate":          fn / (tp + fn + 1e-9),
        "fn_count":         fn,
        "tp_count":         tp,
        # SUPPORTING
        "precision":        tp / (tp + fp + 1e-9),
        "f1":               f1_score(y_test, y_pred, zero_division=0),
        "f2":               fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "f3":               fbeta_score(y_test, y_pred, beta=3, zero_division=0),
        "pr_auc":           average_precision_score(y_test, y_prob),
        "roc_auc":          roc_auc_score(y_test, y_prob),
        # RANKING QUALITY (recall at top-K%)
        "recall_at_top_10": recall_at_top_k(y_test, y_prob, k_pct=0.10),
        "recall_at_top_20": recall_at_top_k(y_test, y_prob, k_pct=0.20),
        "recall_at_top_30": recall_at_top_k(y_test, y_prob, k_pct=0.30),
        # ADDED LATER
        "y_prob":           y_prob,
    }

    print(f"\n=== Evaluation @ threshold={threshold:.3f} ===")
    print("\n--- RECALL METRICS (primary) ---")
    print(f"recall              {metrics['recall']:.4f}    (% of true high-risk caught)")
    print(f"fn_rate             {metrics['fn_rate']:.4f}    (% of high-risk MISSED)")
    print(f"fn_count            {metrics['fn_count']:,}     (high-risk customers missed)")
    print(f"tp_count            {metrics['tp_count']:,}     (high-risk customers caught)")

    print("\n--- SUPPORTING ---")
    print(f"precision           {metrics['precision']:.4f}    (alert quality)")
    print(f"f1                  {metrics['f1']:.4f}")
    print(f"f2 (recall x4)      {metrics['f2']:.4f}")
    print(f"f3 (recall x9)      {metrics['f3']:.4f}")
    print(f"pr_auc              {metrics['pr_auc']:.4f}    (ranking quality)")
    print(f"roc_auc             {metrics['roc_auc']:.4f}")

    print("\n--- RECALL AT TOP-K% (for collector capacity planning) ---")
    print(f"recall@top 10%      {metrics['recall_at_top_10']:.4f}")
    print(f"recall@top 20%      {metrics['recall_at_top_20']:.4f}")
    print(f"recall@top 30%      {metrics['recall_at_top_30']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion matrix:")
    print(f"             pred Low  pred High")
    print(f"act Low      {tn:>8}   {fp:>8}")
    print(f"act High     {fn:>8}   {tp:>8}   <-- {fn} MISSED")
    return metrics


def recall_at_top_k(y_true, y_prob, k_pct=0.20):
    """
    Recall if we only act on top K% scored customers.
    Models capacity-constrained collections workflow.
    """
    n_top = max(1, int(len(y_prob) * k_pct))
    top_idx = np.argsort(y_prob)[-n_top:]
    tp = int(y_true.iloc[top_idx].sum()) if hasattr(y_true, "iloc") \
         else int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())
    return tp / (total_pos + 1e-9)


def run_optuna_training(training_df):
    df, all_features = prepare_ml_data(training_df)
    pdf = spark_to_pandas_safe(df.select(all_features + ["target", "snapshot_date"]))
    # V2 FIX: Spark DateType can land as object dtype (datetime.date) in
    # pandas; quantile()/comparisons need datetime64.
    pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"])
    print(f"\nDataset: {len(pdf):,} rows | {len(all_features)} features")
    print(f"Class balance:\n{pdf['target'].value_counts(normalize=True)}")

    pdf = pdf.sort_values("snapshot_date").reset_index(drop=True)
    cutoff = pdf["snapshot_date"].quantile(0.85, interpolation="lower")
    dev_pdf = pdf[pdf["snapshot_date"] <= cutoff].reset_index(drop=True)
    test_pdf = pdf[pdf["snapshot_date"] > cutoff].reset_index(drop=True)
    print(f"Dev: {len(dev_pdf):,} | Test: {len(test_pdf):,} (cutoff={cutoff})")

    splits = build_time_series_splits(dev_pdf, n_splits=CV_SPLITS, gap_days=LEAKAGE_GAP_DAYS)

    with mlflow.start_run(run_name="optuna_search") as parent_run:

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        )
        study.optimize(
            make_objective(dev_pdf, all_features, splits),
            n_trials=N_TRIALS,
            gc_after_trial=True,
        )

        best_params = study.best_params
        # study.best_value is the COMPOSITE objective:
        #   score = 0.5 * PR-AUC + 0.3 * F2 + 0.2 * recall@top-20%
        print(f"\nBest CV composite score: {study.best_value:.4f}")
        print(f"  (= 0.5 * PR-AUC + 0.3 * F2 + 0.2 * recall@top-20%)")
        print(f"Best params: {best_params}")

        mlflow.log_metric("best_cv_composite_score", study.best_value)
        mlflow.log_param("cv_objective", "0.5*pr_auc + 0.3*f2 + 0.2*recall_at_top_20")
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_splits", CV_SPLITS)

        # Split tunable spw multiplier OUT of XGBoost params
        spw_mult = best_params.pop("scale_pos_weight_mult", 1.0)

        X_dev = dev_pdf[all_features]
        y_dev = dev_pdf["target"]
        X_test = test_pdf[all_features]
        y_test = test_pdf["target"]

        # Apply the tuned multiplier — matches Optuna trials exactly
        natural_spw = (y_dev == 0).sum() / max((y_dev == 1).sum(), 1)
        spw = natural_spw * spw_mult
        print(f"Class balance: natural spw={natural_spw:.2f}, "
              f"tuned mult={spw_mult:.2f}, final spw={spw:.2f}")
        mlflow.log_metric("natural_scale_pos_weight", natural_spw)
        mlflow.log_metric("final_scale_pos_weight", spw)

        # Train/val for learning curve — split dev into train/val time-based
        cutoff_inner = dev_pdf["snapshot_date"].quantile(0.85, interpolation="lower")
        train_mask = dev_pdf["snapshot_date"] <= cutoff_inner
        val_mask = dev_pdf["snapshot_date"] > cutoff_inner
        X_train = X_dev[train_mask]
        y_train = y_dev[train_mask]
        X_val = X_dev[val_mask]
        y_val = y_dev[val_mask]

        # Init model w/ eval_set + early stopping (same as Optuna trials)
        init_model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            random_state=42,
            scale_pos_weight=spw,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        init_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        selected = select_features(init_model, all_features, IMPORTANCE_CUTOFF)
        mlflow.log_param("n_selected_features", len(selected))

        # Final retrain on selected features (same config)
        final_model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            random_state=42,
            scale_pos_weight=spw,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        final_model.fit(
            X_train[selected], y_train,
            eval_set=[(X_train[selected], y_train), (X_val[selected], y_val)],
            verbose=False,
        )

        # V2 FIX: tune the operating threshold on the inner holdout val
        # set, which final_model never trained on. Original used
        # splits[-1]'s val fold (~last 20% of dev dates), which overlaps
        # final_model's training rows (<= 85% quantile) -> threshold was
        # tuned partly on seen data.
        y_val_prob = final_model.predict_proba(X_val[selected])[:, 1]
        y_val_true = y_val

        # Three candidate thresholds:
        t_f2, _ = optimal_threshold(y_val_true, y_val_prob, metric="f2")
        t_f3, _ = optimal_threshold(y_val_true, y_val_prob, metric="f3")
        t_recall = threshold_for_recall(y_val_true, y_val_prob, TARGET_RECALL)
        t_prec_at_recall, prec_at = threshold_for_precision_at_recall(
            y_val_true, y_val_prob, min_recall=TARGET_RECALL
        )

        # PRIMARY = max precision while recall >= TARGET_RECALL
        # Balanced: respects recall floor BUT optimizes precision.
        # Avoids alerting on every customer just to chase recall.
        chosen = t_prec_at_recall

        print(f"\n--- THRESHOLD CANDIDATES ---")
        print(f"F2-optimal                 : {t_f2:.3f}")
        print(f"F3-optimal (recall-heavy)  : {t_f3:.3f}")
        print(f"Recall floor = {TARGET_RECALL:.2f}      : {t_recall:.3f}  (recall-extreme)")
        print(f"Max precision @ recall>={TARGET_RECALL:.2f}: {t_prec_at_recall:.3f}  <-- CHOSEN")
        print(f"  -> precision={prec_at:.3f} while keeping recall floor met")

        mlflow.log_metric("threshold_f2", t_f2)
        mlflow.log_metric("threshold_f3", t_f3)
        mlflow.log_metric("threshold_recall_floor", t_recall)
        mlflow.log_metric("threshold_max_precision_at_recall", t_prec_at_recall)
        mlflow.log_metric("precision_at_recall_floor", prec_at)
        mlflow.log_param("chosen_threshold", chosen)
        mlflow.log_param("threshold_strategy", "max_precision_at_recall_floor")
        mlflow.log_param("target_recall", TARGET_RECALL)

        # ---- Evaluate the early-stopped model on holdout test FIRST
        # This is the honest test metric (model never saw test data)
        results = evaluate(final_model, X_test[selected], y_test, chosen)
        for k in ("recall", "fn_rate", "precision", "f1", "f2", "f3",
                  "pr_auc", "roc_auc",
                  "recall_at_top_10", "recall_at_top_20", "recall_at_top_30"):
            mlflow.log_metric(f"test_{k}", results[k])
        mlflow.log_metric("test_fn_count", results["fn_count"])
        mlflow.log_metric("test_tp_count", results["tp_count"])

        # ---- PRODUCTION REFIT — use full dev set (train + val combined)
        # n_estimators locked to best_iteration discovered via early stopping.
        # This is the model registered to MLflow.
        best_n_trees = int(final_model.best_iteration) + 1 \
            if hasattr(final_model, "best_iteration") and final_model.best_iteration is not None \
            else int(best_params.get("n_estimators", 500))

        prod_params = {k: v for k, v in best_params.items() if k != "n_estimators"}

        production_model = XGBClassifier(
            **prod_params,
            n_estimators=best_n_trees,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            random_state=42,
            scale_pos_weight=spw,
        )
        production_model.fit(X_dev[selected], y_dev, verbose=False)

        print(f"\nProduction refit: trained on {len(X_dev):,} rows "
              f"with n_estimators={best_n_trees} (locked from early stopping)")
        mlflow.log_param("production_n_estimators", best_n_trees)
        mlflow.log_param("production_train_rows", len(X_dev))

        # Sanity check on test set — should match or beat final_model
        prod_results = evaluate(production_model, X_test[selected], y_test, chosen)
        for k in ("recall", "fn_rate", "precision", "f1", "f2",
                  "pr_auc", "roc_auc"):
            mlflow.log_metric(f"prod_test_{k}", prod_results[k])

        # Feature importance from production model
        fi_df = pd.DataFrame({
            "feature": selected,
            "importance": production_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_df.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")

        # ---- Register PRODUCTION model (not final_model)
        mlflow.sklearn.log_model(
            sk_model=production_model,
            artifact_path="model",
            input_example=X_dev[selected].head(5),
        )
        model_uri = f"runs:/{parent_run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"\nProduction model registered as: {MODEL_NAME}")

        return {
            "model": production_model,
            "early_stopped_model": final_model,
            "selected_features": selected,
            "threshold": chosen,
            "best_params": best_params,
            "test_metrics": {k: results[k] for k in ("roc_auc", "pr_auc", "f1", "f2")},
            "study": study,
            "pdf": pdf,
            "test_pdf": test_pdf,
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "y_prob_test": results["y_prob"],
        }


# COMMAND ----------

# =========================================================
# PART D — DIAGNOSTIC PLOTS
# =========================================================

# ---------- D.1 Data exploration ----------

def plot_target_balance(training_df):
    counts = training_df.groupBy("target").count().toPandas()
    counts["target"] = counts["target"].map({0: "Low Risk", 1: "High Risk"})

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts["target"], counts["count"], color=["#2ecc71", "#e74c3c"])
    ax.set_title("Target Class Balance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:,}\n({val/counts['count'].sum()*100:.1f}%)",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_collection_ratio(training_df, sample_frac=0.1):
    pdf_s = training_df.sample(sample_frac, seed=42).select("collection_ratio").toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(pdf_s["collection_ratio"], bins=100, color="#3498db", edgecolor="black")
    axes[0].axvline(0.4, color="red", linestyle="--", linewidth=2,
                    label="HIGH_RISK_THRESHOLD = 0.4")
    axes[0].set_title("Collection Ratio Distribution")
    axes[0].set_xlabel("Collection Ratio")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    sorted_vals = np.sort(pdf_s["collection_ratio"])
    cdf = np.arange(len(sorted_vals)) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf, color="#9b59b6", linewidth=2)
    axes[1].axvline(0.4, color="red", linestyle="--", linewidth=2, label="threshold = 0.4")
    axes[1].set_title("Collection Ratio CDF")
    axes[1].set_xlabel("Collection Ratio")
    axes[1].set_ylabel("Cumulative % of Customers")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


def plot_threshold_sensitivity(training_df):
    thresholds = np.arange(0, 1.05, 0.05)
    rows = []
    for t in thresholds:
        counts = training_df.withColumn(
            "tgt", F.when(F.col("collection_ratio") < t, 1).otherwise(0)
        ).groupBy("tgt").count().collect()
        result = {r["tgt"]: r["count"] for r in counts}
        high = result.get(1, 0)
        total = sum(result.values())
        rows.append({"threshold": t, "high_risk_pct": high / total})
    df_t = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_t["threshold"], df_t["high_risk_pct"] * 100,
            marker="o", color="#e74c3c", linewidth=2)
    ax.axvline(0.4, color="black", linestyle="--", label="Chosen threshold = 0.4")
    ax.set_title("High-Risk Customer % vs Collection-Ratio Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("% of Customers Labeled High Risk")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_snapshots_over_time(training_df):
    counts = training_df.groupBy("snapshot_date").agg(
        F.count("*").alias("rows"),
        F.sum("target").alias("high_risk_count"),
    ).orderBy("snapshot_date").toPandas()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(counts["snapshot_date"], counts["rows"], color="#3498db", alpha=0.6)
    ax1.set_xlabel("Snapshot date")
    ax1.set_ylabel("Total rows", color="#3498db")

    ax2 = ax1.twinx()
    ax2.plot(counts["snapshot_date"],
             counts["high_risk_count"] / counts["rows"] * 100,
             color="#e74c3c", marker="o")
    ax2.set_ylabel("High-Risk %", color="#e74c3c")

    ax1.set_title("Snapshot Volume + High-Risk Rate Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(pdf, top_n=25):
    numeric = pdf.select_dtypes(include=[np.number])
    top_vars = numeric.var().sort_values(ascending=False).head(top_n).index
    corr = numeric[top_vars].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title(f"Feature Correlation (Top {top_n} by variance)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions_by_target(pdf, features_to_plot):
    n = len(features_to_plot)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    last_i = 0
    for i, feat in enumerate(features_to_plot):
        last_i = i
        ax = axes[i]
        for target_val, color, label in [(0, "#2ecc71", "Low Risk"),
                                          (1, "#e74c3c", "High Risk")]:
            data = pdf.loc[pdf["target"] == target_val, feat]
            data = data[data.notna() & np.isfinite(data)]
            if len(data) == 0:
                continue
            ax.hist(data, bins=50, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(feat)
        ax.legend()

    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Feature Distributions by Target Class",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_missing_values(pdf):
    missing = pdf.isnull().sum() / len(pdf) * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.3)))
    ax.barh(missing.index, missing.values, color="#e67e22")
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Values by Feature")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# ---------- D.2 Optuna search ----------

def _resize_optuna_axes(ax, w, h):
    """ovm.* returns Axes (or array of Axes). Resize via .figure."""
    if hasattr(ax, "figure"):
        ax.figure.set_size_inches(w, h)
    elif hasattr(ax, "flat"):  # ndarray of Axes (e.g. plot_slice)
        ax.flat[0].figure.set_size_inches(w, h)
    else:
        plt.gcf().set_size_inches(w, h)


def plot_optuna_history(study):
    ax = ovm.plot_optimization_history(study)
    _resize_optuna_axes(ax, 10, 5)
    plt.title("Optuna Optimization History")
    plt.tight_layout()
    plt.show()


def plot_optuna_param_importance(study):
    ax = ovm.plot_param_importances(study)
    _resize_optuna_axes(ax, 8, 5)
    plt.title("Hyperparameter Importance")
    plt.tight_layout()
    plt.show()


def plot_optuna_parallel(study):
    ax = ovm.plot_parallel_coordinate(study)
    _resize_optuna_axes(ax, 14, 6)
    plt.title("Parallel Coordinates — All Trials")
    plt.tight_layout()
    plt.show()


def plot_optuna_slice(study):
    ax = ovm.plot_slice(study)
    _resize_optuna_axes(ax, 14, 5)
    plt.tight_layout()
    plt.show()


def plot_optuna_contour(study, params=("max_depth", "learning_rate")):
    ax = ovm.plot_contour(study, params=list(params))
    _resize_optuna_axes(ax, 8, 6)
    plt.title(f"Contour: {params[0]} vs {params[1]}")
    plt.tight_layout()
    plt.show()


# ---------- D.3 Model evaluation ----------

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#3498db", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#9b59b6", linewidth=2, label=f"AP = {ap:.4f}")
    ax.axhline(baseline, color="black", linestyle="--", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=("Low Risk", "High Risk")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(labels), yticklabels=list(labels), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_threshold_vs_metrics(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 100)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        })
    df_t = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_t["threshold"], df_t["precision"], label="Precision", color="#3498db", linewidth=2)
    ax.plot(df_t["threshold"], df_t["recall"], label="Recall", color="#e74c3c", linewidth=2)
    ax.plot(df_t["threshold"], df_t["f1"], label="F1", color="#2ecc71", linewidth=2)
    ax.plot(df_t["threshold"], df_t["f2"], label="F2", color="#f39c12", linewidth=2)
    best_f2 = df_t.loc[df_t["f2"].idxmax()]
    ax.axvline(best_f2["threshold"], linestyle="--", color="black",
               label=f"Best F2 t={best_f2['threshold']:.2f}")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision/Recall/F1/F2 vs Threshold")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_calibration(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", color="#9b59b6", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predicted_probs_by_class(y_true, y_prob):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.5,
            color="#2ecc71", label="Actual Low Risk", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.5,
            color="#e74c3c", label="Actual High Risk", density=True)
    ax.set_xlabel("Predicted Probability of High Risk")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability by True Class")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------- D.4 Feature importance + SHAP ----------

def plot_feature_importance(model, feature_cols, top_n=20):
    importance = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_cols, "importance": importance}) \
        .sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.3)))
    ax.barh(fi["feature"], fi["importance"], color="#34495e")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.show()


def _coerce_numeric(X):
    """Force all columns to numeric float. SHAP fails on object dtype."""
    X_num = X.copy()
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    return X_num.fillna(0).astype("float64")


def plot_shap_summary(model, X_sample):
    X_clean = _coerce_numeric(X_sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)

    shap.summary_plot(shap_values, X_clean, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_clean, plot_type="bar", show=False)
    plt.title("SHAP Mean |Value| (Bar)")
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(model, X_sample, feature):
    X_clean = _coerce_numeric(X_sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)
    shap.dependence_plot(feature, shap_values, X_clean, show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(model, X_sample, row_idx=0):
    X_clean = _coerce_numeric(X_sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_clean.iloc[[row_idx]])
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"SHAP Waterfall — Customer #{row_idx}")
    plt.tight_layout()
    plt.show()


# ---------- D.5 Learning curves ----------

def plot_learning_curve(model):
    if not hasattr(model, "evals_result_") or not model.evals_result_:
        print("Model not trained with eval_set. Skipping learning curve.")
        return

    results = model.evals_result_
    metric = list(results["validation_0"].keys())[0]
    train_scores = results["validation_0"][metric]
    val_scores = results.get("validation_1", {}).get(metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_scores, label="Train", color="#3498db", linewidth=2)
    if val_scores:
        ax.plot(val_scores, label="Validation", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning Curve ({metric})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------- D.5b Recall-focused plots ----------

def plot_recall_at_top_k(y_true, y_prob):
    """Recall vs % of customers acted on. Capacity-planning chart."""
    ks = np.linspace(0.01, 1.0, 100)
    recalls = [recall_at_top_k(y_true, y_prob, k_pct=k) for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks * 100, np.array(recalls) * 100, color="#e74c3c", linewidth=2)
    for k_mark in (0.10, 0.20, 0.30, 0.50):
        r = recall_at_top_k(y_true, y_prob, k_pct=k_mark)
        ax.scatter([k_mark * 100], [r * 100], color="black", zorder=5)
        ax.annotate(f"{int(k_mark*100)}% -> R={r*100:.0f}%",
                    (k_mark * 100, r * 100),
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("% of Customers Acted On (Top-K%)")
    ax.set_ylabel("Recall of High-Risk Customers (%)")
    ax.set_title("Recall vs Collector Capacity")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_recall_vs_threshold(y_true, y_prob, target_recall=TARGET_RECALL):
    """Shows where the recall floor lands across thresholds."""
    thresholds = np.linspace(0.01, 0.95, 200)
    recalls = []
    precisions = []
    fn_counts = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        recalls.append(tp / (tp + fn + 1e-9))
        precisions.append(tp / (tp + fp + 1e-9))
        fn_counts.append(fn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(thresholds, recalls, color="#e74c3c", linewidth=2, label="Recall")
    axes[0].plot(thresholds, precisions, color="#3498db", linewidth=2, label="Precision")
    axes[0].axhline(target_recall, color="black", linestyle="--",
                    label=f"Recall floor = {target_recall}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Recall & Precision vs Threshold")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(thresholds, fn_counts, color="#e67e22", linewidth=2)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("False Negatives (missed high-risk)")
    axes[1].set_title("Missed High-Risk Customers vs Threshold")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_recall_by_band(test_pdf, y_prob, y_true,
                         bins=(0, 0.4, 0.7, 1.0),
                         labels=("Low", "Medium", "High")):
    """Where do actual high-risk customers fall across predicted bands."""
    df_r = pd.DataFrame({
        "y_true": (y_true.values if hasattr(y_true, "values") else y_true),
        "y_prob": y_prob,
    })
    df_r["band"] = pd.cut(df_r["y_prob"], bins=list(bins),
                          labels=list(labels), include_lowest=True)

    grp = df_r.groupby("band").agg(
        total=("y_true", "size"),
        actual_high=("y_true", "sum"),
    )
    grp["recall_share"] = grp["actual_high"] / grp["actual_high"].sum()
    grp["high_rate"] = grp["actual_high"] / grp["total"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    axes[0].bar(grp.index.astype(str), grp["recall_share"] * 100, color=colors)
    axes[0].set_title("Where Actual High-Risk Customers Live")
    axes[0].set_ylabel("% of Total High-Risk Caught in Band")
    for i, v in enumerate(grp["recall_share"] * 100):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    axes[1].bar(grp.index.astype(str), grp["high_rate"] * 100, color=colors)
    axes[1].set_title("Precision Within Each Band")
    axes[1].set_ylabel("% Actual High-Risk in Band")
    for i, v in enumerate(grp["high_rate"] * 100):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


# ---------- D.6 Risk bands ----------

def plot_risk_band_distribution(y_prob, bins=(0, 0.4, 0.7, 1.0),
                                 labels=("Low", "Medium", "High")):
    bands = pd.cut(y_prob, bins=list(bins), labels=list(labels), include_lowest=True)
    counts = bands.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_title("Risk Band Distribution")
    ax.set_ylabel("Customers")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:,}\n({val/counts.sum()*100:.1f}%)",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_collection_ratio_by_band(test_pdf, y_prob,
                                    bins=(0, 0.4, 0.7, 1.0),
                                    labels=("Low", "Medium", "High")):
    df_b = test_pdf.copy()
    df_b["risk_band"] = pd.cut(y_prob, bins=list(bins),
                                labels=list(labels), include_lowest=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    df_b.boxplot(column="collection_ratio", by="risk_band", ax=ax)
    ax.set_title("Collection Ratio by Risk Band (Test Set)")
    ax.set_xlabel("Risk Band")
    ax.set_ylabel("Actual Collection Ratio (next 30d)")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()


# ---------- Master plot runner ----------

def run_all_plots(training_df, output):
    """All diagnostic plots end-to-end using the output dict from run_optuna_training()."""

    pdf = output["pdf"]
    model = output["model"]
    selected = output["selected_features"]
    study = output["study"]
    X_test = output["X_test"]
    y_test = output["y_test"]
    test_pdf = output["test_pdf"]
    y_prob = output["y_prob_test"]
    threshold = output["threshold"]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n========== 1. DATA EXPLORATION ==========\n")
    plot_target_balance(training_df)
    plot_collection_ratio(training_df)
    plot_threshold_sensitivity(training_df)
    plot_snapshots_over_time(training_df)
    plot_correlation_heatmap(pdf, top_n=25)

    top_features_for_dist = [
        "total_outstanding", "pct_90_plus", "max_dpd",
        "on_time_ratio", "avg_days_to_pay", "broken_ratio",
    ]
    available = [f for f in top_features_for_dist if f in pdf.columns]
    plot_feature_distributions_by_target(pdf, available)
    plot_missing_values(pdf)

    print("\n========== 2. OPTUNA SEARCH ==========\n")
    for plot_fn, args in [
        (plot_optuna_history, (study,)),
        (plot_optuna_param_importance, (study,)),
        (plot_optuna_parallel, (study,)),
        (plot_optuna_slice, (study,)),
        (plot_optuna_contour, (study, ("max_depth", "learning_rate"))),
    ]:
        try:
            plot_fn(*args)
        except Exception as e:
            print(f"{plot_fn.__name__} skipped: {e}")

    print("\n========== 3. MODEL EVALUATION ==========\n")
    plot_roc_curve(y_test, y_prob)
    plot_pr_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_threshold_vs_metrics(y_test, y_prob)
    plot_calibration(y_test, y_prob)
    plot_predicted_probs_by_class(y_test, y_prob)

    print("\n========== 3b. RECALL-FOCUSED PLOTS ==========\n")
    plot_recall_at_top_k(y_test, y_prob)
    plot_recall_vs_threshold(y_test, y_prob, target_recall=TARGET_RECALL)
    plot_recall_by_band(test_pdf, y_prob, y_test)

    # print("\n========== 4. FEATURE IMPORTANCE + SHAP ==========\n")
    # plot_feature_importance(model, selected, top_n=20)

    # X_sample = X_test[selected].sample(min(500, len(X_test)), random_state=42)
    # plot_shap_summary(model, X_sample)

    # top_feature = selected[int(np.argmax(model.feature_importances_))]
    # plot_shap_dependence(model, X_sample, top_feature)
    # plot_shap_waterfall(model, X_sample, row_idx=0)

    print("\n========== 5. LEARNING CURVE ==========\n")
    plot_learning_curve(model)

    print("\n========== 6. RISK BANDS ==========\n")
    plot_risk_band_distribution(y_prob)
    if "collection_ratio" in test_pdf.columns:
        plot_collection_ratio_by_band(test_pdf, y_prob)

    print("\n========== ALL PLOTS COMPLETE ==========")


# COMMAND ----------

training_df = build_training_dataset()

# COMMAND ----------

training_df.count()

# COMMAND ----------

print(MODEL_NAME)

# COMMAND ----------

mlflow.set_registry_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

(training_df.count(), len(training_df.columns))

# COMMAND ----------

display(training_df.limit(50))

# COMMAND ----------

quantiles = training_df.selectExpr(
    "percentile(collection_ratio, array(0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99)) as q"
).collect()[0]["q"]

percentiles = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

plt.figure(figsize=(8, 5))
plt.plot(percentiles, quantiles, marker="o", color="#3498db")
plt.title("Collection Ratio Percentiles")
plt.xlabel("Percentile")
plt.ylabel("Collection Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Step 2 — training data distribution

display(training_df.groupBy("target").count().orderBy("target"))

display(
    training_df.groupBy("snapshot_date")
    .count()
    .orderBy("snapshot_date")
)

display(
    training_df.select(
        "total_outstanding",
        "num_open_invoices",
        "max_dpd",
        "collection_ratio",
        "target",
    ).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
)

plot_target_balance(training_df)
plot_collection_ratio(training_df, sample_frac=0.1)
plot_snapshots_over_time(training_df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame


def ordinal_encode(
    df: DataFrame,
    source_col: str,
    rank_map: dict,
    target_col: str = None,
    default: int = -1,
    keep_label: bool = False,
    label_col: str = None,
) -> DataFrame:
    """
    Map a coded/categorical column to a single ordinal integer column.

    Args:
        source_col:  raw code column (e.g. "risk_class")
        rank_map:    {value: rank} mapping. The value is whatever source_col
                     actually holds (raw code or label); rank is the integer
                     it encodes to. Ranks need not be contiguous.
        target_col:  output integer column. Defaults to f"{source_col}_ord".
        default:     value for anything not in rank_map (incl. nulls).
        keep_label:  if True, also emit a decoded label column.
        label_col:   name for that label column. Defaults to f"{source_col}_label".

    Returns:
        DataFrame with the ordinal column (and optionally a label column).
    """
    target_col = target_col or f"{source_col}_ord"

    rank_expr = F.create_map(
        [x for k, v in rank_map.items() for x in (F.lit(k), F.lit(v))]
    )[F.col(source_col)]

    df = df.withColumn(target_col, F.coalesce(rank_expr, F.lit(default)))

    if keep_label:
        label_col = label_col or f"{source_col}_label"
        # invert: rank -> value. Assumes ranks are unique (they should be).
        inv = {v: k for k, v in rank_map.items()}
        label_expr = F.create_map(
            [x for k, v in inv.items() for x in (F.lit(k), F.lit(v))]
        )[F.col(target_col)]
        df = df.withColumn(label_col, label_expr)

    return df

# COMMAND ----------

display(training_df.select("risk_class").distinct())

# COMMAND ----------

risk_rank = {
    "A": 0,   # No Risk
    "B": 1,   # Very Low Risk
    "C": 2,   # Low Risk
    "D": 3,   # Medium Risk
    "E": 4,   # High Risk
    "F": 5,   # Very High Risk
}

training_df_new = ordinal_encode(
    training_df,
    source_col="risk_class",
    rank_map=risk_rank,
    target_col="risk_ord",
    default=-1,        # unknown / null
)

display(training_df_new.select("risk_class", "risk_ord").limit(50))

# COMMAND ----------

display(training_df_new.select("credit_group").distinct())

# COMMAND ----------

credit_rank = {
    "0003": 0, 
    "0008": 1,  
    "0004": 2, 
    "0007": 3,   
    "0009": 4,
    "0006": 5,
}

training_df_new = ordinal_encode(
    training_df_new,
    source_col="credit_group",
    rank_map=credit_rank,
    target_col="credit_ord",
    default=-1,
)

display(training_df_new.select("credit_group", "credit_ord").limit(50))

# COMMAND ----------

display(training_df_new.limit(10))

# COMMAND ----------

# Step 2 — Optuna training
output = run_optuna_training(training_df_new)


# COMMAND ----------


# Step 3 — all diagnostic plots
run_all_plots(training_df_new, output)

# COMMAND ----------

print("\n=== DONE ===")
print(f"Best test metrics: {output['test_metrics']}")
print(f"Chosen threshold: {output['threshold']:.3f}")
print(f"Selected feature count: {len(output['selected_features'])}")

# COMMAND ----------

output['selected_features']

# COMMAND ----------

training_df_new.columns