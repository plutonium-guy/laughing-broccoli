# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster Model Trainer (V2)
# MAGIC
# MAGIC **Input:** a list of countries that belong to one behavior cluster
# MAGIC (from `country_cluster_pooled_training_v2.py` dendrogram/silhouette).
# MAGIC
# MAGIC **Output:** one balanced, hyperparameter-tuned XGBoost for that
# MAGIC cluster, registered to MLflow.
# MAGIC
# MAGIC What "balanced" + "tuned" means here:
# MAGIC   - class imbalance: scale_pos_weight = natural ratio x tuned
# MAGIC     multiplier (Optuna decides how aggressive)
# MAGIC   - country imbalance: Optuna objective is the MACRO average of
# MAGIC     per-country PR-AUC inside each CV fold — every member country
# MAGIC     votes equally, the biggest market cannot dominate tuning
# MAGIC   - time safety: walk-forward CV folds with a 30d gap, never random
# MAGIC   - small-data safety: tuning only runs if the pool passes a size
# MAGIC     gate; otherwise falls back to conservative fixed params.
# MAGIC     Search space is deliberately small + regularization-biased.
# MAGIC   - stability: objective = mean(folds) - 0.5*std(folds)
# MAGIC   - threshold: max precision subject to recall >= TARGET_RECALL,
# MAGIC     tuned on a holdout val slice the model never trained on
# MAGIC
# MAGIC Feature engineering = V2 bugfixed unified-view pipeline
# MAGIC (proper DATE parsing, censoring-capped snapshots, due-date on-time,
# MAGIC window-only ratios, snapshot-dated tenure).

# COMMAND ----------

!pip install -U optuna xgboost mlflow shap seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# =========================================================
# INPUT — the cluster. Edit here or pass as job widgets.
# =========================================================
dbutils.widgets.text("cluster_countries", "FR,BE,NL")
dbutils.widgets.text("cluster_name", "c1")

CLUSTER_COUNTRIES = [c.strip().upper() for c in
                     dbutils.widgets.get("cluster_countries").split(",") if c.strip()]
CLUSTER_NAME = dbutils.widgets.get("cluster_name").strip()

assert CLUSTER_COUNTRIES, "Give at least one country code"
print(f"Cluster '{CLUSTER_NAME}': {CLUSTER_COUNTRIES}")

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
    precision_score, recall_score, classification_report, confusion_matrix,
    precision_recall_curve,
)

import optuna.visualization.matplotlib as ovm

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
UNIFIED_VIEW = "f_erp_glide_o2c_12.table_invoice_unified_master"

TODAY = "2026-03-25"
LOOKBACK_DAYS = 730
FUTURE_WINDOW_DAYS = 30
SNAPSHOT_STEP_DAYS = 7
HIGH_RISK_THRESHOLD = 0.4
WINDOWS = [60, 90, 180]

# Splits
TEST_FRACTION = 0.15
CV_SPLITS = 4
LEAKAGE_GAP_DAYS = 30

# Tuning gate — below either bound, Optuna is skipped (it would tune
# to CV noise) and conservative fixed params are used instead.
MIN_ROWS_TO_TUNE = 500
MIN_POSITIVES_TO_TUNE = 100
N_TRIALS = 50
STABILITY_PENALTY = 0.5          # objective = mean - penalty*std

EARLY_STOPPING_ROUNDS = 30
TARGET_RECALL = 0.80
MIN_TEST_POSITIVES = 20

MODEL_NAME = f"collection_risk_model_cluster_v2_{CLUSTER_NAME}"
EXPERIMENT_PATH = ("/Workspace/Users/amiya.x.mandal@gsk.com/APEC/exp/"
                   f"collection_risk_cluster_v2_{CLUSTER_NAME}")

# Fallback when the gate fails — shallow + heavily regularized.
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
    "n_estimators": 500,          # ceiling — early stopping decides
}

# COMMAND ----------

# =========================================================
# PART A — V2-FIXED FEATURE ENGINEERING (pooled over the cluster)
# =========================================================

def to_date_any(col):
    """Parse to DATE whether source is yyyyMMdd string, ISO string, or DATE."""
    c = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.to_date(c, "yyyyMMdd"), F.to_date(c))


def load_base_data(date_val, countries):
    return spark.table(UNIFIED_VIEW).select(
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
        F.col("country"),
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
    ).filter(
        F.col("baseline_date").isNotNull()
        & (F.col("baseline_date") >= F.date_sub(F.lit(date_val), LOOKBACK_DAYS))
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
    # V2 FIX: cap calendar -> every snapshot has a full 30d target window
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


def build_cluster_dataset(countries):
    print(f"Loading master table for {countries}...")
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

    print(f"Pooled dataset: {df.count():,} rows")
    return df

# COMMAND ----------

# =========================================================
# PART B — ML PREP
# =========================================================

DROP_COLS = ["customer_id", "snapshot_date", "target",
             "collected_30d", "collection_ratio", "country"]


def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


def add_country_onehot(pdf, countries):
    """Per-market offset inside one pooled model."""
    for c in sorted(countries):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
    return pdf


def build_time_series_splits(pdf, n_splits=CV_SPLITS, gap_days=LEAKAGE_GAP_DAYS):
    """Walk-forward CV with a leakage gap. Indexes refer to pdf as given."""
    dates = np.sort(pdf["snapshot_date"].unique())
    n = len(dates)
    fold_size = n // (n_splits + 1)

    splits = []
    for i in range(1, n_splits + 1):
        train_end = dates[fold_size * i - 1]
        val_start = train_end + pd.Timedelta(days=gap_days)
        val_end = dates[n - 1] if i == n_splits \
            else dates[min(fold_size * i + fold_size, n - 1)]

        train_mask = pdf["snapshot_date"] <= train_end
        val_mask = (pdf["snapshot_date"] >= val_start) & (pdf["snapshot_date"] <= val_end)
        if val_mask.sum() == 0:
            continue
        splits.append((pdf[train_mask].index, pdf[val_mask].index))

    print(f"Built {len(splits)} walk-forward folds (gap={gap_days}d)")
    for i, (tr, va) in enumerate(splits):
        pos_va = int(pdf.loc[va, "target"].sum())
        by_country = pdf.loc[va].groupby("country")["target"].agg(["size", "sum"])
        flag = "  WARN-sparse" if pos_va < 30 else ""
        print(f"  fold {i+1}: train={len(tr):,} val={len(va):,} val_pos={pos_va}{flag}")
        for c, row in by_country.iterrows():
            print(f"           {c}: n={int(row['size'])}, pos={int(row['sum'])}")
    return splits

# COMMAND ----------

# =========================================================
# PART C — TUNING (macro per-country objective)
# =========================================================

def macro_country_score(model, val_pdf, feature_cols):
    """
    Equal-weight average of per-country PR-AUC on a val slice.
    Degenerate country slices (one class) are skipped, not faked.
    Falls back to pooled PR-AUC if every slice is degenerate.
    """
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
    def objective(trial):
        # Small, regularization-biased space — pooled clusters are still
        # small data. n_estimators NOT tuned (early stopping decides).
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
        # Class balance knob: natural ratio x multiplier
        spw_mult = trial.suggest_float("scale_pos_weight_mult", 0.5, 1.2)

        fold_scores = []
        for step, (tr_idx, va_idx) in enumerate(splits):
            tr, va = pdf.loc[tr_idx], pdf.loc[va_idx]
            spw = ((tr["target"] == 0).sum() / max((tr["target"] == 1).sum(), 1)) * spw_mult

            model = XGBClassifier(**params, scale_pos_weight=spw,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(tr[feature_cols], tr["target"],
                      eval_set=[(va[feature_cols], va["target"])], verbose=False)

            fold_scores.append(macro_country_score(model, va, feature_cols))
            trial.report(float(np.mean(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Stability-penalized: params must work across ALL time folds,
        # not win one fold big and lose the rest.
        return float(np.mean(fold_scores) - STABILITY_PENALTY * np.std(fold_scores))

    return objective


def tune_or_fallback(dev_pdf, feature_cols, splits):
    """Run Optuna if the pool passes the size gate, else conservative params."""
    n_pos = int(dev_pdf["target"].sum())
    if len(dev_pdf) < MIN_ROWS_TO_TUNE or n_pos < MIN_POSITIVES_TO_TUNE:
        print(f"GATE FAILED ({len(dev_pdf)} rows, {n_pos} positives) — "
              f"using conservative fixed params, skipping Optuna.")
        return dict(CONSERVATIVE_PARAMS), 1.0, None

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(make_objective(dev_pdf, feature_cols, splits),
                   n_trials=N_TRIALS, gc_after_trial=True)

    best = dict(study.best_params)
    spw_mult = best.pop("scale_pos_weight_mult", 1.0)
    print(f"Best macro objective: {study.best_value:.4f}")
    print(f"Best params: {best} | spw_mult={spw_mult:.2f}")
    return best, spw_mult, study

# COMMAND ----------

# =========================================================
# PART D — THRESHOLD + EVALUATION
# =========================================================

def threshold_for_precision_at_recall(y_true, y_prob, min_recall=TARGET_RECALL):
    thresholds = np.linspace(0.001, 0.9, 500)
    best_t, best_prec = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        rec = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        if rec >= min_recall and prec > best_prec:
            best_prec, best_t = prec, t
    return float(best_t)


def eval_slice(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
    }
    return out


def evaluate_test(model, test_pdf, feature_cols, threshold, countries):
    prob = model.predict_proba(test_pdf[feature_cols])[:, 1]
    pooled = eval_slice(test_pdf["target"], pd.Series(prob, index=test_pdf.index), threshold)
    if pooled["n_pos"] < MIN_TEST_POSITIVES:
        print(f"WARN: only {pooled['n_pos']} positives in test — treat metrics directionally")

    rows = [{"slice": "POOLED", **pooled}]
    for c in countries:
        tc = test_pdf[test_pdf["country"] == c]
        if len(tc) == 0:
            continue
        pc = model.predict_proba(tc[feature_cols])[:, 1]
        rows.append({"slice": c, **eval_slice(tc["target"], pd.Series(pc, index=tc.index), threshold)})

    report = pd.DataFrame(rows).set_index("slice")
    print(f"\n=== TEST EVALUATION @ threshold={threshold:.3f} ===")
    print(report.round(3).to_string())

    y_pred = (prob >= threshold).astype(int)
    print("\nPooled classification report:")
    print(classification_report(test_pdf["target"], y_pred, zero_division=0))
    return report, prob

# COMMAND ----------

# =========================================================
# PART E — DIAGNOSTIC PLOTS
# =========================================================

def plot_optuna_diagnostics(study):
    if study is None:
        print("No study (gate failed) — skipping Optuna plots.")
        return
    for fn, title in [(ovm.plot_optimization_history, "Optimization History"),
                      (ovm.plot_param_importances, "Hyperparameter Importance")]:
        try:
            ax = fn(study)
            (ax.figure if hasattr(ax, "figure") else plt.gcf()).set_size_inches(9, 4)
            plt.title(title)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"{title} skipped: {e}")


def plot_learning_curve(model):
    if not getattr(model, "evals_result_", None):
        return
    res = model.evals_result_
    metric = list(res["validation_0"].keys())[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, label) in enumerate([("validation_0", "train"), ("validation_1", "val")]):
        if name in res:
            ax.plot(res[name][metric], label=label, linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric)
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_and_threshold(y_true, y_prob, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[0].plot(rec, prec, color="#9b59b6", linewidth=2, label=f"AP={ap:.3f}")
    axes[0].axhline(y_true.mean(), color="black", linestyle="--",
                    label=f"baseline={y_true.mean():.3f}")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall (pooled test)")
    axes[0].legend()

    ts = np.linspace(0.02, 0.9, 100)
    rs = [recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ts]
    ps = [precision_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ts]
    axes[1].plot(ts, rs, label="recall", color="#e74c3c", linewidth=2)
    axes[1].plot(ts, ps, label="precision", color="#3498db", linewidth=2)
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"chosen t={threshold:.2f}")
    axes[1].axhline(TARGET_RECALL, color="gray", linestyle=":", label=f"recall floor={TARGET_RECALL}")
    axes[1].set_xlabel("Threshold"); axes[1].set_title("Threshold choice")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_per_country_report(report):
    d = report.drop(index="POOLED", errors="ignore")
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["recall", "precision", "pr_auc"]):
        ax.bar(d.index, d[m], color="#34495e")
        ax.axhline(report.loc["POOLED", m], color="red", linestyle="--", label="pooled")
        ax.set_title(f"test {m} per country")
        ax.set_ylim(0, 1)
        ax.legend()
        for i, (v, p) in enumerate(zip(d[m], d["n_pos"])):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}\n(p={int(p)})", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Per-Country Test Performance (p = positives in slice)", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_prob, threshold):
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Pooled Test Confusion @ t={threshold:.2f}")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# =========================================================
# RUN — end to end
# =========================================================

training_df = build_cluster_dataset(CLUSTER_COUNTRIES)

pdf = spark_to_pandas_safe(training_df)
pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"])
pdf = pdf.sort_values("snapshot_date").reset_index(drop=True)
pdf = add_country_onehot(pdf, CLUSTER_COUNTRIES)

feature_cols = [c for c in pdf.columns if c not in DROP_COLS]
print(f"\n{len(pdf):,} rows | {len(feature_cols)} features | "
      f"positives={int(pdf['target'].sum()):,} ({pdf['target'].mean()*100:.1f}%)")
print(pdf.groupby("country")["target"].agg(rows="size", positives="sum"))

# Time split: dev / test
cut_test = pdf["snapshot_date"].quantile(1 - TEST_FRACTION, interpolation="lower")
dev_pdf = pdf[pdf["snapshot_date"] <= cut_test].reset_index(drop=True)
test_pdf = pdf[pdf["snapshot_date"] > cut_test].reset_index(drop=True)
print(f"\nDev: {len(dev_pdf):,} | Test: {len(test_pdf):,} (cutoff={cut_test.date()})")

splits = build_time_series_splits(dev_pdf)

# COMMAND ----------

mlflow.set_registry_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name=f"cluster_{CLUSTER_NAME}_tuned") as run:
    mlflow.log_param("cluster_name", CLUSTER_NAME)
    mlflow.log_param("cluster_countries", ",".join(CLUSTER_COUNTRIES))
    mlflow.log_param("n_rows", len(pdf))
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_param("objective", "macro per-country PR-AUC, stability-penalized")

    # ---- 1. tune (or fallback)
    best_params, spw_mult, study = tune_or_fallback(dev_pdf, feature_cols, splits)
    mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
    mlflow.log_param("hp_scale_pos_weight_mult", spw_mult)
    mlflow.log_param("tuned", study is not None)

    # ---- 2. inner train/val for early stopping + threshold
    cut_val = dev_pdf["snapshot_date"].quantile(0.85, interpolation="lower")
    train = dev_pdf[dev_pdf["snapshot_date"] <= cut_val]
    val = dev_pdf[dev_pdf["snapshot_date"] > cut_val]

    spw = ((train["target"] == 0).sum() / max((train["target"] == 1).sum(), 1)) * spw_mult
    print(f"\nscale_pos_weight = {spw:.2f} (natural x {spw_mult:.2f})")
    mlflow.log_metric("final_scale_pos_weight", spw)

    model = XGBClassifier(**FIXED_PARAMS, **best_params, scale_pos_weight=spw,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(train[feature_cols], train["target"],
              eval_set=[(train[feature_cols], train["target"]),
                        (val[feature_cols], val["target"])],
              verbose=False)

    # threshold tuned on val — model never trained on it
    val_prob = model.predict_proba(val[feature_cols])[:, 1]
    threshold = threshold_for_precision_at_recall(val["target"], val_prob)
    mlflow.log_param("chosen_threshold", threshold)
    mlflow.log_param("target_recall", TARGET_RECALL)

    # ---- 3. honest test evaluation (pooled + per country)
    report, test_prob = evaluate_test(model, test_pdf, feature_cols,
                                      threshold, CLUSTER_COUNTRIES)
    for m in ["recall", "precision", "f1", "f2", "pr_auc", "roc_auc"]:
        v = report.loc["POOLED", m]
        if not np.isnan(v):
            mlflow.log_metric(f"test_{m}", float(v))
    report.to_csv("/tmp/per_country_report.csv")
    mlflow.log_artifact("/tmp/per_country_report.csv")

    # ---- 4. production refit on full dev, trees locked from early stopping
    best_n = int(model.best_iteration) + 1 if model.best_iteration is not None \
        else FIXED_PARAMS["n_estimators"]
    prod_fixed = {k: v for k, v in FIXED_PARAMS.items() if k != "n_estimators"}
    production_model = XGBClassifier(**prod_fixed, **best_params,
                                     n_estimators=best_n, scale_pos_weight=spw)
    production_model.fit(dev_pdf[feature_cols], dev_pdf["target"], verbose=False)
    mlflow.log_param("production_n_estimators", best_n)

    prod_report, _ = evaluate_test(production_model, test_pdf, feature_cols,
                                   threshold, CLUSTER_COUNTRIES)
    for m in ["recall", "precision", "pr_auc"]:
        v = prod_report.loc["POOLED", m]
        if not np.isnan(v):
            mlflow.log_metric(f"prod_test_{m}", float(v))

    # ---- 5. register production model
    mlflow.sklearn.log_model(
        sk_model=production_model,
        artifact_path="model",
        input_example=dev_pdf[feature_cols].head(5),
    )
    mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME)
    print(f"\nRegistered: {MODEL_NAME} (threshold={threshold:.3f})")

# COMMAND ----------

# =========================================================
# RUN — diagnostics
# =========================================================

plot_optuna_diagnostics(study)
plot_learning_curve(model)
plot_pr_and_threshold(test_pdf["target"], test_prob, threshold)
plot_per_country_report(report)
plot_confusion(test_pdf["target"], test_prob, threshold)

# COMMAND ----------

print("\n=== DONE ===")
print(f"Cluster: {CLUSTER_NAME} = {CLUSTER_COUNTRIES}")
print(f"Tuned: {study is not None} | threshold={threshold:.3f}")
print(report.round(3).to_string())
