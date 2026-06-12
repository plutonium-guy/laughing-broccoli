# Databricks notebook source
# MAGIC %md
# MAGIC # Country-Cluster Pooled Training (V2)
# MAGIC
# MAGIC **Problem:** per-country training data is tiny (< 200 rows for many
# MAGIC markets) — too small to train a stable per-country model.
# MAGIC
# MAGIC **Approach:** club countries with similar payment behavior into
# MAGIC clusters, then train ONE model per cluster on the pooled rows.
# MAGIC
# MAGIC Pipeline:
# MAGIC   1. Build a behavior fingerprint vector per country
# MAGIC      (DPD, on-time rate, days-to-pay, dunning intensity, P2P keep rate...)
# MAGIC   2. Cluster countries — Hierarchical Agglomerative (Ward) with the
# MAGIC      cluster count picked by silhouette score (KMeans shown as
# MAGIC      cross-check)
# MAGIC   3. Graphical diagnostics — dendrogram, silhouette curve, PCA map,
# MAGIC      behavior heatmap, sample-size bars
# MAGIC   4. Pool training rows per cluster (country one-hot kept as feature)
# MAGIC      and train an XGBoost per cluster
# MAGIC   5. Evaluate per cluster AND per country inside each cluster,
# MAGIC      including leave-one-country-out (does pooling actually help?)
# MAGIC
# MAGIC Feature engineering reuses the V2 bugfixes from
# MAGIC `training_pipeline_unified_view_v2.py`:
# MAGIC   - proper DATE parsing for dunning/P2P/due dates (not yyyyMMdd strings)
# MAGIC   - snapshot calendar capped at TODAY - 30d (no censored labels)
# MAGIC   - on_time measured against due_date
# MAGIC   - windowed on-time ratio over window payments only
# MAGIC   - tenure back-dated to snapshot

# COMMAND ----------

!pip install -U xgboost mlflow shap seaborn scikit-learn scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType

from xgboost import XGBClassifier

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

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

# Clustering
MIN_K = 2
MAX_K = 8                    # upper bound on clusters to test
MIN_CUSTOMERS_PER_COUNTRY = 20   # below this, fingerprint flagged noisy

# Training
TEST_FRACTION = 0.2          # most recent snapshots
EARLY_STOPPING_ROUNDS = 30
TARGET_RECALL = 0.80
MIN_TEST_POSITIVES = 20      # warn below this — metrics unstable

MODEL_NAME_PREFIX = "collection_risk_model_cluster_v2"
EXPERIMENT_PATH = "/Workspace/Users/amiya.x.mandal@gsk.com/APEC/exp/collection_risk_cluster_pooled_v2"

# Small-data-friendly XGBoost — shallow, heavily regularized.
# With a few hundred pooled rows per cluster, deep trees memorize.
SMALL_DATA_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_estimators": 500,            # early stopping decides actual count
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

# COMMAND ----------

# =========================================================
# PART A — V2-FIXED LOADER + FEATURE ENGINEERING
# (same fixes as training_pipeline_unified_view_v2.py, but loads
#  ALL countries and carries `country` through to the snapshot grain)
# =========================================================

def to_date_any(col):
    """Parse to DATE whether source is yyyyMMdd string, ISO string, or DATE."""
    c = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.to_date(c, "yyyyMMdd"), F.to_date(c))


def load_base_data(date_val):
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
        F.col("region"),
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
        & F.col("country").isNotNull()
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
    # V2 FIX: cap at date_val - FUTURE_WINDOW_DAYS -> full target window
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
        F.first("country", ignorenulls=True).alias("country"),   # carried for pooling
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
        # V2 FIX: tenure as of snapshot, not as of data load
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
        # V2 FIX: on-time = cleared by due date
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
            # V2 FIX: window-only denominator
            F.avg(F.when(cond, F.when(F.col("days_late") <= 0, 1).otherwise(0))).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"num_payments_{w}d"),
        ])

    return hist.groupBy("b.customer_id", "s.snapshot_date").agg(*aggs)


def compute_dunning_features(invoice_df, snapshots):
    # Unified-view limitation: dunning_level/count are current-state.
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


def build_training_dataset_all_countries():
    """Snapshot-grain training rows for ALL countries, `country` kept."""
    print("Loading master table (all countries)...")
    base = ensure_due_date(load_base_data(TODAY))

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

    print(f"Training dataset built: {df.count():,} rows")
    return df

# COMMAND ----------

# =========================================================
# PART B — COUNTRY BEHAVIOR FINGERPRINTS
# One vector per country describing HOW that market pays.
# Built at customer grain first so big customers don't dominate.
# =========================================================

FINGERPRINT_COLS = [
    "mean_dpd", "p90_dpd", "on_time_rate", "mean_days_to_pay",
    "dunning_rate", "high_dunning_rate", "p2p_keep_rate", "p2p_activity_rate",
    "dispute_rate", "median_invoice_size", "collection_ratio_mean",
    "high_risk_rate",
]


def build_country_fingerprints(training_df):
    """
    Aggregate snapshot-grain rows -> customer grain -> country grain.
    Returns (pandas fingerprint df indexed by country, sample-size df).
    """
    cust = training_df.groupBy("country", "customer_id").agg(
        F.avg("avg_dpd").alias("c_dpd"),
        F.avg("on_time_ratio").alias("c_on_time"),
        F.avg("avg_days_to_pay").alias("c_days_to_pay"),
        F.max(F.when(F.col("total_dunning_events") > 0, 1).otherwise(0)).alias("c_dunned"),
        F.max(F.when(F.col("high_severity_dunning") > 0, 1).otherwise(0)).alias("c_high_dunned"),
        F.avg("kept_ratio").alias("c_keep"),
        F.max("promise_activity_flag").alias("c_p2p_active"),
        F.max(F.when(F.col("number_of_disputes") > 0, 1).otherwise(0)).alias("c_disputed"),
        F.avg("avg_invoice_size").alias("c_inv_size"),
        F.avg("collection_ratio").alias("c_coll_ratio"),
        F.avg(F.col("target").cast("double")).alias("c_high_risk"),
    )

    country = cust.groupBy("country").agg(
        F.count("*").alias("n_customers"),
        F.avg("c_dpd").alias("mean_dpd"),
        F.expr("percentile(c_dpd, 0.9)").alias("p90_dpd"),
        F.avg("c_on_time").alias("on_time_rate"),
        F.avg("c_days_to_pay").alias("mean_days_to_pay"),
        F.avg("c_dunned").alias("dunning_rate"),
        F.avg("c_high_dunned").alias("high_dunning_rate"),
        F.avg("c_keep").alias("p2p_keep_rate"),
        F.avg("c_p2p_active").alias("p2p_activity_rate"),
        F.avg("c_disputed").alias("dispute_rate"),
        F.expr("percentile(c_inv_size, 0.5)").alias("median_invoice_size"),
        F.avg("c_coll_ratio").alias("collection_ratio_mean"),
        F.avg("c_high_risk").alias("high_risk_rate"),
    ).toPandas().set_index("country").sort_index()

    sizes = country[["n_customers"]].copy()
    fp = country[FINGERPRINT_COLS].fillna(0)

    noisy = sizes[sizes["n_customers"] < MIN_CUSTOMERS_PER_COUNTRY].index.tolist()
    if noisy:
        print(f"WARNING: fingerprints from < {MIN_CUSTOMERS_PER_COUNTRY} customers "
              f"(noisy, will still cluster): {noisy}")

    return fp, sizes

# COMMAND ----------

# =========================================================
# PART C — CLUSTERING (HAC Ward, k by silhouette; KMeans cross-check)
# =========================================================

def cluster_countries(fp):
    """
    Returns dict with z-scored matrix, linkage, chosen k, labels,
    and the silhouette table used to pick k.
    """
    scaler = StandardScaler()
    Z = scaler.fit_transform(fp.values)
    Zdf = pd.DataFrame(Z, index=fp.index, columns=fp.columns)

    link = linkage(Z, method="ward")

    n = len(fp)
    max_k = min(MAX_K, n - 1)
    rows = []
    for k in range(MIN_K, max_k + 1):
        hac_labels = fcluster(link, t=k, criterion="maxclust")
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        rows.append({
            "k": k,
            "silhouette_hac": silhouette_score(Z, hac_labels) if len(set(hac_labels)) > 1 else np.nan,
            "silhouette_kmeans": silhouette_score(Z, km.labels_) if len(set(km.labels_)) > 1 else np.nan,
        })
    sil = pd.DataFrame(rows).set_index("k")

    best_k = int(sil["silhouette_hac"].idxmax())
    labels = fcluster(link, t=best_k, criterion="maxclust")
    assign = pd.Series(labels, index=fp.index, name="cluster")

    print(f"Chosen k={best_k} (HAC silhouette={sil.loc[best_k, 'silhouette_hac']:.3f})")
    print("\nCluster membership:")
    for c in sorted(assign.unique()):
        print(f"  cluster {c}: {sorted(assign[assign == c].index.tolist())}")

    return {"Z": Zdf, "linkage": link, "k": best_k,
            "assign": assign, "silhouette_table": sil}

# COMMAND ----------

# =========================================================
# PART D — CLUSTERING VISUALS
# =========================================================

def plot_dendrogram(clu, sizes):
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"{c} (n={int(sizes.loc[c, 'n_customers'])})" for c in clu["Z"].index]
    dendrogram(clu["linkage"], labels=labels, ax=ax,
               color_threshold=clu["linkage"][-(clu["k"] - 1), 2])
    ax.set_title(f"Country Dendrogram (Ward) — cut at k={clu['k']}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Ward distance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_silhouette_curve(clu):
    sil = clu["silhouette_table"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sil.index, sil["silhouette_hac"], marker="o", label="HAC (Ward)", color="#e74c3c")
    ax.plot(sil.index, sil["silhouette_kmeans"], marker="s", label="KMeans", color="#3498db")
    ax.axvline(clu["k"], color="black", linestyle="--", label=f"chosen k={clu['k']}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Cluster Count Selection")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_silhouette_per_country(clu):
    Z = clu["Z"].values
    labels = clu["assign"].values
    if len(set(labels)) < 2:
        print("Single cluster — silhouette per country skipped.")
        return
    svals = silhouette_samples(Z, labels)
    df_s = pd.DataFrame({"country": clu["Z"].index, "cluster": labels,
                         "silhouette": svals}).sort_values(["cluster", "silhouette"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_s) * 0.3)))
    colors = plt.cm.tab10(df_s["cluster"] % 10)
    ax.barh(df_s["country"] + " [c" + df_s["cluster"].astype(str) + "]",
            df_s["silhouette"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Silhouette (negative = poorly placed, candidate to reassign)")
    ax.set_title("Per-Country Cluster Fit")
    plt.tight_layout()
    plt.show()


def plot_pca_map(clu, sizes):
    Z = clu["Z"].values
    pca = PCA(n_components=2)
    pts = pca.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_sizes = 40 + 200 * (sizes["n_customers"] / sizes["n_customers"].max())
    for c in sorted(clu["assign"].unique()):
        mask = clu["assign"].values == c
        ax.scatter(pts[mask, 0], pts[mask, 1], s=scatter_sizes[mask],
                   label=f"cluster {c}", alpha=0.7)
    for i, name in enumerate(clu["Z"].index):
        ax.annotate(name, (pts[i, 0], pts[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.set_title("Country Behavior Map (PCA of fingerprints, size = #customers)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_fingerprint_heatmap(clu):
    order = clu["assign"].sort_values().index
    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.4)))
    sns.heatmap(clu["Z"].loc[order], cmap="coolwarm", center=0, annot=True,
                fmt=".1f", ax=ax, cbar_kws={"label": "z-score"})
    cluster_of = clu["assign"]
    ax.set_yticklabels([f"{c}  [c{cluster_of[c]}]" for c in order], rotation=0)
    ax.set_title("Country Behavior Fingerprints (z-scored), grouped by cluster",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(clu):
    order = clu["assign"].sort_values().index
    D = pd.DataFrame(squareform(pdist(clu["Z"].values)),
                     index=clu["Z"].index, columns=clu["Z"].index).loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(D, cmap="viridis_r", annot=len(order) <= 15, fmt=".1f", ax=ax,
                cbar_kws={"label": "Euclidean distance (z-space)"})
    ax.set_title("Country-to-Country Behavior Distance")
    plt.tight_layout()
    plt.show()


def plot_rows_per_country(training_pdf, assign):
    counts = training_pdf.groupby("country").size().rename("rows").to_frame()
    counts["cluster"] = assign.reindex(counts.index)
    counts = counts.sort_values(["cluster", "rows"])

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = plt.cm.tab10(counts["cluster"].fillna(-1).astype(int) % 10)
    ax.bar(counts.index, counts["rows"], color=colors)
    ax.axhline(200, color="red", linestyle="--", label="200 rows — too small solo")
    ax.set_ylabel("Training rows")
    ax.set_title("Training Rows per Country (color = cluster) — why we pool")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# =========================================================
# PART E — POOLED TRAINING PER CLUSTER
# =========================================================

DROP_COLS = ["customer_id", "snapshot_date", "target",
             "collected_30d", "collection_ratio", "country", "cluster"]


def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


def add_country_onehot(pdf, countries):
    """One-hot the country inside a cluster so the pooled model can keep
    a small per-market offset without separate models."""
    for c in sorted(countries):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
    return pdf


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


def eval_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }
    # AUCs undefined on single-class slices (tiny countries)
    if y_true.nunique() > 1:
        out["pr_auc"] = average_precision_score(y_true, y_prob)
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["pr_auc"] = np.nan
        out["roc_auc"] = np.nan
    return out


def train_cluster_model(pdf_cluster, cluster_id, countries):
    """
    Pooled model for one cluster. Time-split train/val/test, conservative
    params + early stopping (no Optuna — too little data, it would tune
    to CV noise). Returns bundle dict.
    """
    pdf = pdf_cluster.sort_values("snapshot_date").reset_index(drop=True)
    pdf = add_country_onehot(pdf, countries)
    feature_cols = [c for c in pdf.columns if c not in DROP_COLS]

    cut_test = pdf["snapshot_date"].quantile(1 - TEST_FRACTION, interpolation="lower")
    dev = pdf[pdf["snapshot_date"] <= cut_test]
    test = pdf[pdf["snapshot_date"] > cut_test]
    cut_val = dev["snapshot_date"].quantile(0.8, interpolation="lower")
    train = dev[dev["snapshot_date"] <= cut_val]
    val = dev[dev["snapshot_date"] > cut_val]

    n_pos_test = int(test["target"].sum())
    if n_pos_test < MIN_TEST_POSITIVES:
        print(f"  WARN cluster {cluster_id}: only {n_pos_test} positives in test — "
              f"metrics unstable, treat directionally")

    spw = (train["target"] == 0).sum() / max((train["target"] == 1).sum(), 1)

    model = XGBClassifier(**SMALL_DATA_PARAMS, scale_pos_weight=spw,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(train[feature_cols], train["target"],
              eval_set=[(val[feature_cols], val["target"])], verbose=False)

    # Threshold on val (never trained on)
    val_prob = model.predict_proba(val[feature_cols])[:, 1]
    threshold = threshold_for_precision_at_recall(val["target"], val_prob)

    test_prob = model.predict_proba(test[feature_cols])[:, 1]
    cluster_metrics = eval_metrics(test["target"], pd.Series(test_prob, index=test.index), threshold)

    # Per-country breakdown on the SAME pooled test window
    per_country = {}
    for c in countries:
        tc = test[test["country"] == c]
        if len(tc) == 0:
            continue
        pc_prob = model.predict_proba(tc[feature_cols])[:, 1]
        per_country[c] = eval_metrics(tc["target"], pd.Series(pc_prob, index=tc.index), threshold)

    print(f"  cluster {cluster_id}: {len(pdf):,} rows pooled from {countries} | "
          f"test recall={cluster_metrics['recall']:.2f} "
          f"precision={cluster_metrics['precision']:.2f} "
          f"pr_auc={cluster_metrics['pr_auc']:.3f}")

    return {
        "cluster": cluster_id,
        "countries": countries,
        "model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "metrics": cluster_metrics,
        "per_country": per_country,
        "test": test,
        "test_prob": test_prob,
    }


def loco_check(pdf_cluster, cluster_id, countries):
    """
    Leave-one-country-out: train on cluster minus one country, test on the
    held-out country. If pooled-without-it still scores it well, the
    cluster genuinely shares behavior (pooling is safe for new/low-data
    markets in this cluster).
    """
    rows = []
    for held in countries:
        tr = pdf_cluster[pdf_cluster["country"] != held]
        te = pdf_cluster[pdf_cluster["country"] == held]
        if te["target"].nunique() < 2 or tr["target"].sum() < 10:
            rows.append({"cluster": cluster_id, "held_out": held,
                         "pr_auc": np.nan, "roc_auc": np.nan,
                         "n_test": len(te), "note": "degenerate"})
            continue
        feats = [c for c in pdf_cluster.columns
                 if c not in DROP_COLS and not c.startswith("country_")]
        spw = (tr["target"] == 0).sum() / max((tr["target"] == 1).sum(), 1)
        m = XGBClassifier(**{**SMALL_DATA_PARAMS, "n_estimators": 200},
                          scale_pos_weight=spw)
        m.fit(tr[feats], tr["target"], verbose=False)
        prob = m.predict_proba(te[feats])[:, 1]
        rows.append({
            "cluster": cluster_id, "held_out": held,
            "pr_auc": average_precision_score(te["target"], prob),
            "roc_auc": roc_auc_score(te["target"], prob),
            "n_test": len(te), "note": "",
        })
    return pd.DataFrame(rows)

# COMMAND ----------

# =========================================================
# PART F — EVALUATION VISUALS
# =========================================================

def plot_per_country_metrics(bundles):
    rows = []
    for b in bundles:
        for c, m in b["per_country"].items():
            rows.append({"country": c, "cluster": b["cluster"], **m})
    dfm = pd.DataFrame(rows).sort_values(["cluster", "country"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, metric in zip(axes, ["recall", "precision", "pr_auc"]):
        colors = plt.cm.tab10(dfm["cluster"].astype(int) % 10)
        ax.bar(dfm["country"], dfm[metric], color=colors)
        ax.set_title(f"Test {metric} per country")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        for i, (v, n) in enumerate(zip(dfm[metric], dfm["n_pos"])):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}\n(p={n})", ha="center", va="bottom", fontsize=7)
    plt.suptitle("Pooled Cluster Models — Per-Country Test Performance "
                 "(color = cluster, p = test positives)", fontweight="bold")
    plt.tight_layout()
    plt.show()
    return dfm


def plot_loco_results(loco_df):
    d = loco_df.dropna(subset=["pr_auc"]).sort_values(["cluster", "pr_auc"])
    if d.empty:
        print("No valid LOCO results (degenerate test sets).")
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = plt.cm.tab10(d["cluster"].astype(int) % 10)
    ax.bar(d["held_out"], d["pr_auc"], color=colors)
    ax.set_ylabel("PR-AUC on held-out country")
    ax.set_title("Leave-One-Country-Out — can the cluster score a country "
                 "it never saw? (higher = pooling is safe)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_confusions(bundles):
    n = len(bundles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bundles):
        y_pred = (b["test_prob"] >= b["threshold"]).astype(int)
        cm = confusion_matrix(b["test"]["target"], y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"])
        ax.set_title(f"Cluster {b['cluster']} ({', '.join(b['countries'])})\n"
                     f"t={b['threshold']:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# =========================================================
# RUN — 1. data, 2. fingerprints, 3. cluster, 4. visuals
# =========================================================

training_df = build_training_dataset_all_countries()

fingerprints, sizes = build_country_fingerprints(training_df)
display(fingerprints.assign(n_customers=sizes["n_customers"]))

clu = cluster_countries(fingerprints)

# COMMAND ----------

plot_dendrogram(clu, sizes)
plot_silhouette_curve(clu)
plot_silhouette_per_country(clu)
plot_pca_map(clu, sizes)
plot_fingerprint_heatmap(clu)
plot_distance_matrix(clu)

# COMMAND ----------

# =========================================================
# RUN — 5. pooled training per cluster
# =========================================================

mlflow.set_registry_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)

training_pdf = spark_to_pandas_safe(training_df)
training_pdf["snapshot_date"] = pd.to_datetime(training_pdf["snapshot_date"])
training_pdf["cluster"] = training_pdf["country"].map(clu["assign"])

plot_rows_per_country(training_pdf, clu["assign"])

bundles = []
with mlflow.start_run(run_name="cluster_pooled_training"):
    mlflow.log_param("k_clusters", clu["k"])
    mlflow.log_param("cluster_assignment",
                     clu["assign"].to_dict())

    for cluster_id in sorted(clu["assign"].unique()):
        countries = sorted(clu["assign"][clu["assign"] == cluster_id].index.tolist())
        pdf_c = training_pdf[training_pdf["cluster"] == cluster_id].copy()
        print(f"\n=== Cluster {cluster_id}: {countries} | {len(pdf_c):,} rows ===")

        bundle = train_cluster_model(pdf_c, cluster_id, countries)
        bundles.append(bundle)

        for k, v in bundle["metrics"].items():
            if not (isinstance(v, float) and np.isnan(v)):
                mlflow.log_metric(f"cluster{cluster_id}_test_{k}", float(v))
        mlflow.log_metric(f"cluster{cluster_id}_threshold", bundle["threshold"])

        mlflow.sklearn.log_model(
            sk_model=bundle["model"],
            artifact_path=f"model_cluster_{cluster_id}",
            input_example=pdf_c.head(5)[bundle["feature_cols"]],
        )
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model_cluster_{cluster_id}",
            name=f"{MODEL_NAME_PREFIX}_c{cluster_id}",
        )

# COMMAND ----------

# =========================================================
# RUN — 6. evaluation: per-country breakdown + LOCO + confusions
# =========================================================

per_country_df = plot_per_country_metrics(bundles)
display(per_country_df)

plot_confusions(bundles)

# COMMAND ----------

loco_all = pd.concat([
    loco_check(training_pdf[training_pdf["cluster"] == b["cluster"]],
               b["cluster"], b["countries"])
    for b in bundles
], ignore_index=True)

display(loco_all)
plot_loco_results(loco_all)

# COMMAND ----------

print("\n=== DONE ===")
print(f"k = {clu['k']} clusters")
for b in bundles:
    m = b["metrics"]
    print(f"cluster {b['cluster']} {b['countries']}: "
          f"recall={m['recall']:.2f} precision={m['precision']:.2f} "
          f"pr_auc={m['pr_auc']:.3f} (n_test={m['n']}, pos={m['n_pos']}) "
          f"-> {MODEL_NAME_PREFIX}_c{b['cluster']}")
