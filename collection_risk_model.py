# Databricks notebook source
# MAGIC %md
# MAGIC Changing future window to 30 days

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %pip install category-encoders

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn

mlflow.set_registry_uri("databricks")
mlflow.set_experiment("/Users/nishant.x.guvvada@gsk.com/collection_risk_model_customer_v1")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =========================================================
# CONFIG
# =========================================================
TODAY = "2026-03-25"
LOOKBACK_DAYS = 730
FUTURE_WINDOW_DAYS = 30
SNAPSHOT_STEP_DAYS = 7
HIGH_RISK_THRESHOLD = 0.4
WINDOWS = [60, 90, 180]


# =========================================================
# 1. LOAD BASE DATA
# =========================================================
def load_base_data():

    bsad = spark.table(
        "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsad"
    )

    df = bsad.filter(
        F.trim("zfbdt").rlike("^[0-9]{8}$") &
        (F.trim("zfbdt") != "00000000")
    ).select(
        F.col("KUNNR").alias("customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.col("BUZEI").alias("line_item_id"),
        F.col("WRBTR").alias("invoice_amount"),
        F.to_date("zfbdt", "yyyyMMdd").alias("baseline_date"),
        F.to_date("augdt", "yyyyMMdd").alias("clearing_date"),
        F.col("zbd3t"),
        F.col("zbd2t"),
        F.col("zbd1t")
    )

    return df.filter(
        F.col("baseline_date") >= F.date_sub(F.lit(TODAY), LOOKBACK_DAYS)
    )


# =========================================================
# 2. DUE DATE
# =========================================================
def add_due_date(df):

    return df.withColumn(
        "payment_terms_days",
        F.when(F.col("zbd3t") != 0, F.col("zbd3t"))
         .when(F.col("zbd2t") != 0, F.col("zbd2t"))
         .when(F.col("zbd1t") != 0, F.col("zbd1t"))
         .otherwise(0)
         .cast(IntegerType())
    ).withColumn(
        "due_date",
        F.date_add("baseline_date", F.col("payment_terms_days"))
    )


# =========================================================
# 3. INVOICE LEVEL
# =========================================================
def build_invoice_level(df):

    return df.groupBy(
        "customer_id",
        "invoice_id"
    ).agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.min("baseline_date").alias("baseline_date"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date")
    )


# =========================================================
# 4. SNAPSHOT GENERATION
# =========================================================
def create_snapshots(invoice_df):

    customers = invoice_df.select("customer_id").distinct()

    date_bounds = invoice_df.select(
        F.min("baseline_date").alias("min_date"),
        F.lit(TODAY).alias("max_date")
    ).collect()[0]

    calendar = spark.sql(f"""
        SELECT explode(
            sequence(
                to_date('{date_bounds["min_date"]}'),
                to_date('{date_bounds["max_date"]}'),
                interval {SNAPSHOT_STEP_DAYS} days
            )
        ) AS snapshot_date
    """)

    base_snapshots = customers.crossJoin(calendar)

    base_snapshots = base_snapshots.withColumn(
        "month_bucket",
        F.date_format("snapshot_date", "yyyy-MM")
    ).dropDuplicates(["customer_id", "month_bucket"])

    return base_snapshots.drop("month_bucket")


# =========================================================
# 5. EXPOSURE FEATURES
# =========================================================
def compute_exposure_features(invoice_df, snapshots):

    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")

    joined = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id")) &
        (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner"
    ).filter(
        (F.col("i.clearing_date").isNull()) |
        (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).withColumn(
        "days_past_due",
        F.when(
            F.col("i.due_date") <= F.col("s.snapshot_date"),
            F.datediff("s.snapshot_date", "i.due_date")
        ).otherwise(0)
    ).withColumn(
        "invoice_age",
        F.datediff("s.snapshot_date", "i.baseline_date")
    )

    return joined.groupBy(
        "s.customer_id",
        "s.snapshot_date"
    ).agg(
        F.sum("invoice_amount").alias("total_outstanding"),
        F.countDistinct("invoice_id").alias("num_open_invoices"),

        F.max("days_past_due").alias("max_dpd"),
        F.avg("days_past_due").alias("avg_dpd"),

        # delinquency buckets
        F.sum(F.when(F.col("days_past_due") > 30, F.col("invoice_amount")).otherwise(0)).alias("amt_30_plus"),
        F.sum(F.when(F.col("days_past_due") > 60, F.col("invoice_amount")).otherwise(0)).alias("amt_60_plus"),
        F.sum(F.when(F.col("days_past_due") > 90, F.col("invoice_amount")).otherwise(0)).alias("amt_90_plus"),

        # aging
        F.max("invoice_age").alias("oldest_invoice_age"),
        F.avg("invoice_age").alias("avg_invoice_age")
    ).withColumn(
        "avg_invoice_size",
        F.col("total_outstanding") / F.col("num_open_invoices")
    ).withColumn(
        "pct_30_plus", F.col("amt_30_plus") / F.col("total_outstanding")
    ).withColumn(
        "pct_60_plus", F.col("amt_60_plus") / F.col("total_outstanding")
    ).withColumn(
        "pct_90_plus", F.col("amt_90_plus") / F.col("total_outstanding")
    )


# =========================================================
# 6. BEHAVIOR FEATURES
# =========================================================
def compute_behavior_features(base_df, snapshots):

    b = base_df.alias("b")
    s = snapshots.alias("s")

    hist = b.join(
        s,
        (F.col("b.customer_id") == F.col("s.customer_id")) &
        (F.col("b.clearing_date") <= F.col("s.snapshot_date")),
        "inner"
    ).withColumn(
        "days_to_pay",
        F.datediff("b.clearing_date", "b.baseline_date")
    )

    aggs = [
        F.avg("days_to_pay").alias("avg_days_to_pay"),
        F.max("days_to_pay").alias("max_days_to_pay"),
        F.avg(F.when(F.col("days_to_pay") <= 0, 1).otherwise(0)).alias("on_time_ratio"),
        F.count("*").alias("total_payments"),
        F.datediff(F.col("s.snapshot_date"), F.max("b.clearing_date")).alias("days_since_last_payment")
    ]

    for w in WINDOWS:
        cond = F.col("b.clearing_date") >= F.date_sub(F.col("s.snapshot_date"), w)

        aggs.extend([
            F.avg(F.when(cond, F.col("days_to_pay"))).alias(f"avg_days_to_pay_{w}d"),
            F.max(F.when(cond, F.col("days_to_pay"))).alias(f"max_days_to_pay_{w}d"),
            F.avg(F.when(cond & (F.col("days_to_pay") <= 0), 1).otherwise(0)).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"num_payments_{w}d")
        ])

    return hist.groupBy(
        "b.customer_id",
        "s.snapshot_date"
    ).agg(*aggs)

# =========================================================
# 7. DUNNING FEATURES
# =========================================================
def load_dunning_features(snapshot_df):

    d = spark.table(
        "hive_metastore.t_erp_ordertocash_rbp_conf.mhnd"
    ).select(
        F.col("KUNNR").alias("d_customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.to_date("LAUFD", "yyyyMMdd").alias("dunning_date"),
        F.col("MAHNN").cast("int").alias("dunning_level")
    )

    s = snapshot_df.select(
        F.col("customer_id").alias("s_customer_id"),
        "snapshot_date"
    )

    df = d.join(
        s,
        (F.col("d_customer_id") == F.col("s_customer_id")) &
        (F.col("dunning_date") <= F.col("snapshot_date")),
        "inner"
    )

    aggs = [
        F.max("dunning_level").alias("max_dunning_level"),
        F.count("*").alias("total_dunning_events"),
        F.avg("dunning_level").alias("avg_dunning_level"),
        F.sum(F.when(F.col("dunning_level") >= 3, 1).otherwise(0)).alias("high_severity_dunning")
    ]

    for w in WINDOWS:
        cond = F.col("dunning_date") >= F.date_sub(F.col("snapshot_date"), w)

        aggs.extend([
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"dunning_events_{w}d"),
            F.sum(F.when(cond & (F.col("dunning_level") >= 3), 1).otherwise(0)).alias(f"high_severity_dunning_{w}d")
        ])

    return df.groupBy(
        "s_customer_id",
        "snapshot_date"
    ).agg(*aggs).withColumn(
        "high_dunning_ratio",
        F.when(F.col("total_dunning_events") > 0,
               F.col("high_severity_dunning") / F.col("total_dunning_events"))
         .otherwise(0)
    ).withColumnRenamed("s_customer_id", "customer_id")

# =========================================================
# 8. P2P FEATURES
# =========================================================
def load_p2p_features(snapshot_df):

    p = spark.table(
        "hive_metastore.f_erp_glide_o2c_12.UDM_P2P_ATTR"
    ).select(
        F.col("fin_customer").alias("p_customer_id"),
        F.substring("fin_invoice_key", 5, 10).alias("invoice_id"),
        F.col("fin_promised_amt").cast("double").alias("promised_amt"),
        F.col("fin_p2p_state").cast("int").alias("p2p_state"),
        F.to_date("fin_p2p_date", "yyyyMMdd").alias("promise_date")
    )

    s = snapshot_df.select(
        F.col("customer_id").alias("s_customer_id"),
        "snapshot_date"
    )

    df = p.join(
        s,
        (F.col("p_customer_id") == F.col("s_customer_id")) &
        (F.col("promise_date") <= F.col("snapshot_date")),
        "inner"
    )

    aggs = [
        F.count("*").alias("total_promises"),
        F.sum(F.when(F.col("p2p_state") == 1, 1).otherwise(0)).alias("broken_promises"),
        F.sum(F.when(F.col("p2p_state") == 3, 1).otherwise(0)).alias("kept_promises"),
        F.sum("promised_amt").alias("total_promised_amount")
    ]

    for w in WINDOWS:
        cond = F.col("promise_date") >= F.date_sub(F.col("snapshot_date"), w)

        aggs.extend([
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"promises_{w}d"),
            F.sum(F.when(cond & (F.col("p2p_state") == 1), 1).otherwise(0)).alias(f"broken_{w}d"),
            F.sum(F.when(cond & (F.col("p2p_state") == 3), 1).otherwise(0)).alias(f"kept_{w}d"),
            F.sum(F.when(cond, F.col("promised_amt"))).alias(f"promised_amt_{w}d")
        ])

    return df.groupBy(
        "s_customer_id",
        "snapshot_date"
    ).agg(*aggs).withColumn(
        "broken_ratio",
        F.when(F.col("total_promises") > 0,
               F.col("broken_promises") / F.col("total_promises"))
         .otherwise(0)
    ).withColumn(
        "kept_ratio",
        F.when(F.col("total_promises") > 0,
               F.col("kept_promises") / F.col("total_promises"))
         .otherwise(0)
    ).withColumn(
        "avg_promised_amount",
        F.when(F.col("total_promises") > 0,
               F.col("total_promised_amount") / F.col("total_promises"))
         .otherwise(0)
    ).withColumn(
        "promise_activity_flag",
        F.when(F.col("total_promises") > 0, 1).otherwise(0)
    ).withColumnRenamed("s_customer_id", "customer_id")

# =========================================================
# 9. TARGET (FIXED - NO LEAKAGE)
# =========================================================
def create_target(invoice_df, snapshots):

    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")

    future = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id")) &
        (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner"
    ).filter(
        # MUST be open at snapshot
        (
            F.col("i.clearing_date").isNull() |
            (F.col("i.clearing_date") > F.col("s.snapshot_date"))
        )
    ).filter(
        # MUST be collected within 30 days
        (F.col("i.clearing_date") >= F.col("s.snapshot_date")) &
        (F.col("i.clearing_date") <= F.date_add(F.col("s.snapshot_date"), FUTURE_WINDOW_DAYS))
    )

    target_df = future.groupBy(
        "s.customer_id",
        "s.snapshot_date"
    ).agg(
        F.sum("i.invoice_amount").alias("collected_30d")
    )

    return target_df

# =========================================================
# 10. COUNTRY FILTER
# =========================================================
def filter_by_countries(df, countries):

    customer_details_df = spark.table(
        "hive_metastore.t_erp_ordertocash_rbp_csi.kna1"
    ).select(
        F.col("KUNNR").alias("customer_id"),
        F.col("LAND1").alias("country"),
        F.col("REGIO").alias("region")
    )

    df = df.join(customer_details_df, on="customer_id", how="left")

    if not countries:
        return df

    return df.filter(F.col("country").isin(countries))


# =========================================================
# 11. FINAL DATASET
# =========================================================
def build_training_dataset():

    base = load_base_data()
    base = add_due_date(base)
    base = filter_by_countries(base, ['FR'])

    invoice_df = build_invoice_level(base)
    snapshots = create_snapshots(invoice_df)

    exposure = compute_exposure_features(invoice_df, snapshots)
    behavior = compute_behavior_features(base, snapshots)
    target = create_target(invoice_df, snapshots)

    # NEW FEATURES
    dunning = load_dunning_features(snapshots)
    p2p = load_p2p_features(snapshots)

    df = exposure \
        .join(behavior, ["customer_id", "snapshot_date"], "left") \
        .join(dunning, ["customer_id", "snapshot_date"], "left") \
        .join(p2p, ["customer_id", "snapshot_date"], "left") \
        .join(target, ["customer_id", "snapshot_date"], "left")

    # NULL handling
    df = df.fillna(0)

    df = df.filter(F.col("total_outstanding") > 0)

    df = df.withColumn(
        "collection_ratio",
        F.col("collected_30d") / F.col("total_outstanding")
    )

    df = df.withColumn(
        "target",
        F.when(F.col("collection_ratio") < HIGH_RISK_THRESHOLD, 1)
         .otherwise(0)
    )

    return df


# =========================================================
# RUN
# =========================================================
training_df = build_training_dataset()

# COMMAND ----------

# DBTITLE 1,Cell 8 function summary
# MAGIC %md
# MAGIC | Function | What it does | Output generated |
# MAGIC | --- | --- | --- |
# MAGIC | `load_base_data()` | Loads invoice-level source data from `hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsad`, cleans date fields, selects relevant columns, and filters records to the lookback window. | Spark DataFrame with base invoice/payment records per line item, including `customer_id`, `invoice_id`, `line_item_id`, `invoice_amount`, `baseline_date`, `clearing_date`, and payment-term fields. |
# MAGIC | `add_due_date(df)` | Derives `payment_terms_days` from the payment-term columns and calculates each record's due date. | Spark DataFrame with added `payment_terms_days` and `due_date` columns. |
# MAGIC | `build_invoice_level(df)` | Aggregates line items to invoice level by customer and invoice. | Spark DataFrame with one row per customer-invoice containing aggregated `invoice_amount`, earliest `baseline_date`, earliest `due_date`, and latest `clearing_date`. |
# MAGIC | `create_snapshots(invoice_df)` | Builds periodic customer snapshot dates from the earliest baseline date through `TODAY`, then keeps one snapshot per customer per month. | Spark DataFrame of customer snapshot rows with `customer_id` and `snapshot_date`. |
# MAGIC | `compute_exposure_features(invoice_df, snapshots)` | Joins open invoices to each snapshot and calculates outstanding exposure, delinquency, and aging metrics at snapshot date. | Spark DataFrame with snapshot-level exposure features such as `total_outstanding`, `num_open_invoices`, `max_dpd`, `avg_dpd`, bucketed overdue amounts, invoice age metrics, and percentage delinquency features. |
# MAGIC | `compute_behavior_features(base_df, snapshots)` | Uses historical cleared invoices up to each snapshot to calculate payment behavior metrics overall and over rolling windows. | Spark DataFrame with snapshot-level behavior features like `avg_days_to_pay`, `max_days_to_pay`, `on_time_ratio`, `total_payments`, `days_since_last_payment`, plus 60/90/180-day windowed payment metrics. |
# MAGIC | `load_dunning_features(snapshot_df)` | Pulls dunning history from `hive_metastore.t_erp_ordertocash_rbp_conf.mhnd` and aggregates dunning activity up to each snapshot. | Spark DataFrame with snapshot-level dunning features including `max_dunning_level`, `total_dunning_events`, `avg_dunning_level`, `high_severity_dunning`, rolling-window dunning counts, and `high_dunning_ratio`. |
# MAGIC | `load_p2p_features(snapshot_df)` | Pulls promise-to-pay history from `hive_metastore.f_erp_glide_o2c_12.UDM_P2P_ATTR` and aggregates promise activity up to each snapshot. | Spark DataFrame with snapshot-level P2P features such as `total_promises`, `broken_promises`, `kept_promises`, `total_promised_amount`, rolling-window promise metrics, `broken_ratio`, `kept_ratio`, `avg_promised_amount`, and `promise_activity_flag`. |
# MAGIC | `create_target(invoice_df, snapshots)` | Identifies invoices that were still open at snapshot time and then got cleared within the next 30 days, avoiding leakage from future-open/closed status. | Spark DataFrame with `customer_id`, `snapshot_date`, and `collected_30d` representing amount collected within 30 days after the snapshot. |
# MAGIC | `filter_by_countries(df, countries)` | Joins customer master data from `hive_metastore.t_erp_ordertocash_rbp_csi.kna1` and optionally filters the dataset to selected countries. | Spark DataFrame enriched with `country` and `region`, optionally restricted to the requested country list. |
# MAGIC | `build_training_dataset()` | Orchestrates the full feature-engineering pipeline: loads data, adds due dates, filters countries, creates invoice snapshots, computes exposure/behavior/dunning/P2P features, joins target, fills nulls, filters to open exposure, and derives modeling labels. | Final training Spark DataFrame containing snapshot-level model features plus `collected_30d`, `collection_ratio`, and binary `target`. |

# COMMAND ----------

training_df.printSchema()

# COMMAND ----------

training_df.groupBy("target").count().show()

# COMMAND ----------

training_df.select(
    F.avg(F.when(F.col("amt_90_plus") > 0, 1).otherwise(0))
).show()

# COMMAND ----------

training_df.select(
    F.avg(F.when(F.col("amt_60_plus") > 0, 1).otherwise(0))
).show()

# COMMAND ----------

training_df.select(
    F.avg(F.when(F.col("amt_30_plus") > 0, 1).otherwise(0))
).show()

# COMMAND ----------

training_df.filter(F.col("collection_ratio") > 1).count()

# COMMAND ----------

training_df.select("collection_ratio").describe().show()

# COMMAND ----------

training_df.select("avg_dpd", "amt_90_plus").describe().show()

# COMMAND ----------

training_df.filter(F.col("collected_30d").isNull()).count()

# COMMAND ----------

# MAGIC %md
# MAGIC PLOTTING DISTRIBUTION

# COMMAND ----------

training_df.selectExpr(
    "percentile(collection_ratio, array(0.1,0.25,0.5,0.75,0.9))"
).show(truncate=False)

# COMMAND ----------

# Sample for visualization (adjust fraction if needed)
pdf = training_df.sample(fraction=0.1, seed=42).select("collection_ratio").toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.hist(pdf["collection_ratio"], bins=100)
plt.title("Distribution of Collection Ratio")
plt.xlabel("Collection Ratio")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

import numpy as np

sorted_vals = np.sort(pdf["collection_ratio"])
cdf = np.arange(len(sorted_vals)) / float(len(sorted_vals))

plt.figure(figsize=(10,5))

plt.plot(sorted_vals, cdf)

plt.title("CDF of Collection Ratio")
plt.xlabel("Collection Ratio")
plt.ylabel("Cumulative % of Customers")

plt.grid(True)
plt.show()

# COMMAND ----------

quantiles = training_df.selectExpr(
    "percentile(collection_ratio, array(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)) as q"
).collect()[0]["q"]

for p, q in zip(
    [1,5,10,20,30,40,50,60,70,80,90],
    quantiles
):
    print(f"{p}th percentile: {q:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC The biggest jump happens here:
# MAGIC
# MAGIC 30th percentile -> 0.139
# MAGIC
# MAGIC 40th percentile -> 0.438
# MAGIC
# MAGIC This is a natural separation point in the data
# MAGIC
# MAGIC Best choice for threshold in this case: HIGH_RISK_THRESHOLD = 0.4
# MAGIC
# MAGIC Aligns with natural distribution break
# MAGIC Gives ~38% high risk -> very manageable
# MAGIC
# MAGIC Captures both:
# MAGIC non-payers
# MAGIC weak payers
# MAGIC
# MAGIC Matches business intuition: "If you collect less than ~40%, that's a problem"

# COMMAND ----------

# MAGIC %md
# MAGIC This code evaluates how your definition of "high risk customer" changes as you vary the threshold on the collection ratio.
# MAGIC
# MAGIC It first creates a list of threshold values from 0.0 to 1.0 in steps of 0.1. Each threshold represents a possible business rule: "flag a customer as high risk if their collection ratio is below this value."
# MAGIC
# MAGIC For each threshold, the code creates a temporary label (target_tmp) by comparing every customer's collection_ratio against that threshold. If the ratio is lower than the threshold, the customer is marked as high risk (1), otherwise not high risk (0).
# MAGIC
# MAGIC It then counts how many customers fall into each class (0 and 1) and calculates the proportion of high-risk customers in the dataset. This is done by dividing the number of high-risk customers by the total number of customers.
# MAGIC
# MAGIC Finally, it prints the threshold alongside the percentage of customers classified as high risk at that threshold.
# MAGIC
# MAGIC The output shows a clear pattern: as the threshold increases, more customers are labeled high risk. For example, at threshold 0.1 only 27.5% are high risk, while at 1.0 about 51.8% are high risk. This helps business understand how strict or lenient each threshold is, and choose one based on operational capacity or risk appetite rather than intuition.

# COMMAND ----------

thresholds = [i/10 for i in range(0,11)]

for t in thresholds:
    counts = training_df.withColumn(
        "target_tmp",
        F.when(F.col("collection_ratio") < t, 1).otherwise(0)
    ).groupBy("target_tmp").count().collect()

    result = {row["target_tmp"]: row["count"] for row in counts}

    high_risk = result.get(1, 0)
    total = sum(result.values())

    print(f"Threshold={t:.1f} | High Risk % = {high_risk/total:.3f}")

# COMMAND ----------

pdf2 = training_df.sample(0.1, seed=42).select(
    "collection_ratio",
    "total_outstanding"
).toPandas()

thresholds = np.linspace(0,1,50)

amt_curve = []

for t in thresholds:
    amt = pdf2.loc[
        pdf2["collection_ratio"] < t,
        "total_outstanding"
    ].sum()

    amt_curve.append(amt)

plt.figure(figsize=(10,5))
plt.plot(thresholds, amt_curve)

plt.title("Exposure captured vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Total Outstanding Captured")

plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC TRAINING

# COMMAND ----------

from pyspark.sql.types import DecimalType, DoubleType
from pyspark.sql import DataFrame as SparkDF, functions as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def prepare_ml_data(df):

    # Drop non-feature columns
    drop_cols = [
        "customer_id",
        "snapshot_date",
        "target",
        "collected_30d",
        "collection_ratio",
        "last_payment_date"
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Fill nulls
    df = df.fillna(0)

    print("Features used in the model: ", feature_cols)

    return df, feature_cols

def time_split(df):

    train_df = df.filter(F.col("snapshot_date") <= "2024-06-30")

    valid_df = df.filter(
        (F.col("snapshot_date") > "2024-06-30") &
        (F.col("snapshot_date") <= "2024-08-31")
    )

    test_df = df.filter(F.col("snapshot_date") > "2024-08-31")

    return train_df, valid_df, test_df

def spark_to_pandas_safe(spark_df, sample_frac=None):

    if sample_frac:
        spark_df = spark_df.sample(fraction=sample_frac, seed=42)

    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(
                field.name,
                spark_df[field.name].cast(DoubleType())
            )

    spark_df = spark_df.fillna(0)

    return spark_df.toPandas()

def to_pandas(df, feature_cols):

    pdf = spark_to_pandas_safe(df.select(feature_cols + ["target"]))

    X = pdf[feature_cols]
    y = pdf["target"]

    return X, y

def train_xgb(X_train, y_train, X_valid, y_valid):

    train_counts = y_train.value_counts().to_dict()

    num_neg = train_counts.get(0, 1)
    num_pos = train_counts.get(1, 1)

    sample_weights = np.where(y_train == 1, num_neg / num_pos, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
        sample_weight=sample_weights
    )

    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold-based predictions
    y_pred = (y_prob >= threshold).astype(int)

    best_t = get_threshold_for_recall(y_test, y_prob)
    print(f"\n--- Best Threshold = {t} ---")

    print(f"\n--- Evaluation @ Threshold = {threshold} ---")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # AUC ROC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # PR AUC (VERY IMPORTANT for imbalance)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"PR-AUC: {pr_auc:.4f}")

    f1 = f1_score(y_test, y_pred)

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(6,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    return y_prob, roc_auc, pr_auc, f1

def tune_threshold(y_test, y_prob):

    thresholds = np.linspace(0.01, 0.5, 50)

    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = ((y_test == 1) & (y_pred == 1)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        fn = ((y_test == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        results.append((t, precision, recall))

    df = pd.DataFrame(results, columns=["threshold", "precision", "recall"])

    return df.sort_values(by="recall", ascending=False)

def get_threshold_for_recall(y_test, y_prob, target_recall=0.79):

    thresholds = np.linspace(0.001, 0.5, 200)

    best_t = None
    best_diff = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tp = ((y_test == 1) & (y_pred == 1)).sum()
        fn = ((y_test == 1) & (y_pred == 0)).sum()

        recall = tp / (tp + fn + 1e-9)

        diff = abs(recall - target_recall)

        if diff < best_diff:
            best_diff = diff
            best_t = t

    return best_t

def get_feature_importance(model, feature_cols):

    importance = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return fi_df

# COMMAND ----------

# Prepare
df, feature_cols = prepare_ml_data(training_df)

# Split
train_df, valid_df, test_df = time_split(df)

# Convert
X_train, y_train = to_pandas(train_df, feature_cols)
X_valid, y_valid = to_pandas(valid_df, feature_cols)
X_test, y_test = to_pandas(test_df, feature_cols)

import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

MODEL_NAME = "collection_risk_model_customer_v1"

with mlflow.start_run() as run:

    # Train
    model = train_xgb(X_train, y_train, X_valid, y_valid)
    threshold = 0.4

    y_prob, roc_auc, pr_auc, f1 = evaluate_model(model, X_test, y_test, threshold=threshold)

    # =====================================================
    # LOG METRICS
    # =====================================================
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("f1", f1)

    # =====================================================
    # LOG PARAMS (important for reproducibility)
    # =====================================================
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("features", ",".join(feature_cols))

    # =====================================================
    # LOG MODEL
    # =====================================================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(5)
    )

    model_uri = f"runs:/{run.info.run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Model registered as: {MODEL_NAME}")

    # Feature Importance
    fi_df = get_feature_importance(model, feature_cols)
    print(fi_df.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC # EXPLAINABILITY

# COMMAND ----------

model = mlflow.sklearn.load_model(
    model_uri="models:/collection_risk_model_customer_v1/latest"
)

# COMMAND ----------

# =========================================================
# IMPORTS
# =========================================================
import numpy as np
import pandas as pd
import xgboost as xgb

# =========================================================
# 1. PREDICTIONS
# =========================================================

pred_proba = model.predict_proba(X_test)[:, 1]
pred_class = (pred_proba >= 0.5).astype(int)

# =========================================================
# 2. NATIVE XGBOOST SHAP VALUES
# =========================================================

dmatrix = xgb.DMatrix(X_test)

contribs = model.get_booster().predict(
    dmatrix,
    pred_contribs=True
)

# remove bias term
shap_values = contribs[:, :-1]

# =========================================================
# 3. ALIGN FEATURES SAFELY
# =========================================================

feature_names = X_test.columns

# =========================================================
# 4. GLOBAL EXPLAINABILITY
# =========================================================

global_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": np.abs(shap_values).mean(axis=0)
})

global_importance = global_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nGLOBAL FEATURE IMPORTANCE")
print(global_importance.head(20))

# =========================================================
# 5. LOCAL SHAP DATAFRAME
# =========================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

# =========================================================
# 6. TOP DRIVERS FUNCTION (LOCAL EXPLANATION)
# =========================================================

def get_top_drivers(row_index, top_n=5):

    row_shap = shap_df.iloc[row_index]

    contribution_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": row_shap.values,
        "feature_value": X_test.iloc[row_index].values
    })

    contribution_df["abs_shap"] = contribution_df["shap_value"].abs()

    return contribution_df.sort_values(
        "abs_shap",
        ascending=False
    ).head(top_n)

# =========================================================
# 7. BUSINESS EXPLANATION MAPPER
# =========================================================

def build_business_explanation(contribution_df):

    explanation_lines = []

    for _, row in contribution_df.iterrows():

        feature = row["feature"]
        value = row["feature_value"]
        shap_value = row["shap_value"]

        impact = "increased" if shap_value > 0 else "reduced"

        if feature == "pct_90_plus":
            text = f"High >90 days overdue ({value:.2f}) {impact} risk."

        elif feature == "broken_ratio":
            text = f"High broken promise ratio ({value:.2f}) {impact} risk."

        elif feature == "avg_days_to_pay":
            text = f"Long payment delays ({value:.1f} days) {impact} risk."

        elif feature == "max_dunning_level":
            text = f"High dunning severity (level {value:.0f}) {impact} risk."

        elif feature == "on_time_ratio":
            text = f"Low on-time payment ratio ({value:.2f}) {impact} risk."

        elif feature == "days_since_last_payment":
            text = f"No recent payment ({value:.0f} days) {impact} risk."

        elif feature == "total_outstanding":
            text = f"High outstanding exposure ({value:,.0f}) {impact} risk."

        else:
            text = f"{feature} ({value}) {impact} risk."

        explanation_lines.append(text)

    return explanation_lines

# =========================================================
# 8. GENERATE LOCAL EXPLANATIONS
# =========================================================

collector_output = []

# IMPORTANT: reset index to avoid misalignment
X_test = X_test.reset_index(drop=True)
pdf = pdf.reset_index(drop=True)

for i in range(len(X_test)):

    top_drivers = get_top_drivers(i, top_n=5)

    explanations = build_business_explanation(top_drivers)

    collector_output.append({
        "prediction_probability": float(pred_proba[i]),
        "predicted_class": int(pred_class[i]),
        "top_risk_drivers": explanations
    })

collector_explanations_df = pd.DataFrame(collector_output)

# =========================================================
# 9. GLOBAL SUMMARY OUTPUT (BUSINESS VIEW)
# =========================================================

print("\nTOP GLOBAL DRIVERS")
print(global_importance.head(10))

# =========================================================
# 10. SAMPLE LOCAL OUTPUT
# =========================================================

print("\nSAMPLE COLLECTOR EXPLANATIONS\n")

for i in range(min(3, len(collector_explanations_df))):

    row = collector_explanations_df.iloc[i]

    print("====================================")
    print("Risk Probability:", round(row["prediction_probability"], 3))
    print("Predicted Class:", row["predicted_class"])

    print("\nTop Drivers:")

    for x in row["top_risk_drivers"]:
        print("-", x)

# COMMAND ----------

test_df_pd = test_df.toPandas()
test_df_pd["pred"] = model.predict(X_test)

test_df_pd.groupby("pred")["collection_ratio"].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC # PREDICTION
# MAGIC
# MAGIC BSID (open invoices) + BSAD (closed invoices)
# MAGIC    -> Add due_date (same logic as training)
# MAGIC    -> Build invoice-level features (same grouping)
# MAGIC    -> Create snapshot (TODAY only, but same logic style)
# MAGIC    -> Compute exposure features (identical code)
# MAGIC    -> Compute behavior features (BSAD only)
# MAGIC    -> Join features
# MAGIC    -> Apply SAME feature list
# MAGIC    -> Predict with XGBoost

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DecimalType, DoubleType
import pandas as pd
import mlflow

# =========================================================
# CONFIG
# =========================================================
TODAY = F.current_date()
WINDOWS = [60, 90, 180]

model = mlflow.sklearn.load_model(
    model_uri="models:/collection_risk_model_customer_v1/latest"
)

# MUST MATCH TRAINING FEATURES EXACTLY
FEATURE_COLS = model.get_booster().feature_names
print(FEATURE_COLS)

# =========================================================
# 1. LOAD DATA
# =========================================================
def load_inference_data():

    bsid = spark.table("hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid")
    bsad = spark.table("hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsad")

    open_df = bsid.select(
        F.col("KUNNR").alias("customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.col("BUZEI").alias("line_item_id"),
        F.col("WRBTR").alias("invoice_amount"),
        F.to_date("zfbdt", "yyyyMMdd").alias("baseline_date"),
        F.lit(None).cast("date").alias("clearing_date"),
        "zbd1t","zbd2t","zbd3t"
    )

    closed_df = bsad.select(
        F.col("KUNNR").alias("customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.col("BUZEI").alias("line_item_id"),
        F.col("WRBTR").alias("invoice_amount"),
        F.to_date("zfbdt", "yyyyMMdd").alias("baseline_date"),
        F.to_date("augdt", "yyyyMMdd").alias("clearing_date"),
        "zbd1t","zbd2t","zbd3t"
    )

    return open_df.unionByName(closed_df)

# =========================================================
# 2. DUE DATE
# =========================================================
def add_due_date(df):

    return df.withColumn(
        "payment_terms_days",
        F.when(F.col("zbd3t") != 0, F.col("zbd3t"))
         .when(F.col("zbd2t") != 0, F.col("zbd2t"))
         .when(F.col("zbd1t") != 0, F.col("zbd1t"))
         .otherwise(0)
         .cast(IntegerType())
    ).withColumn(
        "due_date",
        F.date_add("baseline_date", F.col("payment_terms_days"))
    )

# =========================================================
# 3. INVOICE LEVEL
# =========================================================
def build_invoice_level(df):

    return df.groupBy("customer_id","invoice_id").agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.min("baseline_date").alias("baseline_date"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date")
    )

# =========================================================
# 4. SNAPSHOT (TODAY)
# =========================================================
def build_snapshot(invoice_df):

    return invoice_df.filter(
        (F.col("baseline_date") <= TODAY) &
        (
            F.col("clearing_date").isNull() |
            (F.col("clearing_date") > TODAY)
        )
    ).withColumn("snapshot_date", TODAY)

# =========================================================
# 5. CUSTOMER SPINE
# =========================================================
def customer_spine():

    return spark.table(
        "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid"
    ).select(
        F.col("KUNNR").alias("customer_id")
    ).distinct() \
     .withColumn("snapshot_date", TODAY)

# =========================================================
# 5. EXPOSURE FEATURES
# =========================================================
def exposure_features(snapshot_df, spine):

    df = spine.join(snapshot_df, ["customer_id","snapshot_date"], "left")

    df = df.withColumn("days_past_due",
        F.datediff("snapshot_date", "due_date")
    ).withColumn("invoice_age",
        F.datediff("snapshot_date", "baseline_date")
    )

    agg = df.groupBy("customer_id","snapshot_date").agg(

        F.sum("invoice_amount").alias("total_outstanding"),
        F.countDistinct("invoice_id").alias("num_open_invoices"),

        F.max("days_past_due").alias("max_dpd"),
        F.avg("days_past_due").alias("avg_dpd"),

        F.sum(F.when(F.col("days_past_due")>30,F.col("invoice_amount")).otherwise(0)).alias("amt_30_plus"),
        F.sum(F.when(F.col("days_past_due")>60,F.col("invoice_amount")).otherwise(0)).alias("amt_60_plus"),
        F.sum(F.when(F.col("days_past_due")>90,F.col("invoice_amount")).otherwise(0)).alias("amt_90_plus"),

        F.max("invoice_age").alias("oldest_invoice_age"),
        F.avg("invoice_age").alias("avg_invoice_age")
    )

    return agg.fillna(0) \
    .withColumn("avg_invoice_size",
        F.col("total_outstanding")/F.col("num_open_invoices")
    ).withColumn("pct_30_plus",
        F.col("amt_30_plus")/F.col("total_outstanding")
    ).withColumn("pct_60_plus",
        F.col("amt_60_plus")/F.col("total_outstanding")
    ).withColumn("pct_90_plus",
        F.col("amt_90_plus")/F.col("total_outstanding")
    )

# =========================================================
# 6. BEHAVIOR FEATURES
# =========================================================
def behavior_features(base_df, spine_df):

    b = base_df.filter(F.col("clearing_date").isNotNull()).withColumn(
        "days_to_pay",
        F.datediff("clearing_date","baseline_date")
    )

    s = spine_df.select(
        F.col("customer_id").alias("s_customer_id"),
        "snapshot_date"
    )

    df = s.join(
        b,
        (F.col("customer_id")==F.col("s_customer_id")) &
        (F.col("clearing_date")<=F.col("snapshot_date")) &
        (F.col("clearing_date") >= F.date_sub(F.col("snapshot_date"), 730)),
        "left"
    )

    aggs = [
        F.avg("days_to_pay").alias("avg_days_to_pay"),
        F.max("days_to_pay").alias("max_days_to_pay"),
        F.avg(F.when(F.col("days_to_pay")<=0,1).otherwise(0)).alias("on_time_ratio"),
        F.count("*").alias("total_payments"),
        F.datediff(F.col("snapshot_date"),F.max("clearing_date")).alias("days_since_last_payment")
    ]

    for w in WINDOWS:
        cond = F.col("clearing_date") >= F.date_sub(F.col("snapshot_date"), w)
        aggs.extend([
            F.avg(F.when(cond,F.col("days_to_pay"))).alias(f"avg_days_to_pay_{w}d"),
            F.max(F.when(cond,F.col("days_to_pay"))).alias(f"max_days_to_pay_{w}d"),
            F.avg(F.when(cond & (F.col("days_to_pay")<=0),1).otherwise(0)).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond,1).otherwise(0)).alias(f"num_payments_{w}d")
        ])


    return df.groupBy("s_customer_id","snapshot_date").agg(*aggs).withColumnRenamed("s_customer_id", "customer_id")

# =========================================================
# 7. DUNNING
# =========================================================
def dunning_features(spine_df):

    d = spark.table(
        "hive_metastore.t_erp_ordertocash_rbp_conf.mhnd"
    ).select(
        F.col("KUNNR").alias("d_customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.to_date("LAUFD", "yyyyMMdd").alias("dunning_date"),
        F.col("MAHNN").cast("int").alias("dunning_level")
    )

    s = spine_df.select(
        F.col("customer_id").alias("s_customer_id"),
        "snapshot_date"
    )

    df = s.join(
        d,
        (F.col("d_customer_id") == F.col("s_customer_id")) &
        (F.col("dunning_date") <= F.col("snapshot_date")) &
        (F.col("dunning_date") >= F.date_sub(F.col("snapshot_date"), 730)),
        "left"
    )

    aggs = [
        F.max("dunning_level").alias("max_dunning_level"),
        F.count("dunning_level").alias("total_dunning_events"),
        F.avg("dunning_level").alias("avg_dunning_level"),
        F.sum(F.when(F.col("dunning_level")>=3,1).otherwise(0)).alias("high_severity_dunning")
    ]

    for w in WINDOWS:
        cond = F.col("dunning_date") >= F.date_sub(F.col("snapshot_date"), w)
        aggs.extend([
            F.sum(F.when(cond,1).otherwise(0)).alias(f"dunning_events_{w}d"),
            F.sum(F.when(cond & (F.col("dunning_level")>=3),1).otherwise(0)).alias(f"high_severity_dunning_{w}d")
        ])

    out = df.groupBy("s_customer_id", "snapshot_date").agg(*aggs).withColumnRenamed("s_customer_id", "customer_id")

    return out.withColumn(
        "high_dunning_ratio",
        F.when(F.col("total_dunning_events") > 0,
               F.col("high_severity_dunning") / F.col("total_dunning_events"))
         .otherwise(0)
    )

# =========================================================
# 8. P2P
# =========================================================
def p2p_features(spine_df):

    p = spark.table(
        "hive_metastore.f_erp_glide_o2c_12.UDM_P2P_ATTR"
    ).select(
        F.col("fin_customer").alias("p_customer_id"),
        F.substring("fin_invoice_key", 5, 10).alias("invoice_id"),
        F.col("fin_promised_amt").cast("double").alias("promised_amt"),
        F.col("fin_p2p_state").cast("int").alias("p2p_state"),
        F.to_date("fin_p2p_date", "yyyyMMdd").alias("promise_date")
    )

    s = spine_df.select(
        F.col("customer_id").alias("s_customer_id"),
        "snapshot_date"
    )

    df = s.join(
        p,
        (F.col("p_customer_id")==F.col("s_customer_id")) &
        (F.col("promise_date")<= F.col("snapshot_date")) &
        (F.col("promise_date") >= F.date_sub(F.col("snapshot_date"), 730)),
        "left"
    )

    aggs = [
        F.count("promise_date").alias("total_promises"),
        F.sum(F.when(F.col("p2p_state")==1,1).otherwise(0)).alias("broken_promises"),
        F.sum(F.when(F.col("p2p_state")==3,1).otherwise(0)).alias("kept_promises"),
        F.sum("promised_amt").alias("total_promised_amount")
    ]

    for w in WINDOWS:
        cond = F.col("promise_date") >= F.date_sub(F.col("snapshot_date"), w)
        aggs.extend([
            F.sum(F.when(cond,1).otherwise(0)).alias(f"promises_{w}d"),
            F.sum(F.when(cond & (F.col("p2p_state")==1),1).otherwise(0)).alias(f"broken_{w}d"),
            F.sum(F.when(cond & (F.col("p2p_state")==3),1).otherwise(0)).alias(f"kept_{w}d"),
            F.sum(F.when(cond,F.col("promised_amt"))).alias(f"promised_amt_{w}d")
        ])

    out = df.groupBy("s_customer_id", "snapshot_date").agg(*aggs).withColumnRenamed("s_customer_id", "customer_id")

    return out.withColumn(
        "broken_ratio",
        F.when(F.col("total_promises") > 0,
               F.col("broken_promises") / F.col("total_promises"))
         .otherwise(0)
    ).withColumn(
        "kept_ratio",
        F.when(F.col("total_promises") > 0,
               F.col("kept_promises") / F.col("total_promises"))
         .otherwise(0)
    ).withColumn(
        "avg_promised_amount",
        F.when(F.col("total_promises") > 0,
               F.col("total_promised_amount") / F.col("total_promises"))
         .otherwise(0)
    ).withColumn(
        "promise_activity_flag",
        F.when(F.col("total_promises") > 0, 1).otherwise(0)
    ).withColumnRenamed("s_customer_id", "customer_id")

# =========================================================
# 9. COUNTRY FILTER
# =========================================================
def filter_by_countries(df, countries):

    if not countries:
        return df

    eligible_customers = spark.table(
        "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid"
    ).select(
        F.col("KUNNR").alias("customer_id"),
        F.upper(F.substring(F.col("BUKRS"), 1, 2)).alias("country_code")
    ).filter(
        F.col("country_code").isin(
            [c.upper() for c in countries]
        )
    ).select(
        "customer_id"
    ).distinct()

    return df.join(
        eligible_customers,
        on="customer_id",
        how="inner"
    )

# =========================================================
# 10. SPARK TO PANDAS
# =========================================================
def spark_to_pandas_safe(spark_df, sample_frac=None):

    if sample_frac:
        spark_df = spark_df.sample(fraction=sample_frac, seed=42)

    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(
                field.name,
                spark_df[field.name].cast(DoubleType())
            )

    spark_df = spark_df.fillna(0)

    return spark_df.toPandas()

import numpy as np
import pandas as pd
import xgboost as xgb

# =========================================================
# BUSINESS EXPLANATION FUNCTION
# =========================================================

def build_business_explanations(
    shap_row,
    feature_row,
    top_n=5
):

    contribution_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap_value": shap_row,
        "feature_value": feature_row
    })

    contribution_df["abs_shap"] = (
        contribution_df["shap_value"].abs()
    )

    contribution_df = contribution_df.sort_values(
        by="abs_shap",
        ascending=False
    ).head(top_n)

    explanations = []

    for _, row in contribution_df.iterrows():

        feature = row["feature"]
        value = row["feature_value"]
        shap_value = row["shap_value"]

        impact = (
            "increased"
            if shap_value > 0
            else "reduced"
        )

        if feature == "pct_90_plus":

            text = (
                f"High percentage of invoices "
                f"over 90 days past due "
                f"({value:.2f}) "
                f"{impact} risk."
            )

        elif feature == "broken_ratio":

            text = (
                f"High broken promise ratio "
                f"({value:.2f}) "
                f"{impact} risk."
            )

        elif feature == "avg_days_to_pay":

            text = (
                f"Long customer payment delays "
                f"({value:.1f} days) "
                f"{impact} risk."
            )

        elif feature == "max_dunning_level":

            text = (
                f"Severe dunning activity "
                f"(level {value:.0f}) "
                f"{impact} risk."
            )

        elif feature == "on_time_ratio":

            text = (
                f"Low on-time payment ratio "
                f"({value:.2f}) "
                f"{impact} risk."
            )

        elif feature == "days_since_last_payment":

            text = (
                f"Long time since last payment "
                f"({value:.0f} days) "
                f"{impact} risk."
            )

        elif feature == "total_outstanding":

            text = (
                f"Large outstanding balance "
                f"({value:,.2f}) "
                f"{impact} risk."
            )

        else:

            text = (
                f"{feature} ({value}) "
                f"{impact} risk."
            )

        explanations.append(text)

    return explanations

# =========================================================
# 11. SCORING (NO RANKING)
# =========================================================
def score(df):

    # 1. convert full dataset once
    pdf = spark_to_pandas_safe(
        df.select(["customer_id", "snapshot_date"] + FEATURE_COLS)
    ).fillna(0)

    # 2. features
    X = pdf[FEATURE_COLS]

    # 3. predict
    probs = model.predict_proba(X)[:, 1] # 0 - low class, 1 - high class

    # 4. attach directly (NO JOIN, NO SPARK)
    pdf["risk_score"] = probs

    pdf["predicted_class"] = pd.cut(
        pdf["risk_score"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    # =====================================================
    # SHAP VALUES
    # =====================================================

    dmatrix = xgb.DMatrix(X)

    contribs = model.get_booster().predict(
        dmatrix,
        pred_contribs=True
    )

    # remove bias term
    shap_values = contribs[:, :-1]

    # =====================================================
    # BUSINESS EXPLANATIONS
    # =====================================================

    all_explanations = []

    for i in range(len(pdf)):

        explanations = build_business_explanations(
            shap_row=shap_values[i],
            feature_row=X.iloc[i].values,
            top_n=5
        )

        all_explanations.append(
            " | ".join(explanations)
        )

    pdf["collector_explanations"] = all_explanations

    # =====================================================
    # TOP DRIVER FEATURE
    # =====================================================

    top_driver = []

    for i in range(len(pdf)):

        shap_abs = np.abs(shap_values[i])

        max_idx = np.argmax(shap_abs)

        top_driver.append(
            FEATURE_COLS[max_idx]
        )

    pdf["top_risk_driver"] = top_driver

    # =====================================================
    # RETURN TO SPARK
    # =====================================================

    return spark.createDataFrame(pdf)

# =========================================================
# 10. PIPELINE
# =========================================================
def run_prediction():

    spine = filter_by_countries(customer_spine(), ['FR'])
    base = load_inference_data()
    base = add_due_date(base)
    base = filter_by_countries(base, ['FR'])

    inv = build_invoice_level(base)
    snap = build_snapshot(inv)

    exp = exposure_features(snap, spine)
    beh = behavior_features(base, spine)
    dun = dunning_features(spine)
    p2p = p2p_features(spine)

    feat = spine \
        .join(exp,["customer_id","snapshot_date"],"left") \
        .join(beh,["customer_id","snapshot_date"],"left") \
        .join(dun,["customer_id","snapshot_date"],"left") \
        .join(p2p,["customer_id","snapshot_date"],"left") \
        .fillna(0)

    return score(feat)

predictions_df = run_prediction()
display(predictions_df)

# COMMAND ----------

pdf = predictions_df.toPandas()
pdf["risk_score"].describe()

# COMMAND ----------

import matplotlib.pyplot as plt

pdf["risk_score"].hist(bins=50)
plt.title("Risk Score Distribution")
plt.show()

# COMMAND ----------

bands = pd.qcut(
    pdf["risk_score"],
    q=3,
    duplicates="drop"
)

num_bins = len(bands.cat.categories)

labels = ["Low", "Medium", "High"][:num_bins]

pdf["risk_band"] = pd.qcut(
    pdf["risk_score"],
    q=num_bins,
    labels=labels,
    duplicates="drop"
)

# COMMAND ----------

import matplotlib.pyplot as plt

pdf["risk_band"].hist(bins=50)
plt.title("Risk Band Distribution")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # PUSH TO TABLE

# COMMAND ----------

# Define schema and table names
schema_name = "f_erp_glide_o2c_12"
table_name = "collection_ml_customer"
base_location = "abfss://root@coentus6abfsprod001.dfs.core.windows.net/data/raw/vala/vala/global/vala/Foundation/"
table_location = f"{base_location}{table_name}/"

spark.sql("USE CATALOG hive_metastore")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
from pyspark.sql.functions import lit
from pyspark.sql.functions import current_date

spark_df = (
    predictions_df
    .withColumn("country", lit("FR"))
    .withColumn("refresh_date", current_date())
)

spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table_location)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS hive_metastore.{schema_name}.{table_name}
    USING DELTA
    LOCATION '{table_location}'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # DNT

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE f_erp_glide_o2c_12.collection_ml_customer AS
# MAGIC
# MAGIC SELECT
# MAGIC     c.* EXCEPT(predicted_class),
# MAGIC
# MAGIC     CASE
# MAGIC
# MAGIC         WHEN EXISTS (
# MAGIC             SELECT 1
# MAGIC             FROM f_erp_glide_o2c_12.table_invoice_unified u
# MAGIC             WHERE u.customer_id = c.customer_id
# MAGIC               AND u.company_code = 'AT01'
# MAGIC               AND c.customer_id IN (
# MAGIC                     '1100802482',
# MAGIC                     '1400022779',
# MAGIC                     '1100071147',
# MAGIC                     '1100070805'
# MAGIC               )
# MAGIC         )
# MAGIC         THEN 'DNT'
# MAGIC
# MAGIC         WHEN EXISTS (
# MAGIC             SELECT 1
# MAGIC             FROM f_erp_glide_o2c_12.table_invoice_unified u
# MAGIC             WHERE u.customer_id = c.customer_id
# MAGIC               AND u.company_code IN ('FR01', 'FR05')
# MAGIC               AND c.customer_id IN (
# MAGIC                     '1100033624',
# MAGIC                     '1100035172',
# MAGIC                     '1100047129',
# MAGIC                     '1100655377',
# MAGIC                     '1400000296',
# MAGIC                     '1400000486',
# MAGIC                     '1400000714',
# MAGIC                     '1400020340',
# MAGIC                     '1400021622',
# MAGIC                     '1400022347'
# MAGIC               )
# MAGIC         )
# MAGIC         THEN 'DNT'
# MAGIC
# MAGIC         ELSE c.predicted_class
# MAGIC
# MAGIC     END AS predicted_class
# MAGIC
# MAGIC FROM f_erp_glide_o2c_12.collection_ml_customer c

# COMMAND ----------

t = spark.table("f_erp_glide_o2c_12.collection_ml_customer")
display(t)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     country,
# MAGIC     COUNT(*)
# MAGIC FROM f_erp_glide_o2c_12.collection_ml_customer
# MAGIC GROUP BY country

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     predicted_class,
# MAGIC     COUNT(*)
# MAGIC FROM f_erp_glide_o2c_12.collection_ml_customer
# MAGIC GROUP BY predicted_class

# COMMAND ----------

# MAGIC %md
# MAGIC # LLM Summary

# COMMAND ----------

# MAGIC %pip install langchain_openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

kong_client_id=""
kong_client_secret=""
kong_base_url=""
api_version=""
api_deployment_name=""
kong_endpoint_url=""

# COMMAND ----------

import os
import json
import asyncio
import httpx

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

from langchain_openai import AzureChatOpenAI
import requests
import json
import os
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI


# -----------------------------
# Token helper
# -----------------------------
def get_kong_token(kong_endpoint_url, client_id, client_secret):
    """
    Acquires an access token from the Kong endpoint using client credentials.
    """
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
        'scope': 'openid email profile'
    }
    try:
        response = requests.post(kong_endpoint_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        return None

# -----------------------------
# LLM helper
# -----------------------------
def create_llm_client():
    bearer_access_token = get_kong_token(
        kong_endpoint_url=kong_endpoint_url,
        client_id=kong_client_id,
        client_secret=kong_client_secret
    )
    if not bearer_access_token:
        return None

    llm = AzureChatOpenAI(
        api_version=api_version,
        azure_endpoint=kong_base_url,
        azure_ad_token=bearer_access_token,
        model=api_deployment_name,
        temperature=0.00001
    )
    return llm

async def generate_summary(collector_explanations, predicted_class):

    llm = create_llm_client()

    prompt = f"""
    You are a business risk analyst.

    A customer has been classified into the following risk category:
    Predicted Class: {predicted_class}

    Based on the collector explanation below, generate a concise business-friendly reason
    explaining why the customer was classified as this predicted class.

    Collector Explanation:
    {collector_explanations}

    Rules:
    - Keep the response to exactly 10 words.
    - Use simple business-friendly language.
    - Clearly justify the predicted class using the explanation provided.
    - Do not include bullet points or extra text.
    """

    response = await llm.ainvoke([
        {"role": "user", "content": prompt}
    ])

    return response.content.strip()


# -----------------------------
# Sync wrapper for Spark UDF
# -----------------------------
def summarize_explanation(collector_explanations, predicted_class):

    if collector_explanations is None or predicted_class is None:
        return None

    try:
        return asyncio.run(
            generate_summary(
                collector_explanations,
                predicted_class
            )
        )
    except Exception as e:
        return f"ERROR: {str(e)}"


# -----------------------------
# Register UDF
# -----------------------------
summary_udf = udf(summarize_explanation, StringType())


# -----------------------------
# Read dataframe
# -----------------------------
df = spark.table("f_erp_glide_o2c_12.collection_ml_customer")


# -----------------------------
# Apply LLM row by row
# -----------------------------
df_final = df.withColumn(
    "llm_summary",
    summary_udf(
        col("collector_explanations"),
        col("predicted_class")
    )
)


display(
    df_final.select(
        "collector_explanations",
        "predicted_class",
        "llm_summary"
    )
)
