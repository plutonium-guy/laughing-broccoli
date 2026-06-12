# Databricks notebook source
# MAGIC %md
# MAGIC # Collection Risk Pipeline — `table_invoice_unified_master` Version
# MAGIC
# MAGIC Reads ONE pre-joined master table (`table_invoice_unified_master`) instead of
# MAGIC the multi-table SAP joins (BSAD / BSID / MHND / UDM_P2P_ATTR / KNA1).
# MAGIC
# MAGIC Master grain = (customer_id, invoice_id, line_item) with denormalized
# MAGIC dunning + P2P + dispute + credit + customer-master columns already attached.
# MAGIC
# MAGIC Feature parity with the ORIGINAL pipeline (`collection_risk_model.py`):
# MAGIC   exposure + behavior + dunning + p2p + target — identical column names.
# MAGIC
# MAGIC Extra columns carried from the master table (per user request):
# MAGIC   credit_limit, number_of_disputes, open_dispute_amount,
# MAGIC   customer_tenure_days, risk_class, credit_group  (+ derived credit_utilization)
# MAGIC
# MAGIC NOTE: `is_actioned` is NOT available on this master table — column dropped.
# MAGIC
# MAGIC Two entry points:
# MAGIC   build_training_dataset()  -> weekly snapshots + 30d forward target (model training)
# MAGIC   build_inference_dataset() -> single snapshot = TODAY (production scoring puller)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =========================================================
# CONFIG
# =========================================================
MASTER_TABLE = "f_erp_glide_o2c_12.table_invoice_unified_master"   # <-- source master table

TODAY = "2026-03-25"
LOOKBACK_DAYS = 730
FUTURE_WINDOW_DAYS = 30
SNAPSHOT_STEP_DAYS = 7
HIGH_RISK_THRESHOLD = 0.4
WINDOWS = [60, 90, 180]
COUNTRIES = ["FR"]


# =========================================================
# 1. LOAD MASTER TABLE
# =========================================================
def load_base_data():
    """
    Single source — master table already has all joins done.
    Keep only line items with a valid baseline_date and a known source
    ('BSID' = open, 'BSAD' = cleared).
    `is_actioned` intentionally NOT selected — not available on this table.
    """
    df = spark.table(MASTER_TABLE).select(
        F.col("customer_id"),
        F.col("invoice_id"),
        F.col("line_item"),
        F.col("company_code"),
        F.col("currency"),
        F.col("invoice_amount").cast("double"),
        F.col("open_amount").cast("double"),
        F.col("baseline_date").cast("date"),
        F.col("document_entry_date").cast("date"),
        F.col("clearing_date").cast("date"),
        F.col("cash_discount_days_1").cast("int"),
        F.col("cash_discount_days_2").cast("int"),
        F.col("net_payment_days").cast("int"),
        F.col("payment_terms"),
        F.col("due_date").cast("date"),
        F.col("days_past_due").cast("int"),
        F.col("invoice_status"),
        F.col("country"),
        F.col("region"),
        F.col("customer_tenure_days").cast("int"),
        # dunning (denormalized)
        F.col("dunning_level").cast("int"),
        F.col("last_dunned_date").cast("date"),
        F.col("dunning_count").cast("int"),
        # P2P (denormalized)
        F.col("fin_promised_amt").cast("double"),
        F.col("fin_p2p_state").cast("int"),
        F.col("promise_create_dt").cast("date"),
        F.col("promise_dt").cast("date"),
        # credit
        F.col("risk_class"),
        F.col("credit_group"),
        F.col("credit_limit").cast("double"),
        # disputes
        F.col("dispute_create_date").cast("date"),
        F.col("number_of_disputes").cast("int"),
        F.col("open_dispute_amount").cast("double"),
        # ops
        F.col("payment_method"),
        F.upper(F.col("source")).alias("source"),   # 'BSID' = open, 'BSAD' = cleared
    ).filter(
        F.col("baseline_date").isNotNull()
        & (F.col("baseline_date") >= F.date_sub(F.lit(TODAY), LOOKBACK_DAYS))
        & F.col("source").isin(["BSID", "BSAD"])
    )

    return df


# =========================================================
# 2. DUE DATE — already on the table, but derive if null
# =========================================================
def ensure_due_date(df):
    """Master has due_date — if any row null, fall back to payment-terms tiers
    (matches original: net -> cash_disc_2 -> cash_disc_1 -> 0)."""
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


# =========================================================
# 3. COUNTRY FILTER (column already on table — no join needed)
# =========================================================
def filter_by_countries(df, countries):
    if not countries:
        return df
    return df.filter(F.col("country").isin([c.upper() for c in countries]))


# =========================================================
# 4. INVOICE LEVEL — collapse line items
# =========================================================
def build_invoice_level(df):
    """
    Roll up line_item -> invoice. Keep first values of denormalized
    dunning/p2p/credit/dispute columns (same across all line items of an invoice).
    Invoice is considered cleared if ANY line item is BSAD with clearing_date set.
    """
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


# =========================================================
# 5. SNAPSHOTS (training only — weekly, dedup monthly)
# =========================================================
def create_snapshots(invoice_df):
    customers = invoice_df.select("customer_id").distinct()

    bounds = invoice_df.select(
        F.min("baseline_date").alias("min_date"),
        F.lit(TODAY).alias("max_date"),
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


# =========================================================
# 6. EXPOSURE FEATURES
# =========================================================
def compute_exposure_features(invoice_df, snapshots):
    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")

    # OPEN AT SNAPSHOT =
    #   never cleared (is_cleared=0)
    #   OR cleared AFTER snapshot date
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
    )


# =========================================================
# 7. BEHAVIOR FEATURES (cleared invoices up to snapshot)
# =========================================================
def compute_behavior_features(invoice_df, snapshots):
    """Cleared invoices only (is_cleared=1) — payment behavior signal.
    Matches original compute_behavior_features feature set exactly."""
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


# =========================================================
# 8. DUNNING FEATURES (denormalized on table — aggregate to customer)
# =========================================================
def compute_dunning_features(invoice_df, snapshots):
    """
    Master already has dunning_level + dunning_count per invoice.
    Aggregate per (customer, snapshot) using invoices that had dunning
    AND last_dunned_date <= snapshot.
    """
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


# =========================================================
# 9. P2P FEATURES (denormalized on table — aggregate to customer)
# =========================================================
def compute_p2p_features(invoice_df, snapshots):
    """
    Master has one promise per invoice (latest). Aggregate per
    (customer, snapshot) using promises whose promise_dt <= snapshot.
    """
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


# =========================================================
# 10. TARGET (forward 30d collection — no leakage)
# =========================================================
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


# =========================================================
# 11. TRAINING ORCHESTRATOR
# =========================================================
def build_training_dataset():
    print("Loading master table...")
    base = load_base_data()
    base = ensure_due_date(base)
    base = filter_by_countries(base, COUNTRIES)

    print("Rolling up to invoice level...")
    inv = build_invoice_level(base)

    print("Building snapshots...")
    snaps = create_snapshots(inv)

    print("Computing feature layers...")
    exposure = compute_exposure_features(inv, snaps)
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


# =========================================================
# 12. INFERENCE ORCHESTRATOR — snapshot = TODAY only
# =========================================================
def build_inference_dataset(snapshot_date=None):
    """
    Single-snapshot production scoring puller.
    Same feature engineering as training (no leakage by construction —
    only history up to snapshot_date is used), but ONE snapshot and NO target.

    snapshot_date: optional 'yyyy-MM-dd' string; defaults to TODAY config.
    """
    snap_date = snapshot_date or TODAY

    print(f"Loading master table for inference @ {snap_date}...")
    base = load_base_data()
    base = ensure_due_date(base)
    base = filter_by_countries(base, COUNTRIES)

    inv = build_invoice_level(base)

    # Spine = distinct customers with any invoice activity, snapshot = snap_date
    spine = inv.select("customer_id").distinct() \
        .withColumn("snapshot_date", F.lit(snap_date).cast("date"))

    exposure = compute_exposure_features(inv, spine)
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


# =========================================================
# RUN
# =========================================================
training_df = build_training_dataset()
display(training_df.limit(50))

# COMMAND ----------

inference_df = build_inference_dataset()
display(inference_df.limit(50))
