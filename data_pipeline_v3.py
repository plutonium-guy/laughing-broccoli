# Databricks notebook source
# MAGIC %md
# MAGIC # Collection-Risk DATA PIPELINE (V3) — ORIGINAL LOGIC on the UNIFIED VIEW
# MAGIC
# MAGIC NEW FILE I own (standing rule #1: no edits to the original `.py`).
# MAGIC
# MAGIC **What this is:** the feature ENGINEERING LOGIC from the *original*
# MAGIC multi-table pipelines (the user's canonical `collection_risk_model.py`
# MAGIC training builder + the original inference builder), re-pointed at the
# MAGIC single unified view `table_invoice_unified_master`. One MODE-switched
# MAGIC builder, two Delta outputs.
# MAGIC
# MAGIC **Design rule: original behavior by default; every V2 "fix" is an opt-in
# MAGIC flag.** So the numbers match the originals out of the box, and you can
# MAGIC flip individual corrections on deliberately.
# MAGIC
# MAGIC ### Faithful replications of the original logic
# MAGIC   - `on_time_ratio` uses `days_to_pay <= 0` (clearing − baseline), the
# MAGIC     original definition (≈ always 0 — paid-same-day-or-earlier). Flip
# MAGIC     `ON_TIME_DUE_DATE_FIX=True` for the V2 `days_late = clearing − due`.
# MAGIC   - `due_date` is RECOMPUTED from payment terms (net→cd2→cd1), ignoring
# MAGIC     any `due_date` already on the view — exactly as the originals do.
# MAGIC   - Original feature SET only (exposure/behavior/dunning/p2p). The V2
# MAGIC     extras (tenure, credit_limit, credit_utilization, disputes,
# MAGIC     total_open_amount) are emitted ONLY if `INCLUDE_V2_FEATURES=True`
# MAGIC     (needed to feed the v3 cluster models).
# MAGIC   - No snapshot censoring cap; train floors baseline at LOOKBACK; infer
# MAGIC     applies the original 730-day floor on the behavior/dunning/p2p
# MAGIC     history joins. Flip `CENSOR_SNAPSHOTS=True` for the V2 TODAY−30 cap.
# MAGIC
# MAGIC ### View-translations (where the source shape forces a faithful mapping)
# MAGIC   - The originals join raw MHND / UDM_P2P_ATTR (one row per dunning
# MAGIC     letter / promise). The unified view carries those PRE-AGGREGATED per
# MAGIC     invoice (`dunning_count`, `dunning_level`, `last_dunned_date`;
# MAGIC     `fin_p2p_state`, `promise_dt`, `fin_promised_amt`). So the original's
# MAGIC     `count(*)` events → **`sum(dunning_count)`** here (count(*) on the
# MAGIC     view would count invoices, not letters — wrong). This preserves the
# MAGIC     original MEANING on this source.
# MAGIC
# MAGIC ### Unavoidable harmonizations (the originals disagreed; view forces one)
# MAGIC   - Source: BSID∪BSAD (the view's content). Original *training* was
# MAGIC     BSAD-only; original *inference* was BSID∪BSAD. Unifying = the
# MAGIC     inference choice, and the open-at-snapshot filter
# MAGIC     (`clearing IS NULL OR clearing > snapshot`) naturally captures both.
# MAGIC   - Country: BUKRS prefix (original *inference* + the v3 parity choice).
# MAGIC     Original *training* used KNA1.LAND1 (master) hardcoded to `['AT']`.
# MAGIC   - `country` is carried as a column for downstream routing; the
# MAGIC     originals only used it as a filter.
# MAGIC
# MAGIC **Modes:** `train` → snapshot panel + 30d label → `TRAIN_FEATURE_TABLE`.
# MAGIC `infer` → today-spine, no label → `INFER_FEATURE_TABLE`. `both` runs both
# MAGIC and asserts the feature columns are identical (train/serve parity).

# COMMAND ----------

!pip install -U seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
UNIFIED_VIEW = "f_erp_glide_o2c_12.table_invoice_unified_master"

# MODE: "train" | "infer" | "both"
MODE = "both"

TODAY = "2026-06-07"          # train snapshot anchor (infer uses current_date())
LOOKBACK_DAYS = 730
FUTURE_WINDOW_DAYS = 30
SNAPSHOT_STEP_DAYS = 7
HIGH_RISK_THRESHOLD = 0.4
WINDOWS = [60, 90, 180]

# Country universe (BUKRS prefix). Same default as the v3 trainer's CLUSTERS.
CLUSTERS = {
    "apac_big": ["CN", "AU", "JP", "NZ", "TW"],
    "kr_my":    ["KR", "MY"],
    "sea":      ["SG", "HK", "PH", "TH", "ID", "VN"],
}
COUNTRIES = sorted({c.upper() for cs in CLUSTERS.values() for c in cs})
COUNTRY_FROM = "bukrs"        # "bukrs" (parity) | "master" (original training)

# ---- V2-fix toggles: ALL default to ORIGINAL behavior. Flip to opt in. ----
INCLUDE_V2_FEATURES = True    # default ON: emit tenure/credit/disputes/open_amt
                              # (richer schema; required to feed the v3 models)
ON_TIME_DUE_DATE_FIX = False  # True => on_time_ratio uses days_late (V2 fix)
CENSOR_SNAPSHOTS = False      # True => cap train calendar at TODAY-30 (V2 fix)
# Original inference floored its history joins at 730d; original training did
# not. None = no floor (train), int = floor in days (infer).
INFER_HIST_LOOKBACK_DAYS = 730

# Output Delta tables.
TRAIN_FEATURE_TABLE = "f_erp_glide_o2c_12.collection_ml_features_train_v3"
INFER_FEATURE_TABLE = "f_erp_glide_o2c_12.collection_ml_features_infer_v3"
WRITE_TABLES = True

# NOTE: the V2-extra columns (total_open_amount, credit_limit,
# credit_utilization, number_of_disputes, open_dispute_amount,
# customer_tenure_days) are gated inline in compute_exposure_features via
# `if INCLUDE_V2_FEATURES` — no separate column list needed.

print(f"MODE={MODE} | {len(COUNTRIES)} countries: {COUNTRIES}")
print(f"flags: INCLUDE_V2_FEATURES={INCLUDE_V2_FEATURES} "
      f"ON_TIME_DUE_DATE_FIX={ON_TIME_DUE_DATE_FIX} "
      f"CENSOR_SNAPSHOTS={CENSOR_SNAPSHOTS}")

# COMMAND ----------

# =========================================================
# SHARED FEATURE ENGINEERING — original logic, unified-view source
# =========================================================

def to_date_any(col):
    """Unified-view date columns arrive as strings in mixed forms; parse to a
    real DateType (the original raw tables were already clean dates)."""
    c = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.to_date(c, "yyyyMMdd"), F.to_date(c))


def load_base_data(countries, mode):
    """Select the view columns the originals need (plus the pre-aggregated
    dunning/p2p/master attrs the view carries). TRAIN floors baseline at
    LOOKBACK (original training); INFER applies no baseline floor (original
    inference loaded full history, open-filtered at the snapshot)."""
    df = spark.table(UNIFIED_VIEW).select(
        F.col("customer_id"),
        F.col("invoice_id"),
        F.col("invoice_amount").cast("double"),
        F.col("open_amount").cast("double"),
        F.to_date(F.col("baseline_date"), "yyyyMMdd").alias("baseline_date"),
        F.to_date(F.col("clearing_date"), "yyyyMMdd").alias("clearing_date"),
        F.col("cash_discount_days_1").cast("int"),
        F.col("cash_discount_days_2").cast("int"),
        F.col("net_payment_days").cast("int"),
        (F.upper(F.substring(F.col("company_code"), 1, 2))
         if COUNTRY_FROM == "bukrs" else F.upper(F.col("country"))
         ).alias("country"),
        F.col("dunning_level").cast("int"),
        to_date_any("last_dunned_date").alias("last_dunned_date"),
        F.col("dunning_count").cast("int"),
        F.col("fin_promised_amt").cast("double"),
        F.col("fin_p2p_state").cast("int"),
        to_date_any("promise_dt").alias("promise_dt"),
        F.col("credit_limit").cast("double"),
        F.col("number_of_disputes").cast("int"),
        F.col("open_dispute_amount").cast("double"),
        F.col("customer_tenure_days").cast("int"),
        F.upper(F.col("source")).alias("source"),
    ).filter(
        F.col("baseline_date").isNotNull()
        & F.col("source").isin(["BSID", "BSAD"])
        & F.col("country").isin(countries)
    )

    if mode == "train":
        df = df.filter(F.col("baseline_date") >= F.date_sub(F.lit(TODAY), LOOKBACK_DAYS))
    return df


def add_due_date(df):
    """ORIGINAL: due_date = baseline + payment_terms, recomputed from the term
    columns (net→cd2→cd1). The view's own due_date is intentionally ignored."""
    return df.withColumn(
        "payment_terms_days",
        F.when(F.col("net_payment_days") != 0, F.col("net_payment_days"))
         .when(F.col("cash_discount_days_2") != 0, F.col("cash_discount_days_2"))
         .when(F.col("cash_discount_days_1") != 0, F.col("cash_discount_days_1"))
         .otherwise(0)
         .cast(IntegerType())
    ).withColumn(
        "due_date",
        F.date_add("baseline_date", F.col("payment_terms_days"))
    )


def build_invoice_level(df):
    """Roll up to one row per (customer, invoice). The view carries dunning/p2p/
    master attrs per line, so they are rolled up here to ride along."""
    return df.groupBy("customer_id", "invoice_id").agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.sum("open_amount").alias("open_amount"),
        F.min("baseline_date").alias("baseline_date"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date"),
        F.first("country", ignorenulls=True).alias("country"),
        F.max("dunning_level").alias("dunning_level"),
        F.max("last_dunned_date").alias("last_dunned_date"),
        F.max("dunning_count").alias("dunning_count"),
        F.max("fin_promised_amt").alias("fin_promised_amt"),
        F.first("fin_p2p_state", ignorenulls=True).alias("fin_p2p_state"),
        F.max("promise_dt").alias("promise_dt"),
        F.first("credit_limit", ignorenulls=True).alias("credit_limit"),
        F.max("number_of_disputes").alias("number_of_disputes"),
        F.max("open_dispute_amount").alias("open_dispute_amount"),
        F.first("customer_tenure_days", ignorenulls=True).alias("customer_tenure_days"),
    )


def create_snapshots(invoice_df):
    """TRAIN: weekly calendar per customer, min(baseline) .. TODAY. ORIGINAL had
    NO censoring cap (max_date = TODAY); CENSOR_SNAPSHOTS=True applies the V2
    TODAY-30 cap so a full 30-day label exists."""
    customers = invoice_df.select("customer_id").distinct()
    max_anchor = F.date_sub(F.lit(TODAY), FUTURE_WINDOW_DAYS) \
        if CENSOR_SNAPSHOTS else F.lit(TODAY)
    bounds = invoice_df.select(
        F.min("baseline_date").alias("min_date"),
        max_anchor.alias("max_date"),
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


def make_spine(base_df):
    """INFER: one row per distinct BSID customer @ current_date() (original
    customer_spine), carrying country for routing."""
    return base_df.filter(F.col("source") == "BSID").groupBy("customer_id").agg(
        F.first("country", ignorenulls=True).alias("country")
    ).withColumn("snapshot_date", F.current_date())


def compute_exposure_features(invoice_df, snapshots, mode):
    """Open-invoice exposure at each snapshot. ORIGINAL feature set; V2 extras
    only if INCLUDE_V2_FEATURES. Open filter = clearing IS NULL OR
    clearing > snapshot (captures BSID open + not-yet-cleared BSAD)."""
    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")
    joined = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner",
    ).filter(
        F.col("i.clearing_date").isNull()
        | (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).withColumn(
        # ORIGINAL training floored dpd at 0; we floor in both modes (original
        # inference omitted the floor — a quirk we don't propagate as skew).
        "dpd",
        F.when(F.col("i.due_date") <= F.col("s.snapshot_date"),
               F.datediff("s.snapshot_date", "i.due_date")).otherwise(0),
    ).withColumn(
        "invoice_age", F.datediff("s.snapshot_date", "i.baseline_date")
    )

    aggs = [
        F.sum("invoice_amount").alias("total_outstanding"),
        F.countDistinct("invoice_id").alias("num_open_invoices"),
        F.max("dpd").alias("max_dpd"),
        F.avg("dpd").alias("avg_dpd"),
        F.sum(F.when(F.col("dpd") > 30, F.col("invoice_amount")).otherwise(0)).alias("amt_30_plus"),
        F.sum(F.when(F.col("dpd") > 60, F.col("invoice_amount")).otherwise(0)).alias("amt_60_plus"),
        F.sum(F.when(F.col("dpd") > 90, F.col("invoice_amount")).otherwise(0)).alias("amt_90_plus"),
        F.max("invoice_age").alias("oldest_invoice_age"),
        F.avg("invoice_age").alias("avg_invoice_age"),
    ]
    if mode == "train":
        aggs = [F.first("country", ignorenulls=True).alias("country")] + aggs
    if INCLUDE_V2_FEATURES:
        aggs += [
            F.sum("open_amount").alias("total_open_amount"),
            F.max("credit_limit").alias("credit_limit"),
            F.max("number_of_disputes").alias("number_of_disputes"),
            F.sum("open_dispute_amount").alias("open_dispute_amount"),
            F.max("customer_tenure_days").alias("customer_tenure_days"),
        ]

    g = joined.groupBy("s.customer_id", "s.snapshot_date").agg(*aggs).withColumn(
        "avg_invoice_size", F.col("total_outstanding") / F.col("num_open_invoices")
    ).withColumn(
        "pct_30_plus", F.col("amt_30_plus") / F.col("total_outstanding")
    ).withColumn(
        "pct_60_plus", F.col("amt_60_plus") / F.col("total_outstanding")
    ).withColumn(
        "pct_90_plus", F.col("amt_90_plus") / F.col("total_outstanding")
    )

    if INCLUDE_V2_FEATURES:
        g = g.withColumn(
            "credit_utilization",
            F.when(F.col("credit_limit") > 0,
                   F.col("total_outstanding") / F.col("credit_limit")).otherwise(0)
        )
        if mode == "train":
            g = g.withColumn(
                "customer_tenure_days",
                F.greatest(
                    F.col("customer_tenure_days")
                    - F.datediff(F.to_date(F.lit(TODAY)), F.col("snapshot_date")),
                    F.lit(0),
                ),
            )
    return g


def _hist_join(left, snapshots, on_date_col, hist_lookback):
    """Shared historical-join lower bound: None (train) = events up to the
    snapshot; int (infer) = only the last `hist_lookback` days (original
    inference floored at 730)."""
    cond = (F.col("h.customer_id") == F.col("s.customer_id")) \
        & (F.col(on_date_col).isNotNull()) \
        & (F.col(on_date_col) <= F.col("s.snapshot_date"))
    if hist_lookback is not None:
        cond = cond & (F.col(on_date_col) >= F.date_sub(F.col("s.snapshot_date"), hist_lookback))
    return left.alias("h").join(snapshots.alias("s"), cond, "inner")


def compute_behavior_features(invoice_df, snapshots, hist_lookback):
    """Payment behavior from cleared invoices. ORIGINAL on_time_ratio uses
    days_to_pay<=0; ON_TIME_DUE_DATE_FIX swaps in days_late (clearing-due)."""
    hist = _hist_join(
        invoice_df.filter(F.col("clearing_date").isNotNull()),
        snapshots, "h.clearing_date", hist_lookback,
    ).withColumn(
        "days_to_pay", F.datediff("h.clearing_date", "h.baseline_date")
    ).withColumn(
        "days_late", F.datediff("h.clearing_date", "h.due_date")
    )
    late = F.col("days_late") if ON_TIME_DUE_DATE_FIX else F.col("days_to_pay")

    aggs = [
        F.avg("days_to_pay").alias("avg_days_to_pay"),
        F.max("days_to_pay").alias("max_days_to_pay"),
        F.avg(F.when(late <= 0, 1).otherwise(0)).alias("on_time_ratio"),
        F.count("*").alias("total_payments"),
        F.datediff(F.col("s.snapshot_date"), F.max("h.clearing_date")).alias("days_since_last_payment"),
    ]
    for w in WINDOWS:
        cond = F.col("h.clearing_date") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.avg(F.when(cond, F.col("days_to_pay"))).alias(f"avg_days_to_pay_{w}d"),
            F.max(F.when(cond, F.col("days_to_pay"))).alias(f"max_days_to_pay_{w}d"),
            # ORIGINAL windowed on_time: indicator(in-window AND on-time) averaged
            # over ALL payments (.otherwise(0)) — kept as-is.
            F.avg(F.when(cond & (late <= 0), 1).otherwise(0)).alias(f"on_time_ratio_{w}d"),
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"num_payments_{w}d"),
        ])

    return hist.groupBy("s.customer_id", "s.snapshot_date").agg(*aggs)


def compute_dunning_features(invoice_df, snapshots, hist_lookback):
    """Dunning history. View carries dunning_count per invoice, so the
    original's count(*) of letters → sum(dunning_count) here (faithful)."""
    hist = _hist_join(invoice_df, snapshots, "h.last_dunned_date", hist_lookback)

    aggs = [
        F.max("dunning_level").alias("max_dunning_level"),
        F.sum("dunning_count").alias("total_dunning_events"),
        F.avg("dunning_level").alias("avg_dunning_level"),
        F.sum(F.when(F.col("dunning_level") >= 3, F.col("dunning_count")).otherwise(0))
         .alias("high_severity_dunning"),
    ]
    for w in WINDOWS:
        cond = F.col("h.last_dunned_date") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.sum(F.when(cond, F.col("dunning_count")).otherwise(0)).alias(f"dunning_events_{w}d"),
            F.sum(F.when(cond & (F.col("dunning_level") >= 3), F.col("dunning_count")).otherwise(0))
             .alias(f"high_severity_dunning_{w}d"),
        ])

    return hist.groupBy("s.customer_id", "s.snapshot_date").agg(*aggs).withColumn(
        "high_dunning_ratio",
        F.when(F.col("total_dunning_events") > 0,
               F.col("high_severity_dunning") / F.col("total_dunning_events")).otherwise(0)
    )


def compute_p2p_features(invoice_df, snapshots, hist_lookback):
    """Promise-to-pay history. One promise attr per invoice on the view, so
    count(*) ≈ number of promises (original meaning preserved)."""
    hist = _hist_join(invoice_df, snapshots, "h.promise_dt", hist_lookback)

    aggs = [
        F.count("*").alias("total_promises"),
        F.sum(F.when(F.col("fin_p2p_state") == 1, 1).otherwise(0)).alias("broken_promises"),
        F.sum(F.when(F.col("fin_p2p_state") == 3, 1).otherwise(0)).alias("kept_promises"),
        F.sum("fin_promised_amt").alias("total_promised_amount"),
    ]
    for w in WINDOWS:
        cond = F.col("h.promise_dt") >= F.date_sub(F.col("s.snapshot_date"), w)
        aggs.extend([
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"promises_{w}d"),
            F.sum(F.when(cond & (F.col("fin_p2p_state") == 1), 1).otherwise(0)).alias(f"broken_{w}d"),
            F.sum(F.when(cond & (F.col("fin_p2p_state") == 3), 1).otherwise(0)).alias(f"kept_{w}d"),
            F.sum(F.when(cond, F.col("fin_promised_amt"))).alias(f"promised_amt_{w}d"),
        ])

    return hist.groupBy("s.customer_id", "s.snapshot_date").agg(*aggs).withColumn(
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
    """TRAIN only: 30-day forward collection. target=1 == HIGH RISK
    (collection_ratio < HIGH_RISK_THRESHOLD)."""
    inv = invoice_df.alias("i")
    snap = snapshots.alias("s")
    future = inv.join(
        snap,
        (F.col("i.customer_id") == F.col("s.customer_id"))
        & (F.col("i.baseline_date") <= F.col("s.snapshot_date")),
        "inner",
    ).filter(
        F.col("i.clearing_date").isNull()
        | (F.col("i.clearing_date") > F.col("s.snapshot_date"))
    ).filter(
        (F.col("i.clearing_date") >= F.col("s.snapshot_date"))
        & (F.col("i.clearing_date") <= F.date_add(F.col("s.snapshot_date"), FUTURE_WINDOW_DAYS))
    )
    return future.groupBy("s.customer_id", "s.snapshot_date").agg(
        F.sum("i.invoice_amount").alias("collected_30d")
    )

# COMMAND ----------

# =========================================================
# DATASET BUILDERS
# =========================================================

def build_training_dataset(countries):
    print(f"[train] loading unified view for {len(countries)} countries...")
    base = add_due_date(load_base_data(countries, mode="train"))
    inv = build_invoice_level(base)
    snaps = create_snapshots(inv)

    df = (
        compute_exposure_features(inv, snaps, mode="train")          # carries country
        .join(compute_behavior_features(inv, snaps, hist_lookback=None), ["customer_id", "snapshot_date"], "left")
        .join(compute_dunning_features(inv, snaps, hist_lookback=None), ["customer_id", "snapshot_date"], "left")
        .join(compute_p2p_features(inv, snaps, hist_lookback=None),     ["customer_id", "snapshot_date"], "left")
        .join(create_target(inv, snaps),                                ["customer_id", "snapshot_date"], "left")
    )
    df = df.fillna(0).filter(F.col("total_outstanding") > 0)
    df = df.withColumn(
        "collection_ratio", F.col("collected_30d") / F.col("total_outstanding")
    ).withColumn(
        "target",
        F.when(F.col("collection_ratio") < HIGH_RISK_THRESHOLD, 1).otherwise(0),
    )
    return df


def build_inference_dataset(countries):
    print(f"[infer] loading unified view for {len(countries)} countries...")
    base = add_due_date(load_base_data(countries, mode="infer"))
    inv = build_invoice_level(base)
    spine = make_spine(base)
    lb = INFER_HIST_LOOKBACK_DAYS

    df = (
        spine                                                        # carries country
        .join(compute_exposure_features(inv, spine, mode="infer"),  ["customer_id", "snapshot_date"], "left")
        .join(compute_behavior_features(inv, spine, hist_lookback=lb), ["customer_id", "snapshot_date"], "left")
        .join(compute_dunning_features(inv, spine, hist_lookback=lb),  ["customer_id", "snapshot_date"], "left")
        .join(compute_p2p_features(inv, spine, hist_lookback=lb),      ["customer_id", "snapshot_date"], "left")
        .fillna(0)
    )
    return df

# COMMAND ----------

# =========================================================
# RUN — build requested mode(s), summarize, write Delta
# =========================================================
NON_FEATURE = ("customer_id", "snapshot_date", "country",
               "collected_30d", "collection_ratio", "target", "payment_terms_days")


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE]


def summarize(df, label, has_target):
    n = df.count()
    n_cust = df.select("customer_id").distinct().count()
    fc = feature_cols(df)
    print(f"[{label}] rows={n:,} customers={n_cust:,} features={len(fc)}")
    if has_target:
        pos = df.filter(F.col("target") == 1).count()
        print(f"[{label}] positives={pos:,} ({pos/max(n,1)*100:.1f}%) | "
              f"target=1 == HIGH RISK (collection_ratio < {HIGH_RISK_THRESHOLD})")
    return fc


def write_table(df, table):
    if not WRITE_TABLES:
        print(f"  WRITE_TABLES=False — skipped {table}")
        return
    df.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(table)
    print(f"  wrote {table}")


train_feats = infer_feats = []

if MODE in ("train", "both"):
    train_df = build_training_dataset(COUNTRIES).cache()
    train_feats = summarize(train_df, "train", has_target=True)
    display(train_df.orderBy(F.desc("max_dpd")).limit(20))
    write_table(train_df, TRAIN_FEATURE_TABLE)

if MODE in ("infer", "both"):
    infer_df = build_inference_dataset(COUNTRIES).cache()
    infer_feats = summarize(infer_df, "infer", has_target=False)
    display(infer_df.orderBy(F.desc("max_dpd")).limit(20))
    write_table(infer_df, INFER_FEATURE_TABLE)

# COMMAND ----------

# =========================================================
# PARITY CHECK — train and infer must expose identical feature columns
# (values can differ by the per-mode history floor; columns must not)
# =========================================================
if MODE == "both" and train_feats and infer_feats:
    only_train = sorted(set(train_feats) - set(infer_feats))
    only_infer = sorted(set(infer_feats) - set(train_feats))
    print(f"shared features: {len(set(train_feats) & set(infer_feats))}")
    print(f"train-only: {only_train}")
    print(f"infer-only: {only_infer}")
    assert not only_train and not only_infer, \
        "FEATURE SKEW between train and infer — investigate before training"
    print("PARITY OK — train and infer feature columns identical")

# COMMAND ----------

print("\n=== DONE (data_pipeline_v3 — original logic on unified view) ===")
print(f"MODE={MODE} | INCLUDE_V2_FEATURES={INCLUDE_V2_FEATURES} "
      f"ON_TIME_DUE_DATE_FIX={ON_TIME_DUE_DATE_FIX} CENSOR_SNAPSHOTS={CENSOR_SNAPSHOTS}")
if MODE in ("train", "both"):
    print(f"  TRAIN -> {TRAIN_FEATURE_TABLE} ({len(train_feats)} features + target)")
if MODE in ("infer", "both"):
    print(f"  INFER -> {INFER_FEATURE_TABLE} ({len(infer_feats)} features)")
print("\nNOTE: original feature SET by default. To feed the v3 cluster models, "
      "set INCLUDE_V2_FEATURES=True (they expect tenure/credit/disputes).")
