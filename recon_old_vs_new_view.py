# Databricks notebook source
# MAGIC %md
# MAGIC # Reconciliation — Old Multi-Table Pipeline vs Unified View
# MAGIC
# MAGIC Validates that the new `vw_invoice_unified` produces the same
# MAGIC features as the original BSAD + BSID + MHND + UDM_P2P_ATTR + KNA1
# MAGIC pipeline.
# MAGIC
# MAGIC Run both, compare:
# MAGIC   - row counts
# MAGIC   - customer counts
# MAGIC   - total outstanding / amount sums
# MAGIC   - per-feature mean / sum / null counts
# MAGIC   - per-customer deltas (inner join on customer_id + snapshot_date)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
SAMPLE_SNAPSHOT = "2026-03-25"   # single snapshot for clean comparison
COUNTRY = "FR"
TOLERANCE_PCT = 0.01             # 1% = call it a match

# =========================================================
# LOAD BOTH PIPELINES
# =========================================================
# Old pipeline = from collection_risk_model.py (multi-table joins)
# New pipeline = from pipeline_unified_view.py (single view)
#
# Both must be importable in the same notebook session, OR run each
# pipeline separately and write to delta, then read back here.

# Option A — both in scope:
#   from collection_risk_model     import build_training_dataset as build_old
#   from pipeline_unified_view     import build_training_dataset as build_new

# Option B — read pre-computed deltas:
#   old_df = spark.read.format("delta").load("/tmp/recon/old_training_df")
#   new_df = spark.read.format("delta").load("/tmp/recon/new_training_df")

# Demo placeholders — replace with real loads
old_df = spark.table("scratch.training_df_old")
new_df = spark.table("scratch.training_df_new")


# =========================================================
# 1. FILTER TO SINGLE SNAPSHOT + COUNTRY
# =========================================================
def slice_snapshot(df):
    return df.filter(
        (F.col("snapshot_date") == F.lit(SAMPLE_SNAPSHOT))
        & (F.col("country") == COUNTRY)
    )

old = slice_snapshot(old_df).cache()
new = slice_snapshot(new_df).cache()


# =========================================================
# 2. ROW + CUSTOMER COUNTS
# =========================================================
def basic_counts(df, label):
    return {
        "label":     label,
        "rows":      df.count(),
        "customers": df.select("customer_id").distinct().count(),
        "snapshots": df.select("snapshot_date").distinct().count(),
    }

counts_df = pd.DataFrame([basic_counts(old, "OLD"), basic_counts(new, "NEW")])
print("\n=== BASIC COUNTS ===")
print(counts_df.to_string(index=False))


# =========================================================
# 3. COLUMN-LEVEL AGGREGATE COMPARISON
# =========================================================
# For each numeric column shared by both, compare:
#   sum, mean, min, max, null_count

def numeric_cols(df):
    return [
        f.name for f in df.schema.fields
        if str(f.dataType).startswith(("Double", "Decimal", "Long", "Integer"))
    ]

shared_cols = sorted(set(numeric_cols(old)) & set(numeric_cols(new)))
print(f"\n=== SHARED NUMERIC COLUMNS: {len(shared_cols)} ===")

def feature_stats(df, cols):
    aggs = []
    for c in cols:
        aggs.extend([
            F.sum(F.col(c).cast("double")).alias(f"{c}__sum"),
            F.avg(F.col(c).cast("double")).alias(f"{c}__mean"),
            F.sum(F.col(c).isNull().cast("int")).alias(f"{c}__nulls"),
        ])
    return df.agg(*aggs).toPandas().T.reset_index()

stats_old = feature_stats(old, shared_cols)
stats_old.columns = ["metric", "old_value"]

stats_new = feature_stats(new, shared_cols)
stats_new.columns = ["metric", "new_value"]

stats = stats_old.merge(stats_new, on="metric")
stats["old_value"] = pd.to_numeric(stats["old_value"], errors="coerce")
stats["new_value"] = pd.to_numeric(stats["new_value"], errors="coerce")
stats["abs_diff"] = (stats["new_value"] - stats["old_value"]).abs()
stats["pct_diff"] = stats["abs_diff"] / stats["old_value"].abs().replace(0, 1e-9)
stats["match"]    = stats["pct_diff"] <= TOLERANCE_PCT

print("\n=== AGGREGATE DIFFS (top 30 mismatches) ===")
mismatches = stats[~stats["match"]].sort_values("pct_diff", ascending=False)
print(mismatches.head(30).to_string(index=False))
print(f"\nTotal metrics: {len(stats)} | Mismatched: {(~stats['match']).sum()}")


# =========================================================
# 4. PER-CUSTOMER DELTAS (inner join)
# =========================================================
key = ["customer_id", "snapshot_date"]

joined = old.alias("o").join(new.alias("n"), key, "inner")

# Pick most important features to dig into
DRILL_COLS = [
    "total_outstanding", "num_open_invoices",
    "max_dpd", "amt_90_plus", "on_time_ratio",
    "max_dunning_level", "broken_ratio",
    "total_promises", "collected_30d", "target",
]

delta_exprs = []
for c in DRILL_COLS:
    if c not in shared_cols:
        continue
    delta_exprs.append(
        (F.col(f"o.{c}").cast("double") - F.col(f"n.{c}").cast("double"))
        .alias(f"{c}__delta")
    )

per_cust = joined.select(*key, *delta_exprs)

print("\n=== PER-CUSTOMER DELTA SUMMARY ===")
summary_aggs = []
for c in DRILL_COLS:
    col = f"{c}__delta"
    if col not in per_cust.columns:
        continue
    summary_aggs.extend([
        F.avg(col).alias(f"{c}__avg_delta"),
        F.max(F.abs(F.col(col))).alias(f"{c}__max_abs_delta"),
        F.sum(F.when(F.abs(F.col(col)) > 0.01, 1).otherwise(0)).alias(f"{c}__mismatched_rows"),
    ])

summary_pdf = per_cust.agg(*summary_aggs).toPandas().T
summary_pdf.columns = ["value"]
print(summary_pdf.to_string())


# =========================================================
# 5. CUSTOMERS PRESENT IN ONE PIPELINE BUT NOT THE OTHER
# =========================================================
only_old = old.select(*key).subtract(new.select(*key))
only_new = new.select(*key).subtract(old.select(*key))

print("\n=== KEY-LEVEL DIFFS ===")
print(f"customers only in OLD: {only_old.count():,}")
print(f"customers only in NEW: {only_new.count():,}")

if only_old.count() > 0:
    print("Sample OLD-only customers:")
    only_old.limit(10).show()

if only_new.count() > 0:
    print("Sample NEW-only customers:")
    only_new.limit(10).show()


# =========================================================
# 6. WORST OFFENDERS — rows with biggest deltas
# =========================================================
print("\n=== TOP 20 ROWS WITH BIGGEST total_outstanding DELTA ===")
per_cust.orderBy(F.abs(F.col("total_outstanding__delta")).desc()).limit(20).show()


# =========================================================
# 7. WRITE FULL DIFF FOR AUDIT
# =========================================================
per_cust.write.format("delta").mode("overwrite") \
    .saveAsTable("scratch.recon_old_vs_new_diffs")

stats_sdf = spark.createDataFrame(stats)
stats_sdf.write.format("delta").mode("overwrite") \
    .saveAsTable("scratch.recon_old_vs_new_stats")

print("\nFull diffs written to:")
print("  scratch.recon_old_vs_new_diffs   (per customer)")
print("  scratch.recon_old_vs_new_stats   (aggregate)")


# =========================================================
# VERDICT
# =========================================================
total_metrics = len(stats)
matched = stats["match"].sum()
match_pct = matched / total_metrics * 100

print(f"\n{'='*50}")
print(f"OVERALL: {matched}/{total_metrics} metrics match ({match_pct:.1f}%)")
if match_pct >= 99:
    print("VERDICT: pipelines equivalent within tolerance")
elif match_pct >= 95:
    print("VERDICT: minor drift — investigate top mismatches")
else:
    print("VERDICT: significant drift — check view logic before swap")
print(f"{'='*50}")
