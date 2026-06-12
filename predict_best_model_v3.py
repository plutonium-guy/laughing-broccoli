# Databricks notebook source
# MAGIC %md
# MAGIC # Inference — Global Model (V3)
# MAGIC
# MAGIC NEW FILE (standing rule #1: no edits to existing `.py`). Scores the
# MAGIC single global model trained by `train_best_model_v3.py`.
# MAGIC
# MAGIC **Flow**
# MAGIC   1. Load registered `collection_risk_model_global_v3` + its operating
# MAGIC      threshold (auto-fetched from the training run's `chosen_threshold`
# MAGIC      param; `DEFAULT_THRESHOLD` fallback).
# MAGIC   2. Read `INFER_FEATURE_TABLE` (= `collection_ml_features_infer_v3`,
# MAGIC      written by `data_pipeline_v3.py` MODE=infer, INCLUDE_V2_FEATURES=True).
# MAGIC   3. Rebuild the SAME country + cluster one-hots as training, then ALIGN
# MAGIC      to `booster.feature_names` (any feature the booster expects but the
# MAGIC      batch lacks → 0; exact column order). This is what keeps train/serve
# MAGIC      identical even if today's batch is missing a country.
# MAGIC   4. Score → `risk_score = P(target=1)`; `binary_pred = score ≥ threshold`
# MAGIC      (1 = HIGH RISK); Low/Medium/High band.
# MAGIC   5. SHAP per customer (native `pred_contribs`) → top-3 drivers +
# MAGIC      business-English `collector_explanations` (same wording as
# MAGIC      `predict_all_clusters_v2.py`; country_*/cluster_* one-hots excluded).
# MAGIC   6. `risk_band` = per-country score tertiles. Write Delta.
# MAGIC
# MAGIC Conventions (same as training): higher score = worse; `target=1 == HIGH RISK`.

# COMMAND ----------

!pip install -U xgboost mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
from mlflow.tracking import MlflowClient

from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, DoubleType

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
INFER_FEATURE_TABLE = "f_erp_glide_o2c_12.collection_ml_features_infer_v3"
MODEL_NAME = "collection_risk_model_global_v3"
MODEL_URI = f"models:/{MODEL_NAME}/latest"
OUTPUT_TABLE = "f_erp_glide_o2c_12.collection_ml_customer_global_v3"

# Operating threshold: auto-fetched from the training run's logged
# `chosen_threshold` (max precision @ recall>=0.80). Used if the lookup fails.
DEFAULT_THRESHOLD = 0.5

# Same cluster map as the trainer — cluster one-hots must match training.
CLUSTERS = {
    "apac_big": ["CN", "AU", "JP", "NZ", "TW"],
    "kr_my":    ["KR", "MY"],
    "sea":      ["SG", "HK", "PH", "TH", "ID", "VN"],
}
CLUSTER_OF = {c.upper(): name for name, cs in CLUSTERS.items() for c in cs}

TOP_K_DRIVERS = 3
# Band around threshold t: Low < t <= Medium < t+MEDIUM_BAND_WIDTH*(1-t) <= High
MEDIUM_BAND_WIDTH = 0.5

ID_COLS = ["customer_id", "country", "cluster", "snapshot_date"]

# COMMAND ----------

# =========================================================
# LOAD MODEL + THRESHOLD
# =========================================================
model = mlflow.sklearn.load_model(MODEL_URI)
feature_cols = list(model.get_booster().feature_names)
print(f"loaded {MODEL_URI} ({len(feature_cols)} features)")


def fetch_threshold():
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        t = run.data.params.get("chosen_threshold")
        if t is not None:
            print(f"threshold {float(t):.4f} (from run {latest.run_id[:8]}, v{latest.version})")
            return float(t)
    except Exception as e:
        print(f"threshold lookup failed ({e}) — using DEFAULT_THRESHOLD")
    return DEFAULT_THRESHOLD


THRESHOLD = fetch_threshold()

# COMMAND ----------

# =========================================================
# LOAD FEATURES + ALIGN (one-hots must match training)
# =========================================================

def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


pdf = spark_to_pandas_safe(spark.table(INFER_FEATURE_TABLE))
pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"])

# cluster column (carried to output + used for one-hots)
pdf["cluster"] = pdf["country"].map(CLUSTER_OF).fillna("other") \
    if "country" in pdf.columns else "other"

# Rebuild the SAME one-hots as training (only those present in the batch;
# align() then fills any training one-hot the batch lacks).
for c in sorted(pdf["country"].dropna().unique()):
    pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
for cl in sorted(pdf["cluster"].unique()):
    pdf[f"cluster_{cl}"] = (pdf["cluster"] == cl).astype(int)

print(f"rows={len(pdf):,} | countries={pdf['country'].nunique()} "
      f"| clusters={sorted(pdf['cluster'].unique())}")


def align_features(frame, cols):
    """Exact booster column set + order; fill anything missing with 0
    (e.g. a country seen in training but absent from today's batch)."""
    out = frame.copy()
    missing = [f for f in cols if f not in out.columns]
    for f in missing:
        out[f] = 0.0
    if missing:
        print(f"  filled {len(missing)} missing features with 0: {missing[:8]}"
              f"{'...' if len(missing) > 8 else ''}")
    extra = [c for c in out.columns if c.startswith(("country_", "cluster_"))
             and c not in cols]
    if extra:
        print(f"  NOTE: {len(extra)} one-hots not in the model (scored as 0): {extra}")
    X = out[cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0).astype("float64")


X = align_features(pdf, feature_cols)

# COMMAND ----------

# =========================================================
# SCORE
# =========================================================
pdf["risk_score"] = model.predict_proba(X)[:, 1]
pdf["threshold_used"] = THRESHOLD
pdf["binary_pred"] = (pdf["risk_score"] >= THRESHOLD).astype(int)


def band_of(prob, t):
    mid = t + MEDIUM_BAND_WIDTH * (1.0 - t)
    return np.where(prob < t, "Low", np.where(prob < mid, "Medium", "High"))


pdf["predicted_class"] = band_of(pdf["risk_score"].values, THRESHOLD)

n_high = int(pdf["binary_pred"].sum())
print(f"scored {len(pdf):,} customers | {n_high:,} HIGH RISK "
      f"({n_high/max(len(pdf),1)*100:.1f}%) @ threshold={THRESHOLD:.3f}")

# COMMAND ----------

# =========================================================
# SHAP — top drivers + business explanations
# =========================================================

def shap_matrix(model, X):
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    return contribs[:, :-1]                       # drop bias term


def top_drivers(shap_vals, cols, k=TOP_K_DRIVERS):
    order = np.argsort(-np.abs(shap_vals), axis=1)[:, :k]
    cols = np.array(cols)
    signs = np.take_along_axis(shap_vals, order, axis=1) >= 0
    drivers = [
        "; ".join(f"{cols[j]}({'+' if s else '-'})" for j, s in zip(row, sign_row))
        for row, sign_row in zip(order, signs)
    ]
    return drivers, [cols[row[0]] for row in order]


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


def build_business_explanations(shap_row, feature_row, cols, top_n=5):
    """Top-n |SHAP| drivers -> sentences. country_*/cluster_* one-hots are
    routing/grouping artifacts, not behavior — excluded from the narrative."""
    cdf = pd.DataFrame({"feature": cols, "shap_value": shap_row,
                        "feature_value": feature_row})
    cdf = cdf[~cdf["feature"].str.startswith(("country_", "cluster_"))]
    cdf["abs_shap"] = cdf["shap_value"].abs()
    cdf = cdf.sort_values("abs_shap", ascending=False).head(top_n)
    return [explain_one(r["feature"], r["feature_value"], r["shap_value"])
            for _, r in cdf.iterrows()]


shap_vals = shap_matrix(model, X)
drivers, top1 = top_drivers(shap_vals, X.columns)
pdf["top_risk_drivers"] = drivers
pdf["top_risk_driver"] = top1
pdf["collector_explanations"] = [
    " | ".join(build_business_explanations(shap_vals[i], X.iloc[i].values,
                                           list(X.columns), top_n=5))
    for i in range(len(pdf))
]
pdf["model_features"] = ", ".join(feature_cols)
pdf["model_name"] = MODEL_URI

# COMMAND ----------

# =========================================================
# risk_band — per-country score tertiles (relative ranking within country)
# =========================================================

def _risk_band(scores):
    try:
        q = pd.qcut(scores, q=3, duplicates="drop")
        labels = ["Low", "Medium", "High"][:len(q.cat.categories)]
        return pd.qcut(scores, q=len(q.cat.categories), labels=labels,
                       duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("Low", index=scores.index)


pdf["risk_band"] = pdf.groupby("country")["risk_score"].transform(_risk_band)

# COMMAND ----------

# =========================================================
# ASSEMBLE + WRITE
# =========================================================
PRED_COLS = ["risk_score", "binary_pred", "predicted_class", "risk_band",
             "threshold_used", "top_risk_driver", "top_risk_drivers",
             "collector_explanations", "model_features", "model_name"]
# raw feature columns from the infer table (exclude ids + the one-hots we added)
FEATURE_OUT_COLS = [c for c in pdf.columns
                    if c not in ID_COLS + PRED_COLS
                    and not c.startswith(("country_", "cluster_"))]

predictions = pdf[ID_COLS + FEATURE_OUT_COLS + PRED_COLS]
print(f"output: {len(predictions):,} rows x {len(predictions.columns)} columns")
print(predictions.groupby(["cluster", "predicted_class"]).size())

# COMMAND ----------

display(spark.createDataFrame(predictions).orderBy(F.desc("risk_score")).limit(50))

summary = predictions.groupby("country").agg(
    customers=("customer_id", "size"),
    high_risk=("binary_pred", "sum"),
    avg_score=("risk_score", "mean"),
)
summary["high_risk_pct"] = (summary["high_risk"] / summary["customers"] * 100).round(1)
display(summary.reset_index())

# COMMAND ----------

out_sdf = spark.createDataFrame(predictions).withColumn("refresh_date", F.current_date())
out_sdf.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
print(f"Wrote {out_sdf.count():,} rows to {OUTPUT_TABLE}")
