# Databricks notebook source
# MAGIC %md
# MAGIC # Inference — Cluster Models (V3)
# MAGIC
# MAGIC NEW FILE (standing rule #1: no edits to existing `.py`). Matches the
# MAGIC rewired `train_all_clusters_v3.py` (one CalibratedSeedEnsemble per
# MAGIC cluster + a routing table) and reads the SAME feature source as the
# MAGIC trainer — `data_pipeline_v3.py`'s infer table — so no inline feature
# MAGIC engineering and no train/serve skew.
# MAGIC
# MAGIC **Flow**
# MAGIC   1. Read the routing table (country → cluster model + per-country
# MAGIC      threshold) written by the trainer.
# MAGIC   2. Read `INFER_FEATURE_TABLE` (= `collection_ml_features_infer_v3`,
# MAGIC      `data_pipeline_v3.py` MODE=infer, INCLUDE_V2_FEATURES=True).
# MAGIC   3. Per cluster: load the registered CalibratedSeedEnsemble, rebuild the
# MAGIC      country one-hots it trained with, ALIGN to `booster.feature_names`
# MAGIC      (missing→0, exact order), score.
# MAGIC   4. Per-country threshold from routing → `binary_pred` (1 = HIGH RISK)
# MAGIC      + Low/Medium/High band.
# MAGIC   5. SHAP per customer → top-3 drivers + business `collector_explanations`
# MAGIC      (same wording as `predict_all_clusters_v2.py`; country_ one-hots
# MAGIC      excluded). `risk_band` = per-country score tertiles. Write Delta.
# MAGIC
# MAGIC Conventions: higher score = worse; `target=1 == HIGH RISK`. The model is
# MAGIC a `CalibratedSeedEnsemble` (calibrated K-seed) but duck-types the
# MAGIC XGBClassifier surface, so scoring/SHAP are unchanged.

# COMMAND ----------

!pip install -U xgboost mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, DoubleType

# CalibratedSeedEnsemble must be importable to unpickle the registered models
# (also shipped via code_paths, but importing here is belt-and-braces).
CODE_PATH_MODULE = "cluster_xgb_ensemble.py"
_MODULE_DIR = os.path.dirname(os.path.abspath(CODE_PATH_MODULE)) or "."
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
from cluster_xgb_ensemble import CalibratedSeedEnsemble  # noqa: F401 (unpickle)

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
INFER_FEATURE_TABLE = "f_erp_glide_o2c_12.collection_ml_features_infer_v3"
# Routing written by train_all_clusters_v3.py. Default = the staging table it
# always writes; point at the prod map if you ran with WRITE_PROD_ROUTING=True.
ROUTING_TABLE = "f_erp_glide_o2c_12.collection_ml_country_model_map_v3_staging"
OUTPUT_TABLE = "f_erp_glide_o2c_12.collection_ml_customer_clusters_v3"

TOP_K_DRIVERS = 3
# Band around threshold t: Low < t <= Medium < t+MEDIUM_BAND_WIDTH*(1-t) <= High
MEDIUM_BAND_WIDTH = 0.5

# COMMAND ----------

# =========================================================
# ROUTING MAP
# =========================================================
routing = spark.table(ROUTING_TABLE).toPandas()
assert len(routing), f"{ROUTING_TABLE} is empty — run train_all_clusters_v3.py first"
routing["country"] = routing["country"].str.upper()

ALL_COUNTRIES = sorted(routing["country"].unique().tolist())
MODEL_OF = {cl: f"models:/{name}/latest"
            for cl, name in zip(routing["cluster"], routing["model_name"])}
THRESHOLD_OF = dict(zip(routing["country"], routing["threshold"]))   # per-country
COUNTRIES_OF = routing.groupby("cluster")["country"].apply(sorted).to_dict()

print(f"Routing: {len(ALL_COUNTRIES)} countries -> {routing['cluster'].nunique()} clusters")
print(routing[["country", "cluster", "model_name", "threshold"]]
      .sort_values(["cluster", "country"]).to_string(index=False))

# COMMAND ----------

# =========================================================
# SCORING MACHINERY
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
    path = MODEL_OF[cluster]
    model = mlflow.sklearn.load_model(path)
    feature_cols = list(model.get_booster().feature_names)
    bundle = {"model": model, "feature_cols": feature_cols, "model_name": path}
    _MODEL_CACHE[cluster] = bundle
    print(f"[{cluster}] loaded {path} ({len(feature_cols)} features)")
    return bundle


def align_features(pdf, cluster, feature_cols):
    """Rebuild the cluster's country one-hots, fill any booster feature the
    batch lacks with 0, and return columns in the exact booster order."""
    out = pdf.copy()
    for c in COUNTRIES_OF[cluster]:
        out[f"country_{c}"] = (out["country"] == c).astype(int)
    missing = [f for f in feature_cols if f not in out.columns]
    for f in missing:
        out[f] = 0.0
    if missing:
        print(f"[{cluster}] filled {len(missing)} missing features with 0: {missing[:6]}"
              f"{'...' if len(missing) > 6 else ''}")
    X = out[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0).astype("float64")


def band_of(prob, t):
    mid = t + MEDIUM_BAND_WIDTH * (1.0 - t)
    return np.where(prob < t, "Low", np.where(prob < mid, "Medium", "High"))


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
    cdf = pd.DataFrame({"feature": cols, "shap_value": shap_row, "feature_value": feature_row})
    cdf = cdf[~cdf["feature"].str.startswith(("country_", "cluster_"))]
    cdf["abs_shap"] = cdf["shap_value"].abs()
    cdf = cdf.sort_values("abs_shap", ascending=False).head(top_n)
    return [explain_one(r["feature"], r["feature_value"], r["shap_value"])
            for _, r in cdf.iterrows()]


def score_cluster(cluster, feats_pdf):
    members = COUNTRIES_OF[cluster]
    pdf = feats_pdf[feats_pdf["country"].isin(members)].copy()
    if len(pdf) == 0:
        print(f"[{cluster}] no customers, skipping")
        return None

    bundle = load_cluster_model(cluster)
    X = align_features(pdf, cluster, bundle["feature_cols"])

    pdf["risk_score"] = bundle["model"].predict_proba(X)[:, 1]
    pdf["threshold_used"] = pdf["country"].map(THRESHOLD_OF)
    pdf["binary_pred"] = (pdf["risk_score"] >= pdf["threshold_used"]).astype(int)
    pdf["predicted_class"] = [band_of(np.array([p]), t)[0]
                              for p, t in zip(pdf["risk_score"], pdf["threshold_used"])]

    shap_vals = shap_matrix(bundle["model"], X)
    drivers, top1 = top_drivers(shap_vals, X.columns)
    pdf["top_risk_drivers"] = drivers
    pdf["top_risk_driver"] = top1
    pdf["collector_explanations"] = [
        " | ".join(build_business_explanations(shap_vals[i], X.iloc[i].values,
                                               list(X.columns), top_n=5))
        for i in range(len(pdf))
    ]
    pdf["cluster"] = cluster
    pdf["model_name"] = bundle["model_name"]
    pdf["model_features"] = ", ".join(bundle["feature_cols"])

    n_high = int(pdf["binary_pred"].sum())
    print(f"[{cluster}] scored {len(pdf):,} customers, {n_high:,} HIGH RISK "
          f"({n_high/max(len(pdf),1)*100:.1f}%)")
    return pdf

# COMMAND ----------

# =========================================================
# RUN — features once, score per cluster, union, write
# =========================================================
feats_pdf = spark_to_pandas_safe(spark.table(INFER_FEATURE_TABLE))
feats_pdf["snapshot_date"] = pd.to_datetime(feats_pdf["snapshot_date"])
print(f"loaded {len(feats_pdf):,} customers from {INFER_FEATURE_TABLE}")

ID_COLS = ["customer_id", "country", "cluster", "snapshot_date"]
PRED_COLS = ["risk_score", "binary_pred", "predicted_class", "risk_band",
             "threshold_used", "top_risk_driver", "top_risk_drivers",
             "collector_explanations", "model_features", "model_name"]

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


def _risk_band(scores):
    try:
        q = pd.qcut(scores, q=3, duplicates="drop")
        labels = ["Low", "Medium", "High"][:len(q.cat.categories)]
        return pd.qcut(scores, q=len(q.cat.categories), labels=labels,
                       duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("Low", index=scores.index)


predictions["risk_band"] = predictions.groupby("country")["risk_score"].transform(_risk_band)

FEATURE_OUT_COLS = [c for c in predictions.columns
                    if c not in ID_COLS + PRED_COLS
                    and not c.startswith(("country_", "cluster_"))]
predictions = predictions[ID_COLS + FEATURE_OUT_COLS + PRED_COLS]

print(f"\nTotal scored: {len(predictions):,} customers ({len(predictions.columns)} columns)")
print(predictions.groupby(["cluster", "predicted_class"]).size())

# COMMAND ----------

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

out_sdf = spark.createDataFrame(predictions).withColumn("refresh_date", F.current_date())
out_sdf.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
print(f"Wrote {out_sdf.count():,} rows to {OUTPUT_TABLE}")
