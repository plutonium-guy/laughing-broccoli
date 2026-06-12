# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Country Inference Router
# MAGIC
# MAGIC Each country has:
# MAGIC   - own trained model (different hyperparams)
# MAGIC   - own selected feature list (different important features)
# MAGIC   - own threshold (different recall/precision trade-off)
# MAGIC   - own risk bands
# MAGIC
# MAGIC This script routes customers to the correct model based on country,
# MAGIC engineers the right features, scores with the right threshold, and
# MAGIC combines results into one prediction table.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, DoubleType


# =========================================================
# COUNTRY REGISTRY
# =========================================================
# Each country points to its own MLflow registered model.
# Feature list is read from the model itself (no duplication).
# Threshold + risk band cutoffs stored here per country.
# =========================================================

COUNTRY_REGISTRY = {
    "FR": {
        "model_uri":   "models:/collection_risk_model_customer_FR/latest",
        "threshold":   0.22,
        "bands":       (0, 0.4, 0.7, 1.0),
        "band_labels": ("Low", "Medium", "High"),
    },
    "AT": {
        "model_uri":   "models:/collection_risk_model_customer_AT/latest",
        "threshold":   0.31,
        "bands":       (0, 0.35, 0.65, 1.0),
        "band_labels": ("Low", "Medium", "High"),
    },
    "DE": {
        "model_uri":   "models:/collection_risk_model_customer_DE/latest",
        "threshold":   0.28,
        "bands":       (0, 0.4, 0.7, 1.0),
        "band_labels": ("Low", "Medium", "High"),
    },
    # Add more countries here as models get trained
}


# =========================================================
# MODEL CACHE — load each model once, reuse across customers
# =========================================================

_MODEL_CACHE = {}


def get_model_for_country(country):
    """
    Lazy-load model + extract its feature list from the booster.
    Cached per executor so each country's model loaded only once.
    """
    if country not in COUNTRY_REGISTRY:
        raise ValueError(f"No model registered for country '{country}'. "
                         f"Known: {list(COUNTRY_REGISTRY.keys())}")

    if country in _MODEL_CACHE:
        return _MODEL_CACHE[country]

    config = COUNTRY_REGISTRY[country]
    model = mlflow.sklearn.load_model(config["model_uri"])

    # Feature list lives inside the model — never duplicate it here
    feature_cols = model.get_booster().feature_names

    bundle = {
        "model":        model,
        "feature_cols": feature_cols,
        "threshold":    config["threshold"],
        "bands":        config["bands"],
        "band_labels":  config["band_labels"],
    }
    _MODEL_CACHE[country] = bundle
    print(f"[{country}] loaded model with {len(feature_cols)} features, "
          f"threshold={config['threshold']}")
    return bundle


# =========================================================
# FEATURE ENGINEERING — same logic across countries
# =========================================================
# Build the SUPERSET of all features any model might need.
# At scoring time, each model selects only the columns it needs.
# Country-specific filters (country code mapping etc.) applied first.
# =========================================================

def engineer_features_for_country(country):
    """
    Replace the body with your existing inference feature engineering.
    Should produce one Spark DataFrame:
       customer_id | snapshot_date | <all_possible_features>
    filtered to customers in `country`.

    The feature SUPERSET must contain every feature any country's
    model could ask for. Per-model column selection happens later.
    """
    # === call your existing functions here, e.g.: ===
    # spine = filter_by_countries(customer_spine(), [country])
    # base  = load_inference_data()
    # base  = add_due_date(base)
    # base  = filter_by_countries(base, [country])
    # inv   = build_invoice_level(base)
    # snap  = build_snapshot(inv)
    # exp   = exposure_features(snap, spine)
    # beh   = behavior_features(base, spine)
    # dun   = dunning_features(spine)
    # p2p   = p2p_features(spine)
    # feat  = (spine.join(exp, [...], "left")
    #              .join(beh, [...], "left")
    #              .join(dun, [...], "left")
    #              .join(p2p, [...], "left")
    #              .fillna(0))
    # return feat
    raise NotImplementedError("Wire in your inference feature engineering.")


# =========================================================
# SCORING — country aware
# =========================================================

def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(
                field.name, spark_df[field.name].cast(DoubleType())
            )
    return spark_df.fillna(0).toPandas()


def score_country(country, feat_sdf):
    """
    Score one country's customers with its own model.
    Returns Spark DataFrame with: customer_id, country, risk_score,
    predicted_class, top_risk_driver.
    """
    bundle = get_model_for_country(country)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    threshold    = bundle["threshold"]
    bands        = bundle["bands"]
    band_labels  = bundle["band_labels"]

    # Convert to pandas with only the columns this model wants
    needed_cols = ["customer_id", "snapshot_date"] + feature_cols
    pdf = spark_to_pandas_safe(feat_sdf.select(needed_cols)).fillna(0)

    # Coerce all features to numeric (SHAP + XGBoost need float)
    X = pdf[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0).astype("float64")

    # Predict
    probs = model.predict_proba(X)[:, 1]
    pdf["risk_score"] = probs
    pdf["binary_pred"] = (probs >= threshold).astype(int)
    pdf["predicted_class"] = pd.cut(
        probs,
        bins=list(bands),
        labels=list(band_labels),
        include_lowest=True,
    )
    pdf["country"] = country
    pdf["threshold_used"] = threshold
    pdf["model_version"] = COUNTRY_REGISTRY[country]["model_uri"]

    # Top driver per row via native XGBoost SHAP
    dmatrix = xgb.DMatrix(X)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    top_idx = np.argmax(np.abs(shap_values), axis=1)
    pdf["top_risk_driver"] = [feature_cols[i] for i in top_idx]

    return spark.createDataFrame(pdf[[
        "customer_id", "country", "snapshot_date",
        "risk_score", "binary_pred", "predicted_class",
        "threshold_used", "top_risk_driver", "model_version",
    ]])


# =========================================================
# UNIFIED MULTI-COUNTRY RUN
# =========================================================

def run_inference_all_countries(countries=None):
    """
    Score every country with its own model, union results into a
    single Spark DataFrame for downstream tables / dashboards.
    """
    if countries is None:
        countries = list(COUNTRY_REGISTRY.keys())

    combined = None
    for country in countries:
        print(f"\n========== Scoring {country} ==========")
        try:
            feat_sdf = engineer_features_for_country(country)
            scored = score_country(country, feat_sdf)
            combined = scored if combined is None else combined.unionByName(scored)
            print(f"[{country}] scored {scored.count():,} customers")
        except Exception as e:
            print(f"[{country}] FAILED: {e}")
            continue

    if combined is None:
        raise RuntimeError("No country produced results.")

    return combined


# =========================================================
# WRITE COMBINED OUTPUT
# =========================================================

def write_predictions(predictions_sdf, table_name="f_erp_glide_o2c_12.collection_ml_customer"):
    predictions_sdf = predictions_sdf.withColumn(
        "refresh_date", F.current_date()
    )
    predictions_sdf.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(table_name)
    print(f"Wrote {predictions_sdf.count():,} rows to {table_name}")


# =========================================================
# RUN
# =========================================================
# predictions = run_inference_all_countries(["FR", "AT", "DE"])
# write_predictions(predictions)
# display(predictions)
