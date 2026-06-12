# Databricks notebook source
# MAGIC %pip install shap xgboost category-encoders mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

COUNTRY_MODEL_MAP = {
    "FR": "models:/collection_risk_model_customer_v1/latest",
    "AT": "models:/collection_risk_model_customer_v1_AT/latest",
}

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType
from pyspark.sql.functions import lit, current_date

mlflow.set_registry_uri("databricks")

TODAY = F.current_date()
WINDOWS = [60, 90, 180]

# COMMAND ----------

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


def build_invoice_level(df):
    return df.groupBy("customer_id","invoice_id").agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.min("baseline_date").alias("baseline_date"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date")
    )


def build_snapshot(invoice_df):
    return invoice_df.filter(
        (F.col("baseline_date") <= TODAY) &
        (F.col("clearing_date").isNull() | (F.col("clearing_date") > TODAY))
    ).withColumn("snapshot_date", TODAY)


def customer_spine():
    return spark.table("hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid") \
        .select(F.col("KUNNR").alias("customer_id")).distinct() \
        .withColumn("snapshot_date", TODAY)


def exposure_features(snapshot_df, spine):
    df = spine.join(snapshot_df, ["customer_id","snapshot_date"], "left")
    df = df.withColumn("days_past_due", F.datediff("snapshot_date", "due_date")) \
           .withColumn("invoice_age", F.datediff("snapshot_date", "baseline_date"))

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
        .withColumn("avg_invoice_size", F.col("total_outstanding")/F.col("num_open_invoices")) \
        .withColumn("pct_30_plus", F.col("amt_30_plus")/F.col("total_outstanding")) \
        .withColumn("pct_60_plus", F.col("amt_60_plus")/F.col("total_outstanding")) \
        .withColumn("pct_90_plus", F.col("amt_90_plus")/F.col("total_outstanding"))


def behavior_features(base_df, spine_df):
    b = base_df.filter(F.col("clearing_date").isNotNull()).withColumn(
        "days_to_pay", F.datediff("clearing_date","baseline_date")
    )

    s = spine_df.select(F.col("customer_id").alias("s_customer_id"), "snapshot_date")

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


def dunning_features(spine_df):
    d = spark.table("hive_metastore.t_erp_ordertocash_rbp_conf.mhnd").select(
        F.col("KUNNR").alias("d_customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.to_date("LAUFD", "yyyyMMdd").alias("dunning_date"),
        F.col("MAHNN").cast("int").alias("dunning_level")
    )

    s = spine_df.select(F.col("customer_id").alias("s_customer_id"), "snapshot_date")

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
               F.col("high_severity_dunning") / F.col("total_dunning_events")).otherwise(0)
    )


def p2p_features(spine_df):
    p = spark.table("hive_metastore.f_erp_glide_o2c_12.UDM_P2P_ATTR").select(
        F.col("fin_customer").alias("p_customer_id"),
        F.substring("fin_invoice_key", 5, 10).alias("invoice_id"),
        F.col("fin_promised_amt").cast("double").alias("promised_amt"),
        F.col("fin_p2p_state").cast("int").alias("p2p_state"),
        F.to_date("fin_p2p_date", "yyyyMMdd").alias("promise_date")
    )

    s = spine_df.select(F.col("customer_id").alias("s_customer_id"), "snapshot_date")

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
    ).withColumnRenamed("s_customer_id", "customer_id")


def filter_by_countries(df, countries):
    if not countries:
        return df

    eligible_customers = spark.table(
        "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid"
    ).select(
        F.col("KUNNR").alias("customer_id"),
        F.upper(F.substring(F.col("BUKRS"), 1, 2)).alias("country_code")
    ).filter(
        F.col("country_code").isin([c.upper() for c in countries])
    ).select("customer_id").distinct()

    return df.join(eligible_customers, on="customer_id", how="inner")


def spark_to_pandas_safe(spark_df, sample_frac=None):
    if sample_frac:
        spark_df = spark_df.sample(fraction=sample_frac, seed=42)

    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(
                field.name, spark_df[field.name].cast(DoubleType())
            )

    spark_df = spark_df.fillna(0)
    return spark_df.toPandas()

# COMMAND ----------

def explain_one(feature, value, shap_value):
    impact = "increased risk" if shap_value > 0 else "reduced risk"

    if feature == "total_outstanding":
        return f"Total unpaid balance of {value:,.0f} EUR {impact}."
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
    else:
        readable = feature.replace("_", " ")
        return f"{readable.capitalize()} value of {value} {impact}."


def build_business_explanation(contribution_df):
    return [
        explain_one(
            feature=row["feature"],
            value=row["feature_value"],
            shap_value=row["shap_value"],
        )
        for _, row in contribution_df.iterrows()
    ]


def build_business_explanations(shap_row, feature_row, top_n=5, FEATURE_COLS=None):
    if FEATURE_COLS is None:
        raise ValueError("FEATURE_COLS must be provided.")

    contribution_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap_value": shap_row,
        "feature_value": feature_row,
    })

    contribution_df["abs_shap"] = contribution_df["shap_value"].abs()
    contribution_df = contribution_df.sort_values(by="abs_shap", ascending=False).head(top_n)

    return build_business_explanation(contribution_df)

# COMMAND ----------

def score(df, model, FEATURE_COLS):
    pdf = spark_to_pandas_safe(
        df.select(["customer_id", "snapshot_date"] + FEATURE_COLS)
    ).fillna(0)

    X = pdf[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    pdf["risk_score"] = probs

    pdf["predicted_class"] = pd.cut(
        pdf["risk_score"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    dmatrix = xgb.DMatrix(X)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]

    all_explanations = []
    for i in range(len(pdf)):
        explanations = build_business_explanations(
            shap_row=shap_values[i],
            feature_row=X.iloc[i].values,
            FEATURE_COLS=FEATURE_COLS,
            top_n=5
        )
        all_explanations.append(" | ".join(explanations))
    pdf["collector_explanations"] = all_explanations

    top_driver = []
    for i in range(len(pdf)):
        shap_abs = np.abs(shap_values[i])
        max_idx = np.argmax(shap_abs)
        top_driver.append(FEATURE_COLS[max_idx])
    pdf["top_risk_driver"] = top_driver

    return spark.createDataFrame(pdf)

# COMMAND ----------

def run_country(country_code, model_path):
    print(f"\n========== {country_code} ==========")
    print(f"model: {model_path}")

    model = mlflow.sklearn.load_model(model_uri=model_path)
    FEATURE_COLS = model.get_booster().feature_names

    spine = filter_by_countries(customer_spine(), [country_code])
    base = load_inference_data()
    base = add_due_date(base)
    base = filter_by_countries(base, [country_code])

    inv = build_invoice_level(base)
    snap = build_snapshot(inv)

    exp = exposure_features(snap, spine)
    beh = behavior_features(base, spine)
    dun = dunning_features(spine)
    p2p = p2p_features(spine)

    feat = spine \
        .join(exp, ["customer_id","snapshot_date"], "left") \
        .join(beh, ["customer_id","snapshot_date"], "left") \
        .join(dun, ["customer_id","snapshot_date"], "left") \
        .join(p2p, ["customer_id","snapshot_date"], "left") \
        .fillna(0)

    predictions_df = score(feat, model, FEATURE_COLS)

    predictions_df = predictions_df \
        .withColumn("country", lit(country_code)) \
        .withColumn("refresh_date", current_date())

    print(f"{country_code}: {predictions_df.count():,} customers scored")
    return predictions_df

# COMMAND ----------

schema_name = "f_erp_glide_o2c_12"
table_name = "collection_ml_customer"
base_location = "abfss://root@coentus6abfsprod001.dfs.core.windows.net/data/raw/vala/vala/global/vala/Foundation/"
table_location = f"{base_location}{table_name}/"

spark.sql("USE CATALOG hive_metastore")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

# COMMAND ----------

def table_exists(location):
    try:
        spark.read.format("delta").load(location).limit(1).count()
        return True
    except Exception:
        return False

first_run = not table_exists(table_location)
print(f"first_run = {first_run}")

all_predictions = None

for idx, (country_code, model_path) in enumerate(COUNTRY_MODEL_MAP.items()):
    try:
        preds = run_country(country_code, model_path)

        if first_run and idx == 0:
            preds.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(table_location)
            mode_used = "overwrite (first run)"
        else:
            preds.write.format("delta").mode("append").save(table_location)
            mode_used = "append"

        print(f"{country_code}: wrote with mode={mode_used}")
        all_predictions = preds if all_predictions is None else all_predictions.unionByName(preds)
    except Exception as e:
        print(f"{country_code} FAILED: {e}")
        continue

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS hive_metastore.{schema_name}.{table_name}
    USING DELTA
    LOCATION '{table_location}'
""")

# COMMAND ----------

display(spark.table(f"hive_metastore.{schema_name}.{table_name}"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country, COUNT(*) FROM f_erp_glide_o2c_12.collection_ml_customer GROUP BY country

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT predicted_class, COUNT(*) FROM f_erp_glide_o2c_12.collection_ml_customer GROUP BY predicted_class
