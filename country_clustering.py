# Databricks notebook source
# MAGIC %md
# MAGIC # Country Similarity — Behavioral Vector + Ward HAC + Warm-Start
# MAGIC
# MAGIC Goal: low-data countries (NO, SE, IE...) cannot train standalone.
# MAGIC Find nearest big-data neighbour by PAYMENT BEHAVIOR, then warm-start
# MAGIC the small country's model from the neighbour's booster.
# MAGIC
# MAGIC Method:
# MAGIC   1. Customer-grain behavioral aggregates (correct grain, not line-item)
# MAGIC   2. Country feature vector:
# MAGIC        mean DPD | on-time ratio | dunning rate | P2P keep ratio
# MAGIC        | exposure skew | sector mix
# MAGIC   3. Standardize (z-score)
# MAGIC   4. Hierarchical Agglomerative Clustering — Ward linkage
# MAGIC   5. Dendrogram = interpretable proximity ("AT nearest NO at d=0.4")
# MAGIC   6. Warm-start: load neighbour model, continue-train on small country

# COMMAND ----------

# MAGIC %pip install scikit-learn scipy seaborn xgboost mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# =========================================================
# CONFIG
# =========================================================
UNIFIED_VIEW = "f_erp_glide_o2c_12.vw_invoice_unified"
SECTOR_COL = "credit_group"          # proxy for sector/segment mix; swap if a true industry col exists
MIN_CUSTOMERS_FOR_MODEL = 5000       # >= this => "big" (can train own); else borrow + warm-start
N_CLUSTERS = 4
TODAY_EXPR = F.current_date()


# =========================================================
# 1. CUSTOMER-GRAIN AGGREGATION  (correct grain)
# =========================================================
def customer_level():
    """
    Roll the line-item view up to ONE row per (country, customer).
    Produces the raw quantities the country vector is built from.
    """
    df = spark.table(UNIFIED_VIEW).select(
        "country",
        "customer_id",
        "invoice_id",
        F.col("invoice_amount").cast("double").alias("invoice_amount"),
        F.col("open_amount").cast("double").alias("open_amount"),
        F.col("days_past_due").cast("int").alias("days_past_due"),
        F.col("clearing_date").cast("date").alias("clearing_date"),
        F.col("due_date").cast("date").alias("due_date"),
        F.col("dunning_count").cast("int").alias("dunning_count"),
        F.col("fin_p2p_state").cast("int").alias("fin_p2p_state"),
        F.col("credit_limit").cast("double").alias("credit_limit"),
        F.col(SECTOR_COL).alias("sector"),
        F.upper(F.col("source")).alias("source"),
    ).filter(F.col("country").isNotNull())

    # invoice grain first (open_amount aware)
    inv = df.groupBy("country", "customer_id", "invoice_id").agg(
        F.sum("open_amount").alias("open_amount"),
        F.max("days_past_due").alias("days_past_due"),
        F.max(F.when(F.col("source") == "BSAD", 1).otherwise(0)).alias("is_cleared"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date"),
        F.max("dunning_count").alias("dunning_count"),
        F.first("fin_p2p_state", ignorenulls=True).alias("fin_p2p_state"),
        F.first("credit_limit", ignorenulls=True).alias("credit_limit"),
        F.first("sector", ignorenulls=True).alias("sector"),
    ).withColumn(
        # on-time = cleared on/before due
        "paid_on_time",
        F.when(
            (F.col("is_cleared") == 1) &
            (F.col("clearing_date") <= F.col("due_date")), 1
        ).otherwise(0)
    )

    # customer grain
    cust = inv.groupBy("country", "customer_id").agg(
        F.avg("days_past_due").alias("cust_mean_dpd"),
        F.avg("paid_on_time").alias("cust_on_time_ratio"),
        F.sum("dunning_count").alias("cust_dunning_events"),
        F.sum(F.when(F.col("fin_p2p_state") == 3, 1).otherwise(0)).alias("cust_kept"),
        F.sum(F.when(F.col("fin_p2p_state").isin([1, 3]), 1).otherwise(0)).alias("cust_promises"),
        F.sum("open_amount").alias("cust_outstanding"),
        F.max("credit_limit").alias("cust_credit_limit"),
        F.first("sector", ignorenulls=True).alias("sector"),
    )
    return cust


# =========================================================
# 2. COUNTRY FEATURE VECTOR
# =========================================================
def country_vector(cust_pdf):
    """
    cust_pdf: pandas, one row per customer.
    Returns one row per country with the 6 behavioral signals
    (+ sector-mix expanded to share columns).
    """
    rows = []
    sector_levels = (
        cust_pdf["sector"].fillna("UNK").value_counts().head(8).index.tolist()
    )

    for country, g in cust_pdf.groupby("country"):
        n_cust = len(g)

        # exposure skew — whale concentration of outstanding across customers
        out = g["cust_outstanding"].clip(lower=0).values
        exposure_skew = pd.Series(out).skew() if n_cust > 2 else 0.0

        rec = {
            "country": country,
            "n_customers": n_cust,
            # --- 6 core signals ---
            "mean_dpd":        g["cust_mean_dpd"].mean(),
            "on_time_ratio":   g["cust_on_time_ratio"].mean(),
            "dunning_rate":    (g["cust_dunning_events"] > 0).mean(),          # share of customers ever dunned
            "p2p_keep_ratio":  g["cust_kept"].sum() / max(g["cust_promises"].sum(), 1),
            "exposure_skew":   exposure_skew,
        }

        # sector mix — share of customers per sector level
        sec_counts = g["sector"].fillna("UNK").value_counts(normalize=True)
        for lvl in sector_levels:
            rec[f"sector_{lvl}"] = float(sec_counts.get(lvl, 0.0))

        rows.append(rec)

    vec = pd.DataFrame(rows).set_index("country").fillna(0)
    return vec


# =========================================================
# 3. STANDARDIZE
# =========================================================
def standardize(vec_df):
    feature_cols = [c for c in vec_df.columns if c != "n_customers"]
    X = vec_df[feature_cols].astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(X_scaled, index=vec_df.index, columns=feature_cols)


# =========================================================
# 4. WARD HAC + DENDROGRAM
# =========================================================
def ward_clustering(X_scaled, n_clusters=N_CLUSTERS):
    Z = linkage(X_scaled, method="ward")

    # cophenetic correlation — how faithfully the tree preserves distances
    coph_corr, _ = cophenet(Z, pdist(X_scaled))
    print(f"Cophenetic correlation: {coph_corr:.3f}  (closer to 1 = trustworthy tree)")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z, labels=list(X_scaled.index), ax=ax,
        leaf_rotation=0, leaf_font_size=12, color_threshold=None,
    )
    ax.set_title("Country Proximity — Ward Hierarchical Clustering")
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    plt.show()

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, pd.Series(labels, index=X_scaled.index, name="cluster")


# =========================================================
# 5. NEAREST-NEIGHBOUR PROXIMITY TABLE
# =========================================================
def proximity_table(X_scaled, vec_df, min_customers=MIN_CUSTOMERS_FOR_MODEL):
    """
    Pairwise Ward-space (Euclidean on standardized vector) distances.
    For each SMALL country, nearest BIG country = warm-start donor.
    """
    dist = squareform(pdist(X_scaled, metric="euclidean"))
    dist_df = pd.DataFrame(dist, index=X_scaled.index, columns=X_scaled.index)

    big = vec_df[vec_df["n_customers"] >= min_customers].index.tolist()
    small = vec_df[vec_df["n_customers"] < min_customers].index.tolist()

    rows = []
    for sc in small:
        d = dist_df.loc[sc, big].sort_values()
        rows.append({
            "small_country":  sc,
            "n_customers":    int(vec_df.loc[sc, "n_customers"]),
            "nearest_big":    d.index[0],
            "distance":       round(float(d.iloc[0]), 3),
            "runner_up":      d.index[1] if len(d) > 1 else None,
            "runner_up_dist": round(float(d.iloc[1]), 3) if len(d) > 1 else None,
        })

    routing = pd.DataFrame(rows).sort_values("distance")
    print(f"\nBig countries (>= {min_customers} cust): {big}")
    print(f"Small countries (warm-start): {small}")
    print("\n=== PROXIMITY / DONOR TABLE ===")
    print(routing.to_string(index=False))
    return routing, dist_df


def plot_distance_heatmap(dist_df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="viridis_r", ax=ax)
    ax.set_title("Pairwise Country Distance (lower = more similar)")
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. WARM-START TRAINER  (transfer from donor booster)
# =========================================================
def warm_start_train(small_country_df, feature_cols, target_col,
                      donor_model_uri, n_extra_rounds=100, learning_rate=0.03):
    """
    Continue-train a small country's model from the donor's booster.

    Donor booster = learned structure from big neighbour.
    `xgb_model=` continues boosting on the small country's data, so the
    model adapts to local behavior without learning from scratch.

    small_country_df : pandas with feature_cols + target_col
    donor_model_uri  : MLflow uri of nearest-big-country model
    """
    import mlflow.sklearn
    from xgboost import XGBClassifier

    donor = mlflow.sklearn.load_model(donor_model_uri)
    donor_booster = donor.get_booster()

    # Align features to donor's expected order
    donor_features = donor_booster.feature_names
    X = small_country_df[donor_features].astype("float64").fillna(0)
    y = small_country_df[target_col].astype(int)

    n_neg = (y == 0).sum()
    n_pos = max((y == 1).sum(), 1)

    warm = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="aucpr",
        n_estimators=n_extra_rounds,        # extra rounds ON TOP of donor
        learning_rate=learning_rate,        # small LR — gentle local adaptation
        max_depth=donor.get_params().get("max_depth", 6),
        scale_pos_weight=n_neg / n_pos,
        random_state=42,
    )

    # xgb_model = donor booster -> warm start
    warm.fit(X, y, xgb_model=donor_booster, verbose=False)

    print(f"Warm-started from {donor_model_uri} | "
          f"+{n_extra_rounds} rounds on {len(X):,} rows "
          f"({n_pos} positives)")
    return warm, donor_features


# =========================================================
# RUN
# =========================================================
print("Aggregating to customer grain...")
cust_sdf = customer_level()
cust_pdf = cust_sdf.toPandas()
print(f"{cust_pdf['country'].nunique()} countries | {len(cust_pdf):,} customers")

vec = country_vector(cust_pdf)
print("\n=== COUNTRY FEATURE VECTOR ===")
print(vec.round(3).to_string())

X_scaled = standardize(vec)
Z, clusters = ward_clustering(X_scaled, N_CLUSTERS)
routing, dist_df = proximity_table(X_scaled, vec)
plot_distance_heatmap(dist_df)

result = pd.concat([vec[["n_customers"]], clusters], axis=1).sort_values("cluster")
print("\n=== CLUSTER ASSIGNMENTS ===")
print(result.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist donor routing for inference + warm-start

# COMMAND ----------

routing_sdf = spark.createDataFrame(routing)
routing_sdf.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("f_erp_glide_o2c_12.country_model_routing")
print("Saved -> f_erp_glide_o2c_12.country_model_routing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example warm-start for one small country
# MAGIC ```
# MAGIC # small_df = <pandas training frame for NO with feature_cols + target>
# MAGIC # donor_uri = "models:/collection_risk_model_customer_v1_DE/latest"
# MAGIC # no_model, feats = warm_start_train(small_df, feature_cols, "target", donor_uri)
# MAGIC # mlflow.sklearn.log_model(no_model, "model")  # register as NO model
# MAGIC ```
