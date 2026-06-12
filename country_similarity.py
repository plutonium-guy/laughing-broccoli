# Databricks notebook source
# MAGIC %md
# MAGIC # Country Similarity — Cluster + Nearest-Neighbor Lookup
# MAGIC
# MAGIC **Goal.** For low-data countries (e.g. NO, FI, PT) find the most
# MAGIC behaviorally similar high-data country (e.g. FR, DE, AT) so we can:
# MAGIC   1. Borrow its trained model as warm-start / proxy.
# MAGIC   2. Group countries into clusters that share a single model.
# MAGIC   3. Communicate proximity to stakeholders ("AT ~ NO at d=0.41").
# MAGIC
# MAGIC **Method.**
# MAGIC   country-level feature vector
# MAGIC     → z-score standardize
# MAGIC     → Ward-linkage Hierarchical Agglomerative Clustering
# MAGIC     → pairwise distance matrix
# MAGIC     → top-K nearest peer countries per country
# MAGIC
# MAGIC **Output Delta tables.**
# MAGIC   - country_features          : country x feature aggregates
# MAGIC   - country_distance_matrix   : long form (country_a, country_b, dist)
# MAGIC   - country_neighbors         : top-K nearest peers per country
# MAGIC   - country_cluster_labels    : cluster id per country at chosen k
# MAGIC
# MAGIC **Does NOT modify any existing file or trained model.**

# COMMAND ----------

from pyspark.sql import functions as F

import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================
UNIFIED_VIEW = "f_erp_glide_o2c_12.vw_invoice_unified"

OUT_FEATURES  = "f_erp_glide_o2c_12.country_features"
OUT_DISTANCES = "f_erp_glide_o2c_12.country_distance_matrix"
OUT_NEIGHBORS = "f_erp_glide_o2c_12.country_neighbors"
OUT_CLUSTERS  = "f_erp_glide_o2c_12.country_cluster_labels"

TODAY = "2026-03-25"
LOOKBACK_DAYS = 365            # 1y window for stable country profile
MIN_INVOICES_PER_COUNTRY = 50  # below this -> still included but flagged low-confidence
TOP_K_NEIGHBORS = 3
N_CLUSTERS = 4                 # tune via dendrogram inspection
LINKAGE_METHOD = "ward"        # ward | average | complete
DISTANCE_METRIC = "euclidean"  # ward requires euclidean


# =========================================================
# 1. AGGREGATE TO COUNTRY-LEVEL FEATURE VECTOR
# =========================================================
def build_country_features(spark):
    """
    Reduce invoice-grain unified view to ONE row per country.
    Each feature is a statistic that captures payment behavior shape.

    Why these features:
      - dpd_*       : how late countries pay
      - on_time_*   : % of invoices cleared on/before due
      - dunning_*   : how often escalation is needed
      - p2p_*       : promise discipline
      - exposure_*  : invoice size distribution (small vs large markets)
      - dispute_*   : friction signal
    """
    df = spark.table(UNIFIED_VIEW).filter(
        F.col("baseline_date") >= F.date_sub(F.lit(TODAY), LOOKBACK_DAYS)
    )

    # Per-invoice on-time / late flags
    df = df.withColumn(
        "is_on_time",
        F.when(
            (F.col("clearing_date").isNotNull())
            & (F.col("clearing_date") <= F.col("due_date")),
            1,
        ).otherwise(0),
    ).withColumn(
        "is_late",
        F.when(F.col("days_past_due") > 0, 1).otherwise(0),
    ).withColumn(
        "p2p_kept",
        F.when(F.col("fin_p2p_state") == 3, 1)
         .when(F.col("fin_p2p_state") == 1, 0)
         .otherwise(None),
    )

    agg = df.groupBy("country").agg(
        F.count("*").alias("n_invoices"),
        F.countDistinct("customer_id").alias("n_customers"),

        # Days past due distribution
        F.avg("days_past_due").alias("dpd_mean"),
        F.expr("percentile_approx(days_past_due, 0.5)").alias("dpd_median"),
        F.expr("percentile_approx(days_past_due, 0.9)").alias("dpd_p90"),
        F.stddev("days_past_due").alias("dpd_std"),

        # On-time / late behavior
        F.avg("is_on_time").alias("on_time_rate"),
        F.avg("is_late").alias("late_rate"),

        # Dunning intensity
        F.avg("dunning_level").alias("dunning_level_mean"),
        F.avg("dunning_count").alias("dunning_count_mean"),
        F.avg(F.when(F.col("dunning_level") >= 2, 1).otherwise(0))
            .alias("dunning_escalation_rate"),

        # P2P discipline
        F.avg("p2p_kept").alias("p2p_keep_rate"),
        F.avg(
            F.when(F.col("fin_promised_amt").isNotNull(), 1).otherwise(0)
        ).alias("p2p_usage_rate"),

        # Exposure / invoice size shape
        F.avg("invoice_amount").alias("invoice_amt_mean"),
        F.expr("percentile_approx(invoice_amount, 0.5)").alias("invoice_amt_median"),
        F.stddev("invoice_amount").alias("invoice_amt_std"),
        F.avg("open_amount").alias("open_amt_mean"),

        # Disputes
        F.avg("number_of_disputes").alias("dispute_count_mean"),
        F.avg(F.when(F.col("dispute_create_date").isNotNull(), 1).otherwise(0))
            .alias("dispute_rate"),

        # Credit
        F.avg("credit_limit").alias("credit_limit_mean"),
        F.avg("customer_tenure_days").alias("tenure_days_mean"),

        # Payment terms tendency
        F.avg("net_payment_days").alias("payment_terms_mean"),
    )

    agg = agg.withColumn(
        "low_confidence",
        F.when(F.col("n_invoices") < MIN_INVOICES_PER_COUNTRY, 1).otherwise(0),
    )

    return agg


# =========================================================
# 2. STANDARDIZE + COMPUTE DISTANCE MATRIX
# =========================================================
FEATURE_COLS = [
    "dpd_mean", "dpd_median", "dpd_p90", "dpd_std",
    "on_time_rate", "late_rate",
    "dunning_level_mean", "dunning_count_mean", "dunning_escalation_rate",
    "p2p_keep_rate", "p2p_usage_rate",
    "invoice_amt_mean", "invoice_amt_median", "invoice_amt_std", "open_amt_mean",
    "dispute_count_mean", "dispute_rate",
    "credit_limit_mean", "tenure_days_mean",
    "payment_terms_mean",
]


def to_country_matrix(features_pdf):
    """
    Convert Spark agg -> pandas, fill NaN with column median (robust to
    countries that genuinely have no disputes / no P2P usage), then z-score.
    Returns (countries[list], scaled_matrix[np.ndarray], scaler).
    """
    pdf = features_pdf[["country"] + FEATURE_COLS].copy()
    pdf = pdf.set_index("country").sort_index()

    # Median impute then standardize. Median over mean -> robust to outlier
    # countries with very small invoice counts.
    pdf = pdf.fillna(pdf.median(numeric_only=True))

    scaler = StandardScaler()
    X = scaler.fit_transform(pdf.values)

    return list(pdf.index), X, scaler


def pairwise_distance_long(countries, X):
    """
    Return long-form DataFrame: country_a, country_b, distance (sym, no diag).
    """
    D = squareform(pdist(X, metric=DISTANCE_METRIC))
    rows = []
    for i, a in enumerate(countries):
        for j, b in enumerate(countries):
            if i == j:
                continue
            rows.append((a, b, float(D[i, j])))
    return pd.DataFrame(rows, columns=["country_a", "country_b", "distance"])


# =========================================================
# 3. HAC + CLUSTER LABELS
# =========================================================
def hierarchical_cluster(X, n_clusters=N_CLUSTERS):
    """
    Ward linkage on standardized country vectors.
    Returns linkage matrix + flat cluster labels at chosen k.
    """
    Z = linkage(X, method=LINKAGE_METHOD, metric=DISTANCE_METRIC)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, labels


def plot_dendrogram(Z, countries, save_path=None):
    """
    Optional viz. In Databricks notebook the matplotlib fig will render inline.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=countries, leaf_rotation=45, ax=ax)
    ax.set_title("Country Similarity — Ward Linkage")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


# =========================================================
# 4. NEAREST-NEIGHBOR LOOKUP
# =========================================================
def top_k_neighbors(dist_long_pdf, k=TOP_K_NEIGHBORS):
    """
    For each country, return top-k nearest peers ranked by distance asc.
    """
    ranked = (
        dist_long_pdf.sort_values(["country_a", "distance"])
        .groupby("country_a", as_index=False)
        .head(k)
        .copy()
    )
    ranked["rank"] = ranked.groupby("country_a").cumcount() + 1
    return ranked.rename(columns={"country_a": "country", "country_b": "neighbor"})


def suggest_proxy_model(neighbors_pdf, low_data_countries, trained_countries):
    """
    For each low-data country, pick the nearest country that has a trained
    model. This is the actionable bit -> tells the inference router which
    model to use as proxy.
    """
    out = []
    for c in low_data_countries:
        cand = neighbors_pdf[
            (neighbors_pdf["country"] == c)
            & (neighbors_pdf["neighbor"].isin(trained_countries))
        ].sort_values("distance")
        if len(cand) == 0:
            out.append((c, None, None))
        else:
            row = cand.iloc[0]
            out.append((c, row["neighbor"], float(row["distance"])))
    return pd.DataFrame(out, columns=["country", "proxy_country", "distance"])


# =========================================================
# 5. ORCHESTRATION
# =========================================================
def run(spark, n_clusters=N_CLUSTERS, write=True):
    # 5a. country-level features
    feats_sdf = build_country_features(spark)
    feats_pdf = feats_sdf.toPandas()
    print(f"[country_similarity] {len(feats_pdf)} countries aggregated")

    # 5b. matrix + distance
    countries, X, _ = to_country_matrix(feats_pdf)
    dist_long = pairwise_distance_long(countries, X)

    # 5c. clustering
    Z, labels = hierarchical_cluster(X, n_clusters=n_clusters)
    cluster_pdf = pd.DataFrame(
        {"country": countries, "cluster_id": labels.astype(int)}
    )

    # 5d. neighbors
    neighbors_pdf = top_k_neighbors(dist_long, k=TOP_K_NEIGHBORS)

    if write:
        spark.createDataFrame(feats_pdf).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_FEATURES)
        spark.createDataFrame(dist_long).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_DISTANCES)
        spark.createDataFrame(neighbors_pdf).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_NEIGHBORS)
        spark.createDataFrame(cluster_pdf).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_CLUSTERS)
        print("[country_similarity] wrote 4 Delta tables")

    return {
        "features": feats_pdf,
        "distances": dist_long,
        "neighbors": neighbors_pdf,
        "clusters": cluster_pdf,
        "linkage": Z,
        "countries": countries,
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage
# MAGIC
# MAGIC ```python
# MAGIC res = run(spark, n_clusters=4)
# MAGIC
# MAGIC # Inspect dendrogram
# MAGIC plot_dendrogram(res["linkage"], res["countries"])
# MAGIC
# MAGIC # Proxy model picker for low-data countries
# MAGIC proxy = suggest_proxy_model(
# MAGIC     res["neighbors"],
# MAGIC     low_data_countries=["NO", "FI", "PT"],
# MAGIC     trained_countries=["FR", "AT", "DE"],
# MAGIC )
# MAGIC display(proxy)
# MAGIC ```
# MAGIC
# MAGIC **Tuning notes.**
# MAGIC - If clusters look unbalanced, try `LINKAGE_METHOD = "average"`.
# MAGIC - If one feature dominates (e.g. invoice_amt_std), drop it from
# MAGIC   `FEATURE_COLS` or switch to rank-based features.
# MAGIC - `low_confidence=1` countries should be reviewed manually before
# MAGIC   trusting their proxy assignment.
