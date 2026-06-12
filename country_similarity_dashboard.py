# Databricks notebook source
# MAGIC %md
# MAGIC # Country Similarity — Self-Contained: Pipeline + Auto-Tune + Business Dashboard
# MAGIC
# MAGIC One file. Replaces `country_similarity_v2.py` (now deleted) and
# MAGIC consolidates everything.
# MAGIC
# MAGIC **Sections.**
# MAGIC   1. Original SAP data pipeline   (BSAD/BSID/MHND/UDM_P2P/KNA1, customer grain)
# MAGIC   2. Country feature vector       (18 behavioral + sector mix + exposure skew)
# MAGIC   3. Standardize + PCA decorrelate
# MAGIC   4. Optuna auto-tune clustering  (method × k × pca_var)
# MAGIC   5. Multi-method cluster validation backup (silhouette + cophenetic)
# MAGIC   6. Dual-metric neighbors        (Euclidean + Cosine)
# MAGIC   7. Bootstrap stability          (per-country agreement rate)
# MAGIC   8. Routing table                (small → big donor)
# MAGIC   9. Optuna auto-tune warm-start XGB (per small country)
# MAGIC   10. Business dashboard          (8 stakeholder plots)
# MAGIC   11. Orchestration + Delta outputs
# MAGIC
# MAGIC **Sources (original multi-table pipeline only).**
# MAGIC   - BSAD / BSID   `hive_metastore.t_erp_ibp_customerservice_rbp_conf.{bsad,bsid}`
# MAGIC   - MHND          `hive_metastore.t_erp_ordertocash_rbp_conf.mhnd`
# MAGIC   - UDM_P2P_ATTR  `hive_metastore.f_erp_glide_o2c_12.UDM_P2P_ATTR`
# MAGIC   - KNA1          `hive_metastore.t_erp_ordertocash_rbp_csi.kna1`
# MAGIC
# MAGIC **Does NOT modify any production file** (collection_risk_model.py et al).

# COMMAND ----------

# MAGIC %pip install optuna scikit-learn scipy seaborn xgboost mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, average_precision_score, roc_auc_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


# =========================================================
# CONFIG
# =========================================================
BSAD_TABLE = "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsad"
BSID_TABLE = "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsid"
MHND_TABLE = "hive_metastore.t_erp_ordertocash_rbp_conf.mhnd"
P2P_TABLE  = "hive_metastore.f_erp_glide_o2c_12.UDM_P2P_ATTR"
KNA1_TABLE = "hive_metastore.t_erp_ordertocash_rbp_csi.kna1"

OUT_FEATURES  = "f_erp_glide_o2c_12.country_features_v2"
OUT_DISTANCES = "f_erp_glide_o2c_12.country_distance_matrix_v2"
OUT_NEIGHBORS = "f_erp_glide_o2c_12.country_neighbors_v2"
OUT_CLUSTERS  = "f_erp_glide_o2c_12.country_cluster_labels_v2"
OUT_ROUTING   = "f_erp_glide_o2c_12.country_model_routing_v2"
OUT_STABILITY = "f_erp_glide_o2c_12.country_neighbor_stability_v2"

TODAY = "2026-03-25"
LOOKBACK_DAYS = 730
MIN_CUSTOMERS_FOR_MODEL = 5000
PCA_VARIANCE_RETAINED = 0.90
TOP_K_NEIGHBORS = 3
N_CLUSTERS_RANGE = (2, 6)
LINKAGE_METHODS = ["ward", "average", "complete"]
BOOTSTRAP_RUNS = 50
BOOTSTRAP_SAMPLE_FRAC = 0.7

N_TRIALS_CLUSTERING = 80
N_TRIALS_XGB = 50
STABILITY_LOW_THRESHOLD = 0.5
RANDOM_SEED = 42


# =========================================================
# 1. ORIGINAL SAP DATA PIPELINE  (BSAD + BSID + MHND + P2P + KNA1)
# =========================================================
def load_invoices():
    """
    Mirror collection_risk_model.load_base_data() but include BOTH BSAD
    (cleared) and BSID (open) so we capture full payment behavior shape.
    """
    cols_select = lambda src_tag: [
        F.col("KUNNR").alias("customer_id"),
        F.col("BELNR").alias("invoice_id"),
        F.col("BUZEI").alias("line_item_id"),
        F.col("WRBTR").cast("double").alias("invoice_amount"),
        F.to_date("zfbdt", "yyyyMMdd").alias("baseline_date"),
        F.to_date("augdt", "yyyyMMdd").alias("clearing_date"),
        F.col("zbd1t"), F.col("zbd2t"), F.col("zbd3t"),
        F.lit(src_tag).alias("source"),
    ]

    bsad = (
        spark.table(BSAD_TABLE)
        .filter(F.trim("zfbdt").rlike("^[0-9]{8}$") & (F.trim("zfbdt") != "00000000"))
        .select(*cols_select("BSAD"))
    )
    bsid = (
        spark.table(BSID_TABLE)
        .filter(F.trim("zfbdt").rlike("^[0-9]{8}$") & (F.trim("zfbdt") != "00000000"))
        .select(*cols_select("BSID"))
        .withColumn("clearing_date", F.lit(None).cast("date"))
    )

    df = bsad.unionByName(bsid).filter(
        F.col("baseline_date") >= F.date_sub(F.lit(TODAY), LOOKBACK_DAYS)
    )

    df = df.withColumn(
        "payment_terms_days",
        F.when(F.col("zbd3t") != 0, F.col("zbd3t"))
         .when(F.col("zbd2t") != 0, F.col("zbd2t"))
         .when(F.col("zbd1t") != 0, F.col("zbd1t"))
         .otherwise(0).cast(IntegerType()),
    ).withColumn(
        "due_date", F.date_add("baseline_date", F.col("payment_terms_days"))
    ).withColumn(
        "days_past_due",
        F.when(
            F.col("clearing_date").isNotNull(),
            F.datediff("clearing_date", "due_date"),
        ).otherwise(F.datediff(F.lit(TODAY), F.col("due_date"))),
    )

    return df


def load_dunning_per_customer():
    d = spark.table(MHND_TABLE).select(
        F.col("KUNNR").alias("customer_id"),
        F.col("MAHNS").cast("int").alias("dunning_level"),
        F.to_date("LAUFD", "yyyyMMdd").alias("dunning_date"),
    ).filter(F.col("dunning_date") >= F.date_sub(F.lit(TODAY), LOOKBACK_DAYS))

    return d.groupBy("customer_id").agg(
        F.count("*").alias("cust_dunning_events"),
        F.max("dunning_level").alias("cust_max_dunning_level"),
        F.avg("dunning_level").alias("cust_avg_dunning_level"),
        F.sum(F.when(F.col("dunning_level") >= 2, 1).otherwise(0))
            .alias("cust_dunning_escalations"),
    )


def load_p2p_per_customer():
    p = spark.table(P2P_TABLE).select(
        F.col("KUNNR").alias("customer_id"),
        F.col("FIN_P2P_STATE").cast("int").alias("state"),
        F.col("FIN_PROMISED_AMT").cast("double").alias("promised_amt"),
    )
    return p.groupBy("customer_id").agg(
        F.sum(F.when(F.col("state").isin([1, 3]), 1).otherwise(0)).alias("cust_promises"),
        F.sum(F.when(F.col("state") == 3, 1).otherwise(0)).alias("cust_kept"),
        F.sum(F.when(F.col("state") == 1, 1).otherwise(0)).alias("cust_broken"),
        F.sum("promised_amt").alias("cust_promised_total"),
    )


def load_customer_master():
    return spark.table(KNA1_TABLE).select(
        F.col("KUNNR").alias("customer_id"),
        F.col("LAND1").alias("country"),
        F.col("REGIO").alias("region"),
    )


def build_customer_features():
    """Customer-grain aggregation (correct grain for behavioral comparison)."""
    inv = load_invoices()

    inv_grain = inv.groupBy("customer_id", "invoice_id").agg(
        F.sum("invoice_amount").alias("invoice_amount"),
        F.min("due_date").alias("due_date"),
        F.max("clearing_date").alias("clearing_date"),
        F.max("days_past_due").alias("days_past_due"),
        F.max(F.when(F.col("source") == "BSAD", 1).otherwise(0)).alias("is_cleared"),
    ).withColumn(
        "paid_on_time",
        F.when(
            (F.col("is_cleared") == 1) & (F.col("clearing_date") <= F.col("due_date")), 1
        ).otherwise(0),
    )

    cust = inv_grain.groupBy("customer_id").agg(
        F.count("*").alias("cust_n_invoices"),
        F.sum("invoice_amount").alias("cust_total_exposure"),
        F.avg("invoice_amount").alias("cust_avg_invoice_amt"),
        F.expr("percentile_approx(invoice_amount, 0.5)").alias("cust_med_invoice_amt"),
        F.avg("days_past_due").alias("cust_mean_dpd"),
        F.expr("percentile_approx(days_past_due, 0.9)").alias("cust_p90_dpd"),
        F.avg("paid_on_time").alias("cust_on_time_ratio"),
        F.avg("is_cleared").alias("cust_cleared_ratio"),
    )

    cust = (
        cust.join(load_dunning_per_customer(), "customer_id", "left")
            .join(load_p2p_per_customer(),     "customer_id", "left")
            .join(load_customer_master(),      "customer_id", "left")
            .fillna(0, subset=[
                "cust_dunning_events", "cust_max_dunning_level",
                "cust_avg_dunning_level", "cust_dunning_escalations",
                "cust_promises", "cust_kept", "cust_broken", "cust_promised_total",
            ])
            .filter(F.col("country").isNotNull())
    )

    return cust


# =========================================================
# 2. COUNTRY FEATURE VECTOR (18 behavioral + sector mix + exposure skew)
# =========================================================
def build_country_vector(cust_pdf, sector_col="region", n_sector_levels=8):
    rows = []
    sec_levels = (
        cust_pdf[sector_col].fillna("UNK").value_counts().head(n_sector_levels).index.tolist()
    )

    for country, g in cust_pdf.groupby("country"):
        n_cust = len(g)
        out = g["cust_total_exposure"].clip(lower=0).values
        exposure_skew = float(pd.Series(out).skew()) if n_cust > 2 else 0.0

        promises = g["cust_promises"].sum()
        kept = g["cust_kept"].sum()
        n_with_p2p = (g["cust_promises"] > 0).sum()

        rec = {
            "country": country,
            "n_customers": n_cust,
            # DPD shape
            "mean_dpd":           float(g["cust_mean_dpd"].mean()),
            "median_dpd":         float(g["cust_mean_dpd"].median()),
            "p90_dpd":            float(g["cust_p90_dpd"].mean()),
            "std_dpd":            float(g["cust_mean_dpd"].std() or 0.0),
            # timeliness
            "on_time_ratio":      float(g["cust_on_time_ratio"].mean()),
            "cleared_ratio":      float(g["cust_cleared_ratio"].mean()),
            # dunning
            "dunning_rate":       float((g["cust_dunning_events"] > 0).mean()),
            "avg_dunning_level":  float(g["cust_avg_dunning_level"].mean()),
            "escalation_rate":    float((g["cust_dunning_escalations"] > 0).mean()),
            # p2p
            "p2p_usage_rate":     float(n_with_p2p / max(n_cust, 1)),
            "p2p_keep_ratio":     float(kept / max(promises, 1)),
            "p2p_avg_amt":        float(g["cust_promised_total"].sum() / max(promises, 1)),
            # exposure
            "exposure_mean":      float(g["cust_total_exposure"].mean()),
            "exposure_median":    float(g["cust_total_exposure"].median()),
            "exposure_skew":      exposure_skew,
            "invoice_amt_mean":   float(g["cust_avg_invoice_amt"].mean()),
            "invoice_amt_median": float(g["cust_med_invoice_amt"].median()),
            # volume
            "invoices_per_cust":  float(g["cust_n_invoices"].mean()),
        }

        mix = g[sector_col].fillna("UNK").value_counts(normalize=True)
        for lvl in sec_levels:
            rec[f"mix_{lvl}"] = float(mix.get(lvl, 0.0))

        rows.append(rec)

    return pd.DataFrame(rows).set_index("country").fillna(0.0)


# =========================================================
# 3. STANDARDIZE + PCA DECORRELATE
# =========================================================
def standardize_and_pca(vec_df, variance=PCA_VARIANCE_RETAINED):
    feature_cols = [c for c in vec_df.columns if c != "n_customers"]
    X = vec_df[feature_cols].astype(float).values

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=variance, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(Xz)

    print(f"[pca] kept {Xp.shape[1]} components | explained var = "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    return Xz, Xp, scaler, pca, feature_cols


# =========================================================
# 4. OPTUNA AUTO-TUNE CLUSTERING
# =========================================================
def tune_clustering(vec_df, n_trials=N_TRIALS_CLUSTERING, k_min=2, k_max=8):
    """
    Search: method (ward/avg/complete/kmeans) × k × pca_var.
    Score = silhouette + 0.3*cophenetic − 0.15*imbalance.
    """
    feat_cols = [c for c in vec_df.columns if c != "n_customers"]
    Xz = StandardScaler().fit_transform(vec_df[feat_cols].astype(float).values)

    def objective(trial):
        method = trial.suggest_categorical("method",
                                           ["ward", "average", "complete", "kmeans"])
        k = trial.suggest_int("k", k_min, k_max)
        var = trial.suggest_float("pca_var", 0.70, 0.99)

        Xp = PCA(n_components=var, random_state=RANDOM_SEED).fit_transform(Xz)

        if method == "kmeans":
            labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(Xp) + 1
            coph = 0.0
        else:
            Z = linkage(Xp, method=method)
            labels = fcluster(Z, t=k, criterion="maxclust")
            coph, _ = cophenet(Z, pdist(Xp))

        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) < 2 or (counts < 2).any():
            return -1.0

        sil = silhouette_score(Xp, labels, metric="euclidean")
        balance = counts.min() / counts.max()
        return sil + 0.3 * coph - 0.15 * (1.0 - balance)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"[tune-cluster] best={best} score={study.best_value:.3f}")

    Xp = PCA(n_components=best["pca_var"], random_state=RANDOM_SEED).fit_transform(Xz)
    if best["method"] == "kmeans":
        labels = KMeans(n_clusters=best["k"], random_state=RANDOM_SEED, n_init=10).fit_predict(Xp) + 1
        Z = None
    else:
        Z = linkage(Xp, method=best["method"])
        labels = fcluster(Z, t=best["k"], criterion="maxclust")

    return {
        "study": study, "best_params": best, "Xp": Xp,
        "labels": labels, "linkage": Z, "countries": list(vec_df.index),
    }


# =========================================================
# 5. MULTI-METHOD VALIDATION BACKUP (no Optuna, exhaustive grid)
# =========================================================
def best_clustering(X, countries, k_range=N_CLUSTERS_RANGE):
    candidates = []
    pdist_full = pdist(X)

    for method in LINKAGE_METHODS:
        Z = linkage(X, method=method)
        coph_corr, _ = cophenet(Z, pdist_full)
        for k in range(k_range[0], k_range[1] + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            uniq, counts = np.unique(labels, return_counts=True)
            s = (silhouette_score(X, labels) if len(uniq) >= 2 and (counts >= 2).all()
                 else -1.0)
            candidates.append({
                "algo": f"HAC-{method}", "k": k, "silhouette": s,
                "cophenetic": coph_corr, "linkage": Z, "labels": labels,
            })

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X) + 1
        uniq, counts = np.unique(labels, return_counts=True)
        s = (silhouette_score(X, labels) if len(uniq) >= 2 and (counts >= 2).all()
             else -1.0)
        candidates.append({
            "algo": "KMeans", "k": k, "silhouette": s,
            "cophenetic": None, "linkage": None, "labels": labels,
        })

    df = pd.DataFrame([{k: v for k, v in c.items() if k not in ("linkage", "labels")}
                       for c in candidates])
    print("\n=== CLUSTERING CANDIDATES ===")
    print(df.sort_values("silhouette", ascending=False).to_string(index=False))

    winner = max(candidates, key=lambda c: c["silhouette"])
    cluster_pdf = pd.DataFrame(
        {"country": countries, "cluster_id": winner["labels"].astype(int)}
    )
    return winner, cluster_pdf, df


# =========================================================
# 6. DUAL-METRIC NEIGHBOR RANKING (Euclidean + Cosine)
# =========================================================
def neighbor_tables(X, countries, k=TOP_K_NEIGHBORS):
    eucl = squareform(pdist(X, metric="euclidean"))
    cos  = squareform(pdist(X, metric="cosine"))

    rows = []
    for i, a in enumerate(countries):
        for j, b in enumerate(countries):
            if i == j:
                continue
            rows.append((a, b, float(eucl[i, j]), float(cos[i, j])))
    dist_long = pd.DataFrame(rows, columns=["country", "neighbor", "dist_eucl", "dist_cos"])

    def topk(metric_col):
        out = (
            dist_long.sort_values(["country", metric_col])
            .groupby("country", as_index=False).head(k).copy()
        )
        out["rank"] = out.groupby("country").cumcount() + 1
        out["metric"] = metric_col
        return out

    top_eucl = topk("dist_eucl").rename(columns={"dist_eucl": "distance"})[
        ["country", "neighbor", "rank", "distance", "metric"]
    ]
    top_cos = topk("dist_cos").rename(columns={"dist_cos": "distance"})[
        ["country", "neighbor", "rank", "distance", "metric"]
    ]
    return dist_long, pd.concat([top_eucl, top_cos], ignore_index=True)


# =========================================================
# 7. BOOTSTRAP STABILITY (per-country neighbor agreement)
# =========================================================
def bootstrap_stability(
    cust_pdf, runs=BOOTSTRAP_RUNS, frac=BOOTSTRAP_SAMPLE_FRAC,
    sector_col="region", k=TOP_K_NEIGHBORS,
):
    seed_seq = np.random.SeedSequence(RANDOM_SEED).spawn(runs)
    countries = sorted(cust_pdf["country"].unique())
    baseline_nn = {}

    vec0 = build_country_vector(cust_pdf, sector_col=sector_col)
    _, X0, _, _, _ = standardize_and_pca(vec0)
    D0 = squareform(pdist(X0, metric="euclidean"))
    idx0 = list(vec0.index)
    for i, c in enumerate(idx0):
        order = np.argsort(D0[i])
        baseline_nn[c] = [idx0[j] for j in order if j != i][:k]

    agreement = {c: 0 for c in countries}
    runs_seen = {c: 0 for c in countries}

    for r in range(runs):
        sample_rng = np.random.default_rng(seed_seq[r])
        sample = (
            cust_pdf.groupby("country", group_keys=False)
            .apply(lambda g: g.sample(
                frac=frac, random_state=int(sample_rng.integers(0, 2**31 - 1))
            ))
        )
        if sample["country"].nunique() < 2:
            continue
        vec_r = build_country_vector(sample, sector_col=sector_col)
        if len(vec_r) < 2:
            continue
        try:
            _, Xr, _, _, _ = standardize_and_pca(vec_r)
        except Exception:
            continue
        Dr = squareform(pdist(Xr, metric="euclidean"))
        idx_r = list(vec_r.index)
        for i, c in enumerate(idx_r):
            order = np.argsort(Dr[i])
            nn_r = [idx_r[j] for j in order if j != i][:k]
            runs_seen[c] += 1
            if baseline_nn.get(c) and nn_r and nn_r[0] == baseline_nn[c][0]:
                agreement[c] += 1

    stab = pd.DataFrame([
        {
            "country": c,
            "baseline_top1": baseline_nn.get(c, [None])[0],
            "agreement_rate": (agreement[c] / runs_seen[c]) if runs_seen[c] else None,
            "runs": runs_seen[c],
        }
        for c in countries
    ]).sort_values("agreement_rate")

    print("\n=== BOOTSTRAP NEIGHBOR STABILITY ===")
    print(stab.to_string(index=False))
    return stab


# =========================================================
# 8. ROUTING TABLE (small → big donor)
# =========================================================
def routing_table(vec_df, neighbors_pdf, min_customers=MIN_CUSTOMERS_FOR_MODEL):
    big = set(vec_df[vec_df["n_customers"] >= min_customers].index)
    small = vec_df[vec_df["n_customers"] < min_customers].index.tolist()

    rows = []
    nbrs_eucl = neighbors_pdf[neighbors_pdf["metric"] == "dist_eucl"]
    for sc in small:
        cand = nbrs_eucl[
            (nbrs_eucl["country"] == sc) & (nbrs_eucl["neighbor"].isin(big))
        ].sort_values("distance")
        if cand.empty:
            rows.append((sc, int(vec_df.loc[sc, "n_customers"]), None, None, None, None))
        else:
            top = cand.iloc[0]
            ru = cand.iloc[1] if len(cand) > 1 else None
            rows.append((
                sc, int(vec_df.loc[sc, "n_customers"]),
                top["neighbor"], float(top["distance"]),
                ru["neighbor"] if ru is not None else None,
                float(ru["distance"]) if ru is not None else None,
            ))
    return pd.DataFrame(rows, columns=[
        "small_country", "n_customers", "donor_country", "donor_dist",
        "runner_up", "runner_up_dist",
    ]).sort_values("donor_dist")


# =========================================================
# 9. WARM-START XGB + OPTUNA AUTO-TUNE
# =========================================================
def warm_start_train(
    small_country_df, target_col, donor_model_uri,
    n_extra_rounds=100, learning_rate=0.03,
):
    """Vanilla warm-start (no tuning)."""
    import mlflow.sklearn
    from xgboost import XGBClassifier

    donor = mlflow.sklearn.load_model(donor_model_uri)
    booster = donor.get_booster()
    feat_names = booster.feature_names

    X = small_country_df[feat_names].astype("float64").fillna(0)
    y = small_country_df[target_col].astype(int)

    n_neg = (y == 0).sum()
    n_pos = max((y == 1).sum(), 1)

    warm = XGBClassifier(
        objective="binary:logistic", tree_method="hist", eval_metric="aucpr",
        n_estimators=n_extra_rounds, learning_rate=learning_rate,
        max_depth=donor.get_params().get("max_depth", 6),
        scale_pos_weight=n_neg / n_pos, random_state=RANDOM_SEED,
    )
    warm.fit(X, y, xgb_model=booster, verbose=False)
    print(f"Warm-started from {donor_model_uri} | +{n_extra_rounds} rounds on "
          f"{len(X):,} rows ({n_pos} positives)")
    return warm, feat_names


def tune_warm_start_xgb(
    small_country_df, target_col, donor_model_uri,
    valid_frac=0.2, n_trials=N_TRIALS_XGB,
):
    """Optuna-tuned warm-start. Objective = PR-AUC on stratified holdout."""
    import mlflow.sklearn
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    donor = mlflow.sklearn.load_model(donor_model_uri)
    booster = donor.get_booster()
    feat_names = booster.feature_names

    X = small_country_df[feat_names].astype("float64").fillna(0)
    y = small_country_df[target_col].astype(int)
    if y.nunique() < 2:
        raise ValueError("Target has only one class — cannot tune.")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=valid_frac, stratify=y, random_state=RANDOM_SEED,
    )
    n_neg = (y_tr == 0).sum()
    n_pos = max((y_tr == 1).sum(), 1)
    base_spw = n_neg / n_pos

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", base_spw * 0.5, base_spw * 2.0
            ),
        }
        model = XGBClassifier(
            objective="binary:logistic", tree_method="hist",
            eval_metric="aucpr", random_state=RANDOM_SEED, **params,
        )
        model.fit(X_tr, y_tr, xgb_model=booster, verbose=False)
        return float(average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[tune-xgb] best PR-AUC={study.best_value:.4f} params={study.best_params}")

    final = XGBClassifier(
        objective="binary:logistic", tree_method="hist",
        eval_metric="aucpr", random_state=RANDOM_SEED, **study.best_params,
    )
    final.fit(X, y, xgb_model=booster, verbose=False)

    metrics = {
        "best_pr_auc": float(study.best_value),
        "val_roc_auc": float(roc_auc_score(y_va, final.predict_proba(X_va)[:, 1])),
        "n_train": int(len(X)),
        "n_positive": int((y == 1).sum()),
    }
    return final, study, metrics


# =========================================================
# 10. BUSINESS DASHBOARD PLOTS
# =========================================================
def plot_dendrogram(Z, countries, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=countries, leaf_rotation=45, ax=ax)
    ax.set_title("Country Proximity — Hierarchical Clustering")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_distance_heatmap(X, countries, metric="euclidean", save_path=None):
    D = squareform(pdist(X, metric=metric))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(D, index=countries, columns=countries),
                annot=True, fmt=".2f", cmap="viridis_r", ax=ax)
    ax.set_title(f"Pairwise Country Distance ({metric}, lower = more similar)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_cluster_overview(vec_df, labels, save_path=None):
    df = vec_df[["n_customers"]].copy()
    df["cluster"] = labels
    df = df.sort_values(["cluster", "n_customers"], ascending=[True, False]).reset_index()

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("Set2", n_colors=df["cluster"].nunique())
    color_map = {c: palette[i] for i, c in enumerate(sorted(df["cluster"].unique()))}
    bars = ax.bar(df["country"], df["n_customers"],
                  color=[color_map[c] for c in df["cluster"]], edgecolor="black")
    for b, n in zip(bars, df["n_customers"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(n):,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Country Behavioral Groups (color = cluster)", fontsize=13)
    ax.set_ylabel("Customers"); ax.set_xlabel("")
    plt.xticks(rotation=45)
    handles = [mpatches.Patch(color=color_map[c], label=f"Cluster {c}")
               for c in sorted(color_map)]
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_donor_map(routing_df, stability_df=None, save_path=None):
    df = routing_df.dropna(subset=["donor_country"]).copy()
    if stability_df is not None:
        df = df.merge(stability_df[["country", "agreement_rate"]],
                      left_on="small_country", right_on="country", how="left").drop(columns="country")
    else:
        df["agreement_rate"] = np.nan
    df = df.sort_values("donor_dist").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(df))))
    for i, row in df.iterrows():
        conf = row.get("agreement_rate", np.nan)
        color = "lightgray" if pd.isna(conf) else plt.cm.RdYlGn(conf)
        ax.annotate("", xy=(1, i), xytext=(0, i),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2))
        ax.text(-0.02, i, row["small_country"], ha="right", va="center",
                fontsize=11, fontweight="bold")
        ax.text(1.02, i, row["donor_country"], ha="left", va="center",
                fontsize=11, fontweight="bold")
        ax.text(0.5, i + 0.18,
                f"d={row['donor_dist']:.2f}" + (
                    f" | conf={conf:.0%}" if not pd.isna(conf) else ""),
                ha="center", va="bottom", fontsize=9, color="dimgray")
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.5, len(df) - 0.3)
    ax.invert_yaxis(); ax.axis("off")
    ax.set_title("Donor Country Recommendation\n(arrow color = confidence)", fontsize=13)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_why_similar(vec_df, country_a, country_b, top_n=10, save_path=None):
    feat_cols = [c for c in vec_df.columns if c != "n_customers"]
    Xz = StandardScaler().fit_transform(vec_df[feat_cols].values)
    z = pd.DataFrame(Xz, index=vec_df.index, columns=feat_cols)

    diff = (z.loc[country_a] - z.loc[country_b]).abs().sort_values()
    similar = diff.head(top_n)
    different = diff.tail(top_n)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(similar.index, similar.values, color="seagreen", edgecolor="black")
    ax1.set_title(f"Where {country_a} ~ {country_b} ALIGN (small z-diff)")
    ax1.set_xlabel("|z-score difference|"); ax1.invert_yaxis()

    ax2.barh(different.index, different.values, color="indianred", edgecolor="black")
    ax2.set_title(f"Where {country_a} vs {country_b} DIFFER")
    ax2.set_xlabel("|z-score difference|"); ax2.invert_yaxis()

    plt.suptitle(f"Behavioral comparison: {country_a} vs {country_b}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_stability(stability_df, threshold=STABILITY_LOW_THRESHOLD, save_path=None):
    df = stability_df.dropna(subset=["agreement_rate"]).sort_values("agreement_rate")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(df))))
    colors = ["crimson" if r < threshold else "seagreen" for r in df["agreement_rate"]]
    bars = ax.barh(df["country"], df["agreement_rate"], color=colors, edgecolor="black")
    ax.axvline(threshold, color="black", linestyle="--", lw=1,
               label=f"trust line = {threshold:.0%}")
    for b, n, top1 in zip(bars, df["agreement_rate"], df["baseline_top1"]):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{n:.0%}  → {top1}", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_title("Neighbor Stability — Bootstrap Agreement Rate\n(red = unreliable)",
                 fontsize=13)
    ax.set_xlabel("Agreement across resamples"); ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_cluster_profile(vec_df, labels, save_path=None):
    feat_cols = [c for c in vec_df.columns if c != "n_customers"]
    z = pd.DataFrame(
        StandardScaler().fit_transform(vec_df[feat_cols].values),
        index=vec_df.index, columns=feat_cols,
    )
    z["cluster"] = labels
    prof = z.groupby("cluster")[feat_cols].mean().T

    fig, ax = plt.subplots(figsize=(min(14, 1.5 + 1.2 * prof.shape[1]),
                                    max(6, 0.32 * prof.shape[0])))
    sns.heatmap(prof, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                cbar_kws={"label": "z-score vs cross-country mean"}, ax=ax)
    ax.set_title("Cluster Profile — What Defines Each Group", fontsize=13)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_risk_landscape(Xp, vec_df, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("Set2", n_colors=len(set(labels)))
    sizes = 60 + 800 * (vec_df["n_customers"].values / vec_df["n_customers"].max())

    for cid in sorted(set(labels)):
        mask = np.array(labels) == cid
        ax.scatter(Xp[mask, 0], Xp[mask, 1], s=sizes[mask], color=palette[cid - 1],
                   alpha=0.75, edgecolor="black", linewidth=1.2,
                   label=f"Cluster {cid}")
    for i, c in enumerate(vec_df.index):
        ax.annotate(c, (Xp[i, 0], Xp[i, 1]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Behavioral PC1"); ax.set_ylabel("Behavioral PC2")
    ax.set_title("Country Risk Landscape\n(size = customer volume, color = cluster)",
                 fontsize=13)
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_feature_leaderboard(vec_df, feature, ascending=False, save_path=None):
    df = vec_df[[feature, "n_customers"]].sort_values(feature, ascending=ascending)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(df))))
    bars = ax.barh(df.index, df[feature], color="steelblue", edgecolor="black")
    for b, v in zip(bars, df[feature]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f"  {v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Country Ranking — {feature}", fontsize=13)
    ax.set_xlabel(feature)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_executive_cards(routing_df, vec_df, stability_df=None,
                         features_to_show=("mean_dpd", "on_time_ratio",
                                           "dunning_rate", "p2p_keep_ratio"),
                         save_path=None):
    df = routing_df.dropna(subset=["donor_country"]).copy()
    if stability_df is not None:
        df = df.merge(stability_df[["country", "agreement_rate"]],
                      left_on="small_country", right_on="country", how="left").drop(columns="country")
    else:
        df["agreement_rate"] = np.nan

    n = len(df)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        sc, dn = row["small_country"], row["donor_country"]
        conf = row.get("agreement_rate", np.nan)

        small_vals = vec_df.loc[sc, list(features_to_show)].values
        donor_vals = vec_df.loc[dn, list(features_to_show)].values
        x = np.arange(len(features_to_show)); w = 0.4
        ax.bar(x - w/2, small_vals, w, label=sc, color="#4C72B0", edgecolor="black")
        ax.bar(x + w/2, donor_vals, w, label=dn, color="#DD8452", edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(features_to_show, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9, loc="upper right")

        conf_txt = f"{conf:.0%}" if not pd.isna(conf) else "n/a"
        conf_color = ("crimson" if not pd.isna(conf) and conf < STABILITY_LOW_THRESHOLD
                      else "seagreen" if not pd.isna(conf) else "gray")
        ax.set_title(
            f"{sc}  →  {dn}\n"
            f"distance={row['donor_dist']:.2f}   confidence={conf_txt}",
            fontsize=11, color=conf_color, fontweight="bold",
        )

    for k in range(len(df), len(axes)):
        axes[k].axis("off")

    plt.suptitle("Donor Assignment — Executive Summary",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_tuning_diagnostics(study, title_prefix=""):
    from optuna.visualization.matplotlib import (
        plot_optimization_history, plot_param_importances,
    )
    fig1 = plot_optimization_history(study).figure
    fig1.suptitle(f"{title_prefix} Optimization History")
    plt.tight_layout(); plt.show()
    try:
        fig2 = plot_param_importances(study).figure
        fig2.suptitle(f"{title_prefix} Parameter Importance")
        plt.tight_layout(); plt.show()
    except Exception:
        fig2 = None
    return fig1, fig2


# =========================================================
# 11. ORCHESTRATION
# =========================================================
def run_full_dashboard(
    write=True,
    do_xgb_tuning=False,
    small_country_dfs=None,
    donor_uri_map=None,
    target_col="target",
    do_bootstrap=True,
):
    print("[1/8] Loading customer features (original SAP pipeline)...")
    cust_pdf = build_customer_features().toPandas()
    print(f"      {cust_pdf['country'].nunique()} countries | {len(cust_pdf):,} customers")

    print("[2/8] Building country vector...")
    vec = build_country_vector(cust_pdf)

    print("[3/8] Optuna tuning clustering...")
    tuned = tune_clustering(vec)
    labels = tuned["labels"]; Xp = tuned["Xp"]; countries = tuned["countries"]
    plot_tuning_diagnostics(tuned["study"], "Clustering —")

    print("[4/8] Dual-metric neighbor ranking...")
    dist_long, neighbors = neighbor_tables(Xp, countries)

    print("[5/8] Routing table (small -> donor)...")
    routing = routing_table(vec, neighbors)
    print(routing.to_string(index=False))

    stab = None
    if do_bootstrap:
        print("[6/8] Bootstrap neighbor stability...")
        stab = bootstrap_stability(cust_pdf)

    print("[7/8] Business dashboard plots...")
    if tuned["linkage"] is not None:
        plot_dendrogram(tuned["linkage"], countries)
    plot_distance_heatmap(Xp, countries)
    plot_cluster_overview(vec, labels)
    plot_donor_map(routing, stab)
    if stab is not None:
        plot_stability(stab)
    plot_cluster_profile(vec, labels)
    plot_risk_landscape(Xp, vec, labels)
    plot_executive_cards(routing, vec, stab)
    plot_feature_leaderboard(vec, "mean_dpd", ascending=False)
    plot_feature_leaderboard(vec, "p2p_keep_ratio", ascending=True)
    for _, r in routing.dropna(subset=["donor_country"]).iterrows():
        plot_why_similar(vec, r["small_country"], r["donor_country"])

    tuned_models = {}
    if do_xgb_tuning and small_country_dfs and donor_uri_map:
        print("[8/8] Optuna tuning warm-start XGB per small country...")
        for sc, df in small_country_dfs.items():
            donor_uri = donor_uri_map.get(sc)
            if not donor_uri:
                print(f"  [skip] no donor URI for {sc}"); continue
            model, study, metrics = tune_warm_start_xgb(df, target_col, donor_uri)
            plot_tuning_diagnostics(study, f"XGB {sc} —")
            tuned_models[sc] = {"model": model, "metrics": metrics, "study": study}

    if write:
        spark.createDataFrame(vec.reset_index()).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_FEATURES)
        spark.createDataFrame(dist_long).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_DISTANCES)
        spark.createDataFrame(neighbors).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_NEIGHBORS)
        spark.createDataFrame(
            pd.DataFrame({"country": countries, "cluster_id": labels.astype(int)})
        ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT_CLUSTERS)
        spark.createDataFrame(routing).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(OUT_ROUTING)
        if stab is not None:
            spark.createDataFrame(stab).write.mode("overwrite") \
                .option("overwriteSchema", "true").saveAsTable(OUT_STABILITY)
        print("[country_similarity] wrote Delta tables")

    return {
        "vec": vec, "Xp": Xp, "labels": labels, "countries": countries,
        "neighbors": neighbors, "routing": routing, "stability": stab,
        "cluster_study": tuned["study"], "tuned_xgb": tuned_models,
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage
# MAGIC
# MAGIC ```python
# MAGIC # Full pipeline + Optuna cluster tuning + 8-plot dashboard:
# MAGIC res = run_full_dashboard()
# MAGIC
# MAGIC # Include per-small-country warm-start XGB tuning:
# MAGIC small_dfs = {
# MAGIC     "NO": <pandas frame with feature_cols + target>,
# MAGIC     "FI": <pandas frame ...>,
# MAGIC }
# MAGIC donor_uris = {
# MAGIC     "NO": "models:/collection_risk_model_customer_v1_AT/latest",
# MAGIC     "FI": "models:/collection_risk_model_customer_v1_DE/latest",
# MAGIC }
# MAGIC res = run_full_dashboard(
# MAGIC     do_xgb_tuning=True,
# MAGIC     small_country_dfs=small_dfs,
# MAGIC     donor_uri_map=donor_uris,
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ## Plot guide for business stakeholders
# MAGIC | Plot | Question it answers |
# MAGIC |---|---|
# MAGIC | Dendrogram            | Tree of country proximity. |
# MAGIC | Distance heatmap      | Pairwise behavioral distance. |
# MAGIC | Cluster overview      | Which countries behave similarly? |
# MAGIC | Donor map             | Which big country should each small country borrow from? |
# MAGIC | Why-similar waterfall | Why is NO similar to AT? Where do they differ? |
# MAGIC | Stability dashboard   | How reliable is each proxy assignment? |
# MAGIC | Cluster profile       | What defines each group? |
# MAGIC | Risk landscape        | 2D PCA map, marker size = customer volume. |
# MAGIC | Feature leaderboard   | Who pays late? Who keeps promises? |
# MAGIC | Executive cards       | One-glance summary per low-data country. |
# MAGIC | Optuna diagnostics    | What hyperparameters drove the result? |
