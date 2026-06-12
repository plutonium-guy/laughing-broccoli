# Databricks notebook source
# MAGIC %md
# MAGIC # Train ALL Cluster Models (V3 — XGBoost optimization pack)
# MAGIC
# MAGIC NEW FILE (standing rule #1: no edits to `train_all_clusters_v2.py`).
# MAGIC **Reads the feature table from `data_pipeline_v3.py`** (`TRAIN_FEATURE_TABLE`,
# MAGIC run it first with MODE=train, INCLUDE_V2_FEATURES=True) — no inline feature
# MAGIC engineering. One CalibratedSeedEnsemble per cluster + a routing table.
# MAGIC

# MAGIC **Four optimizations over V2:**
# MAGIC   1. **Monotonic constraints** — domain priors via XGBoost
# MAGIC      `monotone_constraints`: ↑max_dpd / ↑dunning / ↑broken_promises ⇒
# MAGIC      ↑risk, ↑on_time_ratio / ↑kept_promises ⇒ ↓risk. Regularizes the
# MAGIC      tiny per-country pools and makes the score defensible.
# MAGIC   2. **Wider Optuna search** — grow_policy depthwise/lossguide +
# MAGIC      max_leaves, colsample_bylevel/bynode, data-adaptive max_depth,
# MAGIC      50→120 trials.
# MAGIC   3. **Probability calibration** — isotonic / Platt on the held-out val
# MAGIC      slice; undoes the `scale_pos_weight` score inflation so the routing
# MAGIC      thresholds mean what they say. Monotone ⇒ ranking (and the
# MAGIC      inference `risk_band` qcut) is unchanged.
# MAGIC   4. **Seed-ensemble** — K boosters averaged on the raw margin, variance
# MAGIC      cut on small pools.
# MAGIC
# MAGIC **Inference is untouched.** Models register as `CalibratedSeedEnsemble`
# MAGIC (in `cluster_xgb_ensemble.py`), which duck-types the XGBClassifier
# MAGIC surface `predict_all_clusters_v2.py` already calls (`predict_proba`,
# MAGIC `get_booster().feature_names`, `get_booster().predict(..., pred_contribs)`).
# MAGIC Registered with `code_paths=[the module]` so MLflow re-imports the class
# MAGIC on `load_model`. Routing is written to a **staging** table by default;
# MAGIC flip `WRITE_PROD_ROUTING=True` to point production inference at V3.

# COMMAND ----------

!pip install -U optuna xgboost mlflow seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# =========================================================
# INPUT — paste your clusters here
# =========================================================
CLUSTERS = {
    "apac_big":   ["CN", "AU", "JP", "NZ", "TW"],
    "kr_my":      ["KR", "MY"],
    "sea":        ["SG", "HK", "PH", "TH", "ID", "VN"],
}

ALL_COUNTRIES = sorted({c.upper() for cs in CLUSTERS.values() for c in cs})
assert len(ALL_COUNTRIES) == sum(len(v) for v in CLUSTERS.values()), \
    "Same country appears in two clusters — fix CLUSTERS"
print(f"{len(CLUSTERS)} clusters, {len(ALL_COUNTRIES)} countries: {ALL_COUNTRIES}")

# COMMAND ----------

import os
import sys

import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve,
)

# --- the V3 estimator (new importable module, side-effect free) ----------
# Point CODE_PATH_MODULE at the module's path; it is both imported here for
# training AND shipped with each MLflow model (code_paths) so inference can
# unpickle the class without any edit to predict_all_clusters_v2.py.
CODE_PATH_MODULE = "cluster_xgb_ensemble.py"
_MODULE_DIR = os.path.dirname(os.path.abspath(CODE_PATH_MODULE)) or "."
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
from cluster_xgb_ensemble import CalibratedSeedEnsemble

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
# Feature table from data_pipeline_v3.py (run first: MODE=train,
# INCLUDE_V2_FEATURES=True). Replaces the inline PART A feature engineering —
# single source of truth for train/serve parity. Feature semantics + universe
# now live in data_pipeline_v3.py (original logic on the unified view).
TRAIN_FEATURE_TABLE = "f_erp_glide_o2c_12.collection_ml_features_train_v3"

# Anchor used only to stamp the routing table's refresh_date.
TODAY = "2026-06-07"

TEST_FRACTION = 0.15
CV_SPLITS = 4
LEAKAGE_GAP_DAYS = 30

MIN_ROWS_TO_TUNE = 500
MIN_POSITIVES_TO_TUNE = 100
N_TRIALS = 120                       # V3: was 50 — wider space needs more trials
STABILITY_PENALTY = 0.5

EARLY_STOPPING_ROUNDS = 30
TARGET_RECALL = 0.80
MIN_TEST_POSITIVES = 20

# V2.1 optimizations (kept)
IMPORTANCE_CUTOFF = 0.003
MIN_FEATURES_AFTER_SELECT = 10
MIN_VAL_POSITIVES_PER_COUNTRY = 10
USE_RECENCY_WEIGHTS = True
RECENCY_HALF_LIFE_DAYS = 180
REFIT_ON_FULL_DATA = True

# ---- V3 optimization knobs ----
N_SEEDS = 5                          # seed-ensemble size (variance reduction)
CALIBRATION_METHOD = "auto"          # "auto" | "isotonic" | "sigmoid"
USE_MONOTONE = True                  # domain-prior constraints

MODEL_NAME_PREFIX = "collection_risk_model_cluster_v3"
EXPERIMENT_PATH = ("/Workspace/Users/amiya.x.mandal@gsk.com/APEC/exp/"
                   "collection_risk_all_clusters_v3")

# Routing: write a staging table by default; only overwrite the production
# routing table (the one inference reads) when you explicitly opt in.
ROUTING_TABLE = "f_erp_glide_o2c_12.collection_ml_country_model_map"
ROUTING_TABLE_STAGING = ROUTING_TABLE + "_v3_staging"
WRITE_PROD_ROUTING = False           # True => inference serves V3 models

CONSERVATIVE_PARAMS = {
    "grow_policy": "depthwise",
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "colsample_bynode": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "max_delta_step": 3,
}

FIXED_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_estimators": 500,
}
# Params handed to CalibratedSeedEnsemble must NOT carry n_estimators /
# random_state / scale_pos_weight — the class owns those per seed.
ENSEMBLE_FIXED = {k: v for k, v in FIXED_PARAMS.items()
                  if k not in ("n_estimators", "random_state")}

# COMMAND ----------

# =========================================================
# PART B — V3 TRAINING MACHINERY (monotone + wide search + calib + ensemble)
# =========================================================

DROP_COLS = ["customer_id", "snapshot_date", "target",
             "collected_30d", "collection_ratio", "country"]


def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


def resolve_monotone(feature_cols):
    """Domain priors for `monotone_constraints`. target=1 == HIGH risk, so
    +1 = 'more of this ⇒ more risk', -1 = 'more ⇒ less risk', 0 = free.
    Conservative: only unambiguous-direction features are pinned; exposure
    magnitudes, raw counts and country one-hots stay unconstrained."""
    if not USE_MONOTONE:
        return {c: 0 for c in feature_cols}

    def sign(n):
        # -1 : more of it ⇒ LOWER risk
        if n.startswith("on_time_ratio"):                                     return -1
        if n.startswith("kept"):                                              return -1
        # +1 : more of it ⇒ HIGHER risk
        if n.startswith("max_dpd") or n.startswith("avg_dpd"):                return 1
        if n.startswith("amt_") and n.endswith("plus"):                       return 1
        if n.startswith("pct_") and n.endswith("plus"):                       return 1
        if n in ("oldest_invoice_age", "avg_invoice_age"):                    return 1
        if n == "credit_utilization":                                         return 1
        if n.startswith("avg_days_to_pay") or n.startswith("max_days_to_pay"):return 1
        if n == "days_since_last_payment":                                    return 1
        if n in ("number_of_disputes", "open_dispute_amount"):               return 1
        if "dunning" in n:                                                    return 1
        if n.startswith("broken"):                                            return 1
        return 0

    return {c: sign(c) for c in feature_cols}


def monotone_tuple(sign_map, feature_cols):
    """Aligned to the given column order; None if nothing is constrained."""
    t = tuple(int(sign_map.get(c, 0)) for c in feature_cols)
    return t if any(t) else None


def build_time_series_splits(pdf, n_splits=CV_SPLITS, gap_days=LEAKAGE_GAP_DAYS):
    dates = np.sort(pdf["snapshot_date"].unique())
    n = len(dates)
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        return []

    splits = []
    for i in range(1, n_splits + 1):
        train_end = dates[fold_size * i - 1]
        val_start = train_end + pd.Timedelta(days=gap_days)
        val_end = dates[n - 1] if i == n_splits \
            else dates[min(fold_size * i + fold_size, n - 1)]

        train_mask = pdf["snapshot_date"] <= train_end
        val_mask = (pdf["snapshot_date"] >= val_start) & (pdf["snapshot_date"] <= val_end)
        if val_mask.sum() == 0 or pdf.loc[train_mask, "target"].nunique() < 2:
            continue
        splits.append((pdf[train_mask].index, pdf[val_mask].index))
    return splits


def recency_weights(dates, as_of):
    """Exponential time-decay sample weights — halve every
    RECENCY_HALF_LIFE_DAYS of snapshot age. None when disabled."""
    if not USE_RECENCY_WEIGHTS:
        return None
    age = (as_of - dates).dt.days.clip(lower=0)
    return np.power(0.5, age / RECENCY_HALF_LIFE_DAYS).values


def macro_country_score(model, val_pdf, feature_cols):
    scores = []
    for c, grp in val_pdf.groupby("country"):
        if grp["target"].nunique() < 2:
            continue
        prob = model.predict_proba(grp[feature_cols])[:, 1]
        scores.append(average_precision_score(grp["target"], prob))
    if not scores:
        prob = model.predict_proba(val_pdf[feature_cols])[:, 1]
        return average_precision_score(val_pdf["target"], prob)
    return float(np.mean(scores))


def make_objective(pdf, feature_cols, splits, sign_map, max_depth_cap):
    """V3 wide search. Single representative booster per fold for tuning speed
    (the seed-ensemble is built once, with the winning params). Monotone
    constraints active during tuning so the search optimizes the constrained
    model it will actually ship."""
    as_of = pdf["snapshot_date"].max()
    mono = monotone_tuple(sign_map, feature_cols)

    def objective(trial):
        grow = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        params = {
            **FIXED_PARAMS,
            "grow_policy": grow,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 6),
        }
        if grow == "lossguide":
            params["max_depth"] = trial.suggest_int("max_depth_lossguide", 0, max_depth_cap)
            params["max_leaves"] = trial.suggest_int("max_leaves", 7, 127, log=True)
        else:
            params["max_depth"] = trial.suggest_int("max_depth_depthwise", 2, max_depth_cap)
        if mono is not None:
            params["monotone_constraints"] = mono
        spw_mult = trial.suggest_float("scale_pos_weight_mult", 0.5, 1.3)

        fold_scores = []
        for step, (tr_idx, va_idx) in enumerate(splits):
            tr, va = pdf.loc[tr_idx], pdf.loc[va_idx]
            spw = ((tr["target"] == 0).sum() / max((tr["target"] == 1).sum(), 1)) * spw_mult

            model = XGBClassifier(**params, scale_pos_weight=spw,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(tr[feature_cols], tr["target"],
                      sample_weight=recency_weights(tr["snapshot_date"], as_of),
                      eval_set=[(va[feature_cols], va["target"])], verbose=False)

            fold_scores.append(macro_country_score(model, va, feature_cols))
            trial.report(float(np.mean(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores) - STABILITY_PENALTY * np.std(fold_scores))

    return objective


def xgb_params_from_optuna(best):
    """Translate Optuna's flat param dict back into XGBoost kwargs (resolving
    the per-grow-policy depth params and popping the spw multiplier)."""
    b = dict(best)
    spw_mult = b.pop("scale_pos_weight_mult", 1.0)
    grow = b.pop("grow_policy", "depthwise")
    p = {
        "grow_policy": grow,
        "learning_rate": b.pop("learning_rate"),
        "subsample": b.pop("subsample"),
        "colsample_bytree": b.pop("colsample_bytree"),
        "colsample_bylevel": b.pop("colsample_bylevel"),
        "colsample_bynode": b.pop("colsample_bynode"),
        "min_child_weight": b.pop("min_child_weight"),
        "gamma": b.pop("gamma"),
        "reg_lambda": b.pop("reg_lambda"),
        "reg_alpha": b.pop("reg_alpha"),
        "max_delta_step": b.pop("max_delta_step"),
    }
    if grow == "lossguide":
        p["max_depth"] = b.pop("max_depth_lossguide")
        p["max_leaves"] = b.pop("max_leaves")
    else:
        p["max_depth"] = b.pop("max_depth_depthwise")
    return p, spw_mult


def tune_or_fallback(dev_pdf, feature_cols, splits, sign_map, max_depth_cap, cluster_name):
    n_pos = int(dev_pdf["target"].sum())
    if len(dev_pdf) < MIN_ROWS_TO_TUNE or n_pos < MIN_POSITIVES_TO_TUNE or len(splits) < 2:
        print(f"  [{cluster_name}] GATE FAILED ({len(dev_pdf)} rows, {n_pos} pos, "
              f"{len(splits)} folds) — conservative params, no Optuna")
        return dict(CONSERVATIVE_PARAMS), 1.0, None

    print(f"  [{cluster_name}] tuning: {N_TRIALS} trials on {len(dev_pdf):,} rows "
          f"(max_depth_cap={max_depth_cap})...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True,
                                           n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(make_objective(dev_pdf, feature_cols, splits, sign_map, max_depth_cap),
                   n_trials=N_TRIALS, gc_after_trial=True)

    best_params, spw_mult = xgb_params_from_optuna(study.best_params)
    print(f"  [{cluster_name}] best objective={study.best_value:.4f} | {best_params}")
    return best_params, spw_mult, study


def threshold_for_precision_at_recall(y_true, y_prob, min_recall=TARGET_RECALL):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= min_recall
    if not ok.any():
        return float(thr[int(np.argmax(rec))])
    return float(thr[ok][int(np.argmax(prec[ok]))])


def eval_slice(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
    }


def train_one_cluster(cluster_name, countries, all_pdf):
    """Full V3 training for one cluster. Returns result bundle."""
    print(f"\n================ {cluster_name}: {countries} ================")
    pdf = all_pdf[all_pdf["country"].isin(countries)].copy()
    pdf = pdf.sort_values("snapshot_date").reset_index(drop=True)
    for c in sorted(countries):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
    feature_cols = [c for c in pdf.columns if c not in DROP_COLS]

    n_pos = int(pdf["target"].sum())
    print(f"  rows={len(pdf):,} positives={n_pos} ({pdf['target'].mean()*100:.1f}%)")
    if len(pdf) < 100 or n_pos < 10:
        print("  SKIPPED — too small to train anything honest "
              "(need >=100 rows, >=10 positives). Route to a neighbor cluster.")
        return None

    cut_test = pdf["snapshot_date"].quantile(1 - TEST_FRACTION, interpolation="lower")
    dev = pdf[pdf["snapshot_date"] <= cut_test].reset_index(drop=True)
    test = pdf[pdf["snapshot_date"] > cut_test].reset_index(drop=True)

    splits = build_time_series_splits(dev)

    # V3: data-adaptive depth cap — deeper trees only when the pool supports it
    max_depth_cap = 6 if len(dev) >= 1500 else (5 if len(dev) >= 600 else 4)

    sign_map = resolve_monotone(feature_cols)
    n_pin = sum(1 for v in sign_map.values() if v != 0)
    print(f"  monotone: {n_pin}/{len(feature_cols)} features pinned "
          f"(+1={sum(v==1 for v in sign_map.values())}, "
          f"-1={sum(v==-1 for v in sign_map.values())})")

    best_params, spw_mult, study = tune_or_fallback(
        dev, feature_cols, splits, sign_map, max_depth_cap, cluster_name)

    # inner train/val split for early stopping, calibration and thresholds
    cut_val = dev["snapshot_date"].quantile(0.85, interpolation="lower")
    train = dev[dev["snapshot_date"] <= cut_val]
    val = dev[dev["snapshot_date"] > cut_val]
    if val["target"].nunique() < 2:
        n_val = max(int(len(dev) * 0.15), 20)
        train, val = dev.iloc[:-n_val], dev.iloc[-n_val:]

    as_of = pdf["snapshot_date"].max()
    spw = ((train["target"] == 0).sum() / max((train["target"] == 1).sum(), 1)) * spw_mult
    w_train = recency_weights(train["snapshot_date"], as_of)

    # --- feature selection: one monotone booster, drop near-zero importance ---
    sel = XGBClassifier(**FIXED_PARAMS, **best_params, scale_pos_weight=spw,
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        monotone_constraints=monotone_tuple(sign_map, feature_cols))
    sel.fit(train[feature_cols], train["target"], sample_weight=w_train,
            eval_set=[(val[feature_cols], val["target"])], verbose=False)
    keep = [f for f, imp in zip(feature_cols, sel.feature_importances_)
            if imp >= IMPORTANCE_CUTOFF]
    if len(keep) >= MIN_FEATURES_AFTER_SELECT and len(keep) < len(feature_cols):
        print(f"  feature selection: {len(feature_cols)} -> {len(keep)}")
        feature_cols = keep
    sign_map_sel = {c: sign_map[c] for c in feature_cols}

    # --- calibration ensemble: K seeds on train, isotonic/Platt on val ---
    cal_ens = CalibratedSeedEnsemble(
        params={**ENSEMBLE_FIXED, **best_params},
        n_estimators=FIXED_PARAMS["n_estimators"],
        scale_pos_weight=spw, sign_map=sign_map_sel, n_seeds=N_SEEDS)
    cal_ens.fit_base(train[feature_cols], train["target"], sample_weight=w_train,
                     eval_set=[(val[feature_cols], val["target"])],
                     early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    cal_ens.fit_calibrator(val[feature_cols], val["target"], method=CALIBRATION_METHOD)
    best_n = int(np.median(cal_ens.best_iterations_))
    print(f"  ensemble: {N_SEEDS} seeds | calibration={cal_ens.calib_kind_} "
          f"| best_n(median)={best_n}")

    # cluster threshold on CALIBRATED val probs
    val_prob = cal_ens.predict_proba(val[feature_cols])[:, 1]
    threshold = threshold_for_precision_at_recall(val["target"], val_prob)

    # per-country thresholds (calibrated)
    country_thresholds = {}
    for c in countries:
        vc = val[val["country"] == c]
        if (vc["target"].sum() >= MIN_VAL_POSITIVES_PER_COUNTRY
                and vc["target"].nunique() > 1):
            pcv = cal_ens.predict_proba(vc[feature_cols])[:, 1]
            country_thresholds[c] = threshold_for_precision_at_recall(vc["target"], pcv)
        else:
            country_thresholds[c] = threshold

    # honest test eval on the calibration ensemble (trained on train, never
    # saw val/test); per country at its own threshold
    test_prob = cal_ens.predict_proba(test[feature_cols])[:, 1]
    rows = [{"cluster": cluster_name, "slice": "POOLED", "threshold": threshold,
             **eval_slice(test["target"], pd.Series(test_prob, index=test.index), threshold)}]
    for c in countries:
        tc = test[test["country"] == c]
        if len(tc) == 0:
            continue
        pc = cal_ens.predict_proba(tc[feature_cols])[:, 1]
        rows.append({"cluster": cluster_name, "slice": c,
                     "threshold": country_thresholds[c],
                     **eval_slice(tc["target"], pd.Series(pc, index=tc.index),
                                  country_thresholds[c])})
    report = pd.DataFrame(rows)

    pooled = report[report["slice"] == "POOLED"].iloc[0]
    if pooled["n_pos"] < MIN_TEST_POSITIVES:
        print(f"  WARN: {int(pooled['n_pos'])} test positives — metrics directional only")
    print(f"  test: recall={pooled['recall']:.2f} precision={pooled['precision']:.2f} "
          f"pr_auc={pooled['pr_auc']:.3f} @ t={threshold:.2f}")

    # --- production ensemble: refit K seeds on FULL data with locked best_n,
    #     reuse the calibrator learned on the honest held-out val slice ---
    fit_pdf = pdf if REFIT_ON_FULL_DATA else dev
    spw_prod = ((fit_pdf["target"] == 0).sum()
                / max((fit_pdf["target"] == 1).sum(), 1)) * spw_mult
    production_model = CalibratedSeedEnsemble(
        params={**ENSEMBLE_FIXED, **best_params},
        n_estimators=best_n, scale_pos_weight=spw_prod,
        sign_map=sign_map_sel, n_seeds=N_SEEDS)
    production_model.fit_base(fit_pdf[feature_cols], fit_pdf["target"],
                              sample_weight=recency_weights(fit_pdf["snapshot_date"], as_of))
    production_model.set_calibrator(cal_ens)

    return {
        "cluster": cluster_name,
        "countries": countries,
        "model": production_model,
        "eval_model": cal_ens,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "country_thresholds": country_thresholds,
        "best_params": best_params,
        "spw_mult": spw_mult,
        "tuned": study is not None,
        "n_estimators": best_n,
        "n_seeds": N_SEEDS,
        "calibration": cal_ens.calib_kind_,
        "n_monotone": sum(1 for c in feature_cols if sign_map_sel.get(c, 0) != 0),
        "report": report,
        "test": test,
        "test_prob": test_prob,
        "n_rows": len(pdf),
        "production_rows": len(fit_pdf),
    }

# COMMAND ----------

# =========================================================
# RUN — load once, train every cluster
# =========================================================

training_df = spark.table(TRAIN_FEATURE_TABLE).filter(F.col("country").isin(ALL_COUNTRIES))

all_pdf = spark_to_pandas_safe(training_df)
all_pdf["snapshot_date"] = pd.to_datetime(all_pdf["snapshot_date"])
print(all_pdf.groupby("country")["target"].agg(rows="size", positives="sum"))

# COMMAND ----------

mlflow.set_registry_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)

results = []
with mlflow.start_run(run_name="train_all_clusters_v3") as parent:
    mlflow.log_param("clusters", {k: ",".join(v) for k, v in CLUSTERS.items()})
    mlflow.log_param("n_seeds", N_SEEDS)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_param("use_monotone", USE_MONOTONE)
    mlflow.log_param("calibration_method", CALIBRATION_METHOD)

    for cluster_name, countries in CLUSTERS.items():
        countries = [c.upper() for c in countries]
        with mlflow.start_run(run_name=f"cluster_{cluster_name}", nested=True) as child:
            res = train_one_cluster(cluster_name, countries, all_pdf)
            if res is None:
                mlflow.log_param("skipped", True)
                continue

            mlflow.log_param("countries", ",".join(countries))
            mlflow.log_param("tuned", res["tuned"])
            mlflow.log_params({f"hp_{k}": v for k, v in res["best_params"].items()})
            mlflow.log_param("hp_scale_pos_weight_mult", res["spw_mult"])
            mlflow.log_param("chosen_threshold", res["threshold"])
            mlflow.log_param("country_thresholds", res["country_thresholds"])
            mlflow.log_param("n_selected_features", len(res["feature_cols"]))
            mlflow.log_param("n_monotone_features", res["n_monotone"])
            mlflow.log_param("production_n_estimators", res["n_estimators"])
            mlflow.log_param("n_seeds", res["n_seeds"])
            mlflow.log_param("calibration", res["calibration"])
            mlflow.log_param("production_rows", res["production_rows"])
            mlflow.log_param("recency_weights", USE_RECENCY_WEIGHTS)
            pooled = res["report"][res["report"]["slice"] == "POOLED"].iloc[0]
            for m in ["recall", "precision", "f2", "pr_auc", "roc_auc"]:
                if not np.isnan(pooled[m]):
                    mlflow.log_metric(f"test_{m}", float(pooled[m]))

            model_name = f"{MODEL_NAME_PREFIX}_{cluster_name}"
            # code_paths ships the CalibratedSeedEnsemble class WITH the model so
            # inference (mlflow.sklearn.load_model) can unpickle it untouched.
            mlflow.sklearn.log_model(
                sk_model=res["model"],
                artifact_path="model",
                code_paths=[CODE_PATH_MODULE],
                input_example=res["test"][res["feature_cols"]].head(5),
            )
            mlflow.register_model(model_uri=f"runs:/{child.info.run_id}/model",
                                  name=model_name)
            res["model_name"] = model_name
            print(f"  registered: {model_name}")
            results.append(res)

# COMMAND ----------

# =========================================================
# COMBINED REPORT + ROUTING MAP
# =========================================================

full_report = pd.concat([r["report"] for r in results], ignore_index=True)
display(full_report.round(3))

routing = pd.DataFrame([
    {"country": c, "cluster": r["cluster"], "model_name": r["model_name"],
     "threshold": r["country_thresholds"][c],
     "cluster_threshold": r["threshold"],
     "trained_rows": r["n_rows"],
     "n_seeds": r["n_seeds"], "calibration": r["calibration"],
     "tuned": r["tuned"], "refresh_date": TODAY}
    for r in results for c in r["countries"]
])
display(routing)

if len(routing):
    # always publish a staging table for inspection / comparison vs V2
    spark.createDataFrame(routing).write.format("delta") \
        .mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(ROUTING_TABLE_STAGING)
    print(f"Routing map written to STAGING {ROUTING_TABLE_STAGING}")

    if WRITE_PROD_ROUTING:
        # flips production inference to the V3 models — opt-in only
        spark.createDataFrame(routing).write.format("delta") \
            .mode("overwrite").option("overwriteSchema", "true") \
            .saveAsTable(ROUTING_TABLE)
        print(f"Routing map written to PROD {ROUTING_TABLE} — inference now serves V3")
    else:
        print(f"PROD routing NOT written (WRITE_PROD_ROUTING=False). "
              f"Set True to point inference at V3.")

# COMMAND ----------

# =========================================================
# SUMMARY PLOTS
# =========================================================

def plot_summary(full_report):
    d = full_report[full_report["slice"] != "POOLED"].copy()
    pooled = full_report[full_report["slice"] == "POOLED"].set_index("cluster")
    clusters = list(pooled.index)
    cluster_color = {c: plt.cm.tab10(i % 10) for i, c in enumerate(clusters)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m in zip(axes, ["recall", "precision", "pr_auc"]):
        colors = [cluster_color[c] for c in d["cluster"]]
        ax.bar(d["slice"], d[m], color=colors)
        for cl in clusters:
            ax.axhline(pooled.loc[cl, m], color=cluster_color[cl],
                       linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(f"test {m} per country (dashes = cluster pooled)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        for i, (v, p) in enumerate(zip(d[m], d["n_pos"])):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}\n(p={int(p)})", ha="center", va="bottom", fontsize=7)
    plt.suptitle("All Cluster Models V3 — Per-Country Test Performance",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_all_confusions(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        y_pred = (r["test_prob"] >= r["threshold"]).astype(int)
        cm = confusion_matrix(r["test"]["target"], y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"])
        ax.set_title(f"{r['cluster']}\nt={r['threshold']:.2f} "
                     f"({'tuned' if r['tuned'] else 'fixed'}, {r['n_seeds']}x)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


plot_summary(full_report)
plot_all_confusions(results)

# COMMAND ----------

print("\n=== DONE (V3) ===")
for r in results:
    pooled = r["report"][r["report"]["slice"] == "POOLED"].iloc[0]
    print(f"{r['cluster']:>12} {str(r['countries']):<40} "
          f"tuned={r['tuned']} seeds={r['n_seeds']} calib={r['calibration']} "
          f"t={r['threshold']:.2f} recall={pooled['recall']:.2f} "
          f"precision={pooled['precision']:.2f} pr_auc={pooled['pr_auc']:.3f} "
          f"-> {r['model_name']}")
skipped = set(CLUSTERS) - {r["cluster"] for r in results}
if skipped:
    print(f"\nSKIPPED (too small): {sorted(skipped)} — route to neighbor cluster")
