# Databricks notebook source
# MAGIC %md
# MAGIC # Train BEST Global Model (V3) — Optuna full search
# MAGIC
# MAGIC NEW FILE (standing rule #1: no edits to existing `.py`). A single global
# MAGIC XGBoost model over ALL countries, consuming the feature table written by
# MAGIC `data_pipeline_v3.py` (run that first, with `INCLUDE_V2_FEATURES=True` —
# MAGIC now the default).
# MAGIC
# MAGIC **Pipeline**
# MAGIC   1. Read `TRAIN_FEATURE_TABLE` (= `collection_ml_features_train_v3`).
# MAGIC   2. Country one-hots; drop ids/leakage columns.
# MAGIC   3. **Time-based split** train / valid / test (most recent = test) —
# MAGIC      snapshot panel data, so a time split avoids look-ahead leakage.
# MAGIC   4. **Optuna** over the FULL XGBoost hyperparameter space
# MAGIC      (depth/leaves/learning-rate/all subsamples/all regularizers/
# MAGIC      grow_policy/max_delta_step), maximizing valid **PR-AUC**.
# MAGIC   5. **Imbalance**: `scale_pos_weight = (neg/pos) × tuned multiplier`.
# MAGIC   6. **Early stopping** on the valid fold every trial (n_estimators is
# MAGIC      effectively tuned via `best_iteration`, not searched).
# MAGIC   7. Honest **test** report from the dev-trained model + operating
# MAGIC      threshold (max precision @ recall ≥ TARGET_RECALL).
# MAGIC   8. **Final model refit on ALL data** (train+valid+test) with the best
# MAGIC      params and `n_estimators` locked to the dev model's `best_iteration`.
# MAGIC   9. Log + register to MLflow.
# MAGIC
# MAGIC Convention (unchanged): `target=1 == HIGH RISK`; risk_score = P(target=1).

# COMMAND ----------

!pip install -U optuna xgboost mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn

from pyspark.sql.types import DecimalType, DoubleType

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix, precision_recall_curve,
    log_loss, brier_score_loss, accuracy_score,
)

# COMMAND ----------

# =========================================================
# CONFIG
# =========================================================
# Feature table written by data_pipeline_v3.py (run it first, INCLUDE_V2_FEATURES=True)
TRAIN_FEATURE_TABLE = "f_erp_glide_o2c_12.collection_ml_features_train_v3"

LABEL = "target"
TIME_COL = "snapshot_date"

# Country -> cluster map (same grouping as data_pipeline_v3 / train_all_clusters_v3).
# Used to add cluster one-hots as features ALONGSIDE the country one-hots: a single
# tree split can peel off a whole behavioral cluster (cheaper than N country splits),
# letting rare countries borrow strength while country dummies keep the detail.
CLUSTERS = {
    "apac_big": ["CN", "AU", "JP", "NZ", "TW"],
    "kr_my":    ["KR", "MY"],
    "sea":      ["SG", "HK", "PH", "TH", "ID", "VN"],
}
CLUSTER_OF = {c.upper(): name for name, cs in CLUSTERS.items() for c in cs}

# Non-feature columns to exclude from X (ids + label + leakage + raw group strings)
DROP_COLS = ["customer_id", "snapshot_date", "country", "cluster",
             "collected_30d", "collection_ratio", "target"]

# Time-based holdout (most recent snapshots = test). Train is the remainder.
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

# Optuna
N_TRIALS = 100
RANDOM_STATE = 42

# Walk-forward time-series CV for the objective (robust across periods, not
# tuned to one valid slice). CV runs over DEV (= train+valid); the TEST holdout
# stays untouched for the honest final eval.
CV_SPLITS = 4              # expanding-window folds over DEV
EMBARGO_DAYS = 30          # gap between each fold's train and val (= label window)
STABILITY_PENALTY = 0.5    # objective = mean(PR-AUC) - penalty * std across folds

# Boosting / early stopping
MAX_BOOST_ROUNDS = 3000        # upper bound; early stopping picks the real n
EARLY_STOPPING_ROUNDS = 50

# Operating point for reporting (the score stays continuous either way)
TARGET_RECALL = 0.80

# MLflow
REGISTER_MODEL = True
MODEL_NAME = "collection_risk_model_global_v3"
EXPERIMENT_PATH = ("/Workspace/Users/amiya.x.mandal@gsk.com/APEC/exp/"
                   "collection_risk_global_v3")

FIXED_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}

# COMMAND ----------

# =========================================================
# LOAD — feature table -> pandas, country one-hots, time split
# =========================================================

def spark_to_pandas_safe(spark_df):
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, spark_df[field.name].cast(DoubleType()))
    return spark_df.fillna(0).toPandas()


pdf = spark_to_pandas_safe(spark.table(TRAIN_FEATURE_TABLE))
assert LABEL in pdf.columns, f"{TRAIN_FEATURE_TABLE} has no '{LABEL}' column — run data_pipeline_v3.py (MODE train)"
pdf[TIME_COL] = pd.to_datetime(pdf[TIME_COL])
pdf = pdf.sort_values(TIME_COL).reset_index(drop=True)

# Country one-hots (global model sees the country it scores)
if "country" in pdf.columns:
    for c in sorted(pdf["country"].dropna().unique()):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)

    # Cluster one-hots (coarse behavioral grouping; complements country one-hots)
    pdf["cluster"] = pdf["country"].map(CLUSTER_OF).fillna("other")
    for cl in sorted(pdf["cluster"].unique()):
        pdf[f"cluster_{cl}"] = (pdf["cluster"] == cl).astype(int)
    n_country = pdf["country"].nunique()
    n_cluster = pdf["cluster"].nunique()
    print(f"one-hots: {n_country} country + {n_cluster} cluster "
          f"({sorted(pdf['cluster'].unique())})")
    if (pdf["cluster"] == "other").any():
        miss = sorted(pdf.loc[pdf["cluster"] == "other", "country"].unique())
        print(f"  WARN: countries not in any CLUSTER -> 'other': {miss}")

feature_cols = [c for c in pdf.columns if c not in DROP_COLS]
print(f"rows={len(pdf):,} | features={len(feature_cols)} | "
      f"positives={int(pdf[LABEL].sum()):,} ({pdf[LABEL].mean()*100:.1f}%)")

# Time-based split: oldest -> train, middle -> valid, newest -> test
q_train = pdf[TIME_COL].quantile(1 - VALID_FRACTION - TEST_FRACTION, interpolation="lower")
q_valid = pdf[TIME_COL].quantile(1 - TEST_FRACTION, interpolation="lower")
train = pdf[pdf[TIME_COL] <= q_train]
valid = pdf[(pdf[TIME_COL] > q_train) & (pdf[TIME_COL] <= q_valid)]
test = pdf[pdf[TIME_COL] > q_valid]
dev = pdf[pdf[TIME_COL] <= q_valid].reset_index(drop=True)   # train+valid, for CV
for name, part in [("train", train), ("valid", valid), ("test", test)]:
    print(f"  {name}: rows={len(part):,} positives={int(part[LABEL].sum()):,} "
          f"({part[LABEL].mean()*100:.1f}%) "
          f"[{part[TIME_COL].min().date()} .. {part[TIME_COL].max().date()}]")

assert train[LABEL].nunique() > 1 and valid[LABEL].nunique() > 1, \
    "train/valid is single-class — widen the date range or adjust fractions"

Xtr, ytr = train[feature_cols], train[LABEL]
Xva, yva = valid[feature_cols], valid[LABEL]
Xte, yte = test[feature_cols], test[LABEL]

# Imbalance baseline (neg/pos on train)
BASE_SPW = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
print(f"base scale_pos_weight (neg/pos) = {BASE_SPW:.2f}")

# COMMAND ----------

# =========================================================
# OPTUNA — full hyperparameter search, WALK-FORWARD CV objective
# =========================================================

def suggest_params(trial):
    grow = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    p = {
        "grow_policy": grow,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
    }
    if grow == "lossguide":
        p["max_depth"] = trial.suggest_int("max_depth_lossguide", 0, 12)   # 0 = unbounded
        p["max_leaves"] = trial.suggest_int("max_leaves", 15, 255, log=True)
    else:
        p["max_depth"] = trial.suggest_int("max_depth_depthwise", 3, 10)
    return p


def params_from_best(best):
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
        "reg_alpha": b.pop("reg_alpha"),
        "reg_lambda": b.pop("reg_lambda"),
        "max_delta_step": b.pop("max_delta_step"),
    }
    if grow == "lossguide":
        p["max_depth"] = b.pop("max_depth_lossguide")
        p["max_leaves"] = b.pop("max_leaves")
    else:
        p["max_depth"] = b.pop("max_depth_depthwise")
    return p, spw_mult


def build_time_series_splits(frame, n_splits=CV_SPLITS, gap_days=EMBARGO_DAYS):
    """Expanding-window walk-forward folds: train on all rows up to a cutoff,
    validate on the next slice, with a `gap_days` embargo between them so the
    30-day label window can't leak across the train/val boundary."""
    dates = np.sort(frame[TIME_COL].unique())
    n = len(dates)
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        return []
    splits = []
    for i in range(1, n_splits + 1):
        train_end = dates[fold_size * i - 1]
        val_start = train_end + np.timedelta64(gap_days, "D")
        val_end = dates[n - 1] if i == n_splits \
            else dates[min(fold_size * i + fold_size, n - 1)]
        tr = (frame[TIME_COL] <= train_end).values
        va = ((frame[TIME_COL] >= val_start) & (frame[TIME_COL] <= val_end)).values
        if va.sum() == 0 or frame.loc[tr, LABEL].nunique() < 2:
            continue
        splits.append((np.where(tr)[0], np.where(va)[0]))
    return splits


DEV_SPLITS = build_time_series_splits(dev)
assert len(DEV_SPLITS) >= 2, \
    f"only {len(DEV_SPLITS)} CV folds — widen the date range or lower CV_SPLITS"
print(f"walk-forward folds: {len(DEV_SPLITS)} (embargo {EMBARGO_DAYS}d)")
for i, (tr_idx, va_idx) in enumerate(DEV_SPLITS):
    print(f"  fold{i+1}: train={len(tr_idx):,} val={len(va_idx):,} "
          f"[val {dev.iloc[va_idx][TIME_COL].min().date()} .. "
          f"{dev.iloc[va_idx][TIME_COL].max().date()}]")


def objective(trial):
    params = suggest_params(trial)
    mult = trial.suggest_float("scale_pos_weight_mult", 0.5, 2.0)
    scores = []
    for step, (tr_idx, va_idx) in enumerate(DEV_SPLITS):
        dtr, dva = dev.iloc[tr_idx], dev.iloc[va_idx]
        # imbalance handled per fold (each fold's train has its own base rate)
        fold_spw = float((dtr[LABEL] == 0).sum()) / max(int((dtr[LABEL] == 1).sum()), 1) * mult
        model = XGBClassifier(**FIXED_PARAMS, **params,
                              n_estimators=MAX_BOOST_ROUNDS, scale_pos_weight=fold_spw,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(dtr[feature_cols], dtr[LABEL],
                  eval_set=[(dva[feature_cols], dva[LABEL])], verbose=False)
        scores.append(average_precision_score(
            dva[LABEL], model.predict_proba(dva[feature_cols])[:, 1]))
        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    # robust objective: penalize fold-to-fold variance
    return float(np.mean(scores) - STABILITY_PENALTY * np.std(scores))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True,
                                       group=True, n_startup_trials=20),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"Optuna: {N_TRIALS} trials, walk-forward CV "
      f"(mean PR-AUC - {STABILITY_PENALTY}*std)...")
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

best_params, spw_mult = params_from_best(study.best_params)
best_spw = BASE_SPW * spw_mult
print(f"\nbest CV objective (mean PR-AUC - {STABILITY_PENALTY}*std) = {study.best_value:.4f}")
print(f"best params = {best_params}")
print(f"scale_pos_weight = {best_spw:.2f} (mult={spw_mult:.2f})")

# COMMAND ----------

# =========================================================
# DEV MODEL — fit on train, early-stop on valid, lock best_n, honest test eval
# =========================================================

dev_model = XGBClassifier(**FIXED_PARAMS, **best_params,
                          n_estimators=MAX_BOOST_ROUNDS, scale_pos_weight=best_spw,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
dev_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
best_n = int(dev_model.best_iteration) + 1 if dev_model.best_iteration is not None \
    else MAX_BOOST_ROUNDS
print(f"early stopping picked n_estimators = {best_n}")


def threshold_at_recall(y_true, y_prob, min_recall=TARGET_RECALL):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= min_recall
    if not ok.any():
        return float(thr[int(np.argmax(rec))])
    return float(thr[ok][int(np.argmax(prec[ok]))])


va_prob = dev_model.predict_proba(Xva)[:, 1]
threshold = threshold_at_recall(yva, va_prob)

te_prob = dev_model.predict_proba(Xte)[:, 1]
te_pred = (te_prob >= threshold).astype(int)
test_metrics = {
    "test_pr_auc": float(average_precision_score(yte, te_prob)),
    "test_roc_auc": float(roc_auc_score(yte, te_prob)),
    "test_precision": float(precision_score(yte, te_pred, zero_division=0)),
    "test_recall": float(recall_score(yte, te_pred, zero_division=0)),
    "test_f1": float(f1_score(yte, te_pred, zero_division=0)),
    "test_f2": float(fbeta_score(yte, te_pred, beta=2, zero_division=0)),
}
print(f"\nthreshold (max precision @ recall>={TARGET_RECALL}) = {threshold:.3f}")
for k, v in test_metrics.items():
    print(f"  {k} = {v:.4f}")
print("confusion (rows=actual Low/High, cols=pred):")
print(confusion_matrix(yte, te_pred))

# top-15 features by gain
imp = pd.Series(dev_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\ntop 15 features:")
print(imp.head(15).to_string())

# COMMAND ----------

# =========================================================
# FINAL MODEL — refit on ALL data with best params + locked n_estimators
# =========================================================

X_all = pdf[feature_cols]
y_all = pdf[LABEL]
full_spw = float((y_all == 0).sum()) / max(int((y_all == 1).sum()), 1) * spw_mult

final_model = XGBClassifier(**FIXED_PARAMS, **best_params,
                            n_estimators=best_n, scale_pos_weight=full_spw)
final_model.fit(X_all, y_all, verbose=False)   # no early stopping: n is locked
print(f"FINAL model trained on ALL {len(pdf):,} rows "
      f"(n_estimators={best_n}, scale_pos_weight={full_spw:.2f})")

# COMMAND ----------

# =========================================================
# MLflow — log params, test metrics, register the final (all-data) model
# =========================================================
mlflow.set_registry_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)


def log_eval_metrics(y_true, y_prob, threshold, prefix):
    """Log a full classification metric set under <prefix>_*: ranking (pr_auc,
    roc_auc), calibration (logloss, brier), operating point (precision/recall/
    f1/f2/accuracy/specificity), confusion (tn/fp/fn/tp) and volumes."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    n, n_pos = len(y_true), int(y_true.sum())
    out = {"n": n, "n_pos": n_pos, "pos_rate": n_pos / max(n, 1),
           "n_flagged": int(y_pred.sum())}
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_prob)
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["logloss"] = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
        out["brier"] = brier_score_loss(y_true, y_prob)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["f2"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    out["accuracy"] = accuracy_score(y_true, y_pred)
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "specificity": tn / max(tn + fp, 1)})
    for k, v in out.items():
        mlflow.log_metric(f"{prefix}_{k}", float(v))
    return out


def log_feature_importance(model, cols, top=15):
    """Full importance vector as a JSON artifact, the top-N as a readable param,
    and the top-10 as metrics (so runs are comparable in the MLflow UI)."""
    imp = {c: float(v) for c, v in zip(cols, model.feature_importances_)}
    mlflow.log_dict(imp, "feature_importances.json")
    ranked = sorted(imp.items(), key=lambda kv: -kv[1])
    mlflow.log_param("top_features", ", ".join(f"{k}={v:.3f}" for k, v in ranked[:top]))
    mlflow.log_metric("n_features_used", int(sum(v > 0 for v in imp.values())))
    for k, v in ranked[:10]:
        mlflow.log_metric(f"imp_{k}", v)


with mlflow.start_run(run_name="train_best_model_v3") as run:
    # --- config / tuning params ---
    mlflow.log_param("feature_table", TRAIN_FEATURE_TABLE)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_param("cv_splits", CV_SPLITS)
    mlflow.log_param("best_n_estimators", best_n)
    mlflow.log_param("scale_pos_weight", round(full_spw, 3))
    mlflow.log_param("chosen_threshold", round(threshold, 4))
    mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
    mlflow.log_param("hp_scale_pos_weight_mult", round(spw_mult, 3))

    # --- dataset shape / class balance per split ---
    mlflow.log_metric("data_rows", len(pdf))
    mlflow.log_metric("data_pos_rate", float(pdf[LABEL].mean()))
    for nm, part in [("train", train), ("valid", valid), ("test", test)]:
        mlflow.log_metric(f"{nm}_rows", len(part))
        mlflow.log_metric(f"{nm}_pos", int(part[LABEL].sum()))

    # --- tuning objective + valid/test metric sets (threshold picked on valid) ---
    mlflow.log_metric("cv_objective", float(study.best_value))
    log_eval_metrics(yva, dev_model.predict_proba(Xva)[:, 1], threshold, "valid")
    log_eval_metrics(yte, te_prob, threshold, "test")

    # --- feature importance of the deployed (all-data) model ---
    log_feature_importance(final_model, feature_cols)

    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        input_example=Xte[feature_cols].head(5),
    )
    if REGISTER_MODEL:
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model",
                              name=MODEL_NAME)
        print(f"registered: {MODEL_NAME}")

print("\n=== DONE (train_best_model_v3) ===")
print(f"cv objective={study.best_value:.4f} | "
      f"test PR-AUC={test_metrics['test_pr_auc']:.4f} "
      f"ROC-AUC={test_metrics['test_roc_auc']:.4f} | "
      f"final n={best_n} on {len(pdf):,} rows -> {MODEL_NAME}")
