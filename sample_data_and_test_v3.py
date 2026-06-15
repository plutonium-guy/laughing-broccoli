"""
sample_data_and_test_v3.py  —  LOCAL integration test (NOT a Databricks notebook).

Generates a synthetic feature table in the exact schema `data_pipeline_v3.py`
emits (panel snapshots, 13 countries across the 3 CLUSTERS, realistic signal +
imbalance, V2 features), then runs BOTH v3 stacks end-to-end on it:

  GLOBAL  : train_best_model_v3 logic  -> predict_best_model_v3 logic
  CLUSTER : train_all_clusters_v3 logic (real CalibratedSeedEnsemble + tiered
            tune) -> predict_clusters_v3 logic (route + align + score + SHAP)

The Databricks notebooks need spark/mlflow/the unified view, so this reproduces
their pandas/xgboost orchestration faithfully and imports the REAL estimator
(`cluster_xgb_ensemble.CalibratedSeedEnsemble`). Run locally:
    python sample_data_and_test_v3.py
Writes the sample data to /tmp/sample_data_v3/{train,infer}.csv for inspection.
"""
import os
import sys

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score, recall_score)
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_xgb_ensemble import CalibratedSeedEnsemble

optuna.logging.set_verbosity(optuna.logging.WARNING)
RNG = np.random.RandomState(7)

CLUSTERS = {"apac_big": ["CN", "AU", "JP", "NZ", "TW"],
            "kr_my": ["KR", "MY"], "sea": ["SG", "HK", "PH", "TH", "ID", "VN"]}
CLUSTER_OF = {c: n for n, cs in CLUSTERS.items() for c in cs}
FIXED = {"objective": "binary:logistic", "tree_method": "hist",
         "eval_metric": "aucpr", "random_state": 42}
N_TRIALS, N_SEEDS, EARLY = 8, 3, 25            # small for a fast local run
TARGET_RECALL = 0.80


# =========================================================
# 1. SAMPLE DATA — schema of collection_ml_features_*_v3
# =========================================================
def gen(country, n_cust, weeks, base_rate, signal):
    rows = []
    start = pd.Timestamp("2025-01-06")
    for cust in range(n_cust):
        cid = f"{country}{cust:04d}"
        # per-customer latent badness
        bad = RNG.beta(1.5, 4)
        for w in range(weeks):
            snap = start + pd.Timedelta(weeks=w)
            dpd = max(0, RNG.normal(bad * 220, 40))
            broken = RNG.binomial(6, bad * 0.5)
            kept = RNG.binomial(6, (1 - bad) * 0.5)
            dun = RNG.binomial(5, bad * 0.6)
            ontime = np.clip(RNG.normal(1 - bad, 0.15), 0, 1)
            outstanding = RNG.gamma(2, 5000) * (1 + bad)
            rows.append(dict(
                customer_id=cid, snapshot_date=snap, country=country,
                total_outstanding=outstanding,
                total_open_amount=outstanding * RNG.uniform(0.5, 1.0),
                num_open_invoices=RNG.randint(1, 20),
                max_dpd=dpd, avg_dpd=dpd * RNG.uniform(0.4, 0.8),
                amt_30_plus=outstanding * np.clip(bad + RNG.normal(0, .1), 0, 1),
                amt_60_plus=outstanding * np.clip(bad - .1 + RNG.normal(0, .1), 0, 1),
                amt_90_plus=outstanding * np.clip(bad - .2 + RNG.normal(0, .1), 0, 1),
                oldest_invoice_age=RNG.randint(10, 900),
                avg_invoice_age=RNG.randint(5, 400),
                avg_invoice_size=outstanding / max(RNG.randint(1, 20), 1),
                pct_30_plus=np.clip(bad + RNG.normal(0, .1), 0, 1),
                pct_60_plus=np.clip(bad - .1 + RNG.normal(0, .1), 0, 1),
                pct_90_plus=np.clip(bad - .2 + RNG.normal(0, .1), 0, 1),
                avg_days_to_pay=RNG.normal(bad * 90, 15),
                max_days_to_pay=RNG.normal(bad * 160, 25),
                on_time_ratio=ontime,
                total_payments=RNG.randint(0, 60),
                days_since_last_payment=max(0, RNG.normal(bad * 120, 30)),
                on_time_ratio_90d=np.clip(ontime + RNG.normal(0, .1), 0, 1),
                num_payments_90d=RNG.randint(0, 12),
                max_dunning_level=min(4, dun),
                total_dunning_events=dun,
                avg_dunning_level=dun * RNG.uniform(0.4, 1.0),
                high_severity_dunning=RNG.binomial(dun, 0.4) if dun else 0,
                high_dunning_ratio=np.clip(bad + RNG.normal(0, .1), 0, 1),
                dunning_events_90d=RNG.binomial(dun, 0.6) if dun else 0,
                total_promises=broken + kept,
                broken_promises=broken, kept_promises=kept,
                broken_ratio=broken / max(broken + kept, 1),
                kept_ratio=kept / max(broken + kept, 1),
                avg_promised_amount=RNG.gamma(2, 1500),
                promise_activity_flag=int((broken + kept) > 0),
                credit_limit=RNG.gamma(3, 8000),
                credit_utilization=np.clip(bad + RNG.normal(0, .15), 0, 2),
                number_of_disputes=RNG.binomial(3, bad * 0.4),
                open_dispute_amount=outstanding * RNG.uniform(0, 0.3) * bad,
                customer_tenure_days=RNG.randint(30, 3000),
                _bad=bad, _signal=signal, _base=base_rate,
            ))
    return rows


def build_sample():
    # heterogeneous clusters. ~45 weekly snapshots so the 30-day-embargo CV
    # forms >=2 folds (exercises real tuning). kr_my deliberately FEW customers
    # so its dev pool is < the full-tune threshold -> triggers the SMALL tier.
    sizes = {"CN": 13, "AU": 12, "JP": 12, "NZ": 11, "TW": 12,     # apac_big -> full
             "KR": 5, "MY": 4,                                     # kr_my -> small
             "SG": 12, "HK": 11, "PH": 10, "TH": 10, "ID": 9, "VN": 8}  # sea -> full
    sig = {"apac_big": 1.0, "kr_my": 1.2, "sea": 0.85}
    rows = []
    for c, n in sizes.items():
        rows += gen(c, n, weeks=45, base_rate=0.22, signal=sig[CLUSTER_OF[c]])
    df = pd.DataFrame(rows)
    # label: badness-driven, cluster-specific weight, thresholded -> imbalance
    z = (1.6 * (df["max_dpd"] / 220) + 1.2 * df["broken_ratio"]
         + 0.9 * df["high_dunning_ratio"] - 1.1 * df["on_time_ratio"]
         + 0.5 * df["credit_utilization"]) * df["_signal"] + RNG.normal(0, 0.4, len(df))
    df["target"] = (z > z.quantile(1 - df["_base"].iloc[0])).astype(int)
    df["collected_30d"] = df["total_outstanding"] * (1 - df["_bad"]) * RNG.uniform(0.5, 1.0, len(df))
    df["collection_ratio"] = df["collected_30d"] / df["total_outstanding"]
    df = df.drop(columns=["_bad", "_signal", "_base"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    train = df.sort_values("snapshot_date").reset_index(drop=True)
    # infer table = latest snapshot per customer, no label columns
    infer = (df.sort_values("snapshot_date").groupby("customer_id").tail(1)
             .drop(columns=["target", "collected_30d", "collection_ratio"])
             .reset_index(drop=True))
    infer["snapshot_date"] = train["snapshot_date"].max()
    return train, infer


DROP = ["customer_id", "snapshot_date", "country", "cluster",
        "collected_30d", "collection_ratio", "target"]


# =========================================================
# shared helpers (copied from the notebooks)
# =========================================================
def threshold_at_recall(y, p, min_recall=TARGET_RECALL):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= min_recall
    return float(thr[ok][int(np.argmax(prec[ok]))]) if ok.any() else float(thr[int(np.argmax(rec))])


def ts_splits(frame, tcol, n_splits=4, gap=30):
    dates = np.sort(frame[tcol].unique()); n = len(dates); fs = n // (n_splits + 1)
    if fs == 0:
        return []
    out = []
    for i in range(1, n_splits + 1):
        te = dates[fs * i - 1]; vs = te + np.timedelta64(gap, "D")
        ve = dates[n - 1] if i == n_splits else dates[min(fs * i + fs, n - 1)]
        tr = (frame[tcol] <= te).values
        va = ((frame[tcol] >= vs) & (frame[tcol] <= ve)).values
        if va.sum() == 0 or frame.loc[tr, "target"].nunique() < 2:
            continue
        out.append((np.where(tr)[0], np.where(va)[0]))
    return out


def shap_top(model, X, k=3):
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    sv = model.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
    cols = np.array(X.columns)
    order = np.argsort(-np.abs(sv), axis=1)[:, :k]
    return [", ".join(cols[row]) for row in order]


# =========================================================
# 2. GLOBAL STACK
# =========================================================
def run_global(train, infer):
    print("\n" + "=" * 60 + "\nGLOBAL STACK\n" + "=" * 60)
    pdf = train.copy()
    pdf["cluster"] = pdf["country"].map(CLUSTER_OF).fillna("other")
    for c in sorted(pdf["country"].unique()):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
    for cl in sorted(pdf["cluster"].unique()):
        pdf[f"cluster_{cl}"] = (pdf["cluster"] == cl).astype(int)
    feats = [c for c in pdf.columns if c not in DROP]

    qv = pdf["snapshot_date"].quantile(0.85, "lower")
    dev = pdf[pdf["snapshot_date"] <= qv].reset_index(drop=True)
    test = pdf[pdf["snapshot_date"] > qv]
    splits = ts_splits(dev, "snapshot_date")
    base_spw = (dev["target"] == 0).sum() / max((dev["target"] == 1).sum(), 1)
    print(f"rows={len(pdf)} feats={len(feats)} pos={pdf['target'].mean()*100:.1f}% "
          f"folds={len(splits)} base_spw={base_spw:.2f}")

    def objective(trial):
        p = {**FIXED, "max_depth": trial.suggest_int("max_depth", 3, 8),
             "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
             "reg_lambda": trial.suggest_float("reg_lambda", 1, 20, log=True)}
        mult = trial.suggest_float("spw", 0.6, 1.6)
        sc = []
        for tr_i, va_i in splits:
            d, v = dev.iloc[tr_i], dev.iloc[va_i]
            spw = (d["target"] == 0).sum() / max((d["target"] == 1).sum(), 1) * mult
            m = XGBClassifier(**p, n_estimators=400, scale_pos_weight=spw,
                              early_stopping_rounds=EARLY)
            m.fit(d[feats], d["target"], eval_set=[(v[feats], v["target"])], verbose=False)
            sc.append(average_precision_score(v["target"], m.predict_proba(v[feats])[:, 1]))
        return float(np.mean(sc) - 0.5 * np.std(sc))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4))
    study.optimize(objective, n_trials=N_TRIALS)
    bp = dict(study.best_params); mult = bp.pop("spw")
    spw = base_spw * mult

    # dev model -> best_n + threshold (last fold as val)
    d, v = dev.iloc[splits[-1][0]], dev.iloc[splits[-1][1]]
    dev_m = XGBClassifier(**FIXED, **bp, n_estimators=400, scale_pos_weight=spw,
                          early_stopping_rounds=EARLY)
    dev_m.fit(d[feats], d["target"], eval_set=[(v[feats], v["target"])], verbose=False)
    best_n = int(dev_m.best_iteration) + 1
    thr = threshold_at_recall(v["target"], dev_m.predict_proba(v[feats])[:, 1])
    tp = dev_m.predict_proba(test[feats])[:, 1]
    print(f"tuned cv={study.best_value:.3f} best_n={best_n} thr={thr:.3f} "
          f"| TEST pr_auc={average_precision_score(test['target'], tp):.3f} "
          f"roc_auc={roc_auc_score(test['target'], tp):.3f} "
          f"recall={recall_score(test['target'], (tp>=thr).astype(int)):.2f}")

    # final refit on ALL
    full_spw = (pdf["target"] == 0).sum() / max((pdf["target"] == 1).sum(), 1) * mult
    final = XGBClassifier(**FIXED, **bp, n_estimators=best_n, scale_pos_weight=full_spw)
    final.fit(pdf[feats], pdf["target"], verbose=False)

    # inference
    ip = infer.copy()
    ip["cluster"] = ip["country"].map(CLUSTER_OF).fillna("other")
    for c in sorted(ip["country"].unique()):
        ip[f"country_{c}"] = (ip["country"] == c).astype(int)
    for cl in sorted(ip["cluster"].unique()):
        ip[f"cluster_{cl}"] = (ip["cluster"] == cl).astype(int)
    bf = list(final.get_booster().feature_names)
    for f in bf:
        if f not in ip.columns:
            ip[f] = 0.0
    X = ip[bf].astype("float64")
    ip["risk_score"] = final.predict_proba(X)[:, 1]
    ip["binary_pred"] = (ip["risk_score"] >= thr).astype(int)
    ip["top_drivers"] = shap_top(final, X)
    assert ip["risk_score"].between(0, 1).all()
    print(f"INFER scored {len(ip)} | HIGH={int(ip['binary_pred'].sum())} "
          f"({ip['binary_pred'].mean()*100:.1f}%)")
    print(ip[["customer_id", "country", "risk_score", "binary_pred", "top_drivers"]]
          .sort_values("risk_score", ascending=False).head(3).to_string(index=False))
    return ip


# =========================================================
# 3. CLUSTER STACK  (real CalibratedSeedEnsemble + tiered tune)
# =========================================================
ENS_FIXED = {k: v for k, v in FIXED.items() if k != "random_state"}


def resolve_monotone(cols):
    def s(n):
        if n.startswith("on_time_ratio") or n.startswith("kept"):
            return -1
        if (n.startswith(("max_dpd", "avg_dpd", "avg_days_to_pay", "max_days_to_pay", "broken"))
                or (n.startswith(("amt_", "pct_")) and n.endswith("plus"))
                or n in ("oldest_invoice_age", "avg_invoice_age", "credit_utilization",
                         "days_since_last_payment", "number_of_disputes", "open_dispute_amount")
                or "dunning" in n):
            return 1
        return 0
    return {c: s(c) for c in cols}


def train_one_cluster(name, members, all_pdf):
    pdf = all_pdf[all_pdf["country"].isin(members)].sort_values("snapshot_date").reset_index(drop=True)
    for c in sorted(members):
        pdf[f"country_{c}"] = (pdf["country"] == c).astype(int)
    feats = [c for c in pdf.columns if c not in DROP]
    if len(pdf) < 100 or pdf["target"].sum() < 10:
        print(f"[{name}] SKIP (too small: {len(pdf)} rows)")
        return None
    qt = pdf["snapshot_date"].quantile(0.85, "lower")
    dev, test = pdf[pdf["snapshot_date"] <= qt].reset_index(drop=True), pdf[pdf["snapshot_date"] > qt]
    splits = ts_splits(dev, "snapshot_date")
    sign = resolve_monotone(feats)
    mono = tuple(sign[c] for c in feats)

    # tiered tune: full vs small by size (mirrors tune_or_fallback)
    full = len(dev) >= 500 and dev["target"].sum() >= 100 and len(splits) >= 2
    mode = "full" if full else ("small" if len(splits) >= 2 else "fixed")
    mcw_hi = int(np.clip(dev["target"].sum() / 10, 2, 20))   # cap so splits form

    def obj(trial):
        if mode == "small":
            p = {**ENS_FIXED, "grow_policy": "depthwise",
                 "max_depth": trial.suggest_int("md", 2, 3),
                 "learning_rate": trial.suggest_float("lr", 0.02, 0.08, log=True),
                 "min_child_weight": trial.suggest_int("mcw", 1, mcw_hi),
                 "gamma": trial.suggest_float("g", 0.0, 2.0),
                 "reg_lambda": trial.suggest_float("rl", 1, 15, log=True)}
        else:
            p = {**ENS_FIXED, "max_depth": trial.suggest_int("md", 2, 6),
                 "learning_rate": trial.suggest_float("lr", 0.02, 0.15, log=True),
                 "min_child_weight": trial.suggest_int("mcw", 3, 30),
                 "reg_lambda": trial.suggest_float("rl", 1, 30, log=True)}
        p["monotone_constraints"] = mono
        mult = trial.suggest_float("spw", 0.6, 1.3)
        sc = []
        for tr_i, va_i in splits:
            d, v = dev.iloc[tr_i], dev.iloc[va_i]
            spw = (d["target"] == 0).sum() / max((d["target"] == 1).sum(), 1) * mult
            m = XGBClassifier(**p, n_estimators=300, scale_pos_weight=spw, early_stopping_rounds=EARLY)
            m.fit(d[feats], d["target"], eval_set=[(v[feats], v["target"])], verbose=False)
            sc.append(average_precision_score(v["target"], m.predict_proba(v[feats])[:, 1]))
        pen = 1.0 if mode == "small" else 0.5
        return float(np.mean(sc) - pen * np.std(sc))

    if mode == "fixed":
        bp, mult = {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 10, "reg_lambda": 5.0}, 1.0
    else:
        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4))
        st.optimize(obj, n_trials=N_TRIALS if full else 6)
        bp = dict(st.best_params); mult = bp.pop("spw")
        bp = {"max_depth": bp["md"], "learning_rate": bp["lr"],
              "min_child_weight": bp["mcw"], "reg_lambda": bp["rl"]}

    cv = dev["snapshot_date"].quantile(0.85, "lower")
    tr, va = dev[dev["snapshot_date"] <= cv], dev[dev["snapshot_date"] > cv]
    if va["target"].nunique() < 2:
        tr, va = dev.iloc[:-max(int(len(dev)*.15), 20)], dev.iloc[-max(int(len(dev)*.15), 20):]
    spw = (tr["target"] == 0).sum() / max((tr["target"] == 1).sum(), 1) * mult
    sign_sel = {c: sign[c] for c in feats}

    cal = CalibratedSeedEnsemble(params={**ENS_FIXED, **bp}, n_estimators=300,
                                 scale_pos_weight=spw, sign_map=sign_sel, n_seeds=N_SEEDS)
    cal.fit_base(tr[feats], tr["target"], eval_set=[(va[feats], va["target"])], early_stopping_rounds=EARLY)
    cal.fit_calibrator(va[feats], va["target"])
    best_n = int(np.median(cal.best_iterations_))
    thr = threshold_at_recall(va["target"], cal.predict_proba(va[feats])[:, 1])
    cthr = {}
    for c in members:
        vc = va[va["country"] == c]
        cthr[c] = (threshold_at_recall(vc["target"], cal.predict_proba(vc[feats])[:, 1])
                   if vc["target"].sum() >= 10 and vc["target"].nunique() > 1 else thr)
    tp = cal.predict_proba(test[feats])[:, 1]
    prod = CalibratedSeedEnsemble(params={**ENS_FIXED, **bp}, n_estimators=best_n,
                                  scale_pos_weight=spw, sign_map=sign_sel, n_seeds=N_SEEDS)
    prod.fit_base(pdf[feats], pdf["target"])
    prod.set_calibrator(cal)

    # DEGENERACY GUARD (mirrors the trainer): rescue zero-importance models
    fi = prod.feature_importances_
    if float(np.sum(fi)) == 0.0:
        print(f"[{name}] WARN zero feature importance -> rescue")
        rescue = {"grow_policy": "depthwise", "max_depth": 3, "learning_rate": 0.05,
                  "min_child_weight": 1, "gamma": 0.0, "reg_lambda": 1.0}
        prod = CalibratedSeedEnsemble(params={**ENS_FIXED, **rescue}, n_estimators=max(best_n, 25),
                                      scale_pos_weight=spw, sign_map=sign_sel, n_seeds=N_SEEDS)
        prod.fit_base(pdf[feats], pdf["target"]); prod.set_calibrator(cal)
        fi = prod.feature_importances_
    top3 = sorted(zip(feats, fi), key=lambda kv: -kv[1])[:3]
    assert float(np.sum(fi)) > 0.0, f"[{name}] STILL zero feature importance"
    print(f"[{name}] mode={mode} rows={len(pdf)} pos={int(pdf['target'].sum())} "
          f"calib={cal.calib_kind_} best_n={best_n} thr={thr:.2f} "
          f"| TEST pr_auc={average_precision_score(test['target'], tp):.3f} "
          f"recall={recall_score(test['target'], (tp>=thr).astype(int)):.2f}")
    print(f"      top3 importance: {[(f, round(float(v), 3)) for f, v in top3]}")
    return dict(name=name, members=members, model=prod, feats=feats,
                cthr=cthr, monotone=sum(1 for c in feats if sign_sel[c]))


def run_cluster(train, infer):
    print("\n" + "=" * 60 + "\nCLUSTER STACK\n" + "=" * 60)
    results = [train_one_cluster(n, m, train) for n, m in CLUSTERS.items()]
    results = [r for r in results if r]
    routing = {c: r for r in results for c in r["members"]}
    COUNTRIES_OF = {r["name"]: sorted(r["members"]) for r in results}

    out = []
    for r in results:
        sub = infer[infer["country"].isin(r["members"])].copy()
        if not len(sub):
            continue
        for c in COUNTRIES_OF[r["name"]]:
            sub[f"country_{c}"] = (sub["country"] == c).astype(int)
        bf = list(r["model"].get_booster().feature_names)
        for f in bf:
            if f not in sub.columns:
                sub[f] = 0.0
        X = sub[bf].astype("float64")
        sub["risk_score"] = r["model"].predict_proba(X)[:, 1]
        sub["threshold_used"] = sub["country"].map(r["cthr"])
        sub["binary_pred"] = (sub["risk_score"] >= sub["threshold_used"]).astype(int)
        sub["cluster"] = r["name"]
        sub["top_drivers"] = shap_top(r["model"], X)
        assert sub["risk_score"].between(0, 1).all()
        out.append(sub)
    preds = pd.concat(out, ignore_index=True)
    assert len(preds) == len(infer), f"scored {len(preds)} != {len(infer)} customers"
    print(f"\nINFER scored {len(preds)} customers across {len(results)} clusters | "
          f"HIGH={int(preds['binary_pred'].sum())} ({preds['binary_pred'].mean()*100:.1f}%)")
    print(preds.groupby("cluster")["binary_pred"].agg(["size", "sum"]).to_string())
    print("\ntop 3 riskiest:")
    print(preds[["customer_id", "country", "cluster", "risk_score", "binary_pred", "top_drivers"]]
          .sort_values("risk_score", ascending=False).head(3).to_string(index=False))
    return preds


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train, infer = build_sample()
    os.makedirs("/tmp/sample_data_v3", exist_ok=True)
    train.to_csv("/tmp/sample_data_v3/train.csv", index=False)
    infer.to_csv("/tmp/sample_data_v3/infer.csv", index=False)
    print(f"SAMPLE DATA: train={train.shape} infer={infer.shape} "
          f"| {train['country'].nunique()} countries, target={train['target'].mean()*100:.1f}% pos")
    print(f"  per-cluster rows: "
          + ", ".join(f"{cl}={train['country'].map(CLUSTER_OF).eq(cl).sum()}" for cl in CLUSTERS))
    print("  saved -> /tmp/sample_data_v3/{train,infer}.csv")

    g = run_global(train, infer)
    c = run_cluster(train, infer)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED — both stacks ran end-to-end on sample data")
    print(f"  global infer rows={len(g)} | cluster infer rows={len(c)}")
