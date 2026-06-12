"""
cluster_xgb_ensemble.py  —  V3 optimization layer for the cluster
collection-risk XGBoost models.

NEW FILE (standing rule #1: no edits to existing files). Side-effect free:
no `!pip`, no `dbutils`, no Spark, no top-level work — safe to `import` from
both the trainer and the (unchanged) inference notebook.

WHY THIS EXISTS
---------------
`train_all_clusters_v3.py` registers a `CalibratedSeedEnsemble` per cluster
instead of a bare `XGBClassifier`. The existing inference
`predict_all_clusters_v2.py` is routing-table driven and only touches the model
through three calls:

    bundle["model"].predict_proba(X)[:, 1]
    model.get_booster().feature_names
    model.get_booster().predict(DMatrix, pred_contribs=True)   # SHAP

`CalibratedSeedEnsemble` duck-types all three, so inference serves V3 models
**unchanged** — provided MLflow can re-import this class at load time. The
trainer registers with `code_paths=[this file]` so MLflow ships the module with
the artifact and re-adds it to sys.path on `load_model`; no inference edit, no
new inference file.

FOUR OPTIMIZATIONS over train_all_clusters_v2.py
------------------------------------------------
  1. Monotonic constraints  — domain priors via XGBoost `monotone_constraints`
                              (built by the trainer, passed in as `sign_map`).
  2. Wider Optuna search     — lives in the trainer (training-time only).
  3. Probability calibration — isotonic / Platt on a held-out val slice; undoes
                              the score distortion `scale_pos_weight` introduces.
  4. Seed-ensemble           — K boosters averaged on the raw margin to cut
                              variance on the tiny per-country pools.

The score path = calibrate(mean over seeds of raw prob). The SHAP path
(`get_booster()`) returns the representative seed (seed 0) — explanations run on
its raw margin. With monotonic constraints the seeds agree on direction, so the
representative SHAP stays faithful to the ensemble's reasoning.
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedSeedEnsemble:
    """K-seed XGBoost ensemble + post-hoc probability calibration.

    Duck-types the slice of the XGBClassifier API the existing inference uses
    (`predict_proba`, `get_booster`, `feature_importances_`) so it is a
    drop-in registered artifact.

    Lifecycle (trainer):
        ens = CalibratedSeedEnsemble(best_params, n_estimators, spw,
                                     sign_map=sign_map, n_seeds=5)
        ens.fit_base(X_train, y_train, sample_weight=w,
                     eval_set=[(X_val, y_val)], early_stopping_rounds=30)
        ens.fit_calibrator(X_val, y_val)          # held-out val, honest
        p = ens.predict_proba(X_test)[:, 1]       # calibrated ensemble score

    Production refit reuses a frozen calibrator (see trainer): fit the base
    seeds on the full data with the locked `best_n`, then assign the calibrator
    learned from the honest held-out val slice.
    """

    def __init__(self, params, n_estimators, scale_pos_weight,
                 sign_map=None, n_seeds=5, base_seed=42):
        self.params = dict(params)
        self.n_estimators = int(n_estimators)
        self.scale_pos_weight = float(scale_pos_weight)
        # name -> {-1, 0, +1}; unmapped features default to 0 (unconstrained)
        self.sign_map = dict(sign_map or {})
        self.n_seeds = int(n_seeds)
        self.base_seed = int(base_seed)

        self.models_ = []
        self.best_iterations_ = []
        self.feature_cols_ = None
        self.calibrator_ = None
        self.calib_kind_ = None      # "isotonic" | "sigmoid" | None

    # ------------------------------------------------------------------ #
    # monotonic constraints                                              #
    # ------------------------------------------------------------------ #
    def _monotone_tuple(self, columns):
        """XGBoost wants a tuple aligned to the *training column order*. Built
        fresh from the actual columns so it survives feature selection."""
        if not self.sign_map:
            return None
        if not any(self.sign_map.get(c, 0) for c in columns):
            return None
        return tuple(int(self.sign_map.get(c, 0)) for c in columns)

    # ------------------------------------------------------------------ #
    # fit                                                                 #
    # ------------------------------------------------------------------ #
    def fit_base(self, X, y, sample_weight=None, eval_set=None,
                 early_stopping_rounds=None):
        """Train the K seed boosters. Each seed differs only by random_state
        (row/column subsampling + tree construction), so averaging cuts
        variance without adding bias."""
        self.feature_cols_ = list(X.columns)
        mono = self._monotone_tuple(self.feature_cols_)
        self.models_ = []
        self.best_iterations_ = []

        for i in range(self.n_seeds):
            p = dict(self.params)
            if mono is not None:
                p["monotone_constraints"] = mono
            kwargs = dict(
                n_estimators=self.n_estimators,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.base_seed + i * 101,
            )
            if early_stopping_rounds is not None:
                kwargs["early_stopping_rounds"] = early_stopping_rounds
            m = XGBClassifier(**p, **kwargs)
            m.fit(X[self.feature_cols_], y, sample_weight=sample_weight,
                  eval_set=eval_set, verbose=False)
            self.models_.append(m)
            bi = getattr(m, "best_iteration", None)
            self.best_iterations_.append(
                (int(bi) + 1) if bi is not None else self.n_estimators)
        return self

    def raw_proba(self, X):
        """Mean P(class=1) across seeds on the raw (uncalibrated) scale."""
        Xc = X[self.feature_cols_]
        return np.mean([m.predict_proba(Xc)[:, 1] for m in self.models_], axis=0)

    def fit_calibrator(self, X_val, y_val, method="auto"):
        """Learn raw-prob -> true-prob on a held-out val slice.

        isotonic = flexible monotone step fit (needs enough positives);
        sigmoid  = Platt (one logistic), stable when positives are scarce.
        "auto" picks sigmoid under 25 val positives, else isotonic. Both are
        monotone non-decreasing, so customer ranking — and therefore the
        per-country `risk_band` qcut downstream — is unchanged.
        """
        raw = self.raw_proba(X_val)
        y = np.asarray(y_val).astype(int)
        n_pos = int(y.sum())

        if method == "auto":
            kind = "sigmoid" if n_pos < 25 else "isotonic"
        else:
            kind = method

        if kind == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(raw, y)
        elif kind == "sigmoid":
            cal = LogisticRegression(solver="lbfgs")
            cal.fit(raw.reshape(-1, 1), y)
        else:
            raise ValueError(f"unknown calibration method: {kind}")

        self.calibrator_ = cal
        self.calib_kind_ = kind
        return self

    def set_calibrator(self, other):
        """Freeze-and-reuse: production base seeds are refit on the full data,
        but the calibrator stays the one learned from the honest held-out val
        slice (calibration relationship is assumed stable across the refit)."""
        self.calibrator_ = other.calibrator_
        self.calib_kind_ = other.calib_kind_
        return self

    def _apply_calibrator(self, raw):
        if self.calibrator_ is None:
            return raw
        if self.calib_kind_ == "isotonic":
            return self.calibrator_.predict(raw)
        return self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]

    # ------------------------------------------------------------------ #
    # predict (the inference surface)                                    #
    # ------------------------------------------------------------------ #
    def predict_proba(self, X):
        p = np.clip(self._apply_calibrator(self.raw_proba(X)), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def get_booster(self):
        """Representative seed (seed 0) for the inference SHAP path: it carries
        `feature_names` and supports `predict(dmat, pred_contribs=True)`. Score
        path uses the calibrated ensemble; SHAP runs on this booster's raw
        margin (seeds agree on direction under the monotone constraints)."""
        return self.models_[0].get_booster()

    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.models_], axis=0)

    def __repr__(self):
        return (f"CalibratedSeedEnsemble(n_seeds={self.n_seeds}, "
                f"n_estimators={self.n_estimators}, "
                f"calib={self.calib_kind_}, "
                f"n_features={len(self.feature_cols_ or [])})")
