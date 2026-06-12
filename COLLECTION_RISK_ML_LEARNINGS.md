# Collection Risk ML — Project Learnings

Knowledge dump for the country-cluster collection-risk system on Databricks/GSK.
Captures bugs found, design decisions, parity rules, and the FR coverage-hole
investigation. Read this before touching any pipeline file.

Last updated: 2026-06-10

---

## 0. Standing rules (non-negotiable)

1. **No edits to old files.** Every existing `.py` in `gsk_work` is read-only.
   Create new files only. (Memory: `feedback_no_edit_old_files`.)
2. **Original multi-table pipeline only.** Use the multi-table SAP joins from
   `collection_risk_model.py` as ground truth for feature semantics — NOT the
   unified view, except where a file already commits to it.
   (Memory: `feedback_original_pipeline_only`.)
3. **Target convention: `target=1 = HIGH RISK`.** High score = high risk =
   collection_ratio < 0.4 (collected_30d / total_outstanding over next 30 days).
   Never invert this.

---

## 1. File map

### Created (mine — safe to edit)
| File | Purpose |
|------|---------|
| `training_pipeline_unified_view_v2.py` | Bug-fixed copy of the unified-view trainer. |
| `country_cluster_pooled_training_v2.py` | Country fingerprints → HAC clustering → pooled training + LOCO. |
| `train_cluster_model_v2.py` | Single-cluster trainer, widget-driven. |
| `train_all_clusters_v2.py` | **Production trainer.** One model per cluster, writes routing table. |
| `predict_all_clusters_v2.py` | **Inference pipeline.** Routing-table-driven, full explain_one, all prod cols. |

### Read-only reference (DO NOT edit)
| File | Why it matters |
|------|----------------|
| `collection_risk_model.py` | Original multi-table pipeline. Ground truth for feature semantics. |
| `inference_original.py` | Source of `explain_one()` wording, BUKRS-prefix country filter, per-country qcut risk_band. |
| `training_pipeline_unified_view_original.py` | Buggy original (see §2). |

---

## 2. THE critical bug (training_pipeline_unified_view_original.py)

**`date_format()` was used where a date was needed.** `date_format` returns a
**string**. Downstream joins/comparisons on dunning (MHND), promise-to-pay
(UDM_P2P_ATTR), and dispute date columns then compared string-vs-date →
silently always-false → **~35 features silently zeroed**. Model trained on
garbage for a third of its features and nobody saw an error.

**Fix (v2):** `to_date_any()` helper that parses any incoming date form to a real
`DateType` before comparison. Applied to every dunning/P2P/dispute date column.

Lesson: in PySpark, `date_format` is display-only. For logic use `to_date` /
`to_timestamp`. A string that *looks* like a date will compare wrong and never throw.

---

## 3. v2 bug-fix pack (training_pipeline_unified_view_v2.py)

- `to_date_any()` date parsing (the §2 fix).
- **Snapshot censoring cap.** Calendar capped at `TODAY − 30d`. Can't build a
  label needing 30 days of future collection if those 30 days don't exist yet.
- **on_time_ratio fix.** `days_late = clearing_date − due_date` (was comparing
  against the wrong baseline).
- **Tenure back-dating.** Customer tenure computed as of the snapshot date, not
  as of today — otherwise every historical snapshot leaks current tenure.
- **Threshold tuned on X_val**, not on train (no threshold leakage).
- Model name bumped: `collection_risk_model_customer_v_2_{country}_optuna`.

---

## 4. Country clustering (country_cluster_pooled_training_v2.py)

**Problem:** per-country data is tiny (<200 rows for many countries). Can't train
a stable model per country. **Solution:** cluster countries with similar
collection behavior, pool their rows, train one model per cluster with country
one-hot features.

Pipeline:
1. **Behavior fingerprint** per country (12-dim): payment-timing stats, dunning
   intensity, dispute rate, exposure distribution, etc.
2. **StandardScaler** (fingerprints are on wildly different scales).
3. **HAC, Ward linkage** → silhouette-selected k. KMeans cross-check.
4. Plots: dendrogram, PCA scatter, behavior heatmap, distance matrix.
5. **Pooled training** per cluster + country one-hots.
6. **LOCO validation** (leave-one-country-out): hold a country out, train on the
   rest of the cluster, score the held-out one. Catches countries that don't
   belong in their cluster.

Glossary the user asked about:
- **Dendrogram** = tree of merge order; cut height → number of clusters.
- **n** = sample/row count; **k** = number of clusters.

Default CLUSTERS:
```python
{"apac_big": ["CN","AU","JP","NZ","TW"],
 "kr_my":    ["KR","MY"],
 "sea":      ["SG","HK","PH","TH","ID","VN"]}
```

---

## 5. Trainer internals (train_all_clusters_v2.py)

Single Spark pass, trains one model per cluster, writes routing table
`f_erp_glide_o2c_12.collection_ml_country_model_map`.

### Config (V2.1 optimizations)
```python
IMPORTANCE_CUTOFF             = 0.003   # drop features below this importance
MIN_FEATURES_AFTER_SELECT     = 10      # never drop below this many
MIN_VAL_POSITIVES_PER_COUNTRY = 10      # gate for per-country threshold
USE_RECENCY_WEIGHTS           = True
RECENCY_HALF_LIFE_DAYS        = 180     # weight = 0.5^(age/180d)
REFIT_ON_FULL_DATA            = True
```

### V2.2 universe-parity flags (must match inference)
```python
TODAY                    = "2026-06-07"
COUNTRY_FROM             = "bukrs"   # country = upper(substring(company_code,1,2))
OPEN_ITEMS_IGNORE_LOOKBACK = True
# Deliberately NO INCLUDE_ZERO_EXPOSURE: training needs total_outstanding>0
# because the label is undefined at zero exposure (can't divide).
```

### Source-aware lookback (the parity heart)
```python
cutoff = F.date_sub(F.lit(date_val), LOOKBACK_DAYS)
if OPEN_ITEMS_IGNORE_LOOKBACK:
    keep = (F.col("source") == "BSID") | \
           ((F.col("source") == "BSAD") &
            (F.coalesce(F.col("clearing_date"), F.col("baseline_date")) >= cutoff))
else:
    keep = F.col("baseline_date") >= cutoff
```
Rationale: **open items (BSID) are always in scope** regardless of age — an
unpaid 2-year-old invoice is still risk today. Only **cleared items (BSAD)** get
capped by clearing date. The original lookback wrongly aged-out open items.

### Country derivation
```python
(F.upper(F.substring(F.col("company_code"), 1, 2))
   if COUNTRY_FROM == "bukrs"
   else F.upper(F.col("country"))).alias("country")
```

### Modeling
- XGBoost binary classifier.
- **Walk-forward time-series CV** with a leakage gap (embargo between train and
  val folds).
- **Macro per-country PR-AUC objective** (not micro — small countries must count).
- **Stability penalty**: score = `mean − 0.5·std` across folds. Punishes models
  that are great on one fold and terrible on another.
- `scale_pos_weight = natural_ratio × tuned_multiplier`.
- **Optuna multivariate TPE**, with a gate: only accept tuned params if they beat
  the fallback (`tune_or_fallback()`).
- **Production refit** on full data with `best_n` (n_estimators) locked from CV.
- **Per-country operating thresholds**: max precision @ recall ≥ 0.80, gated by
  `MIN_VAL_POSITIVES_PER_COUNTRY`, cluster-level fallback when a country fails
  the gate. Stored in the routing table.

---

## 6. Inference internals (predict_all_clusters_v2.py)

Reads `ROUTING_TABLE`, scores per cluster, emits all production columns.

- **Full `explain_one()`** ported from `inference_original.py` (~65-feature dict),
  including V2-only features: `credit_limit`, `credit_utilization`,
  `number_of_disputes`, `open_dispute_amount`, `customer_tenure_days`,
  `total_open_amount`.
- **`build_business_explanations()`** — human-readable per-row reasons; excludes
  `country_*` one-hots (not business-meaningful). Emits `collector_explanations`.
- **SHAP** via native XGBoost `pred_contribs` (no separate shap lib needed).
- **Output = ALL feature columns + predictions.** Added `model_features` (booster
  feature-list joined string) and `risk_band` (per-country `pd.qcut` tertiles).
  **All 71 production-table columns verified covered** (set-diff: MISSING = NONE).
- Parity flags match the trainer:
  ```python
  COUNTRY_FROM             = "bukrs"
  OPEN_ITEMS_IGNORE_LOOKBACK = True
  INCLUDE_ZERO_EXPOSURE    = True   # inference CAN score zero-exposure rows
  RUN_DRIFT_DIAGNOSTIC     = True
  DIAG_COUNTRY             = "FR"
  ```
- Uses inline `F.current_date()` (a `TODAY` constant was added then reverted per
  user request — keep it inline).

**Train/serve asymmetry to remember:** training excludes zero-exposure (label
undefined), inference includes it (`INCLUDE_ZERO_EXPOSURE = True`). That's
intentional, not skew.

---

## 7. Multi-agent adversarial review findings (train_all_clusters_v2.py)

18 confirmed / 15 refuted. Confirmed issues, by severity:

**Critical**
- **Country skew** — train and serve must derive country identically (driven the
  COUNTRY_FROM parity work).
- **`fillna(0)` inverted semantics** — for some features 0 is the *worst* value,
  so filling missing with 0 silently labels unknowns as high-risk (or masks risk,
  depending on sign). Needs sentinel + presence flag, not blind 0.
- **Dunning/P2P current-state leakage** — the unified view carries *current*
  dunning/P2P state, not as-of-snapshot. Inherent to the view. Historical
  snapshots see the future. **Only a multi-table port fixes this.**

**High**
- Open-items lookback skew (fixed by `OPEN_ITEMS_IGNORE_LOOKBACK`).
- Thresholds computed from the wrong (CV) model, not the production-refit model.
- Production `best_n` lock vs. early stopping.

### Remaining fix-pack (NOT yet approved by user)
- Recompute thresholds from `production_model` before the routing write (#5 high).
- Production refit with early stopping instead of locked `best_n` (#6 high).
- `fillna` sentinel features: `has_payment_history` flag +
  `days_since_last_payment` sentinel instead of 0 (#2 critical).
- CV embargo widened to `FUTURE_WINDOW_DAYS`.
- Per-country threshold gate raised ~10 → ~50.
- Deterministic snapshot dedup: `Window.row_number()` instead of `dropDuplicates`.

---

## 8. FR coverage-hole investigation (the big one)

**Symptom:** FR customer count collapsed. Original pipeline 10,183 → new
pipeline 1,651, then 3,444 after parity flags.

**Funnel diagnostic result:**
- Raw BSID FR customers: **10,214**
- Present in unified view (`table_invoice_unified_master`): **3,444** (34%)
- **Absent from the view: 6,786 (66%)**
- Scored by new pipeline: 3,439 (~all of what the view contains)

**Conclusion:** Parity flags lose **zero** customers. The new pipeline is clean.
The 66% gap is a **coverage hole in the unified view itself** — those customers
were never in the view. Country semantics (BUKRS vs master) is NOT the FR cause
(both = 3,444). This is direct evidence for standing rule #2 (multi-table, not
view).

### Diagnostic to characterize the hole (run before blaming the view)
**Check leading-zero join artifact FIRST** — raw `KUNNR` is zero-padded
(`0000123456`), views often trim it. If `matched on stripped` ≫ `matched on
exact`, the hole is a join artifact and the view is fine. Then split missing
customers by `BUKRS` (company-code subset?), `UMSKZ`/`BLART` (special-G/L
exclusion?), baseline year (date cut?), and sum `WRBTR` (EUR invisible to the
model — the business argument). Full snippet lives in the conversation; key idea:
```python
raw  = bsid.filter(upper(substring(BUKRS,1,2))=="FR")
            .withColumn("cust_norm", regexp_replace(KUNNR, "^0+", ""))
missing = raw_cust_norm.join(uv_cust_norm, "cust_norm", "left_anti")
# then groupBy BUKRS / UMSKZ,BLART / year ; sum WRBTR
```

### Two paths forward
1. **Fix the view** — view owner adds missing company codes / special-G/L items.
2. **Port to raw multi-table sources** (BSID+BSAD+MHND+UDM_P2P_ATTR) — fixes the
   66% hole AND the dunning/P2P leakage (#7 critical), and matches standing rule
   #2. **Recommended.** Offered to the user, awaiting decision.

---

## 9. SAP table cheat-sheet

| Table | Contains | Key cols |
|-------|----------|----------|
| BSID | Open (uncleared) AR invoices | KUNNR (customer), BUKRS (company code), BELNR (doc), ZFBDT (baseline date), WRBTR (amount), UMSKZ (special G/L), BLART (doc type) |
| BSAD | Cleared AR invoices | + clearing date |
| MHND | Dunning history | dunning level, dunning date |
| UDM_P2P_ATTR | Promise-to-pay | promise date, promise amount |
| KNA1 | Customer master | credit limit, etc. |
| `f_erp_glide_o2c_12.table_invoice_unified_master` | Unified view (BSID∪BSAD + joins). **Has the coverage hole.** | source, customer_id, company_code, baseline_date, clearing_date |

- **Country = `upper(substring(BUKRS, 1, 2))`** (company-code prefix). This is the
  authoritative derivation both train and serve use.
- KUNNR is **zero-padded** in raw tables. Normalize before joining to anything
  that might have trimmed it.

---

## 10. Validation habits

- After editing any notebook file:
  `sed 's/^!pip/#!pip/' file.py | python3 -c "import ast,sys; ast.parse(sys.stdin.read())"`
  to syntax-check (strip the `!pip` notebook magic first).
- Pyright import-resolution warnings are **expected noise** — `spark`, `dbutils`,
  `display`, Databricks-only packages, and `!pip` lines aren't resolvable
  off-cluster. Not real errors.
- Verify output schema by **set-diff against the production table column list**,
  not by eyeballing.

---

## 11. Open decisions awaiting user

1. Multi-table port of trainer + inference (recommended — fixes coverage hole +
   leakage), OR
2. Apply the remaining §7 fix-pack to the unified-view version, OR
3. Get the view owner to fix the coverage hole.

No new file work should start until the user picks. (Superseded for the
modeling axis by §12 — the V3 XGBoost pack, which is orthogonal to the
data-path decision and still runs on the unified view.)

---

## 12. V3 XGBoost optimization pack (find-a-better-model)

User asked to optimize the trainer for a better XGBoost model. Four levers,
all NEW files (rule #1), inference untouched.

### Files (mine — safe to edit)
| File | Purpose |
|------|---------|
| `cluster_xgb_ensemble.py` | `CalibratedSeedEnsemble` — importable, side-effect-free estimator. The only thing inference must import (to unpickle). |
| `train_all_clusters_v3.py` | V3 cluster trainer — **one model per cluster + routing table**. Now **reads `collection_ml_features_train_v3`** (data_pipeline_v3) instead of inline PART A (removed 2026-06-12); PART B = the four levers (monotone + wide Optuna walk-forward + calibration + seed-ensemble), per-country thresholds, registers `collection_risk_model_cluster_v3_<cluster>`, writes routing to `..._v3_staging` (prod behind `WRITE_PROD_ROUTING`). |
| `predict_clusters_v3.py` | V3 **cluster inference** (matches the rewired trainer). Reads the routing table + `collection_ml_features_infer_v3`; per cluster loads the CalibratedSeedEnsemble, rebuilds the cluster's country one-hots, aligns to `booster.feature_names` (missing→0), scores at the **per-country** threshold from routing, SHAP `collector_explanations` (country_ excluded), per-country `risk_band` → `collection_ml_customer_clusters_v3`. Imports the class for unpickle. Smoke-tested (routing, align, SHAP). |
| `predict_best_model_v3.py` | Inference for the global model. Reads `collection_ml_features_infer_v3`, loads `collection_risk_model_global_v3`; rebuilds country+cluster one-hots and **aligns to `booster.feature_names`** (missing→0, unseen one-hots dropped) so train/serve match even when today's batch misses a country/cluster; threshold auto-fetched from the training run's `chosen_threshold` param (config fallback) → score → `binary_pred`/band + SHAP top-3 drivers + business `collector_explanations` (country_/cluster_ excluded) + per-country `risk_band` qcut → `collection_ml_customer_global_v3`. Plain XGBClassifier ⇒ no code_paths. Smoke-tested (align mismatch, SHAP, bands). |
| `train_best_model_v3.py` | **Single GLOBAL** trainer (not cluster routing). Reads `collection_ml_features_train_v3` (so `data_pipeline_v3.py` must run first, `INCLUDE_V2_FEATURES=True` — now the default). Country **+ cluster** one-hots (cluster from the CLUSTERS map = coarse behavioral group; country = detail; unmapped→`other`) → time split train/valid/test → Optuna over the FULL hyperparam space (depth/leaves/lr/all subsamples+regularizers/grow_policy/max_delta_step); **objective = walk-forward time-series CV over DEV (train+valid): 4 expanding folds, 30d embargo, `mean(PR-AUC) − 0.5·std`** (MedianPruner) → imbalance via tuned `scale_pos_weight=(neg/pos)×mult` per fold → early stopping (n via `best_iteration`, not searched) → honest test report (untouched by CV) + operating threshold → **final refit on ALL data** with locked `best_n` → MLflow register `collection_risk_model_global_v3`. No calibration/ensemble/monotone (not requested). Smoke-tested locally (optuna 4.9/xgboost 3.2): split/embargo/objective/refit. |
| `data_pipeline_v3.py` | Standalone feature ETL — the **ORIGINAL** pipelines' feature logic (user's `collection_risk_model.py` trainer + original inference builder) re-pointed at the unified view. One MODE-switched builder (`train`/`infer`/`both`) → `collection_ml_features_train_v3` + `collection_ml_features_infer_v3`; train/infer columns asserted identical at runtime (parity guard). **Design rule: original behavior by default, every V2 fix is an opt-in flag** (`ON_TIME_DUE_DATE_FIX`, `CENSOR_SNAPSHOTS`, `INCLUDE_V2_FEATURES`). Original quirks reproduced: `on_time_ratio = days_to_pay≤0` (≈always 0), `due_date` recomputed from terms, original feature SET only. **Key view-translation:** originals `count(*)` raw MHND/UDM letters → `sum(dunning_count)` here (view is pre-agg per invoice; `count(*)` would count invoices not letters). Harmonized (originals disagreed): BSID∪BSAD source, BUKRS country. `INCLUDE_V2_FEATURES` now defaults **True** (emits tenure/credit/disputes/open_amount) so the table feeds `train_best_model_v3.py` directly; set False for the strict-original narrow schema. |

### The four optimizations
1. **Monotonic constraints** (`monotone_constraints`). Domain priors, target=1
   = HIGH risk: `+1` ⇒ more-is-riskier (max_dpd, avg_dpd, amt/pct_*_plus,
   oldest/avg_invoice_age, credit_utilization, avg/max_days_to_pay,
   days_since_last_payment, disputes, *every* dunning agg, broken_*); `-1` ⇒
   more-is-safer (on_time_ratio*, kept_*). Everything else free (exposure
   magnitudes, raw counts, country one-hots). ~19/26 features pinned on a full
   list. Resolver = `resolve_monotone()`; tuple is rebuilt **after** feature
   selection (order must match the live columns) by `monotone_tuple()`.
2. **Wider Optuna search**: `grow_policy` depthwise/lossguide (+ `max_leaves`
   7–127 when lossguide), `colsample_bylevel`/`bynode`, data-adaptive
   `max_depth` cap (4 / 5 / 6 by dev size 600/1500), 50→120 trials,
   `n_startup_trials=20`. Optuna's flat params → XGB kwargs via
   `xgb_params_from_optuna()` (resolves the per-grow-policy depth name).
3. **Probability calibration**: `fit_calibrator()` on the held-out val slice.
   `auto` ⇒ sigmoid/Platt under 25 val positives, else isotonic. Both monotone
   non-decreasing ⇒ ranking preserved ⇒ inference `risk_band` qcut unchanged.
   Thresholds are computed on **calibrated** val probs so the routing table
   stays consistent. (Ties from isotonic's flat steps are fine — no inversion.)
4. **Seed-ensemble**: K=5 boosters, seeds `42 + 101*i`, averaged on the **raw
   margin**, then calibrated. Cuts variance on the tiny pools.

### How inference stays UNCHANGED (the key design move)
`predict_all_clusters_v2.py` is routing-table driven and touches the model
only via `predict_proba`, `get_booster().feature_names`, and
`get_booster().predict(DMatrix, pred_contribs=True)` (SHAP). `CalibratedSeedEnsemble`
**duck-types all three**: score = calibrate(mean-over-seeds raw prob);
`get_booster()` returns the representative seed (seed 0) for SHAP. Registered
with `mlflow.sklearn.log_model(..., code_paths=["cluster_xgb_ensemble.py"])` so
MLflow re-imports the class on `load_model` — no import line needed in
inference. V3 writes the routing table → inference loads V3 models → serves
them with **zero edits**. SHAP runs on seed-0's raw margin while the score is
the calibrated ensemble mean — a minor, documented asymmetry; the monotone
constraints keep the seeds directionally agreed.

### Train/serve calibration handshake
`cal_ens` = K seeds on `train`, calibrator fit on held-out `val`, honest test
metrics + thresholds come from it. `production_model` = K seeds refit on FULL
data with `best_n` = median(seed best_iterations) locked, then
`set_calibrator(cal_ens)` **reuses the frozen** val-fit calibrator (refit base,
freeze calibrator — standard). `spw` recomputed on the full-data base rate for
prod.

### Routing safety
Default `WRITE_PROD_ROUTING=False` → writes only `<routing>_v3_staging` for
comparison vs V2. Flip to `True` to point production inference at V3. PART A is
byte-identical to V2, so any count delta vs V2 is the model, not the universe.

### Validated (local venv, xgboost 3.2 / sklearn 1.9, synthetic data)
AST-clean (both files). Smoke test `/tmp/smoke_v3.py` (9 checks) confirmed: no
`n_estimators`/`random_state` kwarg collision, monotone shipped into booster
config, calibration monotone + in-range (isotonic & sigmoid), inference
duck-type (`feature_names` match, contribs shape `(n, nfeat+1)`), feature-select
sign_map rebuild, frozen-calibrator reuse, `predict_proba` ignores extra cols,
pickle round-trip, lossguide+max_leaves+monotone trains. NOT yet run on real
Databricks data.

### Still open / not done
- Per-country threshold gate still 10 (§7 fix-pack item, unchanged).
- `fillna(0)` sentinel issue (§7 critical) — NOT addressed here; orthogonal.
- Dunning/P2P current-state leakage (§7 critical) — inherent to the view; only
  the multi-table port fixes it. V3 inherits it.
