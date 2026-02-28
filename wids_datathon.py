#!/usr/bin/env python3
"""
WiDS Global Datathon 2026 - Advanced Survival Pipeline (v7)
============================================================
Maximises: hybrid = 0.3 * C-index + 0.7 * (1 - weighted_brier)
WB weights: 30% at 24h, 40% at 48h, 30% at 72h.

v7 improvements:
  1. 93 features (81 base + 12 binary tracking quality + interaction terms)
  2. IPCW-weighted LightGBM for direct P(T<=t) optimisation at each horizon
  3. Two-stage adaptive search (StratifiedKFold, faster)
  4. Multi-seed OOF averaging (3 seeds)
  5. Nelder-Mead blend: GBSA + RSF + ExtraTrees + IPCW-LGB
  6. Leave-fold-out calibration: none / clipped / isotonic / platt

Install: pip install pandas numpy scikit-survival scikit-learn scipy lightgbm
Run:     /opt/anaconda3/bin/python3 wids_datathon.py
"""
from __future__ import annotations
import itertools, json, sys, time, warnings
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import brier_score, concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PRED_TIMES   = np.array([12.0, 24.0, 48.0, 72.0])
BRIER_TIMES  = np.array([24.0, 48.0, 72.0])
BRIER_W      = np.array([0.3, 0.4, 0.3])
N_FOLDS      = 5
PROB_FLOOR   = 0.001
PROB_CEIL    = 0.999
OOF_SEEDS    = [42, 123, 789]
FINAL_SEEDS  = [42, 123, 456, 789, 2024, 314, 271]

DATA_DIR   = Path(__file__).resolve().parent
TRAIN_P    = DATA_DIR / "train.csv"
TEST_P     = DATA_DIR / "test.csv"
SAMPLE_P   = DATA_DIR / "sample_submission.csv"
MANIFEST_P = DATA_DIR / "experiment_manifest.json"
ID_COL  = "event_id"
TARGETS = ["event", "time_to_hit_hours"]

# ─── Feature engineering ──────────────────────────────────────────────────────

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Base 81-feature set (34 raw CSV + 47 derived)."""
    d = df.copy()
    dist   = d["dist_min_ci_0_5h"]
    spd    = d["closing_speed_m_per_h"]
    aspd   = d["closing_speed_abs_m_per_h"]
    area   = d["area_first_ha"]
    gr     = d["area_growth_rate_ha_per_h"]
    rad    = d["radial_growth_rate_m_per_h"]
    aln    = d["alignment_abs"]
    alc    = d["alignment_cos"]
    along  = d["along_track_speed"]
    cspd   = d["centroid_speed_m_per_h"]
    dslope = d["dist_slope_ci_0_5h"]
    dstd   = d["dist_std_ci_0_5h"]
    daccel = d["dist_accel_m_per_h2"]
    r2     = d["dist_fit_r2_0_5h"]
    proj   = d["projected_advance_m"]
    dt     = d["dt_first_last_0_5h"]
    nper   = d["num_perimeters_0_5h"]
    safe_spd = spd.replace(0, np.nan)
    safe_rad = rad.replace(0, np.nan)
    d["eta_close"]        = (dist / safe_spd).clip(-500, 500).fillna(999)
    d["eta_radial"]       = (dist / safe_rad).clip(-500, 500).fillna(999)
    d["risk_proxy"]       = spd * aln / (dist + 1.0)
    d["risk_proxy_sq"]    = d["risk_proxy"] ** 2
    d["risk_area_dist"]   = area / (dist + 1.0)
    d["log_dist"]         = np.log1p(dist)
    d["log_area"]         = np.log1p(area)
    d["sqrt_area"]        = np.sqrt(area)
    d["sqrt_dist"]        = np.sqrt(dist)
    d["area_x_growth"]    = area * gr
    d["fire_intensity"]   = np.sqrt(area + 1) * gr
    d["radial_x_area"]    = rad * np.sqrt(area + 1)
    for h in [12, 24, 48, 72]:
        d[f"reach_{h}"]   = rad * h
        d[f"deficit_{h}"] = dist - spd * h
        d[f"phit_{h}"]    = (dist < spd * h).astype(float)
    d["close_frac"]       = spd / (dist + 1.0)
    d["rad_frac"]         = rad / (dist + 1.0)
    d["spd_ratio_rc"]     = rad / (aspd + 1.0)
    d["spd_ratio_cc"]     = spd / (cspd + 1.0)
    d["dir_momentum"]     = along * aln
    d["align_x_speed"]    = alc * spd
    d["cross_abs"]        = d["cross_track_component"].abs()
    d["perim_dens"]       = (nper / dt.replace(0, np.nan)).fillna(0)
    d["adv_ratio"]        = proj / (dist + 1.0)
    d["accel_24"]         = spd + daccel * 24
    d["accel_48"]         = spd + daccel * 48
    d["rel_close"]        = spd * r2
    d["close_sq"]         = spd ** 2
    d["dist_sq"]          = dist ** 2
    d["hr_sin"]           = np.sin(2 * np.pi * d["event_start_hour"] / 24)
    d["hr_cos"]           = np.cos(2 * np.pi * d["event_start_hour"] / 24)
    d["mo_sin"]           = np.sin(2 * np.pi * d["event_start_month"] / 12)
    d["mo_cos"]           = np.cos(2 * np.pi * d["event_start_month"] / 12)
    d["dstd_x_close"]     = dstd * spd
    d["slope_x_align"]    = dslope * aln
    d["slope_x_speed"]    = dslope * spd
    d["dist_x_align"]     = dist * aln
    d["dist_change_norm"] = d["dist_change_ci_0_5h"] / (dist + 1.0)
    d.fillna(0, inplace=True)
    d.replace([np.inf, -np.inf], 0, inplace=True)
    return d


def engineer_v7(df: pd.DataFrame) -> pd.DataFrame:
    """Extended 93-feature set: 81 base + 12 new binary/interaction features.

    EDA findings driving the additions:
      - All events have dist < 5 km; all censored have dist >= 5 km.
      - Well-tracked events (nper>=3, ltr=0) hit at mean 1.6 h; poorly-tracked at 14.8 h.
      - Interaction terms capture group-specific feature effects for tree models.
    """
    d = engineer(df)
    nper = df["num_perimeters_0_5h"].values
    ltr  = df["low_temporal_resolution_0_5h"].values
    dist = df["dist_min_ci_0_5h"].values
    area = df["area_first_ha"].values
    spd  = df["closing_speed_m_per_h"].values
    aln  = df["alignment_abs"].values
    hour = df["event_start_hour"].values
    is_well = ((nper >= 3) & (ltr == 0)).astype(float)
    d["is_well_tracked"]  = is_well
    d["is_close_5km"]     = (dist < 5000).astype(float)
    d["is_close_10km"]    = (dist < 10000).astype(float)
    d["log_area_x_well"]  = np.log1p(area) * is_well
    d["align_x_well"]     = aln * is_well
    d["spd_x_well"]       = spd * is_well
    d["hour_x_well"]      = hour * is_well
    d["track_quality"]    = nper * (1.0 - ltr)
    d["track_x_align"]    = d["track_quality"] * aln
    safe_spd_v            = np.where(np.abs(spd) > 10, spd, 10.0)
    d["time_est"]         = np.clip(dist / safe_spd_v, 0, 200)
    d["is_very_close"]    = (dist < 1000).astype(float)
    d["dist_log_bucket"]  = np.clip(np.floor(np.log10(np.maximum(dist, 1))), 0, 5)
    d.fillna(0, inplace=True)
    d.replace([np.inf, -np.inf], 0, inplace=True)
    return d


def load_data():
    tr   = pd.read_csv(TRAIN_P)
    te   = pd.read_csv(TEST_P)
    feat = [c for c in tr.columns if c not in TARGETS + [ID_COL]]
    Xtr  = engineer_v7(tr[feat])
    Xte  = engineer_v7(te[feat])
    y    = np.array(
        list(zip(tr["event"].astype(bool), tr["time_to_hit_hours"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
    return Xtr, Xte, y, te[ID_COL].values

# ─── Utilities ────────────────────────────────────────────────────────────────

def sf2prob(surv_fns, times=PRED_TIMES):
    n = len(surv_fns)
    out = np.zeros((n, len(times)))
    for i, sf in enumerate(surv_fns):
        xv, yv = sf.x, sf.y
        for j, t in enumerate(times):
            if   t <= xv[0]: s = yv[0]
            elif t >= xv[-1]: s = yv[-1]
            else: s = yv[np.searchsorted(xv, t, side="right") - 1]
            out[i, j] = 1.0 - s
    return out

def mono(p):
    return np.maximum.accumulate(np.clip(p, 0.0, 1.0), axis=1)

def clip_safe(p):
    return mono(np.clip(p, PROB_FLOOR, PROB_CEIL))

def metric(y_tr, y_va, probs, risk):
    c  = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
    sv = 1.0 - probs[:, 1:]
    eps = 0.01
    hi  = min(y_tr["time"].max(), y_va["time"].max()) - eps
    ok  = BRIER_TIMES < hi
    et, ew, es = BRIER_TIMES[ok], BRIER_W[ok], sv[:, ok]
    if len(et) == 0:
        return {"c": c, "wb": np.nan, "h": np.nan}
    try:
        _, bs = brier_score(y_tr, y_va, es, et)
    except ValueError:
        hi2 = min(y_tr["time"].max(), y_va["time"].max()) * 0.99
        ok2 = BRIER_TIMES < hi2
        if not ok2.any():
            return {"c": c, "wb": np.nan, "h": np.nan}
        _, bs = brier_score(y_tr, y_va, sv[:, ok2], BRIER_TIMES[ok2])
        ew    = BRIER_W[ok2]
    wb = float(np.average(bs, weights=ew / ew.sum()))
    return {"c": c, "wb": wb, "h": 0.3 * c + 0.7 * (1.0 - wb)}

# ─── IPCW-LightGBM ────────────────────────────────────────────────────────────

def km_censoring(event, time):
    """G(t) = P(C > t) via Kaplan-Meier on the censoring process."""
    t_km, g_km = kaplan_meier_estimator(~event.astype(bool), time)
    return t_km, g_km

def _g_at(t_km, g_km, t):
    if t > t_km[-1]:
        return max(float(g_km[-1]), 1e-6)
    idx = int(np.searchsorted(t_km, t, side="right")) - 1
    return max(float(g_km[max(idx, 0)]), 1e-6)

def make_ipcw_dataset(X, event, time, horizon, t_km, g_km):
    """Binary IPCW dataset for P(T <= horizon).
    Excludes rows censored at or before horizon (ambiguous label).
    IPCW weight = 1/G(t_i) for positive examples, capped at 20."""
    mask  = event.astype(bool) | (time > horizon)
    Xs    = X[mask]
    y_bin = (event[mask] & (time[mask] <= horizon)).astype(float)
    ws    = np.ones(len(Xs))
    for i, oi in enumerate(np.where(mask)[0]):
        if event[oi] and time[oi] <= horizon:
            ws[i] = 1.0 / _g_at(t_km, g_km, time[oi])
    return Xs, y_bin, np.clip(ws, 1.0, 20.0)

_LGBM_FIXED = {"objective": "binary", "metric": "binary_logloss",
               "verbose": -1, "n_jobs": 2}

def train_lgbm_ipcw(X, event, time, horizon, params, seed=RANDOM_STATE):
    t_km, g_km = km_censoring(event, time)
    Xs, y_bin, ws = make_ipcw_dataset(X, event, time, horizon, t_km, g_km)
    clf = lgb.LGBMClassifier(**{**_LGBM_FIXED, **params, "random_state": seed})
    clf.fit(Xs, y_bin, sample_weight=ws)
    return clf

def oof_lgbm_ipcw(X_arr, y, params, splitter, seeds=None):
    """OOF probabilities for IPCW-LGB across all four horizons."""
    if seeds is None:
        seeds = OOF_SEEDS
    n, oof_p = len(y), np.zeros((len(y), 4))
    for tr_i, va_i in splitter.split(np.arange(n), y["event"].astype(int)):
        acc = np.zeros((len(va_i), 4))
        for seed in seeds:
            fp = np.zeros((len(va_i), 4))
            for j, h in enumerate(PRED_TIMES):
                clf = train_lgbm_ipcw(X_arr[tr_i], y["event"][tr_i],
                                      y["time"][tr_i], h, params, seed=seed)
                fp[:, j] = clf.predict_proba(X_arr[va_i])[:, 1]
            acc += fp
        oof_p[va_i] = acc / len(seeds)
    oof_p = mono(oof_p)
    return oof_p, oof_p[:, -1]

def fit_predict_ipcw_test(X_arr, y, Xte_arr, params):
    """Retrain IPCW-LGB on full training set; average over FINAL_SEEDS."""
    preds = np.zeros((len(Xte_arr), 4))
    for seed in FINAL_SEEDS:
        fp = np.zeros((len(Xte_arr), 4))
        for j, h in enumerate(PRED_TIMES):
            clf = train_lgbm_ipcw(X_arr, y["event"], y["time"], h, params, seed=seed)
            fp[:, j] = clf.predict_proba(Xte_arr)[:, 1]
        preds += fp
    return mono(preds / len(FINAL_SEEDS))

# ─── IPCW-LGB search ──────────────────────────────────────────────────────────

LGBM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "learning_rate":     [0.03, 0.05, 0.08, 0.1],
    "max_depth":         [3, 4, 5],
    "num_leaves":        [8, 15, 31],
    "min_child_samples": [20, 30, 50],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.9, 1.0],
    "reg_lambda":        [1.0, 5.0, 10.0],
}

def search_lgbm_ipcw(X_arr, y, splitter, n_draws=25):
    """Random search for IPCW-LGB hyperparameters via OOF hybrid score."""
    rng  = np.random.default_rng(RANDOM_STATE + 77)
    keys = list(LGBM_GRID)
    pool = list(itertools.product(*(LGBM_GRID[k] for k in keys)))
    rng.shuffle(pool)
    pool = pool[:n_draws]
    best_score  = -np.inf
    best_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
                   "num_leaves": 15, "min_child_samples": 30, "subsample": 0.8,
                   "colsample_bytree": 0.9, "reg_lambda": 5.0}
    for idx, combo in enumerate(pool, start=1):
        params = dict(zip(keys, combo))
        try:
            oof_p, oof_r = oof_lgbm_ipcw(X_arr, y, params, splitter,
                                          seeds=[RANDOM_STATE])
        except Exception:
            continue
        rows = []
        for tr_i, va_i in splitter.split(np.arange(len(y)), y["event"].astype(int)):
            m = metric(y[tr_i], y[va_i], oof_p[va_i], oof_r[va_i])
            if not np.isnan(m.get("h", np.nan)):
                rows.append(m)
        if not rows:
            continue
        score = float(np.mean([m["h"] for m in rows]))
        c_m   = float(np.mean([m["c"] for m in rows]))
        wb_m  = float(np.mean([m["wb"] for m in rows]))
        print(f"  [IPCW {idx:02d}/{n_draws}] h={score:.4f} C={c_m:.4f} WB={wb_m:.4f}")
        if score > best_score:
            best_score  = score
            best_params = params
            print(f"    *** New best: {params}")
    print(f"\n  Best IPCW h={best_score:.4f}  params={best_params}")
    return best_params, best_score

# ─── Survival model search ────────────────────────────────────────────────────

MODEL_SPACE = [
    {"name": "GBSA", "cls": GradientBoostingSurvivalAnalysis,
     "grid": {"n_estimators": [100,150,200,300,400,500],
              "learning_rate": [0.03,0.05,0.08,0.1,0.12],
              "max_depth": [2,3,4], "subsample": [0.6,0.7,0.8,0.9,1.0],
              "min_samples_leaf": [5,8,10,15,20]},
     "fixed": {"random_state": RANDOM_STATE},
     "stage_a_draws": 20, "stage_b_topk": 4, "scale": False},
    {"name": "RSF", "cls": RandomSurvivalForest,
     "grid": {"n_estimators": [200,300,500], "max_depth": [7,10,15,None],
              "min_samples_leaf": [2,3,5], "max_features": [0.25,0.3,0.4,0.5]},
     "fixed": {"random_state": RANDOM_STATE, "n_jobs": 2},
     "stage_a_draws": 15, "stage_b_topk": 3, "scale": False},
    {"name": "ExtraTrees", "cls": ExtraSurvivalTrees,
     "grid": {"n_estimators": [300,500], "max_depth": [10,15,None],
              "min_samples_leaf": [2,3,5], "max_features": [0.5,0.7,0.9]},
     "fixed": {"random_state": RANDOM_STATE, "n_jobs": 2},
     "stage_a_draws": 10, "stage_b_topk": 2, "scale": False},
]

def _draw(grid, n, rng):
    keys = list(grid)
    pool = list(itertools.product(*(grid[k] for k in keys)))
    rng.shuffle(pool)
    return [dict(zip(keys, c)) for c in pool[:n]]

def robust_score(mean_h, std_h):
    return mean_h - 0.30 * std_h

def evaluate_config(X, y, cls, params, splitter, seeds, scale=False):
    fold_metrics = []
    for tr_i, va_i in splitter.split(X, y["event"]):
        pred_acc = np.zeros((len(va_i), 4))
        risk_acc = np.zeros(len(va_i))
        for seed in seeds:
            par = dict(params)
            if "random_state" in par: par["random_state"] = seed
            Xtr, Xva = X.iloc[tr_i].copy(), X.iloc[va_i].copy()
            if scale:
                sc   = StandardScaler()
                cols = Xtr.columns
                Xtr  = pd.DataFrame(sc.fit_transform(Xtr), columns=cols, index=Xtr.index)
                Xva  = pd.DataFrame(sc.transform(Xva),     columns=cols, index=Xva.index)
            m = cls(**par)
            m.fit(Xtr, y[tr_i])
            pred_acc += mono(sf2prob(m.predict_survival_function(Xva)))
            risk_acc += m.predict(Xva)
        pred_acc /= len(seeds)
        risk_acc /= len(seeds)
        fold_metrics.append(metric(y[tr_i], y[va_i], pred_acc, risk_acc))
    hs  = [m["h"]  for m in fold_metrics if not np.isnan(m.get("h",  np.nan))]
    cs  = [m["c"]  for m in fold_metrics if not np.isnan(m.get("c",  np.nan))]
    wbs = [m["wb"] for m in fold_metrics if not np.isnan(m.get("wb", np.nan))]
    if not hs: return None
    return {"mean_h": float(np.mean(hs)), "std_h": float(np.std(hs)),
            "mean_c": float(np.mean(cs)), "mean_wb": float(np.mean(wbs))}

def run_adaptive_search(X, y, rng):
    spl_a = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    spl_b = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE+17)
    results = []
    for spec in MODEL_SPACE:
        print(f"\n{'='*60}\nModel: {spec['name']}\n{'='*60}")
        candidates = _draw(spec["grid"], spec["stage_a_draws"], rng)
        a_rows, best_a = [], -np.inf
        for idx, combo in enumerate(candidates, start=1):
            params = {**spec["fixed"], **combo}
            out    = evaluate_config(X, y, spec["cls"], params, spl_a,
                                     [RANDOM_STATE], scale=spec["scale"])
            if out is None: continue
            rs     = robust_score(out["mean_h"], out["std_h"])
            best_a = max(best_a, rs)
            a_rows.append({"params": params, **out, "robust": rs})
            print(f"  [A {idx:02d}] h={out['mean_h']:.4f}+-{out['std_h']:.4f} "
                  f"C={out['mean_c']:.4f} WB={out['mean_wb']:.4f} rs={rs:.4f}")
        a_rows.sort(key=lambda r: r["robust"], reverse=True)
        topk = a_rows[:spec["stage_b_topk"]]
        if not topk: continue
        print(f"\n  Stage-B ({len(topk)} configs)...")
        best_b, best_entry = -np.inf, None
        for idx, row in enumerate(topk, start=1):
            params = row["params"]
            out    = evaluate_config(X, y, spec["cls"], params, spl_b,
                                     OOF_SEEDS, scale=spec["scale"])
            if out is None: continue
            rs = robust_score(out["mean_h"], out["std_h"])
            print(f"  [B {idx:02d}] h={out['mean_h']:.4f}+-{out['std_h']:.4f} "
                  f"C={out['mean_c']:.4f} WB={out['mean_wb']:.4f} rs={rs:.4f}")
            if rs > best_b:
                best_b     = rs
                best_entry = {"name": spec["name"], "cls": spec["cls"],
                              "params": params, "scale": spec["scale"],
                              "cv_mean_h": out["mean_h"], "cv_std_h": out["std_h"],
                              "cv_mean_c": out["mean_c"], "cv_mean_wb": out["mean_wb"],
                              "robust": rs}
        if best_entry is not None:
            results.append(best_entry)
    return results

def build_oof_for_model(X, y, entry, splitter):
    n, oof_p, oof_r = len(y), np.zeros((len(y), 4)), np.zeros(len(y))
    for tr_i, va_i in splitter.split(X, y["event"]):
        pred_acc = np.zeros((len(va_i), 4))
        risk_acc = np.zeros(len(va_i))
        for seed in OOF_SEEDS:
            par = dict(entry["params"])
            if "random_state" in par: par["random_state"] = seed
            Xtr, Xva = X.iloc[tr_i].copy(), X.iloc[va_i].copy()
            if entry["scale"]:
                sc   = StandardScaler()
                cols = Xtr.columns
                Xtr  = pd.DataFrame(sc.fit_transform(Xtr), columns=cols, index=Xtr.index)
                Xva  = pd.DataFrame(sc.transform(Xva),     columns=cols, index=Xva.index)
            m = entry["cls"](**par)
            m.fit(Xtr, y[tr_i])
            pred_acc += mono(sf2prob(m.predict_survival_function(Xva)))
            risk_acc += m.predict(Xva)
        oof_p[va_i] = pred_acc / len(OOF_SEEDS)
        oof_r[va_i] = risk_acc / len(OOF_SEEDS)
    return oof_p, oof_r

def eval_oof(y, oof_p, oof_r, splitter):
    rows = []
    for tr_i, va_i in splitter.split(np.arange(len(y)), y["event"]):
        m = metric(y[tr_i], y[va_i], oof_p[va_i], oof_r[va_i])
        if not np.isnan(m.get("h", np.nan)): rows.append(m)
    return rows

# ─── Blend ────────────────────────────────────────────────────────────────────

def optimize_blend_weights(entries, y, splitter):
    if len(entries) == 1: return np.array([1.0])
    splits = list(splitter.split(np.arange(len(y)), y["event"]))
    def obj(w_raw):
        w = np.exp(w_raw); w /= w.sum()
        p = mono(sum(wi * e["oof_p"] for wi, e in zip(w, entries)))
        r = sum(wi * e["oof_r"] for wi, e in zip(w, entries))
        hs = []
        for tr_i, va_i in splits:
            m = metric(y[tr_i], y[va_i], p[va_i], r[va_i])
            if not np.isnan(m.get("h", np.nan)): hs.append(m["h"])
        return -float(np.mean(hs)) if hs else 0.0
    k = len(entries)
    best, best_val = np.zeros(k), obj(np.zeros(k))
    for _ in range(30):
        x0  = np.random.randn(k) * 0.7
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-8})
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    w = np.exp(best); w /= w.sum()
    return w

# ─── Calibration ──────────────────────────────────────────────────────────────

def choose_calibration(oof_p, oof_r, y, splitter):
    splits  = list(splitter.split(np.arange(len(y)), y["event"]))
    targets = [(y["event"] & (y["time"] <= t)).astype(float) for t in PRED_TIMES]
    def oof_score(p):
        hs = []
        for tr_i, va_i in splits:
            m = metric(y[tr_i], y[va_i], p[va_i], oof_r[va_i])
            if not np.isnan(m.get("h", np.nan)): hs.append(m["h"])
        return float(np.mean(hs)) if hs else 0.0
    raw_score     = oof_score(oof_p)
    clipped_score = oof_score(clip_safe(oof_p))
    iso_p = np.zeros_like(oof_p)
    for fi, (_, va_i) in enumerate(splits):
        tri = np.concatenate([v for j, (_, v) in enumerate(splits) if j != fi])
        for t in range(4):
            ir = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL, out_of_bounds="clip")
            ir.fit(oof_p[tri, t], targets[t][tri])
            iso_p[va_i, t] = ir.transform(oof_p[va_i, t])
    iso_score = oof_score(clip_safe(iso_p))
    platt_p = np.zeros_like(oof_p)
    for fi, (_, va_i) in enumerate(splits):
        tri = np.concatenate([v for j, (_, v) in enumerate(splits) if j != fi])
        for t in range(4):
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(oof_p[tri, t:t+1], targets[t][tri])
            platt_p[va_i, t] = lr.predict_proba(oof_p[va_i, t:t+1])[:, 1]
    platt_score = oof_score(clip_safe(platt_p))
    scores    = {"none": raw_score, "clipped": clipped_score,
                 "isotonic": iso_score, "platt": platt_score}
    best_name = max(scores, key=scores.get)
    if best_name == "isotonic":
        cal = [IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL, out_of_bounds="clip")
               for _ in range(4)]
        for t in range(4): cal[t].fit(oof_p[:, t], targets[t])
        calibrator = ("isotonic", cal)
    elif best_name == "platt":
        cal = [LogisticRegression(C=1.0, max_iter=1000) for _ in range(4)]
        for t in range(4): cal[t].fit(oof_p[:, t:t+1], targets[t])
        calibrator = ("platt", cal)
    elif best_name == "clipped":
        calibrator = ("clipped", None)
    else:
        calibrator = ("none", None)
    return {"best_name": best_name, "scores": scores, "calibrator": calibrator}

def apply_calibration(p, calibrator):
    name, obj = calibrator
    if name == "none":     return p
    if name == "clipped":  return clip_safe(p)
    if name == "isotonic": return clip_safe(np.column_stack([obj[j].transform(p[:, j]) for j in range(4)]))
    return clip_safe(np.column_stack([obj[j].predict_proba(p[:, j:j+1])[:, 1] for j in range(4)]))

# ─── Final test helpers ───────────────────────────────────────────────────────

def fit_predict_test_model(entry, X, y, Xt):
    preds = []
    for seed in FINAL_SEEDS:
        par = dict(entry["params"])
        if "random_state" in par: par["random_state"] = seed
        if "n_jobs" in par:       par["n_jobs"] = 2
        Xtr, Xte = X.copy(), Xt.copy()
        if entry["scale"]:
            sc = StandardScaler(); cols = Xtr.columns
            Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=cols)
            Xte = pd.DataFrame(sc.transform(Xte),     columns=cols)
        m = entry["cls"](**par)
        m.fit(Xtr, y)
        preds.append(mono(sf2prob(m.predict_survival_function(Xte))))
    return np.mean(preds, axis=0)

def write_submission(path, ids, probs):
    sub = pd.DataFrame({ID_COL: ids, "prob_12h": probs[:, 0],
                        "prob_24h": probs[:, 1], "prob_48h": probs[:, 2],
                        "prob_72h": probs[:, 3]})
    sample = pd.read_csv(SAMPLE_P)
    assert len(sub) == len(sample) and set(sub[ID_COL]) == set(sample[ID_COL])
    sub.sort_values(ID_COL).reset_index(drop=True).to_csv(path, index=False)
    return pd.read_csv(path)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    X, Xt, y, tids = load_data()
    X_arr, Xt_arr  = X.values.astype(float), Xt.values.astype(float)
    base_spl = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    print("=== Data ===")
    print(f"  Train: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Test : {Xt.shape[0]} rows  |  Events: {int(y['event'].sum())}")

    # 1. Survival model search
    print("\n=== Adaptive Survival Model Search ===")
    searched = run_adaptive_search(X, y, rng)
    searched.sort(key=lambda e: e["robust"], reverse=True)
    if not searched:
        raise RuntimeError("No valid survival model candidate found.")
    selected = searched[:3]
    print("\nSelected:")
    for e in selected:
        shown = {k: v for k, v in e["params"].items()
                 if k not in ("random_state", "n_jobs")}
        print(f"  {e['name']}: h={e['cv_mean_h']:.4f}  rs={e['robust']:.4f}  {shown}")

    # 2. IPCW-LGB search
    print("\n=== IPCW-LGB Hyperparameter Search ===")
    lgbm_params, lgbm_cv_h = search_lgbm_ipcw(X_arr, y, base_spl, n_draws=25)

    # 3. Build OOF for all models
    print("\n=== Build Multi-Seed OOF ===")
    for e in selected:
        print(f"  {e['name']}...")
        e["oof_p"], e["oof_r"] = build_oof_for_model(X, y, e, base_spl)
        fm = eval_oof(y, e["oof_p"], e["oof_r"], base_spl)
        print(f"    h={np.mean([m['h'] for m in fm]):.4f} "
              f"C={np.mean([m['c'] for m in fm]):.4f} "
              f"WB={np.mean([m['wb'] for m in fm]):.4f}")

    print("  IPCW-LGB...")
    lgbm_oof_p, lgbm_oof_r = oof_lgbm_ipcw(X_arr, y, lgbm_params, base_spl,
                                             seeds=OOF_SEEDS)
    lgbm_fm = eval_oof(y, lgbm_oof_p, lgbm_oof_r, base_spl)
    lgbm_h   = float(np.mean([m["h"]  for m in lgbm_fm]))
    lgbm_c   = float(np.mean([m["c"]  for m in lgbm_fm]))
    lgbm_wb  = float(np.mean([m["wb"] for m in lgbm_fm]))
    lgbm_std = float(np.std([m["h"]   for m in lgbm_fm]))
    print(f"    h={lgbm_h:.4f} C={lgbm_c:.4f} WB={lgbm_wb:.4f}")

    ipcw_entry = {"name": "IPCW-LGB", "oof_p": lgbm_oof_p, "oof_r": lgbm_oof_r,
                  "cv_mean_h": lgbm_h, "cv_std_h": lgbm_std,
                  "cv_mean_c": lgbm_c, "cv_mean_wb": lgbm_wb,
                  "robust": robust_score(lgbm_h, lgbm_std)}
    all_entries = selected + [ipcw_entry]

    # 4. Blend optimisation
    print("\n=== Blend Optimisation ===")
    weights = optimize_blend_weights(all_entries, y, base_spl)
    print("Blend weights:")
    for e, w in zip(all_entries, weights):
        print(f"  {e['name']}: {w:.4f}")
    blend_oof   = mono(sum(w * e["oof_p"] for w, e in zip(weights, all_entries)))
    blend_oof_r = sum(w * e["oof_r"]      for w, e in zip(weights, all_entries))
    bm = eval_oof(y, blend_oof, blend_oof_r, base_spl)
    b_mean = float(np.mean([m["h"]  for m in bm]))
    b_std  = float(np.std([m["h"]   for m in bm]))
    b_c    = float(np.mean([m["c"]  for m in bm]))
    b_wb   = float(np.mean([m["wb"] for m in bm]))
    print(f"Blend CV: h={b_mean:.4f}+-{b_std:.4f}  C={b_c:.4f}  WB={b_wb:.4f}")

    # 5. Calibration
    print("\n=== Calibration Selection ===")
    cal = choose_calibration(blend_oof, blend_oof_r, y, base_spl)
    print("Scores:", {k: round(v, 4) for k, v in cal["scores"].items()})
    print(f"Chosen: {cal['best_name']}")

    # 6. Final test predictions
    print("\n=== Final Test Predictions ===")
    surv_test = [fit_predict_test_model(e, X, y, Xt) for e in selected]
    ipcw_test  = fit_predict_ipcw_test(X_arr, y, Xt_arr, lgbm_params)
    surv_w, ipcw_w = weights[:len(selected)], weights[len(selected)]
    blend_test = mono(sum(w * p for w, p in zip(surv_w, surv_test)) + ipcw_w * ipcw_test)
    blend_cal  = apply_calibration(blend_test, cal["calibrator"])
    final_test = clip_safe(blend_cal)

    # 7. Write submissions
    sub = write_submission(DATA_DIR / "submission.csv", tids, final_test)
    write_submission(DATA_DIR / "submission_blend.csv", tids, blend_test)
    mono_ok = ((sub["prob_12h"] <= sub["prob_24h"] + 1e-9).all()
               and (sub["prob_24h"] <= sub["prob_48h"] + 1e-9).all()
               and (sub["prob_48h"] <= sub["prob_72h"] + 1e-9).all())
    print("\nSubmission stats:")
    for col in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
        v = sub[col]
        print(f"  {col}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")
    print(f"  monotonic_ok={mono_ok}")

    # 8. Manifest
    manifest = {
        "version": "v7",
        "runtime_seconds": round(time.time() - t0, 2),
        "data": {"train_rows": int(X.shape[0]), "test_rows": int(Xt.shape[0]),
                 "features": int(X.shape[1])},
        "selected_survival_models": [
            {"name": e["name"], "cv_mean_h": e["cv_mean_h"], "cv_std_h": e["cv_std_h"],
             "cv_mean_c": e["cv_mean_c"], "cv_mean_wb": e["cv_mean_wb"],
             "robust": e["robust"],
             "params": {k: v for k, v in e["params"].items()
                        if k not in ("random_state", "n_jobs")}}
            for e in selected],
        "ipcw_lgbm": {"cv_h": lgbm_h, "cv_c": lgbm_c, "cv_wb": lgbm_wb,
                      "params": lgbm_params},
        "blend_weights": {e["name"]: float(w) for e, w in zip(all_entries, weights)},
        "blend_cv": {"mean_h": b_mean, "std_h": b_std, "mean_c": b_c, "mean_wb": b_wb},
        "calibration": {"chosen": cal["best_name"], "scores": cal["scores"]},
        "output_files": ["submission.csv", "submission_blend.csv"],
    }
    MANIFEST_P.write_text(json.dumps(manifest, indent=2))
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal runtime: {elapsed:.1f} minutes")
    print(f"\n{'='*60}")
    print(f"FINAL CV HYBRID SCORE: {b_mean:.4f} +- {b_std:.4f}")
    print(f"  C-index: {b_c:.4f}    Weighted Brier: {b_wb:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
