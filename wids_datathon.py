#!/usr/bin/env python3
"""
WiDS Global Datathon 2026 - v8 Direct-MSE Hybrid Ensemble
==========================================================

Critical fix: use direct MSE Brier (matching competition metric) instead
of IPCW Brier from scikit-survival. Since event=0 means "never hit",
there is no censoring ambiguity — we know the complete truth at every
time horizon.

Models: RSF, GBSA (survival) + XGBoost, LightGBM, GBC, RFC (binary)
Ensemble: per-horizon blend weight optimisation
Feature selection: GBSA-importance based pruning

Install
-------
    pip install pandas numpy scikit-survival scikit-learn scipy xgboost lightgbm

Run
---
    /opt/anaconda3/bin/python3 wids_datathon.py
"""

from __future__ import annotations

import gc
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PRED_TIMES = np.array([12.0, 24.0, 48.0, 72.0])
BRIER_TIMES = np.array([24.0, 48.0, 72.0])
BRIER_W = np.array([0.3, 0.4, 0.3])
N_FOLDS = 5
FINAL_SEEDS = [42, 123, 456, 789, 2024]
PROB_FLOOR = 0.003
PROB_CEIL = 0.997

DATA_DIR = Path(__file__).resolve().parent
TRAIN_P = DATA_DIR / "train.csv"
TEST_P = DATA_DIR / "test.csv"
SAMPLE_P = DATA_DIR / "sample_submission.csv"
ID_COL = "event_id"
TARGETS = ["event", "time_to_hit_hours"]


# ═══════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dist = d["dist_min_ci_0_5h"]
    spd = d["closing_speed_m_per_h"]
    aspd = d["closing_speed_abs_m_per_h"]
    area = d["area_first_ha"]
    gr = d["area_growth_rate_ha_per_h"]
    rad = d["radial_growth_rate_m_per_h"]
    aln = d["alignment_abs"]
    alc = d["alignment_cos"]
    along = d["along_track_speed"]
    cspd = d["centroid_speed_m_per_h"]
    dslope = d["dist_slope_ci_0_5h"]
    dstd = d["dist_std_ci_0_5h"]
    daccel = d["dist_accel_m_per_h2"]
    r2 = d["dist_fit_r2_0_5h"]
    proj = d["projected_advance_m"]
    dt = d["dt_first_last_0_5h"]
    nper = d["num_perimeters_0_5h"]
    cross = d["cross_track_component"]

    safe_spd = spd.replace(0, np.nan)
    safe_rad = rad.replace(0, np.nan)
    safe_dt = dt.replace(0, np.nan)

    d["eta_close"] = (dist / safe_spd).clip(-500, 500).fillna(999)
    d["eta_radial"] = (dist / safe_rad).clip(-500, 500).fillna(999)

    d["risk_proxy"] = spd * aln / (dist + 1.0)
    d["risk_proxy_sq"] = d["risk_proxy"] ** 2
    d["risk_area_dist"] = area / (dist + 1.0)
    d["risk_composite"] = (spd * aln * area) / (dist ** 2 + 1.0)

    d["log_dist"] = np.log1p(dist)
    d["log_area"] = np.log1p(area)
    d["sqrt_area"] = np.sqrt(area)
    d["sqrt_dist"] = np.sqrt(dist)
    d["log_spd"] = np.log1p(aspd)

    d["area_x_growth"] = area * gr
    d["fire_intensity"] = np.sqrt(area + 1) * gr
    d["radial_x_area"] = rad * np.sqrt(area + 1)

    for h in [12, 24, 48, 72]:
        d[f"reach_{h}"] = rad * h
        d[f"deficit_{h}"] = dist - spd * h
        d[f"phit_{h}"] = (dist < spd * h).astype(float)
        d[f"eta_frac_{h}"] = (dist / (safe_spd * h + 1)).clip(0, 10).fillna(10)
        d[f"accel_{h}"] = spd + daccel * h

    d["close_frac"] = spd / (dist + 1.0)
    d["rad_frac"] = rad / (dist + 1.0)
    d["spd_ratio_rc"] = rad / (aspd + 1.0)
    d["spd_ratio_cc"] = spd / (cspd + 1.0)

    d["dir_momentum"] = along * aln
    d["align_x_speed"] = alc * spd
    d["cross_abs"] = cross.abs()
    d["align_x_close_frac"] = alc * spd / (dist + 1.0)

    d["perim_dens"] = (nper / safe_dt).fillna(0)
    d["adv_ratio"] = proj / (dist + 1.0)

    d["rel_close"] = spd * r2
    d["close_sq"] = spd ** 2
    d["dist_sq"] = dist ** 2

    d["hr_sin"] = np.sin(2 * np.pi * d["event_start_hour"] / 24)
    d["hr_cos"] = np.cos(2 * np.pi * d["event_start_hour"] / 24)
    d["mo_sin"] = np.sin(2 * np.pi * d["event_start_month"] / 12)
    d["mo_cos"] = np.cos(2 * np.pi * d["event_start_month"] / 12)

    d["dstd_x_close"] = dstd * spd
    d["slope_x_align"] = dslope * aln
    d["slope_x_speed"] = dslope * spd
    d["dist_x_align"] = dist * aln
    d["dist_change_norm"] = d["dist_change_ci_0_5h"] / (dist + 1.0)

    d["perimeter_est"] = 2 * np.sqrt(np.pi * area)
    d["fire_aspect"] = rad / (gr + 0.01)
    d["spread_efficiency"] = proj / (cspd * safe_dt + 1).fillna(0)
    d["r2_x_deficit_48"] = r2 * d["deficit_48"]
    d["r2_x_risk"] = r2 * d["risk_proxy"]

    # v10: ranking-focused features (top correlates with time_to_hit)
    d["nper_x_align"] = nper * aln
    d["nper_x_spd"] = nper * spd
    d["nper_x_dist_inv"] = nper / (dist + 1.0)
    d["dt_x_align"] = dt * aln
    d["lowtres_x_dist"] = d["low_temporal_resolution_0_5h"] * dist
    d["lowtres_x_risk"] = d["low_temporal_resolution_0_5h"] * d["risk_proxy"]
    d["bear_sin_x_spd"] = d["spread_bearing_sin"] * spd
    d["bear_cos_x_dist"] = d["spread_bearing_cos"] * dist
    d["growth_dist_ratio"] = gr / (dist + 1.0)
    d["cspd_x_align"] = cspd * aln
    d["area_growth_dist"] = area * gr / (dist + 1.0)

    # ETA with acceleration correction
    safe_accel_spd = (spd + daccel * 12).clip(0.01, None)
    d["eta_accel_12"] = (dist / safe_accel_spd).clip(-200, 200).fillna(999)

    d.fillna(0, inplace=True)
    d.replace([np.inf, -np.inf], 0, inplace=True)
    return d


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    tr = pd.read_csv(TRAIN_P)
    te = pd.read_csv(TEST_P)
    feat = [c for c in tr.columns if c not in TARGETS + [ID_COL]]
    Xtr = engineer(tr[feat])
    Xte = engineer(te[feat])
    y_surv = np.array(
        list(zip(tr["event"].astype(bool), tr["time_to_hit_hours"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
    return Xtr, Xte, y_surv, te[ID_COL].values, tr


def make_binary_targets(train_df):
    event = train_df["event"].values
    tth = train_df["time_to_hit_hours"].values
    return {t: ((event == 1) & (tth <= t)).astype(int) for t in PRED_TIMES}


# ═══════════════════════════════════════════════════════════════════════
# Metric: Direct MSE Brier (matching competition scoring)
# ═══════════════════════════════════════════════════════════════════════

def mse_brier(probs, bin_targets_slice):
    """Weighted Brier score via direct MSE at horizons 24, 48, 72."""
    bs = []
    for j, t in enumerate(BRIER_TIMES):
        yt = bin_targets_slice[t]
        p = probs[:, j + 1]  # cols 1,2,3 for 24,48,72
        bs.append(float(np.mean((p - yt) ** 2)))
    return float(np.average(bs, weights=BRIER_W))


def hybrid_score(c_index, wb):
    return 0.3 * c_index + 0.7 * (1.0 - wb)


def eval_fold(y_va, probs_va, bt_va):
    """Evaluate one fold: C-index (from prob_72h) + direct-MSE Brier → hybrid.
    Uses prob_72h as risk to match competition evaluation."""
    risk = probs_va[:, 3]  # prob_72h = risk score for C-index
    c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
    wb = mse_brier(probs_va, bt_va)
    return {"c": c, "wb": wb, "h": hybrid_score(c, wb)}


def eval_cv(y_surv, probs, bin_targets, skf):
    """Full CV evaluation using direct MSE Brier + prob_72h C-index."""
    ms = []
    for _, va_i in skf.split(np.arange(len(y_surv)), y_surv["event"]):
        bt_va = {t: bin_targets[t][va_i] for t in PRED_TIMES}
        m = eval_fold(y_surv[va_i], probs[va_i], bt_va)
        ms.append(m)
    return ms


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def sf2prob(surv_fns, times=PRED_TIMES):
    n = len(surv_fns)
    out = np.zeros((n, len(times)))
    for i, sf in enumerate(surv_fns):
        xv, yv = sf.x, sf.y
        for j, t in enumerate(times):
            if t <= xv[0]:
                s = yv[0]
            elif t >= xv[-1]:
                s = yv[-1]
            else:
                s = yv[np.searchsorted(xv, t, side="right") - 1]
            out[i, j] = 1.0 - s
    return out


def mono(p):
    return np.maximum.accumulate(np.clip(p, 0.0, 1.0), axis=1)


def clip_safe(p):
    return mono(np.clip(p, PROB_FLOOR, PROB_CEIL))


def _draw(grid, n, rng):
    keys = list(grid)
    pool = list(itertools.product(*(grid[k] for k in keys)))
    rng.shuffle(pool)
    return [dict(zip(keys, c)) for c in pool[:n]]


# ═══════════════════════════════════════════════════════════════════════
# Feature selection
# ═══════════════════════════════════════════════════════════════════════

def select_features(X, y_surv, n_keep=40):
    imp_acc = np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for seed in [42, 123, 789]:
        for tr_i, _ in skf.split(X, y_surv["event"]):
            m = GradientBoostingSurvivalAnalysis(
                n_estimators=150, max_depth=3, learning_rate=0.1,
                subsample=0.8, min_samples_leaf=5, random_state=seed,
            )
            m.fit(X.iloc[tr_i], y_surv[tr_i])
            imp_acc += m.feature_importances_
    ranked = np.argsort(imp_acc)[::-1]
    keep = list(X.columns[ranked[:n_keep]])
    print(f"  Feature selection: kept {len(keep)}/{X.shape[1]} features")
    top10 = [(X.columns[ranked[i]], imp_acc[ranked[i]]) for i in range(min(10, n_keep))]
    for fname, fval in top10:
        print(f"    {fname:30s} imp={fval:.4f}")
    return keep


# ═══════════════════════════════════════════════════════════════════════
# PART A: Survival model OOF
# ═══════════════════════════════════════════════════════════════════════

SURV_CONFIGS = [
    {
        "name": "RSF",
        "cls": RandomSurvivalForest,
        "grid": {
            "n_estimators": [300, 500, 800, 1000],
            "max_depth": [5, 7, 10, None],
            "min_samples_leaf": [2, 3, 5, 7],
            "max_features": [0.3, 0.5, 0.7],
        },
        "fixed": {"random_state": RANDOM_STATE, "n_jobs": 2},
        "n_draw": 20,
    },
    {
        "name": "GBSA",
        "cls": GradientBoostingSurvivalAnalysis,
        "grid": {
            "n_estimators": [100, 150, 200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.12, 0.15],
            "max_depth": [1, 2, 3, 4],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [3, 5, 7, 10],
        },
        "fixed": {"random_state": RANDOM_STATE},
        "n_draw": 35,
    },
]

# CoxPH needs standardised features, handled separately
COXPH_GRID = {
    "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
}
COXPH_N_DRAW = 7


def surv_oof(X, y, cls, params, skf, seeds):
    n = len(y)
    acc_p = np.zeros((n, 4))
    acc_r = np.zeros(n)
    for seed in seeds:
        par = dict(params)
        if "random_state" in par:
            par["random_state"] = seed
        for tr_i, va_i in skf.split(X, y["event"]):
            m = cls(**par)
            m.fit(X.iloc[tr_i], y[tr_i])
            acc_p[va_i] += mono(sf2prob(m.predict_survival_function(X.iloc[va_i])))
            acc_r[va_i] += m.predict(X.iloc[va_i])
    acc_p /= len(seeds)
    acc_r /= len(seeds)
    return acc_p, acc_r


def search_surv(X, y, bin_targets, skf, rng):
    results = []
    best_c_gbsa = None  # track best C-index GBSA separately
    for cfg in SURV_CONFIGS:
        combos = _draw(cfg["grid"], cfg["n_draw"], rng)
        print(f"\n  Survival: {cfg['name']} ({len(combos)} configs)")
        best_h, best = -np.inf, None
        best_c_val = -np.inf
        for i, combo in enumerate(combos, 1):
            params = {**cfg["fixed"], **combo}
            try:
                oof_p, oof_r = surv_oof(X, y, cfg["cls"], params, skf, [RANDOM_STATE])
            except Exception as e:
                print(f"    [{i:2d}] FAIL: {e}")
                continue
            ms = eval_cv(y, oof_p, bin_targets, skf)
            mh = float(np.mean([m["h"] for m in ms]))
            mc = float(np.mean([m["c"] for m in ms]))
            sh = float(np.std([m["h"] for m in ms]))
            tag = " ***" if mh > best_h else ""
            print(f"    [{i:2d}] h={mh:.4f}±{sh:.4f}  C={mc:.4f}  WB={np.mean([m['wb'] for m in ms]):.4f}{tag}")
            if mh > best_h:
                best_h = mh
                best = {"name": cfg["name"], "cls": cfg["cls"], "params": params,
                        "oof_p": oof_p, "oof_r": oof_r, "cv": mh}
            if cfg["name"] == "GBSA" and mc > best_c_val:
                best_c_val = mc
                best_c_gbsa = {"name": "GBSA-rank", "cls": cfg["cls"], "params": params,
                               "oof_p": oof_p, "oof_r": oof_r, "cv": mh,
                               "cv_c": mc}
            gc.collect()
        if best:
            results.append(best)
    return results, best_c_gbsa


def refine_gbsa(X, y, bin_targets, skf, best_params, n_refine=30):
    """Stage 2: fine-grained search around the best GBSA params."""
    b = {k: v for k, v in best_params.items() if k not in ("random_state",)}
    ref_grid = {
        "n_estimators": sorted({max(50, b["n_estimators"] - 100),
                                max(50, b["n_estimators"] - 50),
                                b["n_estimators"],
                                b["n_estimators"] + 50,
                                b["n_estimators"] + 100,
                                b["n_estimators"] + 200}),
        "learning_rate": sorted({max(0.005, round(b["learning_rate"] * 0.6, 4)),
                                 max(0.005, round(b["learning_rate"] * 0.8, 4)),
                                 round(b["learning_rate"], 4),
                                 round(b["learning_rate"] * 1.2, 4),
                                 min(0.3, round(b["learning_rate"] * 1.5, 4))}),
        "max_depth": sorted({max(1, b["max_depth"] - 1),
                             b["max_depth"],
                             b["max_depth"] + 1}),
        "subsample": sorted({max(0.5, round(b["subsample"] - 0.1, 2)),
                             max(0.5, round(b["subsample"] - 0.05, 2)),
                             round(b["subsample"], 2),
                             min(1.0, round(b["subsample"] + 0.05, 2)),
                             min(1.0, round(b["subsample"] + 0.1, 2))}),
        "min_samples_leaf": sorted({max(1, b["min_samples_leaf"] - 2),
                                    max(1, b["min_samples_leaf"] - 1),
                                    b["min_samples_leaf"],
                                    b["min_samples_leaf"] + 1,
                                    b["min_samples_leaf"] + 2}),
    }
    rng2 = np.random.default_rng(123)
    combos = _draw(ref_grid, n_refine, rng2)
    print(f"\n  GBSA Stage 2 refinement ({len(combos)} configs around best)")
    best_h, best = -np.inf, None
    for i, combo in enumerate(combos, 1):
        params = {"random_state": RANDOM_STATE, **combo}
        try:
            oof_p, oof_r = surv_oof(X, y, GradientBoostingSurvivalAnalysis, params, skf, [RANDOM_STATE])
        except Exception as e:
            print(f"    [{i:2d}] FAIL: {e}")
            continue
        ms = eval_cv(y, oof_p, bin_targets, skf)
        mh = float(np.mean([m["h"] for m in ms]))
        sh = float(np.std([m["h"] for m in ms]))
        tag = " ***" if mh > best_h else ""
        print(f"    [{i:2d}] h={mh:.4f}±{sh:.4f}  C={np.mean([m['c'] for m in ms]):.4f}  WB={np.mean([m['wb'] for m in ms]):.4f}{tag}")
        if mh > best_h:
            best_h = mh
            best = {"name": "GBSA", "cls": GradientBoostingSurvivalAnalysis,
                    "params": params, "oof_p": oof_p, "oof_r": oof_r, "cv": mh}
        gc.collect()
    return best


def search_gbsa_rank(X, y, bin_targets, skf, rng, n_draw=40, top_k=2):
    """Dedicated GBSA search optimizing for C-index (ranking quality).
    Returns list of top-k candidates sorted by C-index."""
    grid = {
        "n_estimators": [50, 100, 150, 200, 300, 500, 800],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_leaf": [3, 5, 7, 10, 15, 20],
    }
    combos = _draw(grid, n_draw, rng)
    print(f"\n  GBSA C-index search ({len(combos)} configs)")
    top_list = []
    for i, combo in enumerate(combos, 1):
        params = {"random_state": RANDOM_STATE, **combo}
        try:
            oof_p, oof_r = surv_oof(X, y, GradientBoostingSurvivalAnalysis, params, skf, [RANDOM_STATE])
        except Exception as e:
            print(f"    [{i:2d}] FAIL: {e}")
            continue
        ms = eval_cv(y, oof_p, bin_targets, skf)
        mc = float(np.mean([m["c"] for m in ms]))
        mh = float(np.mean([m["h"] for m in ms]))
        tag = ""
        if not top_list or mc > top_list[0]["cv_c"]:
            tag = " ***"
        print(f"    [{i:2d}] C={mc:.4f}  h={mh:.4f}  WB={np.mean([m['wb'] for m in ms]):.4f}{tag}")
        top_list.append({"cls": GradientBoostingSurvivalAnalysis,
                         "params": params, "oof_p": oof_p, "oof_r": oof_r,
                         "cv": mh, "cv_c": mc})
        gc.collect()
    top_list.sort(key=lambda x: -x["cv_c"])
    result = []
    for idx, entry in enumerate(top_list[:top_k]):
        entry["name"] = f"GBSA-rank{idx+1}" if top_k > 1 else "GBSA-rank"
        result.append(entry)
    return result


def search_coxph(X, y, bin_targets, skf, rng):
    """Search CoxPH with standardised features."""
    combos = _draw(COXPH_GRID, COXPH_N_DRAW, rng)
    print(f"\n  Survival: CoxPH ({len(combos)} configs)")
    best_h, best = -np.inf, None
    for i, combo in enumerate(combos, 1):
        n = len(y)
        oof_p = np.zeros((n, 4))
        oof_r = np.zeros(n)
        try:
            for tr_i, va_i in skf.split(X, y["event"]):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X.iloc[tr_i])
                Xva = sc.transform(X.iloc[va_i])
                m = CoxPHSurvivalAnalysis(alpha=combo["alpha"])
                m.fit(Xtr, y[tr_i])
                oof_p[va_i] = mono(sf2prob(m.predict_survival_function(Xva)))
                oof_r[va_i] = m.predict(Xva)
        except Exception as e:
            print(f"    [{i:2d}] FAIL: {e}")
            continue
        ms = eval_cv(y, oof_p, bin_targets, skf)
        mh = float(np.mean([m["h"] for m in ms]))
        sh = float(np.std([m["h"] for m in ms]))
        tag = " ***" if mh > best_h else ""
        print(f"    [{i:2d}] a={combo['alpha']:.2f}  h={mh:.4f}±{sh:.4f}  C={np.mean([m['c'] for m in ms]):.4f}  WB={np.mean([m['wb'] for m in ms]):.4f}{tag}")
        if mh > best_h:
            best_h = mh
            best = {"name": "CoxPH", "cls": CoxPHSurvivalAnalysis,
                    "params": combo, "oof_p": oof_p, "oof_r": oof_r,
                    "cv": mh, "needs_scale": True}
        gc.collect()
    return best


# ═══════════════════════════════════════════════════════════════════════
# PART B: Binary classification OOF
# ═══════════════════════════════════════════════════════════════════════

def bin_oof_single(X, bin_targets, fit_fn, skf, seeds):
    n = len(next(iter(bin_targets.values())))
    acc_p = np.zeros((n, 4))
    for seed in seeds:
        for j, t in enumerate(PRED_TIMES):
            yt = bin_targets[t]
            for tr_i, va_i in skf.split(X, yt):
                acc_p[va_i, j] += fit_fn(X.iloc[tr_i], yt[tr_i], X.iloc[va_i], seed)
    acc_p /= len(seeds)
    return mono(acc_p)


def _gbc_fit(Xtr, ytr, Xva, seed, **hp):
    m = GradientBoostingClassifier(random_state=seed, **hp)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def _xgb_fit(Xtr, ytr, Xva, seed, **hp):
    m = xgb.XGBClassifier(
        random_state=seed, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=2, **hp
    )
    m.fit(Xtr, ytr, verbose=False)
    return m.predict_proba(Xva)[:, 1]


def _lgb_fit(Xtr, ytr, Xva, seed, **hp):
    m = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=2, **hp)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def _rfc_fit(Xtr, ytr, Xva, seed, **hp):
    m = RandomForestClassifier(random_state=seed, n_jobs=2, **hp)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def _etc_fit(Xtr, ytr, Xva, seed, **hp):
    m = ExtraTreesClassifier(random_state=seed, n_jobs=2, **hp)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


BIN_CONFIGS = [
    {
        "name": "XGB",
        "fit_fn": _xgb_fit,
        "grid": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.5, 0.7, 0.9],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [1, 3, 5],
            "min_child_weight": [3, 5, 10],
        },
        "n_draw": 25,
    },
    {
        "name": "LGB",
        "fit_fn": _lgb_fit,
        "grid": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "max_depth": [3, 4, 5, 7, -1],
            "num_leaves": [7, 15, 31, 63],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.5, 0.7, 0.9],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 1, 3, 5],
            "min_child_samples": [5, 10, 20],
        },
        "n_draw": 30,
    },
    {
        "name": "GBC",
        "fit_fn": _gbc_fit,
        "grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.02, 0.05, 0.08, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.8, 0.9],
            "min_samples_leaf": [5, 10, 15],
        },
        "n_draw": 15,
    },
    {
        "name": "RFC",
        "fit_fn": _rfc_fit,
        "grid": {
            "n_estimators": [300, 500],
            "max_depth": [5, 7, 10, None],
            "min_samples_leaf": [3, 5, 10],
            "max_features": [0.3, 0.5, "sqrt"],
        },
        "n_draw": 10,
    },
    # ETC removed: consistently worst model, adds noise to blend optimizer
]


def search_bin(X, y_surv, bin_targets, skf, rng):
    results = []
    for cfg in BIN_CONFIGS:
        combos = _draw(cfg["grid"], cfg["n_draw"], rng)
        print(f"\n  Binary: {cfg['name']} ({len(combos)} configs)")
        best_h, best = -np.inf, None
        for i, combo in enumerate(combos, 1):
            try:
                fit = lambda Xtr, ytr, Xva, seed, _hp=combo, _fn=cfg["fit_fn"]: _fn(Xtr, ytr, Xva, seed, **_hp)
                oof_p = bin_oof_single(X, bin_targets, fit, skf, [RANDOM_STATE])
            except Exception as e:
                print(f"    [{i:2d}] FAIL: {e}")
                continue
            ms = eval_cv(y_surv, oof_p, bin_targets, skf)
            mh = float(np.mean([m["h"] for m in ms]))
            sh = float(np.std([m["h"] for m in ms]))
            tag = " ***" if mh > best_h else ""
            print(f"    [{i:2d}] h={mh:.4f}±{sh:.4f}  C={np.mean([m['c'] for m in ms]):.4f}  WB={np.mean([m['wb'] for m in ms]):.4f}{tag}")
            if mh > best_h:
                best_h = mh
                best = {"name": cfg["name"], "fit_fn": cfg["fit_fn"],
                        "params": combo, "oof_p": oof_p, "cv": mh}
            gc.collect()
        if best:
            results.append(best)
    return results


# ═══════════════════════════════════════════════════════════════════════
# PART C: Blend weight optimisation
# ═══════════════════════════════════════════════════════════════════════

def optimise_global_blend(all_candidates, y_surv, bin_targets, skf,
                          entropy_lambda=0.0):
    splits = list(skf.split(np.arange(len(y_surv)), y_surv["event"]))

    def objective(w_raw):
        w = np.exp(w_raw)
        w /= w.sum()
        bp = mono(sum(wi * c["oof_p"] for wi, c in zip(w, all_candidates)))
        hs = []
        for _, va_i in splits:
            bt_va = {t: bin_targets[t][va_i] for t in PRED_TIMES}
            m = eval_fold(y_surv[va_i], bp[va_i], bt_va)
            hs.append(m["h"])
        penalty = 0.0
        if entropy_lambda > 0:
            penalty = -entropy_lambda * float(np.sum(w * np.log(w + 1e-10)))
        return -float(np.mean(hs)) - penalty

    k = len(all_candidates)
    best_w, best_v = np.zeros(k), objective(np.zeros(k))
    for _ in range(60):
        x0 = np.random.randn(k) * 0.8
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-8})
        if res.fun < best_v:
            best_v, best_w = res.fun, res.x
    w = np.exp(best_w)
    w /= w.sum()
    return w


def optimise_per_horizon_blend(all_oof, y_surv, bin_targets, skf):
    """Brier-only per-horizon (for 12h, 24h, 48h); keep for comparison."""
    splits = list(skf.split(np.arange(len(y_surv)), y_surv["event"]))
    k = len(all_oof)
    horizon_weights = np.zeros((4, k))

    for j, t in enumerate(PRED_TIMES):
        yt = bin_targets[t]

        def objective(w_raw):
            w = np.exp(w_raw)
            w /= w.sum()
            bp = np.clip(sum(wi * m[:, j] for wi, m in zip(w, all_oof)), 0, 1)
            bs_vals = []
            for _, va_i in splits:
                bs_vals.append(float(np.mean((bp[va_i] - yt[va_i]) ** 2)))
            return float(np.mean(bs_vals))

        best_w, best_v = np.zeros(k), objective(np.zeros(k))
        for _ in range(20):
            x0 = np.random.randn(k) * 0.5
            res = minimize(objective, x0, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-7})
            if res.fun < best_v:
                best_v, best_w = res.fun, res.x
        w = np.exp(best_w)
        w /= w.sum()
        horizon_weights[j] = w

    return horizon_weights


def optimise_joint_hybrid_blend(all_candidates, all_oof, y_surv, bin_targets, skf):
    """Joint per-horizon weights optimised for the full HYBRID metric.
    This ensures 72h column preserves ranking (C-index) while others
    optimise calibration (Brier)."""
    splits = list(skf.split(np.arange(len(y_surv)), y_surv["event"]))
    k = len(all_oof)
    n_params = 4 * k  # 4 horizons × k models

    def objective(w_raw_flat):
        w_mat = w_raw_flat.reshape(4, k)
        for j in range(4):
            row = np.exp(w_mat[j])
            w_mat[j] = row / row.sum()

        bp = np.column_stack([
            np.clip(sum(w_mat[j, ci] * m[:, j] for ci, m in enumerate(all_oof)), 0, 1)
            for j in range(4)
        ])
        bp = mono(bp)

        hs = []
        for _, va_i in splits:
            bt_va = {t: bin_targets[t][va_i] for t in PRED_TIMES}
            m = eval_fold(y_surv[va_i], bp[va_i], bt_va)
            hs.append(m["h"])
        return -float(np.mean(hs))

    best_w, best_v = np.zeros(n_params), objective(np.zeros(n_params))
    for _ in range(40):
        x0 = np.random.randn(n_params) * 0.3
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-8})
        if res.fun < best_v:
            best_v, best_w = res.fun, res.x

    w_mat = best_w.reshape(4, k)
    for j in range(4):
        row = np.exp(w_mat[j])
        w_mat[j] = row / row.sum()
    return w_mat


# ═══════════════════════════════════════════════════════════════════════
# PART D: Post-hoc calibration options
# ═══════════════════════════════════════════════════════════════════════

def calibrate_isotonic_cv(oof_probs, bin_targets, skf, y_surv):
    n = oof_probs.shape[0]
    cal_probs = oof_probs.copy()
    splits = list(skf.split(np.arange(n), y_surv["event"]))
    for j, t in enumerate(PRED_TIMES):
        yt = bin_targets[t]
        for fi, (_, va_i) in enumerate(splits):
            tr_idx = np.concatenate([v for fj, (_, v) in enumerate(splits) if fj != fi])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=PROB_FLOOR, y_max=PROB_CEIL)
            iso.fit(oof_probs[tr_idx, j], yt[tr_idx])
            cal_probs[va_i, j] = iso.predict(oof_probs[va_i, j])
    return mono(cal_probs)


def calibrate_platt_cv(oof_probs, bin_targets, skf, y_surv):
    n = oof_probs.shape[0]
    cal_probs = oof_probs.copy()
    splits = list(skf.split(np.arange(n), y_surv["event"]))
    for j, t in enumerate(PRED_TIMES):
        yt = bin_targets[t]
        for fi, (_, va_i) in enumerate(splits):
            tr_idx = np.concatenate([v for fj, (_, v) in enumerate(splits) if fj != fi])
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(oof_probs[tr_idx, j:j+1], yt[tr_idx])
            cal_probs[va_i, j] = lr.predict_proba(oof_probs[va_i, j:j+1])[:, 1]
    return mono(cal_probs)


def stack_lgb_cv(all_oof, y_surv, bin_targets, skf):
    """LightGBM stacking: use all base model OOF probs as features.
    Returns stacked OOF probs + fitted models for test prediction."""
    splits = list(skf.split(np.arange(len(y_surv)), y_surv["event"]))
    n = len(y_surv)
    k = len(all_oof)
    stack_oof = np.zeros((n, 4))
    stack_models = []

    for j, t in enumerate(PRED_TIMES):
        yt = bin_targets[t]
        meta_X = np.column_stack([m[:, j] for m in all_oof])
        fold_models = []
        for fi, (_, va_i) in enumerate(splits):
            tr_idx = np.concatenate([v for fj, (_, v) in enumerate(splits) if fj != fi])
            m = lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, num_leaves=7,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0, min_child_samples=10,
                random_state=RANDOM_STATE, verbose=-1, n_jobs=2,
            )
            m.fit(meta_X[tr_idx], yt[tr_idx])
            stack_oof[va_i, j] = m.predict_proba(meta_X[va_i])[:, 1]
            fold_models.append(m)
        # Final model on all data
        m_final = lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, num_leaves=7,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, min_child_samples=10,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=2,
        )
        m_final.fit(meta_X, yt)
        stack_models.append(m_final)

    return clip_safe(stack_oof), stack_models


# ═══════════════════════════════════════════════════════════════════════
# PART E: Final test predictions
# ═══════════════════════════════════════════════════════════════════════

def predict_surv_test(entry, X, y, Xt):
    preds = []
    for seed in FINAL_SEEDS:
        par = dict(entry["params"])
        if "random_state" in par:
            par["random_state"] = seed
        if "n_jobs" in par:
            par["n_jobs"] = 2
        m = entry["cls"](**par)
        m.fit(X, y)
        preds.append(mono(sf2prob(m.predict_survival_function(Xt))))
    return np.mean(preds, axis=0)


def predict_bin_test(entry, X, bin_targets, Xt):
    preds = np.zeros((len(Xt), 4))
    for seed in FINAL_SEEDS:
        for j, t in enumerate(PRED_TIMES):
            yt = bin_targets[t]
            preds[:, j] += entry["fit_fn"](X, yt, Xt, seed, **entry["params"])
    preds /= len(FINAL_SEEDS)
    return mono(preds)


# ═══════════════════════════════════════════════════════════════════════
# Submission
# ═══════════════════════════════════════════════════════════════════════

def write_sub(path, ids, probs):
    sub = pd.DataFrame({
        ID_COL: ids,
        "prob_12h": probs[:, 0], "prob_24h": probs[:, 1],
        "prob_48h": probs[:, 2], "prob_72h": probs[:, 3],
    })
    sample = pd.read_csv(SAMPLE_P)
    assert len(sub) == len(sample) and set(sub[ID_COL]) == set(sample[ID_COL])
    sub = sub.sort_values(ID_COL).reset_index(drop=True)
    sub.to_csv(path, index=False)
    return sub


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(line_buffering=True)
    start = time.time()
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("Loading data …")
    X_full, Xt_full, y_surv, tids, train_df = load_data()
    bin_targets = make_binary_targets(train_df)
    print(f"  Train : {X_full.shape[0]} × {X_full.shape[1]} features")
    print(f"  Test  : {Xt_full.shape[0]} rows")
    print(f"  Events: {int(y_surv['event'].sum())} / {len(y_surv)}")
    for t in PRED_TIMES:
        print(f"  Positives at {int(t)}h: {int(bin_targets[t].sum())}")

    print("\nFeature selection …")
    keep = select_features(X_full, y_surv, n_keep=40)
    X = X_full[keep].copy()
    Xt = Xt_full[keep].copy()

    # ── PART A ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PART A: Survival Model Search (direct-MSE metric)")
    print("=" * 64)
    surv_results, best_c_gbsa = search_surv(X, y_surv, bin_targets, skf, rng)
    gc.collect()

    # Stage 2: refine around best GBSA (hybrid)
    gbsa_entry = next((r for r in surv_results if r["name"] == "GBSA"), None)
    if gbsa_entry:
        refined = refine_gbsa(X, y_surv, bin_targets, skf, gbsa_entry["params"], n_refine=30)
        if refined and refined["cv"] > gbsa_entry["cv"]:
            print(f"  GBSA refined: {gbsa_entry['cv']:.4f} → {refined['cv']:.4f}")
            idx = next(i for i, r in enumerate(surv_results) if r["name"] == "GBSA")
            surv_results[idx] = refined
        else:
            print(f"  GBSA refinement did not improve (stage1 {gbsa_entry['cv']:.4f})")

    # Dedicated GBSA C-index search (ranking models for tie-breaking)
    print("\n" + "=" * 64)
    print("PART A2: GBSA Ranking Model Search (C-index focused)")
    print("=" * 64)
    gbsa_rank_list = search_gbsa_rank(X, y_surv, bin_targets, skf,
                                       np.random.default_rng(777), n_draw=40, top_k=2)

    # Also consider the passive C-index tracker from main search
    if best_c_gbsa:
        already_better = any(r["cv_c"] >= best_c_gbsa.get("cv_c", 0) for r in gbsa_rank_list)
        if not already_better:
            best_c_gbsa["name"] = "GBSA-rank0"
            gbsa_rank_list.insert(0, best_c_gbsa)

    gbsa_h_entry = next((r for r in surv_results if r["name"] == "GBSA"), None)
    for rank_entry in gbsa_rank_list:
        if gbsa_h_entry and np.array_equal(rank_entry["oof_p"], gbsa_h_entry["oof_p"]):
            continue
        print(f"  {rank_entry['name']} added (C={rank_entry['cv_c']:.4f}, h={rank_entry['cv']:.4f})")
        surv_results.append(rank_entry)
    gc.collect()

    # ── PART B ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PART B: Binary Classifier Search (direct-MSE metric)")
    print("=" * 64)
    bin_results = search_bin(X, y_surv, bin_targets, skf, rng)
    gc.collect()

    # ── Multi-seed refinement ─────────────────────────────────────────
    refine_seeds = [42, 123, 789]
    print(f"\nMulti-seed OOF refinement ({len(refine_seeds)} seeds) …")
    for r in surv_results:
        if r.get("needs_scale"):
            print(f"  Refining {r['name']} (scaled) …")
            n = len(y_surv)
            oof_p = np.zeros((n, 4))
            oof_r = np.zeros(n)
            for seed in refine_seeds:
                for tr_i, va_i in skf.split(X, y_surv["event"]):
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(X.iloc[tr_i])
                    Xva = sc.transform(X.iloc[va_i])
                    m = CoxPHSurvivalAnalysis(**r["params"])
                    m.fit(Xtr, y_surv[tr_i])
                    oof_p[va_i] += mono(sf2prob(m.predict_survival_function(Xva)))
                    oof_r[va_i] += m.predict(Xva)
            r["oof_p"] = oof_p / len(refine_seeds)
            r["oof_r"] = oof_r / len(refine_seeds)
        else:
            print(f"  Refining {r['name']} …")
            r["oof_p"], r["oof_r"] = surv_oof(X, y_surv, r["cls"], r["params"], skf, refine_seeds)
        gc.collect()
    for r in bin_results:
        print(f"  Refining {r['name']} …")
        fit = lambda Xtr, ytr, Xva, seed, _hp=r["params"], _fn=r["fit_fn"]: _fn(Xtr, ytr, Xva, seed, **_hp)
        r["oof_p"] = bin_oof_single(X, bin_targets, fit, skf, refine_seeds)
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'#' * 64}")
    print("Base model summary (direct-MSE + prob_72h C-index):")
    for r in surv_results:
        ms = eval_cv(y_surv, r["oof_p"], bin_targets, skf)
        mh = np.mean([m["h"] for m in ms])
        mc = np.mean([m["c"] for m in ms])
        mwb = np.mean([m["wb"] for m in ms])
        p = {k: v for k, v in r["params"].items() if k not in ("random_state", "n_jobs")}
        print(f"  [surv] {r['name']:6s}  h={mh:.4f}  C={mc:.4f}  WB={mwb:.4f}  {p}")
    for r in bin_results:
        ms = eval_cv(y_surv, r["oof_p"], bin_targets, skf)
        mh = np.mean([m["h"] for m in ms])
        mc = np.mean([m["c"] for m in ms])
        mwb = np.mean([m["wb"] for m in ms])
        print(f"  [bin]  {r['name']:6s}  h={mh:.4f}  C={mc:.4f}  WB={mwb:.4f}  {r['params']}")
    print(f"{'#' * 64}")

    # ── PART C: Global blend ──────────────────────────────────────────
    print("\nPART C: Global blend optimisation (hybrid = C-index + Brier) …")
    all_candidates = []
    for r in surv_results:
        all_candidates.append(r)
    for r in bin_results:
        all_candidates.append({"oof_p": r["oof_p"], "name": r["name"], "cv": r["cv"]})

    g_weights = optimise_global_blend(all_candidates, y_surv, bin_targets, skf,
                                      entropy_lambda=0.0)
    print("  Global blend weights:")
    for c, w in zip(all_candidates, g_weights):
        print(f"    {c['name']:6s}  w={w:.4f}")

    g_blend_oof = mono(sum(w * c["oof_p"] for w, c in zip(g_weights, all_candidates)))
    gms = eval_cv(y_surv, g_blend_oof, bin_targets, skf)
    g_h = float(np.mean([m["h"] for m in gms]))
    print(f"  Global blend CV: h={g_h:.4f}  C={np.mean([m['c'] for m in gms]):.4f}  WB={np.mean([m['wb'] for m in gms]):.4f}")

    # Regularized blend (encourages weight diversity)
    gr_weights = optimise_global_blend(all_candidates, y_surv, bin_targets, skf,
                                       entropy_lambda=0.002)
    print("  Regularized blend weights:")
    for c, w in zip(all_candidates, gr_weights):
        print(f"    {c['name']:6s}  w={w:.4f}")
    gr_blend_oof = mono(sum(w * c["oof_p"] for w, c in zip(gr_weights, all_candidates)))
    grms = eval_cv(y_surv, gr_blend_oof, bin_targets, skf)
    gr_h = float(np.mean([m["h"] for m in grms]))
    print(f"  Regularized blend CV: h={gr_h:.4f}  C={np.mean([m['c'] for m in grms]):.4f}  WB={np.mean([m['wb'] for m in grms]):.4f}")

    # Simple average baseline
    avg_oof = mono(np.mean([c["oof_p"] for c in all_candidates], axis=0))
    avg_ms = eval_cv(y_surv, avg_oof, bin_targets, skf)
    avg_h = float(np.mean([m["h"] for m in avg_ms]))
    print(f"  Simple average CV: h={avg_h:.4f}  C={np.mean([m['c'] for m in avg_ms]):.4f}  WB={np.mean([m['wb'] for m in avg_ms]):.4f}")

    # Top-3 average (by individual hybrid)
    ranked_cands = sorted(all_candidates, key=lambda c: c["cv"], reverse=True)
    top3_oof = mono(np.mean([c["oof_p"] for c in ranked_cands[:3]], axis=0))
    top3_ms = eval_cv(y_surv, top3_oof, bin_targets, skf)
    top3_h = float(np.mean([m["h"] for m in top3_ms]))
    top3_names = [c["name"] for c in ranked_cands[:3]]
    print(f"  Top-3 average ({top3_names}) CV: h={top3_h:.4f}  C={np.mean([m['c'] for m in top3_ms]):.4f}  WB={np.mean([m['wb'] for m in top3_ms]):.4f}")

    # ── PART D: Per-horizon Brier blend ─────────────────────────────────
    print("\nPART D: Per-horizon Brier blend …")
    all_oof = [c["oof_p"] for c in all_candidates]
    h_weights = optimise_per_horizon_blend(all_oof, y_surv, bin_targets, skf)
    print("  Per-horizon Brier weights:")
    for j, t in enumerate(PRED_TIMES):
        ws = "  ".join(f"{c['name']}={h_weights[j, ci]:.3f}" for ci, c in enumerate(all_candidates))
        print(f"    {int(t):2d}h: {ws}")

    h_blend_oof = np.column_stack([
        np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(4)
    ])
    h_blend_oof = mono(h_blend_oof)
    hms = eval_cv(y_surv, h_blend_oof, bin_targets, skf)
    h_h = float(np.mean([m["h"] for m in hms]))
    print(f"  Per-horizon Brier CV: h={h_h:.4f}  C={np.mean([m['c'] for m in hms]):.4f}  WB={np.mean([m['wb'] for m in hms]):.4f}")

    # ── PART D2: Joint hybrid per-horizon blend ───────────────────────
    print("\nPART D2: Joint hybrid per-horizon blend …")
    jh_weights = optimise_joint_hybrid_blend(all_candidates, all_oof, y_surv, bin_targets, skf)
    print("  Joint hybrid weights:")
    for j, t in enumerate(PRED_TIMES):
        ws = "  ".join(f"{c['name']}={jh_weights[j, ci]:.3f}" for ci, c in enumerate(all_candidates))
        print(f"    {int(t):2d}h: {ws}")

    jh_blend_oof = np.column_stack([
        np.clip(sum(jh_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(4)
    ])
    jh_blend_oof = mono(jh_blend_oof)
    jhms = eval_cv(y_surv, jh_blend_oof, bin_targets, skf)
    jh_h = float(np.mean([m["h"] for m in jhms]))
    print(f"  Joint hybrid CV: h={jh_h:.4f}  C={np.mean([m['c'] for m in jhms]):.4f}  WB={np.mean([m['wb'] for m in jhms]):.4f}")

    # ── Custom hybrid: Brier for 12/24/48h, global for 72h ──────────
    print("\nCustom hybrid blend (Brier@12-48h, global@72h) …")
    custom_oof = np.column_stack([
        np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(3)  # 12h, 24h, 48h from per-horizon Brier
    ] + [
        g_blend_oof[:, 3]  # 72h from global blend (preserves ranking)
    ])
    custom_oof = mono(custom_oof)
    custom_ms = eval_cv(y_surv, custom_oof, bin_targets, skf)
    custom_h = float(np.mean([m["h"] for m in custom_ms]))
    print(f"  Custom hybrid CV: h={custom_h:.4f}  C={np.mean([m['c'] for m in custom_ms]):.4f}  WB={np.mean([m['wb'] for m in custom_ms]):.4f}")

    # ── C-index optimized 72h: use model with best C-index for 72h column ──
    print("\nC-optimized 72h blend …")
    cand_cindexes = []
    for c in all_candidates:
        ms_c = eval_cv(y_surv, c["oof_p"], bin_targets, skf)
        avg_c = float(np.mean([m["c"] for m in ms_c]))
        cand_cindexes.append((c["name"], avg_c))
    cand_cindexes.sort(key=lambda x: -x[1])
    print(f"  Candidate C-indexes: {[(n, f'{c:.4f}') for n, c in cand_cindexes]}")
    best_c_name = cand_cindexes[0][0]
    best_c_cand_oof = next(c for c in all_candidates if c["name"] == best_c_name)

    # Build: GBC for 12/24/48h (best Brier), best-C for 72h
    best_brier_cand = min(all_candidates, key=lambda c: float(np.mean(
        [m["wb"] for m in eval_cv(y_surv, c["oof_p"], bin_targets, skf)])))
    copt_oof = np.column_stack([
        best_brier_cand["oof_p"][:, j] for j in range(3)
    ] + [
        best_c_cand_oof["oof_p"][:, 3]
    ])
    copt_oof = mono(copt_oof)
    copt_ms = eval_cv(y_surv, copt_oof, bin_targets, skf)
    copt_h = float(np.mean([m["h"] for m in copt_ms]))
    print(f"  C-opt 72h ({best_c_name} + {best_brier_cand['name']}) CV: h={copt_h:.4f}  C={np.mean([m['c'] for m in copt_ms]):.4f}  WB={np.mean([m['wb'] for m in copt_ms]):.4f}")

    # GBC for 12/24/48h + GBC tie-broken by best-C for 72h
    eps = 1e-4
    tiebreak_72 = best_brier_cand["oof_p"][:, 3] + eps * best_c_cand_oof["oof_p"][:, 3]
    tiebreak_oof = np.column_stack([
        best_brier_cand["oof_p"][:, j] for j in range(3)
    ] + [tiebreak_72])
    tiebreak_oof = mono(np.clip(tiebreak_oof, 0, 1))
    tiebreak_ms = eval_cv(y_surv, tiebreak_oof, bin_targets, skf)
    tiebreak_h = float(np.mean([m["h"] for m in tiebreak_ms]))
    print(f"  Tiebreak 72h (eps={eps}) CV: h={tiebreak_h:.4f}  C={np.mean([m['c'] for m in tiebreak_ms]):.4f}  WB={np.mean([m['wb'] for m in tiebreak_ms]):.4f}")

    # Search for optimal tiebreak epsilon
    best_tb_eps, best_tb_h = eps, tiebreak_h
    for e in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2]:
        tb_72 = best_brier_cand["oof_p"][:, 3] + e * best_c_cand_oof["oof_p"][:, 3]
        tb = np.column_stack([best_brier_cand["oof_p"][:, j] for j in range(3)] + [tb_72])
        tb = mono(np.clip(tb, 0, 1))
        tb_ms = eval_cv(y_surv, tb, bin_targets, skf)
        tb_h = float(np.mean([m["h"] for m in tb_ms]))
        if tb_h > best_tb_h:
            best_tb_h, best_tb_eps = tb_h, e
    print(f"  Best tiebreak eps={best_tb_eps}, h={best_tb_h:.4f}")
    tiebreak_72_final = best_brier_cand["oof_p"][:, 3] + best_tb_eps * best_c_cand_oof["oof_p"][:, 3]
    tiebreak_oof_final = np.column_stack([
        best_brier_cand["oof_p"][:, j] for j in range(3)
    ] + [tiebreak_72_final])
    tiebreak_oof_final = mono(np.clip(tiebreak_oof_final, 0, 1))
    tiebreak_ms_final = eval_cv(y_surv, tiebreak_oof_final, bin_targets, skf)
    tiebreak_h_final = float(np.mean([m["h"] for m in tiebreak_ms_final]))
    print(f"  Tiebreak final: h={tiebreak_h_final:.4f}  C={np.mean([m['c'] for m in tiebreak_ms_final]):.4f}  WB={np.mean([m['wb'] for m in tiebreak_ms_final]):.4f}")

    # ── Mixed hybrid: Brier@12-48h, joint-hybrid@72h ────────────────
    print("\nMixed hybrid (Brier@12-48h, JH@72h) …")
    mixed_oof = np.column_stack([
        np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(3)
    ] + [
        jh_blend_oof[:, 3]  # 72h from joint-hybrid (best C-index)
    ])
    mixed_oof = mono(mixed_oof)
    mixed_ms = eval_cv(y_surv, mixed_oof, bin_targets, skf)
    mixed_h = float(np.mean([m["h"] for m in mixed_ms]))
    print(f"  Mixed hybrid CV: h={mixed_h:.4f}  C={np.mean([m['c'] for m in mixed_ms]):.4f}  WB={np.mean([m['wb'] for m in mixed_ms]):.4f}")

    # ── Alpha-blend 72h: mix global and JH for 72h column ─────────
    print("\nAlpha-blend 72h search (coarse) …")
    best_alpha, best_alpha_h = 0.0, 0.0
    for alpha in np.arange(0.0, 1.05, 0.05):
        blended_72 = alpha * jh_blend_oof[:, 3] + (1 - alpha) * g_blend_oof[:, 3]
        ab_oof = np.column_stack([
            np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
            for j in range(3)
        ] + [blended_72])
        ab_oof = mono(ab_oof)
        ab_ms = eval_cv(y_surv, ab_oof, bin_targets, skf)
        ab_h = float(np.mean([m["h"] for m in ab_ms]))
        if ab_h > best_alpha_h:
            best_alpha_h, best_alpha = ab_h, alpha
    print(f"  Coarse best: alpha={best_alpha:.2f}, h={best_alpha_h:.4f}")

    # Fine search around best alpha
    print("  Fine alpha search …")
    lo = max(0.0, best_alpha - 0.08)
    hi = min(1.0, best_alpha + 0.08)
    for alpha in np.arange(lo, hi + 0.005, 0.01):
        blended_72 = alpha * jh_blend_oof[:, 3] + (1 - alpha) * g_blend_oof[:, 3]
        ab_oof = np.column_stack([
            np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
            for j in range(3)
        ] + [blended_72])
        ab_oof = mono(ab_oof)
        ab_ms = eval_cv(y_surv, ab_oof, bin_targets, skf)
        ab_h = float(np.mean([m["h"] for m in ab_ms]))
        if ab_h > best_alpha_h:
            best_alpha_h, best_alpha = ab_h, alpha
    print(f"  Fine best: alpha={best_alpha:.2f}, h={best_alpha_h:.4f}")

    alpha72_oof = np.column_stack([
        np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(3)
    ] + [best_alpha * jh_blend_oof[:, 3] + (1 - best_alpha) * g_blend_oof[:, 3]])
    alpha72_oof = mono(alpha72_oof)
    alpha72_ms = eval_cv(y_surv, alpha72_oof, bin_targets, skf)
    alpha72_h = float(np.mean([m["h"] for m in alpha72_ms]))
    print(f"  Alpha-72h CV: h={alpha72_h:.4f}  C={np.mean([m['c'] for m in alpha72_ms]):.4f}  WB={np.mean([m['wb'] for m in alpha72_ms]):.4f}")

    # ── Platt calibration on 72h only (preserves ranking) ────────
    print("\nPlatt-calibrated 72h variants …")
    for tag, base_oof in [("alpha72", alpha72_oof), ("global", g_blend_oof)]:
        cal_oof = base_oof.copy()
        splits = list(skf.split(np.arange(len(y_surv)), y_surv["event"]))
        yt72 = bin_targets[72.0]
        for fi, (_, va_i) in enumerate(splits):
            tr_idx = np.concatenate([v for fj, (_, v) in enumerate(splits) if fj != fi])
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(base_oof[tr_idx, 3:4], yt72[tr_idx])
            cal_oof[va_i, 3] = lr.predict_proba(base_oof[va_i, 3:4])[:, 1]
        cal_oof = mono(np.clip(cal_oof, PROB_FLOOR, PROB_CEIL))
        cal_ms = eval_cv(y_surv, cal_oof, bin_targets, skf)
        cal_h = float(np.mean([m["h"] for m in cal_ms]))
        print(f"  {tag}+platt72: h={cal_h:.4f}  C={np.mean([m['c'] for m in cal_ms]):.4f}  WB={np.mean([m['wb'] for m in cal_ms]):.4f}")

    # ── PART F: LGB stacking ────────────────────────────────────────────
    print("\nPART F: LightGBM stacking meta-learner …")
    stack_oof_probs, stack_models = stack_lgb_cv(all_oof, y_surv, bin_targets, skf)
    sms = eval_cv(y_surv, stack_oof_probs, bin_targets, skf)
    stack_h = float(np.mean([m["h"] for m in sms]))
    print(f"  LGB-stack CV: h={stack_h:.4f}  C={np.mean([m['c'] for m in sms]):.4f}  WB={np.mean([m['wb'] for m in sms]):.4f}")

    # ── Calibration evaluation ────────────────────────────────────────
    print("\nCalibration evaluation …")
    approaches = {}
    approaches["global"] = (g_h, g_blend_oof, g_weights, "global")
    approaches["reg-blend"] = (gr_h, gr_blend_oof, gr_weights, "reg")
    approaches["simple-avg"] = (avg_h, avg_oof, None, "avg")
    approaches["top3-avg"] = (top3_h, top3_oof, None, "top3")
    approaches["per-horizon-brier"] = (h_h, h_blend_oof, h_weights, "ph_brier")
    approaches["joint-hybrid"] = (jh_h, jh_blend_oof, jh_weights, "joint")
    approaches["custom-hybrid"] = (custom_h, custom_oof, None, "custom")
    approaches["c-opt-72h"] = (copt_h, copt_oof, None, "copt")
    approaches["tiebreak-72h"] = (tiebreak_h_final, tiebreak_oof_final, best_tb_eps, "tiebreak")
    approaches["mixed-hybrid"] = (mixed_h, mixed_oof, None, "mixed")
    approaches["alpha72-hybrid"] = (alpha72_h, alpha72_oof, best_alpha, "alpha72")
    approaches["lgb-stack"] = (stack_h, stack_oof_probs, stack_models, "stack")

    for tag, base_oof in [("global", g_blend_oof), ("joint-hybrid", jh_blend_oof)]:
        cal = calibrate_isotonic_cv(base_oof, bin_targets, skf, y_surv)
        cms = eval_cv(y_surv, cal, bin_targets, skf)
        cal_h = float(np.mean([m["h"] for m in cms]))
        approaches[f"{tag}+isotonic"] = (cal_h, cal, None, "cal")
        print(f"  {tag}+isotonic: h={cal_h:.4f}  C={np.mean([m['c'] for m in cms]):.4f}  WB={np.mean([m['wb'] for m in cms]):.4f}")

    for r in surv_results:
        ms = eval_cv(y_surv, r["oof_p"], bin_targets, skf)
        approaches[f"single-{r['name']}"] = (float(np.mean([m["h"] for m in ms])), r["oof_p"], None, "single")
    for r in bin_results:
        ms = eval_cv(y_surv, r["oof_p"], bin_targets, skf)
        approaches[f"single-{r['name']}"] = (float(np.mean([m["h"] for m in ms])), r["oof_p"], None, "single")

    best_tag = max(approaches, key=lambda k: approaches[k][0])
    best_h = approaches[best_tag][0]
    print(f"\n  Best approach: {best_tag} (h={best_h:.4f})")
    sorted_app = sorted(approaches.items(), key=lambda x: -x[1][0])
    for tag, (h, _, _w, _t) in sorted_app:
        print(f"    {tag:30s}  h={h:.4f}")

    # ── Final test predictions ────────────────────────────────────────
    print(f"\nFinal predictions ({len(FINAL_SEEDS)} seeds) …")
    surv_test_preds = []
    for r in surv_results:
        if r.get("needs_scale"):
            print(f"  Predicting {r['name']} (scaled) …")
            preds = []
            for seed in FINAL_SEEDS:
                sc = StandardScaler()
                Xsc = sc.fit_transform(X)
                Xtsc = sc.transform(Xt)
                m = CoxPHSurvivalAnalysis(**r["params"])
                m.fit(Xsc, y_surv)
                preds.append(mono(sf2prob(m.predict_survival_function(Xtsc))))
            surv_test_preds.append(np.mean(preds, axis=0))
        else:
            print(f"  Predicting {r['name']} …")
            p = predict_surv_test(r, X, y_surv, Xt)
            surv_test_preds.append(p)
        gc.collect()

    bin_test_preds = []
    for r in bin_results:
        print(f"  Predicting {r['name']} …")
        p = predict_bin_test(r, X, bin_targets, Xt)
        bin_test_preds.append(p)
        gc.collect()

    all_test = surv_test_preds + bin_test_preds

    # Build all test variants
    g_test = mono(sum(w * p for w, p in zip(g_weights, all_test)))

    jh_test = np.column_stack([
        np.clip(sum(jh_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(4)
    ])
    jh_test = mono(jh_test)

    def calibrate_test(oof, test_probs):
        cal_test = test_probs.copy()
        for j, t in enumerate(PRED_TIMES):
            yt = bin_targets[t]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=PROB_FLOOR, y_max=PROB_CEIL)
            iso.fit(oof[:, j], yt)
            cal_test[:, j] = iso.predict(test_probs[:, j])
        return mono(cal_test)

    # Custom test: Brier for 12/24/48h, global for 72h
    h_test = np.column_stack([
        np.clip(sum(h_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(4)
    ])
    h_test = mono(h_test)

    custom_test = np.column_stack([
        np.clip(sum(h_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(3)
    ] + [
        g_test[:, 3]
    ])
    custom_test = mono(custom_test)

    # LGB-stack test predictions
    stack_test = np.column_stack([
        stack_models[j].predict_proba(
            np.column_stack([p[:, j] for p in all_test])
        )[:, 1]
        for j in range(4)
    ])
    stack_test = clip_safe(stack_test)

    # Regularized blend test
    gr_test = mono(sum(w * p for w, p in zip(gr_weights, all_test)))
    # Simple average test
    avg_test = mono(np.mean(all_test, axis=0))
    # Top-3 average test
    top3_indices = [i for i, c in enumerate(all_candidates) if c["name"] in top3_names]
    top3_test = mono(np.mean([all_test[i] for i in top3_indices], axis=0))

    # Build tiebreak test predictions
    best_brier_idx = next(i for i, c in enumerate(all_candidates) if c["name"] == best_brier_cand["name"])
    best_c_idx = next(i for i, c in enumerate(all_candidates) if c["name"] == best_c_name)
    tiebreak_test = np.column_stack([
        all_test[best_brier_idx][:, j] for j in range(3)
    ] + [all_test[best_brier_idx][:, 3] + best_tb_eps * all_test[best_c_idx][:, 3]])
    tiebreak_test = mono(np.clip(tiebreak_test, 0, 1))

    copt_test = np.column_stack([
        all_test[best_brier_idx][:, j] for j in range(3)
    ] + [all_test[best_c_idx][:, 3]])
    copt_test = mono(copt_test)

    # Mixed hybrid test: Brier 12-48h + JH 72h
    mixed_test = np.column_stack([
        np.clip(sum(h_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(3)
    ] + [jh_test[:, 3]])
    mixed_test = mono(mixed_test)

    # Alpha-72h test
    alpha72_test = np.column_stack([
        np.clip(sum(h_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(3)
    ] + [best_alpha * jh_test[:, 3] + (1 - best_alpha) * g_test[:, 3]])
    alpha72_test = mono(alpha72_test)

    if best_tag == "global":
        main_probs = g_test
    elif best_tag == "reg-blend":
        main_probs = gr_test
    elif best_tag == "simple-avg":
        main_probs = avg_test
    elif best_tag == "top3-avg":
        main_probs = top3_test
    elif best_tag == "joint-hybrid":
        main_probs = jh_test
    elif best_tag == "custom-hybrid":
        main_probs = custom_test
    elif best_tag == "mixed-hybrid":
        main_probs = mixed_test
    elif best_tag == "alpha72-hybrid":
        main_probs = alpha72_test
    elif best_tag == "tiebreak-72h":
        main_probs = tiebreak_test
    elif best_tag == "c-opt-72h":
        main_probs = copt_test
    elif best_tag == "lgb-stack":
        main_probs = stack_test
    elif best_tag == "global+isotonic":
        main_probs = calibrate_test(g_blend_oof, g_test)
    elif best_tag == "joint-hybrid+isotonic":
        main_probs = calibrate_test(jh_blend_oof, jh_test)
    else:
        main_probs = g_test

    main_probs = clip_safe(main_probs)
    sub_main = write_sub(DATA_DIR / "submission.csv", tids, main_probs)

    # Generate multiple submission variants for experimentation
    write_sub(DATA_DIR / "submission_blend.csv", tids, clip_safe(g_test))
    write_sub(DATA_DIR / "submission_regblend.csv", tids, clip_safe(gr_test))
    write_sub(DATA_DIR / "submission_custom.csv", tids, clip_safe(custom_test))
    write_sub(DATA_DIR / "submission_tiebreak.csv", tids, clip_safe(tiebreak_test))
    write_sub(DATA_DIR / "submission_mixed.csv", tids, clip_safe(mixed_test))
    write_sub(DATA_DIR / "submission_alpha72.csv", tids, clip_safe(alpha72_test))
    write_sub(DATA_DIR / "submission_stack.csv", tids, stack_test)
    # Single best models
    for i, r in enumerate(surv_results):
        write_sub(DATA_DIR / f"submission_{r['name'].lower()}.csv", tids, clip_safe(surv_test_preds[i]))
    for i, r in enumerate(bin_results):
        write_sub(DATA_DIR / f"submission_{r['name'].lower()}.csv", tids, clip_safe(bin_test_preds[i]))

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"\nSubmission stats (submission.csv — {best_tag}):")
    for c in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
        v = sub_main[c]
        print(f"  {c}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")

    ok = all(
        (sub_main[a] <= sub_main[b] + 1e-9).all()
        for a, b in [("prob_12h", "prob_24h"), ("prob_24h", "prob_48h"), ("prob_48h", "prob_72h")]
    )
    nz = int((sub_main[["prob_12h", "prob_24h", "prob_48h", "prob_72h"]] < 0.001).sum().sum())
    print(f"  Monotonicity: {'OK' if ok else 'FAIL'}")
    print(f"  Near-zero (<0.001): {nz}")

    manifest = {
        "runtime_min": round(elapsed / 60, 1),
        "n_features": len(keep),
        "metric": "direct-MSE Brier (matching competition)",
        "approaches": [(tag, round(h, 4)) for tag, (h, _, _w, _t) in sorted_app],
        "chosen": best_tag,
        "chosen_cv_h": round(best_h, 4),
        "surv_models": [{"name": r["name"], "cv": round(r["cv"], 4),
                         "params": {k: v for k, v in r["params"].items() if k not in ("random_state", "n_jobs")}}
                        for r in surv_results],
        "bin_models": [{"name": r["name"], "cv": round(r["cv"], 4), "params": r["params"]}
                       for r in bin_results],
        "global_weights": {c["name"]: round(float(w), 4) for c, w in zip(all_candidates, g_weights)},
    }
    (DATA_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Manifest → experiment_manifest.json")


if __name__ == "__main__":
    main()
