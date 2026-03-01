#!/usr/bin/env python3
"""
WiDS Global Datathon 2026 – v8s (v8 + stacking)
=================================================

Proven v8 base (0.96394 on public LB) + minimal LightGBM stacking.
NO changes to feature engineering, model configs, or feature selection.

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
# Feature Engineering  (EXACT v8 — do NOT modify)
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
    bs = []
    for j, t in enumerate(BRIER_TIMES):
        yt = bin_targets_slice[t]
        p = probs[:, j + 1]
        bs.append(float(np.mean((p - yt) ** 2)))
    return float(np.average(bs, weights=BRIER_W))


def hybrid_score(c_index, wb):
    return 0.3 * c_index + 0.7 * (1.0 - wb)


def eval_fold(y_va, probs_va, bt_va):
    risk = probs_va[:, 3]
    c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
    wb = mse_brier(probs_va, bt_va)
    return {"c": c, "wb": wb, "h": hybrid_score(c, wb)}


def eval_cv(y_surv, probs, bin_targets, skf):
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
# Feature selection  (EXACT v8 — n_keep=55)
# ═══════════════════════════════════════════════════════════════════════

def select_features(X, y_surv, n_keep=55):
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
# PART A: Survival model OOF  (EXACT v8)
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

COXPH_GRID = {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}
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
    for cfg in SURV_CONFIGS:
        combos = _draw(cfg["grid"], cfg["n_draw"], rng)
        print(f"\n  Survival: {cfg['name']} ({len(combos)} configs)")
        best_h, best = -np.inf, None
        for i, combo in enumerate(combos, 1):
            params = {**cfg["fixed"], **combo}
            try:
                oof_p, oof_r = surv_oof(X, y, cfg["cls"], params, skf, [RANDOM_STATE])
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
                best = {"name": cfg["name"], "cls": cfg["cls"], "params": params,
                        "oof_p": oof_p, "oof_r": oof_r, "cv": mh}
            gc.collect()
        if best:
            results.append(best)
    return results


def search_coxph(X, y, bin_targets, skf, rng):
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
        tag = " ***" if mh > best_h else ""
        print(f"    [{i:2d}] a={combo['alpha']:.2f}  h={mh:.4f}  C={np.mean([m['c'] for m in ms]):.4f}{tag}")
        if mh > best_h:
            best_h = mh
            best = {"name": "CoxPH", "cls": CoxPHSurvivalAnalysis,
                    "params": combo, "oof_p": oof_p, "oof_r": oof_r,
                    "cv": mh, "needs_scale": True}
        gc.collect()
    return best


# ═══════════════════════════════════════════════════════════════════════
# PART B: Binary classification OOF  (EXACT v8 — 4 models, no ETC)
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
# PART C: Blend weight optimisation  (EXACT v8)
# ═══════════════════════════════════════════════════════════════════════

def optimise_global_blend(all_candidates, y_surv, bin_targets, skf):
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
        return -float(np.mean(hs))

    k = len(all_candidates)
    best_w, best_v = np.zeros(k), objective(np.zeros(k))
    for _ in range(30):
        x0 = np.random.randn(k) * 0.5
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-7})
        if res.fun < best_v:
            best_v, best_w = res.fun, res.x
    w = np.exp(best_w)
    w /= w.sum()
    return w


def optimise_per_horizon_blend(all_oof, y_surv, bin_targets, skf):
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
    splits = list(skf.split(np.arange(len(y_surv)), y_surv["event"]))
    k = len(all_oof)
    n_params = 4 * k

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
# Calibration
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


# ═══════════════════════════════════════════════════════════════════════
# PART S: LightGBM Stacking  (base probs ONLY, heavy regularisation)
# ═══════════════════════════════════════════════════════════════════════

def stack_lgb_cv(base_oof_list, bin_targets, y_surv, skf, seed=42):
    """
    Per-timepoint LightGBM meta-learner on base model OOF probs only.
    Heavy regularisation to prevent overfitting the small dataset.
    """
    n = len(y_surv)
    meta_oof = np.zeros((n, 4))

    meta_cols = []
    for oof in base_oof_list:
        for j in range(4):
            meta_cols.append(oof[:, j])
    meta_X = np.column_stack(meta_cols)

    lgb_params = {
        "n_estimators": 200,
        "max_depth": 3,
        "num_leaves": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_samples": 15,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": 2,
    }

    models_per_t = {}
    splits = list(skf.split(np.arange(n), y_surv["event"]))

    for j, t in enumerate(PRED_TIMES):
        yt = bin_targets[t]
        fold_models = []
        for tr_i, va_i in splits:
            m = lgb.LGBMClassifier(**lgb_params)
            m.fit(meta_X[tr_i], yt[tr_i])
            meta_oof[va_i, j] = m.predict_proba(meta_X[va_i])[:, 1]
            fold_models.append(m)
        models_per_t[j] = fold_models

    meta_oof = mono(meta_oof)
    return meta_oof, models_per_t


def stack_predict_test(models_per_t, base_test_list):
    meta_cols = []
    for tp in base_test_list:
        for j in range(4):
            meta_cols.append(tp[:, j])
    meta_X = np.column_stack(meta_cols)

    n = len(meta_X)
    preds = np.zeros((n, 4))
    for j in range(4):
        for m in models_per_t[j]:
            preds[:, j] += m.predict_proba(meta_X)[:, 1]
        preds[:, j] /= len(models_per_t[j])
    return mono(preds)


# ═══════════════════════════════════════════════════════════════════════
# Final test predictions
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
    keep = select_features(X_full, y_surv, n_keep=55)
    X = X_full[keep].copy()
    Xt = Xt_full[keep].copy()

    # ── PART A ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PART A: Survival Model Search (direct-MSE metric)")
    print("=" * 64)
    surv_results = search_surv(X, y_surv, bin_targets, skf, rng)
    gc.collect()

    # ── PART B ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PART B: Binary Classifier Search (direct-MSE metric)")
    print("=" * 64)
    bin_results = search_bin(X, y_surv, bin_targets, skf, rng)
    gc.collect()

    # ── Multi-seed OOF refinement ─────────────────────────────────────
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

    g_weights = optimise_global_blend(all_candidates, y_surv, bin_targets, skf)
    print("  Global blend weights:")
    for c, w in zip(all_candidates, g_weights):
        print(f"    {c['name']:6s}  w={w:.4f}")

    g_blend_oof = mono(sum(w * c["oof_p"] for w, c in zip(g_weights, all_candidates)))
    gms = eval_cv(y_surv, g_blend_oof, bin_targets, skf)
    g_h = float(np.mean([m["h"] for m in gms]))
    print(f"  Global blend CV: h={g_h:.4f}  C={np.mean([m['c'] for m in gms]):.4f}  WB={np.mean([m['wb'] for m in gms]):.4f}")

    # ── Per-horizon Brier blend ───────────────────────────────────────
    all_oof = [c["oof_p"] for c in all_candidates]
    h_weights = optimise_per_horizon_blend(all_oof, y_surv, bin_targets, skf)

    h_blend_oof = np.column_stack([
        np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(4)
    ])
    h_blend_oof = mono(h_blend_oof)
    hms = eval_cv(y_surv, h_blend_oof, bin_targets, skf)
    h_h = float(np.mean([m["h"] for m in hms]))
    print(f"  Per-horizon Brier CV: h={h_h:.4f}")

    # ── Joint hybrid blend ────────────────────────────────────────────
    jh_weights = optimise_joint_hybrid_blend(all_candidates, all_oof, y_surv, bin_targets, skf)
    jh_blend_oof = np.column_stack([
        np.clip(sum(jh_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(4)
    ])
    jh_blend_oof = mono(jh_blend_oof)
    jhms = eval_cv(y_surv, jh_blend_oof, bin_targets, skf)
    jh_h = float(np.mean([m["h"] for m in jhms]))
    print(f"  Joint hybrid CV: h={jh_h:.4f}")

    # ── Custom hybrid: Brier for 12/24/48h, global for 72h ───────────
    custom_oof = np.column_stack([
        np.clip(sum(h_weights[j, ci] * c["oof_p"][:, j] for ci, c in enumerate(all_candidates)), 0, 1)
        for j in range(3)
    ] + [
        g_blend_oof[:, 3]
    ])
    custom_oof = mono(custom_oof)
    custom_ms = eval_cv(y_surv, custom_oof, bin_targets, skf)
    custom_h = float(np.mean([m["h"] for m in custom_ms]))
    print(f"  Custom hybrid CV: h={custom_h:.4f}")

    # ── PART S: LightGBM Stacking ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("PART S: LightGBM Stacking (base probs only, heavy regularisation)")
    print("=" * 64)

    base_oof_list = [c["oof_p"] for c in all_candidates]

    stack_seeds = [42, 123, 789]
    stack_oof_acc = np.zeros((len(y_surv), 4))
    all_stack_models = []

    for si, sseed in enumerate(stack_seeds):
        print(f"  Stack seed {si+1}/{len(stack_seeds)} (seed={sseed}) …")
        soof, smodels = stack_lgb_cv(base_oof_list, bin_targets, y_surv, skf, seed=sseed)
        stack_oof_acc += soof
        all_stack_models.append(smodels)
        ms = eval_cv(y_surv, soof, bin_targets, skf)
        sh = float(np.mean([m["h"] for m in ms]))
        print(f"    h={sh:.4f}  C={np.mean([m['c'] for m in ms]):.4f}  WB={np.mean([m['wb'] for m in ms]):.4f}")

    stack_oof = mono(stack_oof_acc / len(stack_seeds))
    sms = eval_cv(y_surv, stack_oof, bin_targets, skf)
    stack_h = float(np.mean([m["h"] for m in sms]))
    print(f"\n  Multi-seed stack CV: h={stack_h:.4f}  C={np.mean([m['c'] for m in sms]):.4f}  WB={np.mean([m['wb'] for m in sms]):.4f}")

    # ── Stack + blend mixing ──────────────────────────────────────────
    print("\nStack + blend mixing …")
    best_alpha, best_alpha_h = 1.0, stack_h
    for alpha in np.arange(0.5, 1.01, 0.05):
        combo = mono(alpha * stack_oof + (1 - alpha) * g_blend_oof)
        cms = eval_cv(y_surv, combo, bin_targets, skf)
        ch = float(np.mean([m["h"] for m in cms]))
        tag = " ***" if ch > best_alpha_h else ""
        print(f"  α={alpha:.2f}  h={ch:.4f}{tag}")
        if ch > best_alpha_h:
            best_alpha_h = ch
            best_alpha = alpha
    stack_blend_oof = mono(best_alpha * stack_oof + (1 - best_alpha) * g_blend_oof)
    print(f"  Best α={best_alpha:.2f}  h={best_alpha_h:.4f}")

    # ── Calibration ───────────────────────────────────────────────────
    print("\nCalibration evaluation …")
    approaches = {}
    approaches["global"] = (g_h, g_blend_oof, "global")
    approaches["per-horizon-brier"] = (h_h, h_blend_oof, "ph_brier")
    approaches["joint-hybrid"] = (jh_h, jh_blend_oof, "joint")
    approaches["custom-hybrid"] = (custom_h, custom_oof, "custom")
    approaches["stack"] = (stack_h, stack_oof, "stack")
    approaches["stack+blend"] = (best_alpha_h, stack_blend_oof, "stackblend")

    for tag, base_oof in [("global", g_blend_oof), ("joint-hybrid", jh_blend_oof)]:
        cal = calibrate_isotonic_cv(base_oof, bin_targets, skf, y_surv)
        cms = eval_cv(y_surv, cal, bin_targets, skf)
        cal_h = float(np.mean([m["h"] for m in cms]))
        approaches[f"{tag}+isotonic"] = (cal_h, cal, "cal")
        print(f"  {tag}+isotonic: h={cal_h:.4f}")

    for r in surv_results:
        ms = eval_cv(y_surv, r["oof_p"], bin_targets, skf)
        approaches[f"single-{r['name']}"] = (float(np.mean([m["h"] for m in ms])), r["oof_p"], "single")
    for r in bin_results:
        ms = eval_cv(y_surv, r["oof_p"], bin_targets, skf)
        approaches[f"single-{r['name']}"] = (float(np.mean([m["h"] for m in ms])), r["oof_p"], "single")

    best_tag = max(approaches, key=lambda k: approaches[k][0])
    best_h_val = approaches[best_tag][0]
    print(f"\n  Best approach: {best_tag} (h={best_h_val:.4f})")
    sorted_app = sorted(approaches.items(), key=lambda x: -x[1][0])
    for tag, (h, _, _t) in sorted_app:
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

    g_test = mono(sum(w * p for w, p in zip(g_weights, all_test)))

    jh_test = np.column_stack([
        np.clip(sum(jh_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(4)
    ])
    jh_test = mono(jh_test)

    h_test = np.column_stack([
        np.clip(sum(h_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(4)
    ])
    h_test = mono(h_test)

    custom_test = np.column_stack([
        np.clip(sum(h_weights[j, ci] * p[:, j] for ci, p in enumerate(all_test)), 0, 1)
        for j in range(3)
    ] + [g_test[:, 3]])
    custom_test = mono(custom_test)

    # Stack test
    print("  Stack test predictions …")
    stack_test_acc = np.zeros((len(Xt), 4))
    for smodels in all_stack_models:
        st = stack_predict_test(smodels, all_test)
        stack_test_acc += st
    stack_test = mono(stack_test_acc / len(all_stack_models))

    stack_blend_test = mono(best_alpha * stack_test + (1 - best_alpha) * g_test)

    # Determine main submission
    if best_tag == "stack":
        main_probs = stack_test
    elif best_tag == "stack+blend":
        main_probs = stack_blend_test
    elif best_tag == "global":
        main_probs = g_test
    elif best_tag == "joint-hybrid":
        main_probs = jh_test
    elif best_tag == "custom-hybrid":
        main_probs = custom_test
    elif best_tag == "global+isotonic":
        cal_test = g_test.copy()
        for j, t in enumerate(PRED_TIMES):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=PROB_FLOOR, y_max=PROB_CEIL)
            iso.fit(g_blend_oof[:, j], bin_targets[t])
            cal_test[:, j] = iso.predict(g_test[:, j])
        main_probs = mono(cal_test)
    else:
        main_probs = g_test

    main_probs = clip_safe(main_probs)
    sub_main = write_sub(DATA_DIR / "submission.csv", tids, main_probs)
    write_sub(DATA_DIR / "submission_blend.csv", tids, clip_safe(g_test))
    write_sub(DATA_DIR / "submission_stack.csv", tids, clip_safe(stack_test))
    write_sub(DATA_DIR / "submission_stack_blend.csv", tids, clip_safe(stack_blend_test))
    write_sub(DATA_DIR / "submission_jh.csv", tids, clip_safe(jh_test))
    write_sub(DATA_DIR / "submission_custom.csv", tids, clip_safe(custom_test))

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
    print(f"  Monotonicity: {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 64)
    print("Files generated:")
    for fname in ["submission.csv", "submission_blend.csv", "submission_stack.csv",
                  "submission_stack_blend.csv", "submission_jh.csv", "submission_custom.csv"]:
        fp = DATA_DIR / fname
        if fp.exists():
            print(f"  {fname}")

    manifest = {
        "version": "v8s",
        "runtime_min": round(elapsed / 60, 1),
        "n_features": len(keep),
        "metric": "direct-MSE Brier (matching competition)",
        "approaches": [(tag, round(h, 4)) for tag, (h, _, _t) in sorted_app],
        "chosen": best_tag,
        "chosen_cv_h": round(best_h_val, 4),
        "stack_alpha": best_alpha,
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
