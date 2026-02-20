#!/usr/bin/env python3
"""
WiDS Global Datathon 2026 – Wildfire Evacuation Survival Analysis (v3)
======================================================================

Five-model ensemble (RSF, ExtraSurvivalTrees, GradientBoostingSA, CoxPH,
ComponentwiseGBSA) with aggressive feature engineering, multi-seed final
predictions, and post-hoc probability calibration optimised for the
competition hybrid metric  0.3*C-index + 0.7*(1 - Weighted Brier).

Dependencies
------------
    pip install pandas numpy scikit-survival scikit-learn scipy

How to run
----------
    /opt/anaconda3/bin/python3 wids_datathon.py
    # or whichever python has scikit-survival installed

Outputs
-------
    submission.csv  – ready-to-upload Kaggle submission
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_censored

warnings.filterwarnings("ignore")

# ── Global constants ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
PREDICTION_TIMES = np.array([12.0, 24.0, 48.0, 72.0])
BRIER_TIMES = np.array([24.0, 48.0, 72.0])
BRIER_WEIGHTS = np.array([0.3, 0.4, 0.3])
N_FOLDS = 5
MULTI_SEEDS = [42, 123, 789]

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

TARGET_COLS = ["event", "time_to_hit_hours"]
ID_COL = "event_id"


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ~70 features from the raw 34."""
    df = df.copy()

    dist = df["dist_min_ci_0_5h"]
    speed = df["closing_speed_m_per_h"]
    abs_speed = df["closing_speed_abs_m_per_h"]
    area = df["area_first_ha"]
    growth = df["area_growth_rate_ha_per_h"]
    radial = df["radial_growth_rate_m_per_h"]
    align = df["alignment_abs"]
    along = df["along_track_speed"]
    c_speed = df["centroid_speed_m_per_h"]

    # Time-to-arrival proxy
    safe_speed = speed.replace(0, np.nan)
    df["eta_hours"] = (dist / safe_speed).clip(-500, 500).fillna(999)

    # Composite risk: closing fast + aligned + close
    df["risk_proxy"] = speed * align / (dist + 1.0)

    # Log transforms for skewed distance / area
    df["log_dist"] = np.log1p(dist)
    df["log_area"] = np.log1p(area)

    # Fire size * growth interactions
    df["area_x_growth"] = area * growth
    df["fire_intensity"] = np.sqrt(area + 1) * growth

    # Radial reach and distance deficit at each prediction horizon
    for h in [12, 24, 48, 72]:
        df[f"radial_reach_{h}h"] = radial * h
        df[f"dist_deficit_{h}h"] = dist - speed * h

    # Binary: projected to hit at horizon?
    for h in [24, 48, 72]:
        df[f"proj_hit_{h}h"] = (dist < speed * h).astype(float)

    # Speed ratios
    df["closing_frac"] = speed / (dist + 1.0)
    df["speed_ratio_rc"] = radial / (abs_speed + 1.0)
    df["speed_ratio_cc"] = speed / (c_speed + 1.0)

    # Directional momentum
    df["dir_momentum"] = along * align
    df["align_x_speed"] = df["alignment_cos"] * speed

    # Perimeter density
    dt = df["dt_first_last_0_5h"].replace(0, np.nan)
    df["perim_density"] = (df["num_perimeters_0_5h"] / dt).fillna(0)

    # Advance ratio
    df["advance_ratio"] = df["projected_advance_m"] / (dist + 1.0)

    # Acceleration-adjusted closing (projected 24 h forward)
    df["accel_close_24h"] = speed + df["dist_accel_m_per_h2"] * 24

    # Reliability-weighted closing
    df["reliable_closing"] = speed * df["dist_fit_r2_0_5h"]

    # Polynomial / squared terms
    df["closing_sq"] = speed ** 2
    df["dist_sq"] = dist ** 2

    # Absolute cross-track
    df["cross_track_abs"] = df["cross_track_component"].abs()

    # Cyclical temporal encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["event_start_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["event_start_hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["event_start_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["event_start_month"] / 12)

    # Sqrt area ≈ fire radius proxy
    df["sqrt_area"] = np.sqrt(area)

    # Interaction pairs
    df["dist_std_x_close"] = df["dist_std_ci_0_5h"] * speed
    df["slope_x_align"] = df["dist_slope_ci_0_5h"] * align

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feat = [c for c in train.columns if c not in TARGET_COLS + [ID_COL]]

    X_train = engineer_features(train[feat])
    X_test = engineer_features(test[feat])

    y = _make_y(train["event"].values, train["time_to_hit_hours"].values)
    return X_train, X_test, y, test[ID_COL].values


def _make_y(event, time):
    return np.array(
        list(zip(event.astype(bool), time.astype(float))),
        dtype=[("event", bool), ("time", float)],
    )


# ── Survival-function → cumulative-incidence probability ────────────────────

def sf_to_probs(surv_fns, times=PREDICTION_TIMES):
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


def mono(probs):
    """Clip to [0,1] and enforce non-decreasing across time columns."""
    return np.maximum.accumulate(np.clip(probs, 0.0, 1.0), axis=1)


# ── Competition metric ───────────────────────────────────────────────────────

def hybrid_score(y_tr, y_va, probs, risk):
    c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]

    surv_b = 1.0 - probs[:, 1:]
    eps = 0.01
    hi = min(y_tr["time"].max(), y_va["time"].max()) - eps
    lo = max(y_tr["time"].min(), y_va["time"].min()) + eps
    ok = (BRIER_TIMES > lo) & (BRIER_TIMES < hi)
    et, ew, es = BRIER_TIMES[ok], BRIER_WEIGHTS[ok], surv_b[:, ok]

    if len(et) == 0:
        return {"c_index": c, "weighted_brier": np.nan, "hybrid": np.nan}

    _, bs = brier_score(y_tr, y_va, es, et)
    bd = {f"brier_{int(t)}": float(b) for t, b in zip(et, bs)}
    wb = float(np.average(bs, weights=ew / ew.sum()))
    return {"c_index": c, **bd, "weighted_brier": wb,
            "hybrid": 0.3 * c + 0.7 * (1.0 - wb)}


# ── Hyperparameter grid helpers ──────────────────────────────────────────────

def _draw(grid, n, rng):
    keys = list(grid)
    pool = list(itertools.product(*(grid[k] for k in keys)))
    rng.shuffle(pool)
    return [dict(zip(keys, c)) for c in pool[:n]]


MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "RSF",
        "cls": RandomSurvivalForest,
        "grid": {
            "n_estimators": [300, 500, 800],
            "max_depth": [5, 7, 10, None],
            "min_samples_leaf": [2, 3, 5, 7],
            "max_features": ["sqrt", 0.3, 0.5, 0.7],
        },
        "fixed": {"random_state": RANDOM_STATE, "n_jobs": 2},
        "n_draw": 30,
        "scale": False,
    },
    {
        "name": "ExtraTrees",
        "cls": ExtraSurvivalTrees,
        "grid": {
            "n_estimators": [300, 500, 800],
            "max_depth": [7, 10, 15, None],
            "min_samples_leaf": [2, 3, 5],
            "max_features": [0.5, 0.7, 0.9, 1.0],
        },
        "fixed": {"random_state": RANDOM_STATE, "n_jobs": 2},
        "n_draw": 25,
        "scale": False,
    },
    {
        "name": "GBSA",
        "cls": GradientBoostingSurvivalAnalysis,
        "grid": {
            "n_estimators": [100, 150, 200, 300, 400],
            "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
            "max_depth": [1, 2, 3, 4],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [3, 5, 10],
        },
        "fixed": {"random_state": RANDOM_STATE},
        "n_draw": 40,
        "scale": False,
    },
    {
        "name": "CoxPH",
        "cls": CoxPHSurvivalAnalysis,
        "grid": {
            "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        },
        "fixed": {"n_iter": 300, "tol": 1e-9},
        "n_draw": 9,
        "scale": True,
    },
    {
        "name": "CGBSA",
        "cls": ComponentwiseGradientBoostingSurvivalAnalysis,
        "grid": {
            "n_estimators": [100, 200, 300, 500, 800],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        },
        "fixed": {"random_state": RANDOM_STATE},
        "n_draw": 16,
        "scale": False,
    },
]


# ── Cross-validation with OOF collection ────────────────────────────────────

def cv_one(X, y, cls, params, skf, scale):
    n = len(y)
    oof_p = np.zeros((n, 4))
    oof_r = np.zeros(n)
    folds = []

    for tr_i, va_i in skf.split(X, y["event"]):
        Xtr, Xva = X.iloc[tr_i].copy(), X.iloc[va_i].copy()
        ytr, yva = y[tr_i], y[va_i]

        if scale:
            sc = StandardScaler()
            cols = Xtr.columns
            Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=cols, index=Xtr.index)
            Xva = pd.DataFrame(sc.transform(Xva), columns=cols, index=Xva.index)

        m = cls(**params)
        m.fit(Xtr, ytr)

        p = mono(sf_to_probs(m.predict_survival_function(Xva)))
        r = m.predict(Xva)

        oof_p[va_i] = p
        oof_r[va_i] = r
        folds.append(hybrid_score(ytr, yva, p, r))

    return folds, oof_p, oof_r


def search_all(X, y):
    rng = np.random.default_rng(RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for cfg in MODEL_CONFIGS:
        nm, cls = cfg["name"], cfg["cls"]
        fixed, sc = cfg["fixed"], cfg["scale"]
        combos = _draw(cfg["grid"], cfg["n_draw"], rng)

        print(f"\n{'='*60}")
        print(f"  {nm}  ({len(combos)} configs)")
        print(f"{'='*60}")

        best_s, best_e = -np.inf, {}

        for i, combo in enumerate(combos, 1):
            params = {**fixed, **combo}
            try:
                fm, op, orr = cv_one(X, y, cls, params, skf, sc)
            except Exception as exc:
                print(f"  [{i:2d}] FAILED: {exc}")
                continue

            hs = [m["hybrid"] for m in fm if not np.isnan(m["hybrid"])]
            if not hs:
                continue

            mh, sh = float(np.mean(hs)), float(np.std(hs))
            mc = float(np.mean([m["c_index"] for m in fm]))
            mw = float(np.nanmean([m["weighted_brier"] for m in fm]))
            tag = " ***" if mh > best_s else ""
            print(f"  [{i:2d}] hybrid={mh:.4f}±{sh:.4f}  C={mc:.4f}  WB={mw:.4f}{tag}")

            if mh > best_s:
                best_s = mh
                best_e = {"name": nm, "cls": cls, "params": params,
                          "cv": mh, "oof_p": op, "oof_r": orr, "scale": sc}

        if best_e:
            results.append(best_e)

    return results


# ── Blend-weight optimisation ────────────────────────────────────────────────

def opt_weights(results, y):
    if len(results) == 1:
        return np.array([1.0])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.arange(len(y)), y["event"]))

    def neg_h(w_raw):
        w = np.exp(w_raw); w /= w.sum()
        bp = mono(sum(wi * r["oof_p"] for wi, r in zip(w, results)))
        br = sum(wi * r["oof_r"] for wi, r in zip(w, results))
        vals = []
        for ti, vi in splits:
            m = hybrid_score(y[ti], y[vi], bp[vi], br[vi])
            if not np.isnan(m["hybrid"]):
                vals.append(m["hybrid"])
        return -np.mean(vals) if vals else 0.0

    k = len(results)
    best_w = np.zeros(k)
    best_v = neg_h(best_w)
    for _ in range(15):
        x0 = np.random.randn(k) * 0.5
        res = minimize(neg_h, x0, method="Nelder-Mead",
                       options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-6})
        if res.fun < best_v:
            best_v, best_w = res.fun, res.x

    w = np.exp(best_w); w /= w.sum()
    return w


# ── Probability calibration ─────────────────────────────────────────────────

PROB_FLOOR = 0.005
PROB_CEIL = 0.995


def _binary_targets(y):
    """Binary label for each prediction horizon: 1 if event occurred by t."""
    return [(y["event"] & (y["time"] <= t)).astype(float) for t in PREDICTION_TIMES]


def _clip_probs(probs):
    return mono(np.clip(probs, PROB_FLOOR, PROB_CEIL))


def fit_isotonic(oof, y):
    targets = _binary_targets(y)
    cals = []
    for j in range(4):
        ir = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL,
                                out_of_bounds="clip")
        ir.fit(oof[:, j], targets[j])
        cals.append(ir)
    return cals


def apply_isotonic(probs, cals):
    out = np.column_stack([c.transform(probs[:, j]) for j, c in enumerate(cals)])
    return _clip_probs(out)


def fit_platt(oof, y):
    targets = _binary_targets(y)
    cals = []
    for j in range(4):
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(oof[:, j:j + 1], targets[j])
        cals.append(lr)
    return cals


def apply_platt(probs, cals):
    out = np.column_stack([c.predict_proba(probs[:, j:j + 1])[:, 1]
                           for j, c in enumerate(cals)])
    return _clip_probs(out)


def pick_calibration(oof, oof_risk, y, skf):
    """Leave-fold-out evaluation: fit calibrators on K-1 folds, score on fold K.

    Returns (name, score, apply_fn | None, calibrators | None).
    """
    splits = list(skf.split(np.arange(len(y)), y["event"]))

    def eval_raw(probs):
        vals = []
        for ti, vi in splits:
            m = hybrid_score(y[ti], y[vi], probs[vi], oof_risk[vi])
            if not np.isnan(m["hybrid"]):
                vals.append(m["hybrid"])
        return np.mean(vals) if vals else 0.0

    raw_score = eval_raw(oof)
    raw_clipped_score = eval_raw(_clip_probs(oof))

    # Leave-fold-out isotonic
    iso_oof = np.zeros_like(oof)
    for fi, (_, va_i) in enumerate(splits):
        other = np.concatenate([vi for fj, (_, vi) in enumerate(splits) if fj != fi])
        targets = _binary_targets(y)
        for j in range(4):
            ir = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL,
                                    out_of_bounds="clip")
            ir.fit(oof[other, j], targets[j][other])
            iso_oof[va_i, j] = ir.transform(oof[va_i, j])
    iso_oof = _clip_probs(iso_oof)
    iso_score = eval_raw(iso_oof)

    # Leave-fold-out Platt
    platt_oof = np.zeros_like(oof)
    for fi, (_, va_i) in enumerate(splits):
        other = np.concatenate([vi for fj, (_, vi) in enumerate(splits) if fj != fi])
        targets = _binary_targets(y)
        for j in range(4):
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(oof[other, j:j + 1], targets[j][other])
            platt_oof[va_i, j] = lr.predict_proba(oof[va_i, j:j + 1])[:, 1]
    platt_oof = _clip_probs(platt_oof)
    platt_score = eval_raw(platt_oof)

    print(f"    raw       = {raw_score:.4f}")
    print(f"    clipped   = {raw_clipped_score:.4f}")
    print(f"    isotonic  = {iso_score:.4f}  (leave-fold-out)")
    print(f"    platt     = {platt_score:.4f}  (leave-fold-out)")

    # Pick best; if calibration wins, fit final calibrators on all OOF
    options = [
        ("none", raw_score, None, None),
        ("clipped", raw_clipped_score, lambda p, _: _clip_probs(p), None),
        ("isotonic", iso_score, apply_isotonic, fit_isotonic(oof, y)),
        ("platt", platt_score, apply_platt, fit_platt(oof, y)),
    ]
    best = max(options, key=lambda x: x[1])
    return best


# ── Multi-seed final predictions ─────────────────────────────────────────────

def final_predict(results, weights, X_train, y, X_test):
    all_blended = []
    for seed in MULTI_SEEDS:
        wp = np.zeros((len(X_test), 4))
        for r, w in zip(results, weights):
            params = dict(r["params"])
            if "random_state" in params:
                params["random_state"] = seed
            if "n_jobs" in params:
                params["n_jobs"] = 2

            Xtr, Xte = X_train.copy(), X_test.copy()
            if r["scale"]:
                sc = StandardScaler()
                cols = Xtr.columns
                Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=cols)
                Xte = pd.DataFrame(sc.transform(Xte), columns=cols)

            m = r["cls"](**params)
            m.fit(Xtr, y)
            wp += w * mono(sf_to_probs(m.predict_survival_function(Xte)))

        all_blended.append(wp)

    return mono(np.mean(all_blended, axis=0))


# ── Submission ───────────────────────────────────────────────────────────────

def write_sub(ids, probs, path=DATA_DIR / "submission.csv"):
    sub = pd.DataFrame({
        ID_COL: ids,
        "prob_12h": probs[:, 0], "prob_24h": probs[:, 1],
        "prob_48h": probs[:, 2], "prob_72h": probs[:, 3],
    })
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    assert set(sub[ID_COL]) == set(sample[ID_COL]), "ID mismatch"
    assert len(sub) == len(sample), "Row-count mismatch"
    sub = sub.sort_values(ID_COL).reset_index(drop=True)
    sub.to_csv(path, index=False)
    return sub


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(RANDOM_STATE)

    # ── Load ─────────────────────────────────────────────────────────────
    print("Loading data + feature engineering …")
    X_train, X_test, y, test_ids = load_data()
    print(f"  Train : {X_train.shape[0]} × {X_train.shape[1]} features")
    print(f"  Test  : {X_test.shape[0]} rows")
    print(f"  Events: {int(y['event'].sum())} / {len(y)}")

    # ── Phase 1 : per-family hyperparameter search ───────────────────────
    results = search_all(X_train, y)

    print(f"\n{'#'*60}")
    print("  Best per model family:")
    for r in results:
        show = {k: v for k, v in r["params"].items()
                if k not in ("random_state", "n_jobs", "n_iter", "tol")}
        print(f"    {r['name']:10s}  hybrid={r['cv']:.4f}  {show}")
    print(f"{'#'*60}")

    # ── Phase 2 : optimise ensemble blend weights ────────────────────────
    print("\nOptimising blend weights …")
    weights = opt_weights(results, y)
    for r, w in zip(results, weights):
        print(f"  {r['name']:10s}  w = {w:.4f}")

    # Blended OOF
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_blend = mono(sum(w * r["oof_p"] for w, r in zip(weights, results)))
    oof_risk = sum(w * r["oof_r"] for w, r in zip(weights, results))

    fh = []
    for ti, vi in skf.split(np.arange(len(y)), y["event"]):
        m = hybrid_score(y[ti], y[vi], oof_blend[vi], oof_risk[vi])
        fh.append(m)
    print(f"\n  Blend CV hybrid : {np.mean([m['hybrid'] for m in fh]):.4f} "
          f"± {np.std([m['hybrid'] for m in fh]):.4f}")
    print(f"  Blend CV C-index: {np.mean([m['c_index'] for m in fh]):.4f}")
    print(f"  Blend CV W-Brier: {np.nanmean([m['weighted_brier'] for m in fh]):.4f}")

    # ── Phase 3 : probability calibration ────────────────────────────────
    print("\nEvaluating calibration strategies …")
    cal_name, cal_score, cal_apply, cal_objs = pick_calibration(
        oof_blend, oof_risk, y, skf
    )
    print(f"  Best calibration: {cal_name} (score={cal_score:.4f})")

    # ── Phase 4 : multi-seed final predictions ───────────────────────────
    print(f"\nFinal predictions ({len(MULTI_SEEDS)} seeds) …")
    test_probs = final_predict(results, weights, X_train, y, X_test)

    if cal_apply is not None:
        test_probs = cal_apply(test_probs, cal_objs)

    # ── Phase 5 : write submission ───────────────────────────────────────
    sub = write_sub(test_ids, test_probs)
    print(f"\nSubmission → {DATA_DIR / 'submission.csv'}")
    print(f"  Rows: {len(sub)}")
    for col in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
        v = sub[col]
        print(f"    {col}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")

    ok = all(
        (sub[a] <= sub[b] + 1e-9).all()
        for a, b in [("prob_12h", "prob_24h"), ("prob_24h", "prob_48h"),
                      ("prob_48h", "prob_72h")]
    )
    print(f"  Monotonicity: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
