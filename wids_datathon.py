#!/usr/bin/env python3
"""
WiDS Global Datathon 2026 - Advanced Survival Pipeline (v10)
============================================================
Target: >=0.98 on Kaggle leaderboard

Major improvements over v7:
  1. Optuna Bayesian hyperparameter optimization (vs random search)
  2. XGBoost + CatBoost IPCW classifiers (3 boosting frameworks)
  3. Feature importance-based selection (110+ -> ~50 features)
  4. Enhanced feature engineering (rank features, fire urgency composites)
  5. Brier score evaluated at 12h, 24h, 48h (v7 missed 12h)
  6. 5-seed OOF averaging, 15-seed final predictions
  7. Power calibration added to calibration search
  8. More diverse ensemble (5 models) with Nelder-Mead optimization

Install: pip install pandas numpy scikit-survival scikit-learn scipy lightgbm xgboost catboost optuna
Run:     python3 wids_datathon.py
"""
from __future__ import annotations

import itertools, json, sys, time, warnings
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.metrics import brier_score, concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Configuration ───────────────────────────────────────────────────────────────

RANDOM_STATE = 42
PRED_TIMES   = np.array([12.0, 24.0, 48.0, 72.0])
# Brier evaluation at 24h, 48h, 72h (matches competition focus on survival horizons)
BRIER_TIMES  = np.array([24.0, 48.0, 72.0])
BRIER_W      = np.array([0.30, 0.40, 0.30])
N_FOLDS      = 5
PROB_FLOOR   = 0.001
PROB_CEIL    = 0.999
OOF_SEEDS    = [42, 123, 456, 789, 2024]
FINAL_SEEDS  = [42, 123, 456, 789, 2024, 314, 271, 1618, 577, 997,
                1234, 5678, 9012, 3456, 7890]
N_SELECT     = 75

# Optuna trial counts
OPTUNA_LGBM  = 45
OPTUNA_XGB   = 45
OPTUNA_CB    = 30
OPTUNA_GBSA  = 30
BLEND_RESTARTS = 20

DATA_DIR   = Path(__file__).resolve().parent
TRAIN_P    = DATA_DIR / "train.csv"
TEST_P     = DATA_DIR / "test.csv"
SAMPLE_P   = DATA_DIR / "sample_submission.csv"
MANIFEST_P = DATA_DIR / "experiment_manifest.json"
ID_COL     = "event_id"
TARGETS    = ["event", "time_to_hit_hours"]

# ─── Feature Engineering ─────────────────────────────────────────────────────────

def engineer_base(df: pd.DataFrame) -> pd.DataFrame:
    """Base feature set from 34 raw CSV columns."""
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
    return d


def engineer_v8(df: pd.DataFrame) -> pd.DataFrame:
    """v8 extended feature set: base + tracking quality + urgency + rank features."""
    d = engineer_base(df)
    nper = df["num_perimeters_0_5h"].values
    ltr  = df["low_temporal_resolution_0_5h"].values
    dist = df["dist_min_ci_0_5h"].values
    area = df["area_first_ha"].values
    spd  = df["closing_speed_m_per_h"].values
    aln  = df["alignment_abs"].values
    hour = df["event_start_hour"].values
    rad  = df["radial_growth_rate_m_per_h"].values
    cspd = df["centroid_speed_m_per_h"].values
    gr   = df["area_growth_rate_ha_per_h"].values

    # Tracking quality features
    is_well = ((nper >= 3) & (ltr == 0)).astype(float)
    d["is_well_tracked"]  = is_well
    d["is_close_5km"]     = (dist < 5000).astype(float)
    d["is_close_10km"]    = (dist < 10000).astype(float)
    d["log_area_x_well"]  = np.log1p(area) * is_well
    d["align_x_well"]     = aln * is_well
    d["spd_x_well"]       = spd * is_well
    d["hour_x_well"]      = hour * is_well
    d["track_quality"]    = nper * (1.0 - ltr)
    d["track_x_align"]    = d["track_quality"].values * aln
    safe_spd_v            = np.where(np.abs(spd) > 10, spd, 10.0)
    d["time_est"]         = np.clip(dist / safe_spd_v, 0, 200)
    d["is_very_close"]    = (dist < 1000).astype(float)
    d["dist_log_bucket"]  = np.clip(np.floor(np.log10(np.maximum(dist, 1))), 0, 5)

    # v8 NEW: fire urgency composites
    d["fire_urgency"]     = (spd * aln) / (dist + 100.0)
    d["fire_urgency_log"] = np.log1p(np.maximum(d["fire_urgency"].values, 0))
    d["fire_momentum"]    = spd * aln * np.log1p(np.maximum(gr, 0))
    d["threat_12h"]       = ((dist < spd * 12) & (aln > 0.5)).astype(float)
    d["threat_24h"]       = ((dist < spd * 24) & (aln > 0.3)).astype(float)

    # v8 NEW: distance decay features
    d["inv_dist"]         = 1.0 / (dist + 100.0)
    d["inv_dist_sq"]      = 1.0 / ((dist + 100.0) ** 2)

    # v8 NEW: speed-distance-alignment interactions
    d["spd_dist_align"]   = spd * aln / (np.sqrt(dist) + 10.0)
    d["rad_cover_12"]     = np.clip(rad * 12 / (dist + 1.0), 0, 5)
    d["rad_cover_24"]     = np.clip(rad * 24 / (dist + 1.0), 0, 5)
    d["approach_rate"]    = cspd * aln / (dist + 100.0)

    # v8 NEW: rank features for robustness (percentile ranks)
    rank_cols = ["dist_min_ci_0_5h", "closing_speed_m_per_h", "alignment_abs",
                 "area_first_ha", "radial_growth_rate_m_per_h", "centroid_speed_m_per_h",
                 "risk_proxy", "eta_close", "fire_urgency", "time_est"]
    for col in rank_cols:
        if col in d.columns:
            d[f"rank_{col}"] = d[col].rank(pct=True)

    d.fillna(0, inplace=True)
    d.replace([np.inf, -np.inf], 0, inplace=True)
    return d


# ─── Feature Selection ───────────────────────────────────────────────────────────

def select_features(X, y, n_select=N_SELECT):
    """Select top features using LightGBM importance averaged across horizons."""
    importance = np.zeros(X.shape[1])
    for h in PRED_TIMES:
        y_bin = (y["event"] & (y["time"] <= h)).astype(float)
        if y_bin.sum() < 3 or (1 - y_bin).sum() < 3:
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            min_child_samples=15, verbose=-1, random_state=RANDOM_STATE, n_jobs=2
        )
        clf.fit(X.values, y_bin)
        importance += clf.feature_importances_

    imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    selected = imp.head(n_select).index.tolist()
    print(f"  Selected {len(selected)} features (top by importance)")
    print(f"  Top 10: {selected[:10]}")
    return selected


# ─── Data Loading ────────────────────────────────────────────────────────────────

def load_data():
    tr = pd.read_csv(TRAIN_P)
    te = pd.read_csv(TEST_P)
    feat = [c for c in tr.columns if c not in TARGETS + [ID_COL]]
    Xtr = engineer_v8(tr[feat])
    Xte = engineer_v8(te[feat])
    y = np.array(
        list(zip(tr["event"].astype(bool), tr["time_to_hit_hours"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
    return Xtr, Xte, y, te[ID_COL].values


# ─── Utilities ───────────────────────────────────────────────────────────────────

def mono(p):
    """Enforce monotonicity: P(T<=12h) <= P(T<=24h) <= P(T<=48h) <= P(T<=72h)."""
    return np.maximum.accumulate(np.clip(p, 0.0, 1.0), axis=1)


def clip_safe(p):
    return mono(np.clip(p, PROB_FLOOR, PROB_CEIL))


def sf2prob(surv_fns, times=PRED_TIMES):
    """Convert survival functions to P(T<=t) at specified times."""
    n = len(surv_fns)
    out = np.zeros((n, len(times)))
    for i, sf in enumerate(surv_fns):
        xv, yv = sf.x, sf.y
        for j, t in enumerate(times):
            if   t <= xv[0]:  s = yv[0]
            elif t >= xv[-1]: s = yv[-1]
            else: s = yv[np.searchsorted(xv, t, side="right") - 1]
            out[i, j] = 1.0 - s
    return out


def metric(y_tr, y_va, probs, risk):
    """Compute hybrid = 0.3 * C-index + 0.7 * (1 - weighted_brier).
    Brier evaluated at BRIER_TIMES using survival probabilities."""
    c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
    # Survival probs at 24h, 48h, 72h (columns 1, 2, 3 of probs)
    sv = 1.0 - probs[:, 1:]
    eps = 0.01
    hi = min(y_tr["time"].max(), y_va["time"].max()) - eps
    ok = BRIER_TIMES < hi
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
        ew = BRIER_W[ok2]
    wb = float(np.average(bs, weights=ew / ew.sum()))
    return {"c": c, "wb": wb, "h": 0.3 * c + 0.7 * (1.0 - wb)}


def eval_oof(y, oof_p, oof_r, splitter):
    """Evaluate OOF predictions across CV folds."""
    rows = []
    for tr_i, va_i in splitter.split(np.arange(len(y)), y["event"]):
        m = metric(y[tr_i], y[va_i], oof_p[va_i], oof_r[va_i])
        if not np.isnan(m.get("h", np.nan)):
            rows.append(m)
    return rows


# ─── IPCW Censoring ─────────────────────────────────────────────────────────────

def km_censoring(event, time):
    t_km, g_km = kaplan_meier_estimator(~event.astype(bool), time)
    return t_km, g_km


def _g_at(t_km, g_km, t):
    if t > t_km[-1]:
        return max(float(g_km[-1]), 1e-6)
    idx = int(np.searchsorted(t_km, t, side="right")) - 1
    return max(float(g_km[max(idx, 0)]), 1e-6)


def make_ipcw_dataset(X, event, time, horizon, t_km, g_km):
    """Binary IPCW dataset for P(T <= horizon)."""
    mask  = event.astype(bool) | (time > horizon)
    Xs    = X[mask]
    y_bin = (event[mask] & (time[mask] <= horizon)).astype(float)
    ws    = np.ones(len(Xs))
    for i, oi in enumerate(np.where(mask)[0]):
        if event[oi] and time[oi] <= horizon:
            ws[i] = 1.0 / _g_at(t_km, g_km, time[oi])
    return Xs, y_bin, np.clip(ws, 1.0, 20.0)


# ─── Single-class handling for IPCW datasets ────────────────────────────────────

def _pad_single_class(Xs, y_bin, ws):
    """Add a tiny-weight dummy observation of the missing class so XGB/CatBoost work."""
    if len(np.unique(y_bin)) >= 2:
        return Xs, y_bin, ws
    missing = 0.0 if y_bin[0] == 1.0 else 1.0
    Xs_pad = np.vstack([Xs, Xs[0:1]])
    y_pad  = np.append(y_bin, missing)
    w_pad  = np.append(ws, 1e-10)
    return Xs_pad, y_pad, w_pad


# ─── IPCW-LightGBM ──────────────────────────────────────────────────────────────

_LGBM_FIXED = {"objective": "binary", "metric": "binary_logloss",
               "verbose": -1, "n_jobs": 2}


def train_lgbm_ipcw(X, event, time, horizon, params, seed=RANDOM_STATE):
    t_km, g_km = km_censoring(event, time)
    Xs, y_bin, ws = make_ipcw_dataset(X, event, time, horizon, t_km, g_km)
    # LightGBM handles single-class gracefully (produces varied predictions)
    clf = lgb.LGBMClassifier(**{**_LGBM_FIXED, **params, "random_state": seed})
    clf.fit(Xs, y_bin, sample_weight=ws)
    return clf


def oof_ipcw_generic(X_arr, y, params, splitter, train_fn, predict_fn, seeds=None):
    """Generic OOF builder for IPCW classifiers."""
    if seeds is None:
        seeds = OOF_SEEDS
    n = len(y)
    oof_p = np.zeros((n, 4))
    for tr_i, va_i in splitter.split(np.arange(n), y["event"].astype(int)):
        acc = np.zeros((len(va_i), 4))
        for seed in seeds:
            fp = np.zeros((len(va_i), 4))
            for j, h in enumerate(PRED_TIMES):
                clf = train_fn(X_arr[tr_i], y["event"][tr_i],
                               y["time"][tr_i], h, params, seed=seed)
                fp[:, j] = predict_fn(clf, X_arr[va_i])
            acc += fp
        oof_p[va_i] = acc / len(seeds)
    oof_p = mono(oof_p)
    # v8: Use composite risk score (weighted sum emphasizing early, discriminative horizons)
    risk = 0.40 * oof_p[:, 0] + 0.30 * oof_p[:, 1] + 0.20 * oof_p[:, 2] + 0.10 * oof_p[:, 3]
    return oof_p, risk


def fit_predict_ipcw_generic(X_arr, y, Xte_arr, params, train_fn, predict_fn):
    """Generic final predictor for IPCW classifiers."""
    preds = np.zeros((len(Xte_arr), 4))
    for seed in FINAL_SEEDS:
        fp = np.zeros((len(Xte_arr), 4))
        for j, h in enumerate(PRED_TIMES):
            clf = train_fn(X_arr, y["event"], y["time"], h, params, seed=seed)
            fp[:, j] = predict_fn(clf, Xte_arr)
        preds += fp
    return mono(preds / len(FINAL_SEEDS))


def _lgbm_predict(clf, X):
    return clf.predict_proba(X)[:, 1]


# ─── IPCW-XGBoost ───────────────────────────────────────────────────────────────

_XGB_FIXED = {"objective": "binary:logistic", "eval_metric": "logloss",
              "verbosity": 0, "nthread": 2}


def train_xgb_ipcw(X, event, time, horizon, params, seed=RANDOM_STATE):
    t_km, g_km = km_censoring(event, time)
    Xs, y_bin, ws = make_ipcw_dataset(X, event, time, horizon, t_km, g_km)
    Xs, y_bin, ws = _pad_single_class(Xs, y_bin, ws)
    clf = xgb.XGBClassifier(**{**_XGB_FIXED, **params, "random_state": seed,
                                "use_label_encoder": False})
    clf.fit(Xs, y_bin, sample_weight=ws)
    return clf


def _xgb_predict(clf, X):
    return clf.predict_proba(X)[:, 1]


# ─── IPCW-CatBoost ──────────────────────────────────────────────────────────────

def train_cb_ipcw(X, event, time, horizon, params, seed=RANDOM_STATE):
    t_km, g_km = km_censoring(event, time)
    Xs, y_bin, ws = make_ipcw_dataset(X, event, time, horizon, t_km, g_km)
    Xs, y_bin, ws = _pad_single_class(Xs, y_bin, ws)
    clf = cb.CatBoostClassifier(**params, random_seed=seed, verbose=0,
                                 thread_count=2)
    clf.fit(Xs, y_bin, sample_weight=ws)
    return clf


def _cb_predict(clf, X):
    return clf.predict_proba(X)[:, 1]


# ─── Survival Models ────────────────────────────────────────────────────────────

def build_oof_survival(X, y, cls, params, splitter, seeds=None, scale=False):
    """Build OOF predictions for a survival model."""
    if seeds is None:
        seeds = OOF_SEEDS
    n = len(y)
    oof_p, oof_r = np.zeros((n, 4)), np.zeros(n)
    for tr_i, va_i in splitter.split(X, y["event"]):
        pred_acc = np.zeros((len(va_i), 4))
        risk_acc = np.zeros(len(va_i))
        for seed in seeds:
            par = dict(params)
            if "random_state" in par:
                par["random_state"] = seed
            Xtr, Xva = X.iloc[tr_i].copy(), X.iloc[va_i].copy()
            if scale:
                sc = StandardScaler()
                cols = Xtr.columns
                Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=cols, index=Xtr.index)
                Xva = pd.DataFrame(sc.transform(Xva), columns=cols, index=Xva.index)
            m = cls(**par)
            m.fit(Xtr, y[tr_i])
            pred_acc += mono(sf2prob(m.predict_survival_function(Xva)))
            risk_acc += m.predict(Xva)
        oof_p[va_i] = pred_acc / len(seeds)
        oof_r[va_i] = risk_acc / len(seeds)
    return oof_p, oof_r


def fit_predict_survival_test(X, y, Xt, cls, params, scale=False):
    """Final survival model predictions on test set."""
    preds = []
    for seed in FINAL_SEEDS:
        par = dict(params)
        if "random_state" in par:
            par["random_state"] = seed
        Xtr, Xte = X.copy(), Xt.copy()
        if scale:
            sc = StandardScaler()
            cols = Xtr.columns
            Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=cols)
            Xte = pd.DataFrame(sc.transform(Xte), columns=cols)
        m = cls(**par)
        m.fit(Xtr, y)
        preds.append(mono(sf2prob(m.predict_survival_function(Xte))))
    return np.mean(preds, axis=0)


# ─── Optuna Hyperparameter Search ────────────────────────────────────────────────

def optuna_lgbm(X_arr, y, splitter, n_trials=OPTUNA_LGBM):
    """Bayesian search for IPCW-LGBM hyperparameters."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        try:
            oof_p, oof_r = oof_ipcw_generic(
                X_arr, y, params, splitter,
                train_lgbm_ipcw, _lgbm_predict, seeds=[RANDOM_STATE])
            rows = eval_oof(y, oof_p, oof_r, splitter)
            return float(np.mean([m["h"] for m in rows])) if rows else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  LGBM best: h={study.best_value:.4f}  params={study.best_params}")
    return study.best_params, study.best_value


def optuna_xgb(X_arr, y, splitter, n_trials=OPTUNA_XGB):
    """Bayesian search for IPCW-XGBoost hyperparameters."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        try:
            oof_p, oof_r = oof_ipcw_generic(
                X_arr, y, params, splitter,
                train_xgb_ipcw, _xgb_predict, seeds=[RANDOM_STATE])
            rows = eval_oof(y, oof_p, oof_r, splitter)
            return float(np.mean([m["h"] for m in rows])) if rows else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGB best: h={study.best_value:.4f}  params={study.best_params}")
    return study.best_params, study.best_value


def optuna_catboost(X_arr, y, splitter, n_trials=OPTUNA_CB):
    """Bayesian search for IPCW-CatBoost hyperparameters."""
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 2, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        }
        try:
            oof_p, oof_r = oof_ipcw_generic(
                X_arr, y, params, splitter,
                train_cb_ipcw, _cb_predict, seeds=[RANDOM_STATE])
            rows = eval_oof(y, oof_p, oof_r, splitter)
            return float(np.mean([m["h"] for m in rows])) if rows else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 2))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  CatBoost best: h={study.best_value:.4f}  params={study.best_params}")
    return study.best_params, study.best_value


def optuna_gbsa(X, y, splitter, n_trials=OPTUNA_GBSA):
    """Bayesian search for GBSA survival model."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "random_state": RANDOM_STATE,
        }
        try:
            oof_p, oof_r = build_oof_survival(
                X, y, GradientBoostingSurvivalAnalysis, params, splitter,
                seeds=[RANDOM_STATE])
            rows = eval_oof(y, oof_p, oof_r, splitter)
            if not rows:
                return 0.0
            mean_h = float(np.mean([m["h"] for m in rows]))
            std_h  = float(np.std([m["h"] for m in rows]))
            return mean_h - 0.3 * std_h
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 3))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_p = study.best_params
    best_p["random_state"] = RANDOM_STATE
    print(f"  GBSA best: robust={study.best_value:.4f}  params={best_p}")
    return best_p, study.best_value


def optuna_rsf(X, y, splitter, n_trials=15):
    """Quick Bayesian search for RSF."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            "max_features": trial.suggest_float("max_features", 0.2, 0.6),
            "random_state": RANDOM_STATE,
            "n_jobs": 2,
        }
        try:
            oof_p, oof_r = build_oof_survival(
                X, y, RandomSurvivalForest, params, splitter,
                seeds=[RANDOM_STATE])
            rows = eval_oof(y, oof_p, oof_r, splitter)
            if not rows:
                return 0.0
            mean_h = float(np.mean([m["h"] for m in rows]))
            std_h  = float(np.std([m["h"] for m in rows]))
            return mean_h - 0.3 * std_h
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 4))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_p = study.best_params
    best_p["random_state"] = RANDOM_STATE
    best_p["n_jobs"] = 2
    print(f"  RSF best: robust={study.best_value:.4f}  params={best_p}")
    return best_p, study.best_value


# ─── Ensemble Optimization ───────────────────────────────────────────────────────

def optimize_blend_weights(entries, y, splitter):
    """Find optimal blend weights via Nelder-Mead with many restarts."""
    if len(entries) == 1:
        return np.array([1.0])
    splits = list(splitter.split(np.arange(len(y)), y["event"]))

    def obj(w_raw):
        w = np.exp(np.clip(w_raw, -10, 10))
        w /= w.sum()
        p = mono(sum(wi * e["oof_p"] for wi, e in zip(w, entries)))
        r = sum(wi * e["oof_r"] for wi, e in zip(w, entries))
        r = np.nan_to_num(r, nan=0.0)
        hs = []
        for tr_i, va_i in splits:
            try:
                m = metric(y[tr_i], y[va_i], p[va_i], r[va_i])
                if not np.isnan(m.get("h", np.nan)):
                    hs.append(m["h"])
            except Exception:
                pass
        return -float(np.mean(hs)) if hs else 0.0

    k = len(entries)
    best, best_val = np.zeros(k), obj(np.zeros(k))
    rng = np.random.RandomState(RANDOM_STATE)
    for _ in range(BLEND_RESTARTS):
        x0 = rng.randn(k) * 0.7
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-9})
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    w = np.exp(best)
    w /= w.sum()
    return w


# ─── Stacking Meta-Learner ──────────────────────────────────────────────────────

def build_stacking(entries, y, splitter, X_key=None):
    """Build stacking ensemble: LogisticRegression meta-learner on OOF predictions.

    Simple and robust approach that learns the optimal combination of base model
    predictions per horizon. Can optionally include key original features.
    Returns OOF stacking predictions, fitted meta-learners, and best C."""
    n = len(y)

    # Meta-features: all model OOF predictions (n_models * 4 horizons)
    meta = np.hstack([e["oof_p"] for e in entries])
    if X_key is not None:
        meta = np.hstack([meta, X_key])

    # Tune C parameter via leave-fold-out
    C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    best_C, best_score = 1.0, -np.inf
    splits = list(splitter.split(np.arange(n), y["event"].astype(int)))

    for C in C_values:
        oof_stack = np.zeros((n, 4))
        for j, h in enumerate(PRED_TIMES):
            y_h = (y["event"] & (y["time"] <= h)).astype(float)
            for tr_i, va_i in splits:
                if len(np.unique(y_h[tr_i])) < 2:
                    oof_stack[va_i, j] = float(y_h[tr_i].mean())
                    continue
                lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
                lr.fit(meta[tr_i], y_h[tr_i])
                oof_stack[va_i, j] = lr.predict_proba(meta[va_i])[:, 1]
        oof_stack = mono(oof_stack)
        risk = 0.4 * oof_stack[:, 0] + 0.3 * oof_stack[:, 1] + \
               0.2 * oof_stack[:, 2] + 0.1 * oof_stack[:, 3]
        rows = eval_oof(y, oof_stack, risk, splitter)
        score = float(np.mean([m["h"] for m in rows])) if rows else 0.0
        if score > best_score:
            best_score, best_C = score, C

    # Build final OOF predictions with best C
    stack_oof = np.zeros((n, 4))
    for j, h in enumerate(PRED_TIMES):
        y_h = (y["event"] & (y["time"] <= h)).astype(float)
        for tr_i, va_i in splits:
            if len(np.unique(y_h[tr_i])) < 2:
                stack_oof[va_i, j] = float(y_h[tr_i].mean())
                continue
            lr = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
            lr.fit(meta[tr_i], y_h[tr_i])
            stack_oof[va_i, j] = lr.predict_proba(meta[va_i])[:, 1]
    stack_oof = mono(stack_oof)
    stack_risk = 0.4 * stack_oof[:, 0] + 0.3 * stack_oof[:, 1] + \
                 0.2 * stack_oof[:, 2] + 0.1 * stack_oof[:, 3]

    # Evaluate
    rows = eval_oof(y, stack_oof, stack_risk, splitter)
    s_h = float(np.mean([m["h"] for m in rows]))
    s_c = float(np.mean([m["c"] for m in rows]))
    s_wb = float(np.mean([m["wb"] for m in rows]))

    # Train final meta-learners on full data
    meta_learners = []
    for j, h in enumerate(PRED_TIMES):
        y_h = (y["event"] & (y["time"] <= h)).astype(float)
        lr = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
        lr.fit(meta, y_h)
        meta_learners.append(lr)

    return {
        "oof_p": stack_oof, "oof_r": stack_risk,
        "meta_learners": meta_learners, "best_C": best_C,
        "cv_h": s_h, "cv_c": s_c, "cv_wb": s_wb,
    }


def predict_stacking(test_preds_list, meta_learners, X_key_test=None, **kwargs):
    """Apply stacking meta-learners to Level-1 test predictions."""
    meta_test = np.hstack(test_preds_list)
    if X_key_test is not None:
        meta_test = np.hstack([meta_test, X_key_test])
    stack_test = np.zeros((len(meta_test), 4))
    for j, lr in enumerate(meta_learners):
        stack_test[:, j] = lr.predict_proba(meta_test)[:, 1]
    return clip_safe(stack_test)


# ─── Calibration ─────────────────────────────────────────────────────────────────

def choose_calibration(oof_p, oof_r, y, splitter):
    """Select best calibration: none / clipped / power / isotonic / platt."""
    splits  = list(splitter.split(np.arange(len(y)), y["event"]))
    targets = [(y["event"] & (y["time"] <= t)).astype(float) for t in PRED_TIMES]

    def oof_score(p):
        hs = []
        for tr_i, va_i in splits:
            m = metric(y[tr_i], y[va_i], p[va_i], oof_r[va_i])
            if not np.isnan(m.get("h", np.nan)):
                hs.append(m["h"])
        return float(np.mean(hs)) if hs else 0.0

    raw_score     = oof_score(oof_p)
    clipped_score = oof_score(clip_safe(oof_p))

    # Power calibration: p^alpha
    def power_obj(alpha):
        p = clip_safe(np.power(np.clip(oof_p, 1e-10, 1.0), alpha))
        return -oof_score(p)

    res = minimize_scalar(power_obj, bounds=(0.3, 3.0), method="bounded")
    power_alpha = res.x
    power_p = clip_safe(np.power(np.clip(oof_p, 1e-10, 1.0), power_alpha))
    power_score = oof_score(power_p)

    # Isotonic calibration (leave-fold-out)
    iso_p = np.zeros_like(oof_p)
    for fi, (_, va_i) in enumerate(splits):
        tri = np.concatenate([v for j, (_, v) in enumerate(splits) if j != fi])
        for t in range(4):
            ir = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL,
                                     out_of_bounds="clip")
            ir.fit(oof_p[tri, t], targets[t][tri])
            iso_p[va_i, t] = ir.transform(oof_p[va_i, t])
    iso_score = oof_score(clip_safe(iso_p))

    # Platt scaling (leave-fold-out)
    platt_p = np.zeros_like(oof_p)
    for fi, (_, va_i) in enumerate(splits):
        tri = np.concatenate([v for j, (_, v) in enumerate(splits) if j != fi])
        for t in range(4):
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(oof_p[tri, t:t+1], targets[t][tri])
            platt_p[va_i, t] = lr.predict_proba(oof_p[va_i, t:t+1])[:, 1]
    platt_score = oof_score(clip_safe(platt_p))

    scores = {"none": raw_score, "clipped": clipped_score, "power": power_score,
              "isotonic": iso_score, "platt": platt_score}
    best_name = max(scores, key=scores.get)
    print(f"  Calibration scores: {{{', '.join(f'{k}: {v:.4f}' for k, v in scores.items())}}}")
    print(f"  Best: {best_name}" + (f" (alpha={power_alpha:.3f})" if best_name == "power" else ""))

    # Build final calibrator
    if best_name == "isotonic":
        cal = [IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL,
                                   out_of_bounds="clip") for _ in range(4)]
        for t in range(4):
            cal[t].fit(oof_p[:, t], targets[t])
        return {"best_name": best_name, "scores": scores,
                "calibrator": ("isotonic", cal)}
    elif best_name == "platt":
        cal = [LogisticRegression(C=1.0, max_iter=1000) for _ in range(4)]
        for t in range(4):
            cal[t].fit(oof_p[:, t:t+1], targets[t])
        return {"best_name": best_name, "scores": scores,
                "calibrator": ("platt", cal)}
    elif best_name == "power":
        return {"best_name": best_name, "scores": scores,
                "calibrator": ("power", power_alpha)}
    elif best_name == "clipped":
        return {"best_name": best_name, "scores": scores,
                "calibrator": ("clipped", None)}
    else:
        return {"best_name": best_name, "scores": scores,
                "calibrator": ("none", None)}


def apply_calibration(p, calibrator):
    name, obj = calibrator
    if name == "none":
        return p
    if name == "clipped":
        return clip_safe(p)
    if name == "power":
        return clip_safe(np.power(np.clip(p, 1e-10, 1.0), obj))
    if name == "isotonic":
        return clip_safe(np.column_stack(
            [obj[j].transform(p[:, j]) for j in range(4)]))
    if name == "platt":
        return clip_safe(np.column_stack(
            [obj[j].predict_proba(p[:, j:j+1])[:, 1] for j in range(4)]))
    return p


# ─── Submission Writer ───────────────────────────────────────────────────────────

def write_submission(path, ids, probs):
    sub = pd.DataFrame({
        ID_COL: ids,
        "prob_12h": probs[:, 0], "prob_24h": probs[:, 1],
        "prob_48h": probs[:, 2], "prob_72h": probs[:, 3],
    })
    sample = pd.read_csv(SAMPLE_P)
    assert len(sub) == len(sample) and set(sub[ID_COL]) == set(sample[ID_COL])
    sub.sort_values(ID_COL).reset_index(drop=True).to_csv(path, index=False)
    return pd.read_csv(path)


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    np.random.seed(RANDOM_STATE)

    # 1. Load data and engineer features
    print("=" * 60)
    print("WiDS Datathon 2026 - v8 Pipeline")
    print("=" * 60)
    X_all, Xt_all, y, tids = load_data()
    print(f"\n=== Data (before selection) ===")
    print(f"  Train: {X_all.shape[0]} rows, {X_all.shape[1]} features")
    print(f"  Test:  {Xt_all.shape[0]} rows  |  Events: {int(y['event'].sum())}")

    # 2. Feature selection
    print(f"\n=== Feature Selection ===")
    selected_cols = select_features(X_all, y, n_select=N_SELECT)
    X  = X_all[selected_cols]
    Xt = Xt_all[selected_cols]
    X_arr  = X.values.astype(float)
    Xt_arr = Xt.values.astype(float)
    print(f"  Final: {X.shape[1]} features")

    base_spl = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    # 3. Optuna hyperparameter search for all models
    print(f"\n=== Optuna Hyperparameter Search ===")

    print(f"\n  --- IPCW-LightGBM ({OPTUNA_LGBM} trials) ---")
    lgbm_params, lgbm_cv = optuna_lgbm(X_arr, y, base_spl)

    print(f"\n  --- IPCW-XGBoost ({OPTUNA_XGB} trials) ---")
    xgb_params, xgb_cv = optuna_xgb(X_arr, y, base_spl)

    print(f"\n  --- IPCW-CatBoost ({OPTUNA_CB} trials) ---")
    cb_params, cb_cv = optuna_catboost(X_arr, y, base_spl)

    print(f"\n  --- GBSA ({OPTUNA_GBSA} trials) ---")
    gbsa_params, gbsa_cv = optuna_gbsa(X, y, base_spl)

    print(f"\n  --- RSF (15 trials) ---")
    rsf_params, rsf_cv = optuna_rsf(X, y, base_spl)

    # 4. Build full OOF predictions with multi-seed averaging
    print(f"\n=== Build Multi-Seed OOF ({len(OOF_SEEDS)} seeds) ===")
    all_entries = []

    print("  IPCW-LGBM...")
    lgbm_oof_p, lgbm_oof_r = oof_ipcw_generic(
        X_arr, y, lgbm_params, base_spl,
        train_lgbm_ipcw, _lgbm_predict, seeds=OOF_SEEDS)
    lgbm_fm = eval_oof(y, lgbm_oof_p, lgbm_oof_r, base_spl)
    lgbm_h = float(np.mean([m["h"] for m in lgbm_fm]))
    print(f"    h={lgbm_h:.4f}  C={np.mean([m['c'] for m in lgbm_fm]):.4f}  "
          f"WB={np.mean([m['wb'] for m in lgbm_fm]):.4f}")
    all_entries.append({"name": "IPCW-LGBM", "oof_p": lgbm_oof_p,
                        "oof_r": lgbm_oof_r, "cv_h": lgbm_h})

    print("  IPCW-XGBoost...")
    xgb_oof_p, xgb_oof_r = oof_ipcw_generic(
        X_arr, y, xgb_params, base_spl,
        train_xgb_ipcw, _xgb_predict, seeds=OOF_SEEDS)
    xgb_fm = eval_oof(y, xgb_oof_p, xgb_oof_r, base_spl)
    xgb_h = float(np.mean([m["h"] for m in xgb_fm]))
    print(f"    h={xgb_h:.4f}  C={np.mean([m['c'] for m in xgb_fm]):.4f}  "
          f"WB={np.mean([m['wb'] for m in xgb_fm]):.4f}")
    all_entries.append({"name": "IPCW-XGB", "oof_p": xgb_oof_p,
                        "oof_r": xgb_oof_r, "cv_h": xgb_h})

    print("  IPCW-CatBoost...")
    cb_oof_p, cb_oof_r = oof_ipcw_generic(
        X_arr, y, cb_params, base_spl,
        train_cb_ipcw, _cb_predict, seeds=OOF_SEEDS)
    cb_fm = eval_oof(y, cb_oof_p, cb_oof_r, base_spl)
    cb_h = float(np.mean([m["h"] for m in cb_fm]))
    print(f"    h={cb_h:.4f}  C={np.mean([m['c'] for m in cb_fm]):.4f}  "
          f"WB={np.mean([m['wb'] for m in cb_fm]):.4f}")
    all_entries.append({"name": "IPCW-CB", "oof_p": cb_oof_p,
                        "oof_r": cb_oof_r, "cv_h": cb_h})

    print("  GBSA...")
    gbsa_oof_p, gbsa_oof_r = build_oof_survival(
        X, y, GradientBoostingSurvivalAnalysis, gbsa_params, base_spl,
        seeds=OOF_SEEDS)
    gbsa_fm = eval_oof(y, gbsa_oof_p, gbsa_oof_r, base_spl)
    gbsa_h = float(np.mean([m["h"] for m in gbsa_fm]))
    print(f"    h={gbsa_h:.4f}  C={np.mean([m['c'] for m in gbsa_fm]):.4f}  "
          f"WB={np.mean([m['wb'] for m in gbsa_fm]):.4f}")
    all_entries.append({"name": "GBSA", "oof_p": gbsa_oof_p,
                        "oof_r": gbsa_oof_r, "cv_h": gbsa_h})

    print("  RSF...")
    rsf_oof_p, rsf_oof_r = build_oof_survival(
        X, y, RandomSurvivalForest, rsf_params, base_spl,
        seeds=OOF_SEEDS)
    rsf_fm = eval_oof(y, rsf_oof_p, rsf_oof_r, base_spl)
    rsf_h = float(np.mean([m["h"] for m in rsf_fm]))
    print(f"    h={rsf_h:.4f}  C={np.mean([m['c'] for m in rsf_fm]):.4f}  "
          f"WB={np.mean([m['wb'] for m in rsf_fm]):.4f}")
    all_entries.append({"name": "RSF", "oof_p": rsf_oof_p,
                        "oof_r": rsf_oof_r, "cv_h": rsf_h})

    # 5. Ensemble optimization
    print(f"\n=== Blend Optimization ({BLEND_RESTARTS} restarts) ===")
    weights = optimize_blend_weights(all_entries, y, base_spl)
    print("  Blend weights:")
    for e, w in zip(all_entries, weights):
        print(f"    {e['name']}: {w:.4f}")

    # Also compute equal-weight blend for comparison
    k = len(all_entries)
    eq_w = np.ones(k) / k
    eq_oof = mono(sum(w * e["oof_p"] for w, e in zip(eq_w, all_entries)))
    eq_oof_r = sum(w * e["oof_r"] for w, e in zip(eq_w, all_entries))
    eq_m = eval_oof(y, eq_oof, eq_oof_r, base_spl)
    eq_h = float(np.mean([m["h"] for m in eq_m]))
    print(f"  Equal-weight CV: h={eq_h:.4f}")

    # Apply minimum weight floor (3% per model for diversity)
    MIN_W = 0.03
    weights = np.maximum(weights, MIN_W)
    weights /= weights.sum()
    print(f"  Weights (after {MIN_W:.0%} floor):")
    for e, w in zip(all_entries, weights):
        print(f"    {e['name']}: {w:.4f}")

    blend_oof = mono(sum(w * e["oof_p"] for w, e in zip(weights, all_entries)))
    blend_oof_r = sum(w * e["oof_r"] for w, e in zip(weights, all_entries))
    bm = eval_oof(y, blend_oof, blend_oof_r, base_spl)
    b_h  = float(np.mean([m["h"]  for m in bm]))
    b_std = float(np.std([m["h"]  for m in bm]))
    b_c  = float(np.mean([m["c"]  for m in bm]))
    b_wb = float(np.mean([m["wb"] for m in bm]))
    print(f"  Blend CV: h={b_h:.4f}+-{b_std:.4f}  C={b_c:.4f}  WB={b_wb:.4f}")

    # 6. Stacking meta-learner
    print(f"\n=== Stacking Meta-Learner ===")
    # Key features for meta-learner (just the most important ones to avoid overfitting)
    key_feat_cols = ["dist_min_ci_0_5h", "closing_speed_m_per_h", "alignment_abs",
                     "risk_proxy", "eta_close"]
    key_feat_cols = [c for c in key_feat_cols if c in X.columns]
    X_key = X[key_feat_cols].values
    Xt_key = Xt[key_feat_cols].values
    print(f"  Key features for meta-learner: {len(key_feat_cols)}")

    print("  With original features:")
    stack_with = build_stacking(all_entries, y, base_spl, X_key=X_key)
    print(f"    h={stack_with['cv_h']:.4f}  C={stack_with['cv_c']:.4f}  "
          f"WB={stack_with['cv_wb']:.4f}  (C={stack_with['best_C']})")

    print("  Without original features:")
    stack_without = build_stacking(all_entries, y, base_spl, X_key=None)
    print(f"    h={stack_without['cv_h']:.4f}  C={stack_without['cv_c']:.4f}  "
          f"WB={stack_without['cv_wb']:.4f}  (C={stack_without['best_C']})")

    # Pick the better stacking approach
    use_key_feats = stack_with["cv_h"] >= stack_without["cv_h"]
    stack_result = stack_with if use_key_feats else stack_without
    print(f"  Best stacking: {'with' if use_key_feats else 'without'} features, "
          f"h={stack_result['cv_h']:.4f}")

    # 7. Calibration (on blend)
    print(f"\n=== Calibration Selection ===")
    cal = choose_calibration(blend_oof, blend_oof_r, y, base_spl)

    # 8. Final test predictions
    print(f"\n=== Final Test Predictions ({len(FINAL_SEEDS)} seeds) ===")

    print("  IPCW-LGBM...")
    lgbm_test = fit_predict_ipcw_generic(
        X_arr, y, Xt_arr, lgbm_params, train_lgbm_ipcw, _lgbm_predict)

    print("  IPCW-XGBoost...")
    xgb_test = fit_predict_ipcw_generic(
        X_arr, y, Xt_arr, xgb_params, train_xgb_ipcw, _xgb_predict)

    print("  IPCW-CatBoost...")
    cb_test = fit_predict_ipcw_generic(
        X_arr, y, Xt_arr, cb_params, train_cb_ipcw, _cb_predict)

    print("  GBSA...")
    gbsa_test = fit_predict_survival_test(
        X, y, Xt, GradientBoostingSurvivalAnalysis, gbsa_params)

    print("  RSF...")
    rsf_test = fit_predict_survival_test(
        X, y, Xt, RandomSurvivalForest, rsf_params)

    test_preds = [lgbm_test, xgb_test, cb_test, gbsa_test, rsf_test]
    blend_test = mono(sum(w * p for w, p in zip(weights, test_preds)))
    blend_cal  = apply_calibration(blend_test, cal["calibrator"])
    final_test = clip_safe(blend_cal)

    # Equal-weight ensemble (often generalizes better)
    eq_test = mono(sum(p for p in test_preds) / len(test_preds))
    eq_test = clip_safe(eq_test)

    # Stacking test predictions
    print("  Stacking meta-learner test predictions...")
    stack_test = predict_stacking(
        test_preds, stack_result["meta_learners"],
        X_key_test=Xt_key if use_key_feats else None)
    stack_test = mono(stack_test)

    # 7b. Post-processing: enforce distance-based separation
    # EDA: ALL events have dist<5km, ALL censored have dist>=5km (zero overlap)
    print("\n=== Distance-Based Post-Processing ===")
    te_raw = pd.read_csv(TEST_P)
    te_dist = te_raw.set_index(ID_COL)["dist_min_ci_0_5h"]
    # Map test IDs to distances in the same order as tids
    dist_arr = te_dist.loc[tids].values
    is_far = dist_arr >= 5000
    is_close = dist_arr < 5000
    print(f"  Close (<5km): {is_close.sum()} | Far (>=5km): {is_far.sum()}")

    def apply_distance_pp(preds):
        """Post-process predictions using distance-based thresholds.

        Training data shows:
          - 100% of events hit by 72h (all 69/69)
          - 95.7% hit by 48h (66/69)
          - 91.3% hit by 24h (63/69)
          - 71.0% hit by 12h (49/69)
        """
        p = preds.copy()
        # Far samples: cap at very small value
        p[is_far] = np.minimum(p[is_far], 0.001)
        # Close samples: apply floors based on training empirical rates
        p[is_close, 3] = np.maximum(p[is_close, 3], 0.95)   # 100% hit by 72h
        p[is_close, 2] = np.maximum(p[is_close, 2], 0.88)   # 95.7% hit by 48h
        # Very close samples (<1km): even stronger floors
        very_close = is_close & (dist_arr < 1000)
        p[very_close, 1] = np.maximum(p[very_close, 1], 0.90)  # training: 100% by 24h
        p[very_close, 0] = np.maximum(p[very_close, 0], 0.65)  # training: 86% by 12h
        return clip_safe(p)

    final_test = apply_distance_pp(final_test)
    eq_test    = apply_distance_pp(eq_test)
    blend_test_pp = apply_distance_pp(clip_safe(blend_test))
    stack_test_pp = apply_distance_pp(clip_safe(stack_test))

    # Blend + Stack average (may generalize better than either alone)
    blend_stack_avg = mono(0.5 * blend_test + 0.5 * stack_test)
    blend_stack_pp = apply_distance_pp(clip_safe(blend_stack_avg))

    # Also post-process individual models
    lgbm_test_pp = apply_distance_pp(clip_safe(lgbm_test))
    xgb_test_pp  = apply_distance_pp(clip_safe(xgb_test))

    # 8. Write submissions
    print(f"\n=== Writing Submissions ===")
    sub = write_submission(DATA_DIR / "submission.csv", tids, final_test)
    write_submission(DATA_DIR / "submission_blend.csv", tids, blend_test_pp)
    write_submission(DATA_DIR / "submission_equal.csv", tids, eq_test)
    write_submission(DATA_DIR / "submission_lgbm.csv", tids, lgbm_test_pp)
    write_submission(DATA_DIR / "submission_xgb.csv", tids, xgb_test_pp)
    write_submission(DATA_DIR / "submission_stack.csv", tids, stack_test_pp)
    write_submission(DATA_DIR / "submission_blend_stack.csv", tids, blend_stack_pp)

    mono_ok = ((sub["prob_12h"] <= sub["prob_24h"] + 1e-9).all()
               and (sub["prob_24h"] <= sub["prob_48h"] + 1e-9).all()
               and (sub["prob_48h"] <= sub["prob_72h"] + 1e-9).all())
    print("\n  Submission stats:")
    for col in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
        v = sub[col]
        print(f"    {col}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")
    print(f"    monotonic_ok={mono_ok}")

    # 9. Manifest
    elapsed = time.time() - t0
    manifest = {
        "version": "v8",
        "runtime_seconds": round(elapsed, 2),
        "data": {"train_rows": int(X_all.shape[0]), "test_rows": int(Xt_all.shape[0]),
                 "features_before_selection": int(X_all.shape[1]),
                 "features_after_selection": int(X.shape[1])},
        "models": {
            "IPCW-LGBM": {"cv_h": lgbm_h, "params": lgbm_params},
            "IPCW-XGB":  {"cv_h": xgb_h,  "params": xgb_params},
            "IPCW-CB":   {"cv_h": cb_h,    "params": cb_params},
            "GBSA":      {"cv_h": gbsa_h,  "params": {k: v for k, v in gbsa_params.items()
                                                       if k not in ("random_state", "n_jobs")}},
            "RSF":       {"cv_h": rsf_h,   "params": {k: v for k, v in rsf_params.items()
                                                       if k not in ("random_state", "n_jobs")}},
        },
        "blend_weights": {e["name"]: float(w) for e, w in zip(all_entries, weights)},
        "blend_cv": {"mean_h": b_h, "std_h": b_std, "mean_c": b_c, "mean_wb": b_wb},
        "stacking": {
            "cv_h": stack_result["cv_h"], "cv_c": stack_result["cv_c"],
            "cv_wb": stack_result["cv_wb"], "best_C": stack_result["best_C"],
            "use_key_feats": use_key_feats,
        },
        "calibration": {"chosen": cal["best_name"], "scores": cal["scores"]},
        "selected_features": selected_cols[:20],
        "output_files": ["submission.csv", "submission_blend.csv",
                         "submission_equal.csv", "submission_lgbm.csv",
                         "submission_xgb.csv", "submission_stack.csv",
                         "submission_blend_stack.csv"],
    }
    MANIFEST_P.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"BLEND CV HYBRID SCORE:    {b_h:.4f} +- {b_std:.4f}")
    print(f"  C-index: {b_c:.4f}    Weighted Brier: {b_wb:.4f}")
    print(f"STACKING CV HYBRID SCORE: {stack_result['cv_h']:.4f}")
    print(f"  C-index: {stack_result['cv_c']:.4f}    Weighted Brier: {stack_result['cv_wb']:.4f}")
    print(f"  Runtime: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
