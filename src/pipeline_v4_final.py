"""
PIPELINE FINAL SKRIPSI - GABUNGAN TERBAIK
Evaluasi Performa Model XGBoost dan SARIMAX dalam Peramalan Penerimaan
PKB di Provinsi DKI Jakarta dengan Integrasi Variabel Eksogen
================================================================================
PENULIS: Muhammad Rafi Adipratama (NPM 4131220149)
POLITEKNIK KEUANGAN NEGARA STAN - 2026

Arsitektur Pipeline:
1. Machine Learning:   XGBoost Baseline (AR + kalender)
                        XGBoost Penuh (AR + kalender + eksogen)
                        + Ablasi fitur `tahun` (seleksi via CV)
                        + Tuning Bayesian Optimization (Optuna/TPE)
                        + Nested TSCV + early stopping (anti-leakage)
2. Statistik:           SARIMA (baseline time-series, tanpa eksogen)
                        SARIMAX (dengan eksogen)
                        + Grid search (p,d,q)(P,D,Q,12) + filter validitas
                        + Validasi ulang via TSCV (bukan murni AIC)
                        + Rule transformasi log berbasis uji Breakvar
                        + Diagnostik lengkap (Ljung-Box, Jarque-Bera, Breakvar)
3. Baseline Linear:    Ridge Regression (TSCV kausal)
4. Baseline Trivial:   Naive Persistence (y_hat_t = y_{t-1})
5. Ensemble:           Rata-rata sederhana + berbobot invers-CV-RMSE + Oracle (eksploratif)

Validitas Ilmiah:
- Perbaikan data leakage pada fitur rolling/EWMA/diff (pakai .shift(1))
- Uji stasioneritas ADF + KPSS (4 varian differencing)
- Bootstrap block (block=3, B=5000) untuk CI 95%
- Matriks Diebold-Mariano dengan distribusi-t (df=n-1, tepat untuk sampel kecil)
- Anti-test-set-peeking: seleksi model hanya via CV pada data latih
- Ablasi log dipicu rule diagnostik (konsisten proposal 3.4.4)
"""
from __future__ import annotations
import itertools
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats
from scipy.stats import jarque_bera
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# 1. KONFIGURASI
# ==============================================================================
@dataclass
class Config:
    # Dataset harus diperoleh secara resmi dan ditempatkan secara lokal.
    data_path: str = "data/dataset_skripsi.csv"
    modeling_start_date: str = "2015-01-01"   # BARU: awal sampel pemodelan
    output_dir: str = "outputs_final"

    # Split data (sesuai Ruang Lingkup: 80:20 = 105 latih, 27 uji)
    train_end_date: str = "2023-09-01"
    test_start_date: str = "2023-10-01"

    # Reproducibility
    random_state: int = 42

    # Time Series Cross-Validation
    n_cv_splits: int = 5

    # XGBoost (Bayesian Optimization)
    n_optuna_trials: int = 100
    max_boost_rounds: int = 3000
    early_stopping_rounds: int = 50
    inner_val_fraction: float = 0.15
    xgb_search_space: dict = field(default_factory=lambda: {
        "max_depth": (2, 6),
        "learning_rate": (0.005, 0.3),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 15),
        "gamma": (0.0, 5.0),
        "reg_alpha": (1e-4, 10.0),
        "reg_lambda": (1e-3, 20.0),
    })

    # Ablasi fitur `tahun` (seleksi via CV, bukan test)
    xgb_tahun_ablation: bool = True

    # SARIMA/SARIMAX grid
    seasonal_period: int = 12
    p_range: tuple = (0, 3)
    d_range: tuple = (0, 1)
    q_range: tuple = (0, 3)
    P_range: tuple = (0, 2)
    D_range: tuple = (0, 1)
    Q_range: tuple = (0, 2)
    max_total_arma_params: int = 6
    n_aic_shortlist: int = 8

    # Ablasi log-transform (dipicu rule Breakvar)
    log_decision_rule: bool = True

    # Bootstrap CI
    n_bootstrap: int = 5000
    bootstrap_block_size: int = 3

    # Target & date
    target_col: str = "pkb"
    date_col: str = "tanggal"


CFG = Config()


# ==============================================================================
# 1b. CHECKPOINTING (supaya proses panjang bisa dilanjutkan lintas sesi)
# ==============================================================================
import pickle


def ckpt_path(cfg: Config, name: str) -> Path:
    d = Path(cfg.output_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pkl"


def ckpt_save(cfg: Config, name: str, obj) -> None:
    with open(ckpt_path(cfg, name), "wb") as f:
        pickle.dump(obj, f)
    print(f"[CHECKPOINT] Disimpan: {name}")


def ckpt_load(cfg: Config, name: str):
    p = ckpt_path(cfg, name)
    if p.exists():
        with open(p, "rb") as f:
            obj = pickle.load(f)
        print(f"[CHECKPOINT] Memuat ulang hasil tersimpan: {name} (skip komputasi)")
        return obj
    return None


# ==============================================================================
# 2. DATA MENTAH & FEATURE ENGINEERING ANTI-LEAKAGE (REBUILD PENUH)
# ==============================================================================
def load_raw_data(cfg: Config) -> pd.DataFrame:
    """Muat data mentah dan konversi ke Miliar Rupiah."""
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    # === PERBAIKAN KRUSIAL: Konversi ke Miliar Rupiah ===
    # Mencegah SARIMAX 'meledak' akibat perbedaan skala 1 Miliar kali lipat
    df["pkb"] = df["pkb"] / 1e9
    df["bbnkb"] = df["bbnkb"] / 1e9

    print(f"[INFO] Data mentah: {len(df)} obs | "
          f"{df[cfg.date_col].min().date()} s.d. {df[cfg.date_col].max().date()}")
    print("[INFO] Skala `pkb` dan `bbnkb` dikonversi ke Miliar Rupiah.")
    return df


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Rekayasa fitur lengkap & kausal dari data mentah.

    Prinsip anti-leakage:
      - Fitur turunan TARGET hanya memakai informasi s.d. t-1
        (lag eksplisit, atau rolling/EWMA/diff atas y.shift(1)).
      - Fitur kalender deterministik (fungsi waktu) -> bebas leakage.
      - Lag eksogen memakai t-1 dan t-2 (kausal).
      - Eksogen KONTEMPORER (bbnkb, ikk, inflasi_yoy, covid19, pemutihan)
        dipertahankan apa adanya sebagai asumsi CONDITIONAL FORECAST
        (realisasi eksogen bulan t tersedia saat meramal bulan t);
        didokumentasikan sebagai keterbatasan di Bab 5.
    """
    df = df.copy()
    t, y = df[cfg.date_col], df[cfg.target_col]

    # ---- A. Kalender (deterministik) ----
    df["bulan"]     = t.dt.month
    df["kuartal"]   = t.dt.quarter
    df["tahun"]     = t.dt.year
    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)

    # ---- B. Lag target (kausal) ----
    for k in (1, 2, 3, 6, 12):
        df[f"pkb_lag_{k}"] = y.shift(k)

    # ---- C. Rolling / EWMA / diff atas informasi t-1 (kausal) ----
    y1 = y.shift(1)
    df["pkb_roll_mean_3_safe"] = y1.rolling(3).mean()
    df["pkb_roll_mean_6_safe"] = y1.rolling(6).mean()
    df["pkb_roll_std_3_safe"]  = y1.rolling(3).std()
    df["pkb_ewma_3_safe"]      = y1.ewm(span=3, adjust=False).mean()
    df["pkb_diff_1_safe"]      = y.shift(1) - y.shift(2)

    # ---- D. Lag eksogen (kausal) ----
    # Semua eksogen dipakai dalam bentuk LAG-1 dan LAG-2 (tidak ada kontemporer).
    # Alasan per variabel (dibuktikan dari data, lihat analisis korelasi lag):
    #   - BBNKB   : dikompilasi bersamaan PKB (leakage temporal, dominansi lag-0 = 2.14x)
    #   - IKK     : BI rilis survei bulan t di awal bulan t+1 (belum tersedia saat forecast)
    #   - Inflasi : BPS rilis bulan t di awal t+1; lag-1 empiris lebih prediktif (RMSE lebih rendah)
    # Keputusan ini memastikan pipeline bersifat EX ANTE (peramalan murni) tanpa
    # asumsi "conditional forecast" yang sulit diverifikasi.
    for c in ("bbnkb", "ikk", "inflasi_yoy"):
        df[f"{c}_lag_1"] = df[c].shift(1)
        df[f"{c}_lag_2"] = df[c].shift(2)

    # ---- E. Versi terdiferensiasi untuk eksogen non-stasioner ----
    # bbnkb_lag_1 & ikk_lag_1 terbukti non-stasioner di level (ADF gagal tolak H0).
    # Diferensiasi dihitung dari lag-1 (bukan dari kontemporer) agar konsisten
    # dengan keputusan di atas: bbnkb_lag_1_diff = bbnkb_{t-1} - bbnkb_{t-2}.
    df["bbnkb_lag_1_diff"] = df["bbnkb_lag_1"].diff()
    df["ikk_lag_1_diff"]   = df["ikk_lag_1"].diff()

    return df


def prepare_modeling_sample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter ke periode pemodelan (>= 2015-01) + dropna baris tanpa histori."""
    n_before = len(df)
    df_filtered = df[df[cfg.date_col] >= cfg.modeling_start_date]
    n_after_filter = len(df_filtered)
    df_final = df_filtered.dropna().reset_index(drop=True)
    n_dropped_na = n_after_filter - len(df_final)
    print(f"[INFO] Sampel pemodelan: {len(df_final)} obs "
          f"(burn-in 2014 dipakai utk lag-12 & rolling; {n_dropped_na} baris dibuang krn NaN)")
    return df_final


def load_and_engineer_features(cfg: Config) -> pd.DataFrame:
    """Entry point: data mentah -> fitur kausal -> sampel pemodelan (132 obs)."""
    return prepare_modeling_sample(engineer_features(load_raw_data(cfg), cfg), cfg)


def get_xgb_feature_sets():
    autoregressive = [
        "pkb_lag_1", "pkb_lag_2", "pkb_lag_3", "pkb_lag_6", "pkb_lag_12",
        "pkb_roll_mean_3_safe", "pkb_roll_mean_6_safe",
        "pkb_roll_std_3_safe", "pkb_ewma_3_safe", "pkb_diff_1_safe",
    ]
    calendar_with_year = ["kuartal", "bulan_sin", "bulan_cos", "tahun"]
    calendar_no_year   = ["kuartal", "bulan_sin", "bulan_cos"]
    # Hanya lag-1 dan lag-2 — tidak ada kontemporer (lihat catatan engineer_features).
    exogenous = [
        "bbnkb_lag_1", "bbnkb_lag_2",
        "ikk_lag_1", "ikk_lag_2",
        "inflasi_yoy_lag_1", "inflasi_yoy_lag_2",
        "covid19", "pemutihan",
    ]
    return autoregressive, calendar_with_year, calendar_no_year, exogenous


def get_sarimax_exog_cols():
    # Lag-1 kontemporer sudah diganti lag-1 (ex ante, konsisten dengan XGBoost).
    return ["bbnkb_lag_1", "ikk_lag_1", "inflasi_yoy_lag_1", "covid19", "pemutihan"]


def get_sarimax_exog_cols_diff():
    """Varian eksogen dgn bbnkb_lag_1 & ikk_lag_1 didiferensiasi.
    diff dihitung dari lag-1 (bukan kontemporer): bbnkb_{t-1} - bbnkb_{t-2}.
    Dipakai sebagai kandidat ablasi CV — keputusan berbasis data, bukan asumsi."""
    return ["bbnkb_lag_1_diff", "ikk_lag_1_diff", "inflasi_yoy_lag_1", "covid19", "pemutihan"]


def split_train_test(df: pd.DataFrame, cfg: Config):
    train = df[df[cfg.date_col] <= cfg.train_end_date].reset_index(drop=True)
    test  = df[df[cfg.date_col] >= cfg.test_start_date].reset_index(drop=True)
    print(f"[INFO] Data latih : {train[cfg.date_col].min().date()} s.d. "
          f"{train[cfg.date_col].max().date()}  ({len(train)} observasi)")
    print(f"[INFO] Data uji   : {test[cfg.date_col].min().date()} s.d. "
          f"{test[cfg.date_col].max().date()}  ({len(test)} observasi)")
    return train, test


# ==============================================================================
# 3. METRIK, BOOTSTRAP CI, DAN DM TEST (DISTRIBUSI-T)
# ==============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


def block_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> dict:
    """CI 95% via moving block bootstrap (menghormati dependensi temporal)."""
    n, block = len(y_true), cfg.bootstrap_block_size
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(cfg.random_state)
    rmses, maes, mapes = [], [], []
    for _ in range(cfg.n_bootstrap):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yt, yp = y_true[idx], y_pred[idx]
        m = compute_metrics(yt, yp)
        rmses.append(m["RMSE"]); maes.append(m["MAE"]); mapes.append(m["MAPE (%)"])

    def ci(a): return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return {"RMSE_CI95": ci(rmses), "MAE_CI95": ci(maes), "MAPE_CI95": ci(mapes)}


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1
                         ) -> tuple[float, float, float]:
    """DM test dengan 3 keluaran: statistik, p-value Normal, p-value Student-t (rekomendasi)."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dm_mean = float(d.mean())
    var_d = float(np.var(d, ddof=0))
    for lag in range(1, h):
        if lag < n:
            cov = np.cov(d[lag:], d[:-lag])[0, 1]
            var_d += 2 * cov
    var_d /= n
    if var_d <= 0: return np.nan, np.nan, np.nan
    dm = dm_mean / np.sqrt(var_d)
    p_norm = float(2 * (1 - stats.norm.cdf(abs(dm))))
    p_t = float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))   # rekomendasi sampel kecil
    return float(dm), p_norm, p_t


def build_dm_matrix(residuals: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = list(residuals.keys())
    stat_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    p_norm_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    p_t_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in itertools.product(names, names):
        if a == b:
            stat_mat.loc[a, b] = 0.0
            p_norm_mat.loc[a, b] = 1.0
            p_t_mat.loc[a, b] = 1.0
            continue
        dm, pn, pt = diebold_mariano_test(residuals[a], residuals[b])
        stat_mat.loc[a, b] = dm
        p_norm_mat.loc[a, b] = pn
        p_t_mat.loc[a, b] = pt
    return stat_mat, p_norm_mat, p_t_mat


# ==============================================================================
# 4. XGBOOST (BAYESIAN OPT + NESTED TSCV + ABLASI `tahun`)
# ==============================================================================
def _fit_with_inner_early_stopping(X_tr, y_tr, params, cfg):
    n_inner_val = max(3, int(np.ceil(len(X_tr) * cfg.inner_val_fraction)))
    X_fit, y_fit = X_tr.iloc[:-n_inner_val], y_tr.iloc[:-n_inner_val]
    X_val, y_val = X_tr.iloc[-n_inner_val:], y_tr.iloc[-n_inner_val:]
    model = xgb.XGBRegressor(
        n_estimators=cfg.max_boost_rounds,
        early_stopping_rounds=cfg.early_stopping_rounds,
        eval_metric="rmse",
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=1,
        **params,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def make_xgb_objective(X_train, y_train, cfg):
    tscv = TimeSeriesSplit(n_splits=cfg.n_cv_splits)
    ss = cfg.xgb_search_space

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *ss["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
            "gamma": trial.suggest_float("gamma", *ss["gamma"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"], log=True),
        }
        rmses = []
        for ft_idx, fv_idx in tscv.split(X_train):
            X_ft, y_ft = X_train.iloc[ft_idx], y_train.iloc[ft_idx]
            X_fv, y_fv = X_train.iloc[fv_idx], y_train.iloc[fv_idx]
            if len(X_ft) < 15: continue
            model = _fit_with_inner_early_stopping(X_ft, y_ft, params, cfg)
            preds = model.predict(X_fv)
            rmses.append(np.sqrt(np.mean((y_fv.values - preds) ** 2)))
        return float(np.mean(rmses))
    return objective


def tune_xgb(X_train, y_train, cfg, label):
    print(f"\n[TUNING] XGBoost {label}: Bayesian Optimization + Nested TSCV")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )
    study.optimize(
        make_xgb_objective(X_train, y_train, cfg),
        n_trials=cfg.n_optuna_trials,
        show_progress_bar=False,
    )
    print(f"[TUNING] RMSE-CV terbaik = {study.best_value:,.2f}")
    return study.best_params, study.best_value


def run_xgboost_model(model_label, AR, CAL_T, CAL_NT, EXO, train, test, cfg, fig_dir):
    print("\n" + "=" * 80)
    print(f"MODEL: XGBoost {model_label}")
    print("=" * 80)

    # Tentukan kandidat berdasarkan label (Baseline vs Penuh)
    if model_label == "Baseline":
        candidates = {
            "dengan_tahun": AR + CAL_T,
            "tanpa_tahun": AR + CAL_NT,
        }
    else:  # Penuh
        candidates = {
            "dengan_tahun": AR + CAL_T + EXO,
            "tanpa_tahun": AR + CAL_NT + EXO,
        }

    results = {}
    for tag, feats in candidates.items():
        print(f"\n[ABLATION] XGBoost {model_label} ({tag}) ...")
        X_tr, y_tr = train[feats], train[cfg.target_col]
        params, cv = tune_xgb(X_tr, y_tr, cfg, f"{model_label}_{tag}")
        results[tag] = {"feats": feats, "params": params, "cv_rmse": cv}
        print(f"[ABLATION] CV-RMSE ({tag}) = {cv:,.0f}")

    # Pilih varian terbaik berdasarkan CV (bukan test!)
    best_tag = min(results, key=lambda t: results[t]["cv_rmse"])
    feats = results[best_tag]["feats"]
    params = results[best_tag]["params"]
    print(f"[SELECT] XGBoost {model_label}: {best_tag}")

    # Latih model final
    probe = _fit_with_inner_early_stopping(train[feats], train[cfg.target_col], params, cfg)
    best_n_trees = probe.best_iteration + 1
    final_model = xgb.XGBRegressor(
        n_estimators=best_n_trees, objective="reg:squarederror",
        random_state=cfg.random_state, n_jobs=1, **params,
    )
    final_model.fit(train[feats], train[cfg.target_col], verbose=False)

    pred_train = final_model.predict(train[feats])
    pred_test = final_model.predict(test[feats])

    return {
        "name": f"XGBoost {model_label}",
        "model": final_model,
        "features": feats,
        "cv_rmse": results[best_tag]["cv_rmse"],
        "ablation": {t: r["cv_rmse"] for t, r in results.items()},
        "chosen_variant": best_tag,
        "best_params": params,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "metrics_train": compute_metrics(train[cfg.target_col].values, pred_train),
        "metrics_test": compute_metrics(test[cfg.target_col].values, pred_test),
        "residual_test": test[cfg.target_col].values - pred_test,
    }


# ==============================================================================
# 5. SARIMA / SARIMAX (GRID + TSCV + RULE LOG + DIAGNOSTIK)
# ==============================================================================
def stationarity_report(y):
    print("\n[STASIONERITAS] ADF & KPSS (data latih):")
    for label, s in [
        ("level", y), ("diff(1)", y.diff()),
        ("diff(12)", y.diff(12)), ("diff(1)+diff(12)", y.diff().diff(12)),
    ]:
        s = s.dropna()
        a_p = adfuller(s, autolag="AIC")[1]
        try: k_p = kpss(s, regression="c", nlags="auto")[1]
        except Exception: k_p = np.nan
        print(f"  {label:25s} ADF p={a_p:.4f} | KPSS p={k_p:.4f}")


def _is_valid_fit(res, order6, cfg):
    p, d, q, P, D, Q = order6
    if (p + q + P + Q) > cfg.max_total_arma_params: return False
    if not np.isfinite(res.llf) or abs(res.llf) < 1e-6: return False  # FIX: buang solusi degenerate (loglik~0)
    if np.any(~np.isfinite(res.bse)): return False
    if res.mle_retvals.get("converged", True) is False: return False
    return True


def grid_search_sarimax_order(y, cfg, exog=None, label="", ckpt_key=None):
    """Grid search (p,d,q)(P,D,Q,12) dengan filter solusi degenerate.
    Jika ckpt_key diberikan, progres disimpan tiap 50 kombinasi supaya bisa
    dilanjutkan tanpa mengulang dari awal jika proses terhenti di tengah."""
    combos = list(itertools.product(
        range(cfg.p_range[0], cfg.p_range[1] + 1),
        range(cfg.d_range[0], cfg.d_range[1] + 1),
        range(cfg.q_range[0], cfg.q_range[1] + 1),
        range(cfg.P_range[0], cfg.P_range[1] + 1),
        range(cfg.D_range[0], cfg.D_range[1] + 1),
        range(cfg.Q_range[0], cfg.Q_range[1] + 1),
    ))

    start_idx = 0
    candidates = []
    if ckpt_key is not None:
        cached = ckpt_load(cfg, ckpt_key)
        if cached is not None:
            start_idx = cached["next_idx"]
            candidates = cached["candidates"]
            if start_idx >= len(combos):
                print(f"[GRID] {label}: {len(candidates)} kandidat valid (dari checkpoint, sudah lengkap)")
                candidates.sort(key=lambda x: x[1])
                return candidates
            print(f"[GRID] {label}: melanjutkan dari kombinasi ke-{start_idx}/{len(combos)} "
                  f"({len(candidates)} kandidat valid sejauh ini)")

    print(f"\n[GRID] {label}: mencoba {len(combos)} kombinasi (p,d,q)(P,D,Q,12)...")
    for i in range(start_idx, len(combos)):
        o6 = combos[i]
        try:
            mod = SARIMAX(y, exog=exog, order=o6[:3],
                          seasonal_order=(*o6[3:], cfg.seasonal_period),
                          enforce_stationarity=True, enforce_invertibility=True)
            res = mod.fit(disp=False, maxiter=100)
            if _is_valid_fit(res, o6, cfg):
                candidates.append((o6, res.aic))
        except Exception:
            continue
        if ckpt_key is not None and (i + 1) % 50 == 0:
            ckpt_save(cfg, ckpt_key, {"next_idx": i + 1, "candidates": candidates})

    if ckpt_key is not None:
        ckpt_save(cfg, ckpt_key, {"next_idx": len(combos), "candidates": candidates})

    candidates.sort(key=lambda x: x[1])
    print(f"[GRID] {label}: {len(candidates)} kandidat valid")
    return candidates


def cv_validate_sarima(y, candidates, cfg, exog=None, label="", use_log=False):
    """Validasi ulang kandidat teratas via TSCV (RMSE out-of-sample).

    PERBAIKAN PENTING: parameter `y` SELALU berskala LEVEL (bukan log), apa pun
    nilai `use_log`. Jika `use_log=True`, fit dilakukan pada log(y) di internal,
    tapi forecast dikembalikan ke skala level (dgn koreksi smearing/log-normal,
    exp(mu + sigma^2/2)) SEBELUM RMSE dihitung. Ini krusial: RMSE yang dihitung
    di skala log (~0.05-0.3) TIDAK BISA dibandingkan langsung dengan RMSE skala
    level (puluhan-ratusan miliar Rupiah) - keduanya beda satuan sepenuhnya.
    Membandingkan mentah-mentah akan HAMPIR SELALU memenangkan log secara
    spurious, terlepas dari kualitas peramalan yang sesungguhnya."""
    tscv = TimeSeriesSplit(n_splits=cfg.n_cv_splits)
    shortlist = candidates[:cfg.n_aic_shortlist]
    print(f"\n[CV] {label}: menguji {len(shortlist)} kandidat via TSCV "
          f"(skala evaluasi: {'LEVEL, via back-transform log' if use_log else 'LEVEL'})...")
    cv_results = []
    for o6, aic in shortlist:
        rmses = []
        for ft_idx, fv_idx in tscv.split(y):
            if len(ft_idx) < 30: continue
            y_ft, y_fv = y.iloc[ft_idx], y.iloc[fv_idx]
            ex_ft = exog.iloc[ft_idx] if exog is not None else None
            ex_fv = exog.iloc[fv_idx] if exog is not None else None
            try:
                y_fit_scale = np.log(y_ft) if use_log else y_ft
                mod = SARIMAX(y_fit_scale, exog=ex_ft, order=o6[:3],
                              seasonal_order=(*o6[3:], cfg.seasonal_period),
                              enforce_stationarity=True, enforce_invertibility=True)
                res = mod.fit(disp=False, maxiter=100)
                fc = res.get_forecast(steps=len(fv_idx), exog=ex_fv)
                pred = fc.predicted_mean.values
                if use_log:
                    # Koreksi smearing (Duan, 1983) supaya E[Y] bukan E[log Y]
                    sigma2 = float(np.var(res.resid, ddof=0))
                    pred = np.exp(pred + 0.5 * sigma2)
                # RMSE SELALU di skala level (y_fv sudah level) -> apple-to-apple
                rmses.append(np.sqrt(np.mean((y_fv.values - pred) ** 2)))
            except Exception:
                continue
        if rmses:
            cv_results.append((o6, float(np.mean(rmses)), aic))
    cv_results.sort(key=lambda x: x[1])
    print(f"[CV] {label}: hasil (order, RMSE-CV [skala level], AIC):")
    for o6, rmse_cv, aic in cv_results[:3]:
        print(f"    {o6}  RMSE-CV={rmse_cv:,.2f}  AIC={aic:.0f}")
    best_o6, best_cv, _ = cv_results[0]
    return best_o6, best_cv


def run_sarimax_model(model_label, train, test, cfg, fig_dir, use_exog):
    print("\n" + "=" * 80)
    print(f"MODEL: {model_label}")
    print("=" * 80)

    y_train = train[cfg.target_col]
    y_test = test[cfg.target_col]

    if model_label == "SARIMA":
        stationarity_report(y_train)

    exog_ablation_info = None
    if use_exog:
        # --- ABLASI EKSOGEN (TAMBAHAN): level vs terdiferensiasi ---
        # bbnkb & ikk non-stasioner di level (lihat engineer_features), yang
        # secara teoretis berisiko membuat SARIMAX tidak stabil saat
        # ekstrapolasi ke luar rentang data latih. Diuji via CV (bukan
        # diasumsikan otomatis benar) - konsisten dgn filosofi ablasi
        # `tahun`/log yang sudah kamu terapkan di tempat lain pada pipeline.
        exog_variants = {
            "level": get_sarimax_exog_cols(),
            "terdiferensiasi": get_sarimax_exog_cols_diff(),
        }
        variant_results = {}
        for tag, cols in exog_variants.items():
            ckpt_key = f"sarimax_exog_{tag}"
            cached = ckpt_load(cfg, ckpt_key)
            if cached is not None:
                variant_results[tag] = cached
                continue
            ex_tr = train[cols]
            print(f"\n[ABLASI EKSOGEN] SARIMAX ({tag}) ...")
            cands = grid_search_sarimax_order(y_train, cfg, exog=ex_tr, label=f"SARIMAX_{tag}",
                                               ckpt_key=f"grid_sarimax_{tag}")
            o6, cv = cv_validate_sarima(y_train, cands, cfg, exog=ex_tr, label=f"SARIMAX_{tag}")
            variant_results[tag] = {"cols": cols, "candidates": cands, "order": o6, "cv_rmse": cv}
            print(f"[ABLASI EKSOGEN] CV-RMSE ({tag}) = {cv:,.2f}")
            ckpt_save(cfg, ckpt_key, variant_results[tag])

        best_tag = min(variant_results, key=lambda t: variant_results[t]["cv_rmse"])
        print(f"[ABLASI EKSOGEN] Terpilih: eksogen '{best_tag}' "
              f"(CV level={variant_results['level']['cv_rmse']:,.2f} vs "
              f"terdiferensiasi={variant_results['terdiferensiasi']['cv_rmse']:,.2f})")
        exog_cols = variant_results[best_tag]["cols"]
        exog_train = train[exog_cols]
        exog_test = test[exog_cols]
        candidates = variant_results[best_tag]["candidates"]
        best_o6, best_cv = variant_results[best_tag]["order"], variant_results[best_tag]["cv_rmse"]
        exog_ablation_info = {tag: v["cv_rmse"] for tag, v in variant_results.items()}
        exog_ablation_info["chosen"] = best_tag
    else:
        exog_cols = None
        exog_train = None
        exog_test = None
        # Grid + CV validation (SARIMA: tanpa eksogen, tidak perlu ablasi)
        candidates = grid_search_sarimax_order(y_train, cfg, exog=exog_train, label=model_label,
                                                ckpt_key=f"grid_{model_label.lower()}")
        best_o6, best_cv = cv_validate_sarima(y_train, candidates, cfg, exog=exog_train, label=model_label)

    # Fit final
    mod = SARIMAX(y_train, exog=exog_train, order=best_o6[:3],
                  seasonal_order=(*best_o6[3:], cfg.seasonal_period),
                  enforce_stationarity=True, enforce_invertibility=True)
    res = mod.fit(disp=False, maxiter=200)

    # Rule log-transform based on Breakvar test (konsisten proposal 3.4.4)
    use_log = False
    try:
        het = res.test_heteroskedasticity(method="breakvar")[0]
        het_p = float(het[0, 1] if het.ndim == 2 else het[1])
    except Exception:
        het_p = np.nan

    if cfg.log_decision_rule and (het_p == het_p) and het_p < 0.05:
        print(f"[{model_label}] Breakvar p={het_p:.4f} -> mencoba ablasi log-transform...")
        try:
            y_log = np.log(y_train)
            cands_log = grid_search_sarimax_order(y_log, cfg, exog=exog_train, label=f"{model_label}_log",
                                                   ckpt_key=f"grid_{model_label.lower()}_log")
            # FIX: y_train (LEVEL) dipakai di sini, bukan y_log -> cv_validate_sarima
            # menangani transformasi log secara internal & mengembalikan RMSE
            # yang SUDAH di skala level, sehingga sepadan (apple-to-apple)
            # dengan best_cv (level) yang dihitung sebelumnya.
            best_o6_log, best_cv_log = cv_validate_sarima(
                y_train, cands_log, cfg, exog=exog_train,
                label=f"{model_label}_log", use_log=True)
            print(f"[{model_label}] CV (skala level) - level={best_cv:,.2f} vs log={best_cv_log:,.2f}")
            if best_cv_log < best_cv:
                use_log = True
                best_o6 = best_o6_log
                best_cv = best_cv_log
                mod = SARIMAX(y_log, exog=exog_train, order=best_o6[:3],
                              seasonal_order=(*best_o6[3:], cfg.seasonal_period),
                              enforce_stationarity=True, enforce_invertibility=True)
                res = mod.fit(disp=False, maxiter=200)
                try:
                    het2 = res.test_heteroskedasticity(method="breakvar")[0]
                    het_p = float(het2[0, 1] if het2.ndim == 2 else het2[1])
                except Exception:
                    pass
        except Exception:
            pass

    # Diagnostik residual
    lb = acorr_ljungbox(res.resid, lags=[12], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    jb_p = float(jarque_bera(res.resid).pvalue)

    print(f"[FINAL] {model_label}: orde {best_o6[:3]}x{best_o6[3:]} log={use_log}")
    print(f"[DIAG]  Ljung-Box(12) p={lb_p:.4f} | Jarque-Bera p={jb_p:.4f} | Breakvar p={het_p if het_p==het_p else np.nan}")

    # Forecast
    fc = res.get_forecast(steps=len(y_test), exog=exog_test)
    pred_test = fc.predicted_mean.values
    if use_log:
        pred_test = np.exp(pred_test + 0.5 * float(np.var(res.resid, ddof=0)))

    # Fitted values (buang burn-in differencing)
    burn = best_o6[1] + best_o6[4] * cfg.seasonal_period
    fitted = res.fittedvalues.iloc[burn:]
    if use_log:
        fitted = np.exp(fitted + 0.5 * float(np.var(res.resid, ddof=0)))
    y_train_eval = y_train.iloc[burn:]

    return {
        "name": model_label,
        "order": best_o6[:3],
        "seasonal_order": best_o6[3:],
        "use_log": use_log,
        "exog_cols": exog_cols,
        "exog_ablation": exog_ablation_info,
        "cv_rmse": best_cv,
        "pred_train": np.asarray(fitted),
        "y_train_eval": y_train_eval.values,
        "pred_test": pred_test,
        "metrics_train": compute_metrics(y_train_eval.values, np.asarray(fitted)),
        "metrics_test": compute_metrics(y_test.values, pred_test),
        "residual_test": y_test.values - pred_test,
        "diagnostics": {
            "ljung_box_p": lb_p,
            "jarque_bera_p": jb_p,
            "breakvar_p": het_p if het_p == het_p else np.nan,
        },
    }


# ==============================================================================
# 6. RIDGE REGRESSION (TSCV KASUAL)
# ==============================================================================
def run_ridge_model(train, test, features, cfg):
    print("\n" + "=" * 80)
    print("MODEL: Ridge Regression (baseline linear)")
    print("=" * 80)
    X_tr, y_tr = train[features], train[cfg.target_col]
    X_te, y_te = test[features], test[cfg.target_col]

    scaler = StandardScaler()
    Xs_tr, Xs_te = scaler.fit_transform(X_tr), scaler.transform(X_te)

    tscv = TimeSeriesSplit(n_splits=cfg.n_cv_splits)
    model = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=tscv).fit(Xs_tr, y_tr)

    # Hitung RMSE-CV manual untuk konsistensi ensemble weighting
    fold_rmses = []
    for ft_idx, fv_idx in tscv.split(Xs_tr):
        m = Ridge(alpha=model.alpha_).fit(Xs_tr[ft_idx], y_tr.iloc[ft_idx])
        preds = m.predict(Xs_tr[fv_idx])
        fold_rmses.append(np.sqrt(np.mean((y_tr.iloc[fv_idx].values - preds) ** 2)))
    cv_rmse = float(np.mean(fold_rmses))

    print(f"[RIDGE] alpha={model.alpha_:.4f} | CV-RMSE={cv_rmse:,.0f}")
    pred_train = model.predict(Xs_tr)
    pred_test = model.predict(Xs_te)
    return {
        "name": "Ridge Regression",
        "cv_rmse": cv_rmse,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "metrics_train": compute_metrics(y_tr.values, pred_train),
        "metrics_test": compute_metrics(y_te.values, pred_test),
        "residual_test": y_te.values - pred_test,
    }


# ==============================================================================
# 7. ENSEMBLE (BATES & GRANGER, 1969)
# ==============================================================================
def run_ensemble(model_a, model_b, y_test):
    """Ensemble: rata-rata sederhana + berbobot invers-CV-RMSE."""
    pred_a, pred_b = model_a["pred_test"], model_b["pred_test"]
    out = []

    # (a) Simple average
    p_simple = (pred_a + pred_b) / 2
    out.append({
        "name": f"Ensemble Rata-rata ({model_a['name']}+{model_b['name']})",
        "pred_test": p_simple,
        "metrics_test": compute_metrics(y_test, p_simple),
        "residual_test": y_test - p_simple,
    })

    # (b) Inverse-CV weighted
    ra = model_a.get("cv_rmse", model_a["metrics_test"]["RMSE"])
    rb = model_b.get("cv_rmse", model_b["metrics_test"]["RMSE"])
    wa = (1 / ra) / (1 / ra + 1 / rb)
    wb = 1 - wa
    p_w = wa * pred_a + wb * pred_b
    out.append({
        "name": f"Ensemble Berbobot ({model_a['name']}+{model_b['name']})",
        "pred_test": p_w,
        "metrics_test": compute_metrics(y_test, p_w),
        "residual_test": y_test - p_w,
        "weights": {model_a["name"]: wa, model_b["name"]: wb},
    })
    print(f"[ENSEMBLE] w_{model_a['name']}={wa:.3f}, w_{model_b['name']}={wb:.3f}")
    return out



# ==============================================================================
# 8. VISUALISASI
# ==============================================================================

# Nama model inti yang tampil di plot perbandingan, CI, dan heatmap DM.
# Ensemble & oracle tetap dihitung dan masuk tabel CSV, tapi tidak diplot
# supaya grafik tetap terbaca. Penguji bisa lihat detailnya di tabel.
CORE_MODEL_NAMES = [
    "SARIMA", "SARIMAX",
    "XGBoost Baseline", "XGBoost Penuh",
    "Ridge Regression", "Naive (Persistence)",
]


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# (1) PERBANDINGAN METRIK + CI — hanya core models, diurutkan RMSE
def plot_metrics_with_ci(summary_df, fig_dir):
    df = summary_df[summary_df["Model"].isin(CORE_MODEL_NAMES)].copy()
    df = df.sort_values("RMSE_Uji", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.55 * len(df) + 2)))
    for ax, metric, ci_col in zip(
        axes, ["RMSE_Uji", "MAPE_Uji (%)"], ["RMSE_CI95", "MAPE_CI95"]
    ):
        vals   = df[metric].values.astype(float)
        lowers = np.array([x[0] if isinstance(x, (tuple, list)) else np.nan
                           for x in df[ci_col]])
        uppers = np.array([x[1] if isinstance(x, (tuple, list)) else np.nan
                           for x in df[ci_col]])
        err_lo = np.clip(vals - lowers, 0, None)
        err_hi = np.clip(uppers - vals, 0, None)
        ax.barh(df["Model"], vals, xerr=[err_lo, err_hi],
                color="#2b6cb0", capsize=4, ecolor="#c05621", height=0.6)
        ax.set_title(f"{metric} dengan CI 95% (Block Bootstrap)")
        ax.set_xlabel(metric)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(fig_dir / "perbandingan_metrik_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] perbandingan_metrik_ci.png")


# (2) AKTUAL vs PREDIKSI PER MODEL — semua model (incl. ensemble)
def plot_actual_vs_predicted(r, train, test, cfg, fig_dir):
    d_tr, d_te = train[cfg.date_col], test[cfg.date_col]
    y_tr = train[cfg.target_col].values
    y_te = test[cfg.target_col].values
    ptr  = r.get("pred_train")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(d_tr, y_tr, color="#1a202c", lw=1.2, label="Aktual (Latih)")
    if ptr is not None:
        # SARIMA/SARIMAX punya burn-in -> pred_train lebih pendek dari d_tr
        if len(ptr) < len(d_tr):
            ax.plot(d_tr.iloc[-len(ptr):], ptr, "--", color="#63b3ed",
                    lw=1.1, label="Prediksi (Latih)")
        else:
            ax.plot(d_tr, ptr, "--", color="#63b3ed", lw=1.1, label="Prediksi (Latih)")
    ax.plot(d_te, y_te, color="#c05621", lw=1.6, label="Aktual (Uji)")
    ax.plot(d_te, r["pred_test"], "--", color="#f6ad55", lw=1.5, label="Prediksi (Uji)")
    ax.axvline(d_te.iloc[0], color="gray", ls=":", lw=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylabel("PKB (Miliar Rupiah)")
    ax.set_xlabel("Periode")
    ax.set_title(f"Aktual vs Prediksi — {r['name']}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"actual_vs_predicted_{_slug(r['name'])}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# (3) PERBANDINGAN SEMUA MODEL — hanya core models agar legenda terbaca
def plot_all_models_forecast(results, dates_test, y_test, fig_dir):
    core = [r for r in results if r["name"] in CORE_MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_test, y_test, color="black", lw=2.2, label="Aktual", zorder=10)
    colors = plt.cm.tab10(np.linspace(0, 1, len(core)))
    for r, c in zip(core, colors):
        ax.plot(dates_test, r["pred_test"], "--", lw=1.5, color=c,
                alpha=0.85, label=r["name"])
    ax.set_title("Perbandingan Prediksi Model Inti pada Data Uji")
    ax.set_xlabel("Periode")
    ax.set_ylabel("PKB (Miliar Rupiah)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "perbandingan_semua_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] perbandingan_semua_model.png")


# (4) DIAGNOSTIK SARIMA/SARIMAX — 4-panel statsmodels (residual, hist+KDE, QQ, correlogram)
def plot_sarima_diagnostics(model_result, label, fig_dir):
    """Plot diagnostik residual 4-panel dari statsmodels: standardized residual,
    histogram + estimated density, Normal Q-Q, dan correlogram (ACF)."""
    fig = model_result.plot_diagnostics(figsize=(12, 8))
    fig.suptitle(f"Diagnostik Residual — {label}", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{_slug(label)}_diagnostics.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {_slug(label)}_diagnostics.png")


# (5) GAIN & COVER — modular, terbaca, dengan grid
def plot_gain_cover(model, features, label, fig_dir):
    booster = model.get_booster()
    booster.feature_names = list(features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, imp, title in zip(axes, ["gain", "cover"], ["Gain", "Cover"]):
        score = booster.get_score(importance_type=imp)
        if score:
            s = pd.Series(score).sort_values(ascending=True)
            (s / s.sum() * 100).plot(kind="barh", ax=ax, color="#2b6cb0")
            ax.set_xlabel(f"{title} Relatif (%)")
        ax.set_title(f"Feature Importance ({title}) — XGBoost {label}")
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(fig_dir / f"xgb_{label.lower()}_gain_cover.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] xgb_{label.lower()}_gain_cover.png")


# (6) SHAP SUMMARY — dengan error handling
def plot_shap_summary(model, X, label, fig_dir):
    try:
        expl = shap.TreeExplainer(model)
        sv   = expl.shap_values(X)
        fig  = plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X, show=False, plot_size=None)
        plt.title(f"SHAP Summary — XGBoost {label}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"xgb_{label.lower()}_shap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] xgb_{label.lower()}_shap.png")
    except Exception as e:
        print(f"[WARN] SHAP {label} gagal: {e}")


# (7) HEATMAP DM — hanya core models, label terbaca, font lebih besar
def plot_dm_heatmap(p_mat, fig_dir):
    core = [m for m in CORE_MODEL_NAMES if m in p_mat.index]
    pm   = p_mat.loc[core, core]
    n    = len(pm)
    fig, ax = plt.subplots(figsize=(max(7, n * 1.3), max(6, n * 1.1)))
    im = ax.imshow(pm.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(pm.columns, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(pm.index, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = pm.values[i, j]
            c = "white" if v < 0.15 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=c)
    ax.set_title("Matriks p-value Uji Diebold-Mariano (distribusi-t)\n"
                 "(hijau = tidak signifikan, merah = signifikan, α = 5%)")
    plt.colorbar(im, ax=ax, label="p-value")
    plt.tight_layout()
    plt.savefig(fig_dir / "dm_test_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] dm_test_heatmap.png")


# ==============================================================================
# 9. MAIN PIPELINE
# ==============================================================================
def main(cfg: Config = CFG):
    # Setup direktori
    out = Path(cfg.output_dir)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    mod_dir = out / "models"
    for d in [fig_dir, tab_dir, mod_dir]: d.mkdir(parents=True, exist_ok=True)

    # Data
    df = load_and_engineer_features(cfg)
    AR, CAL_T, CAL_NT, EXO = get_xgb_feature_sets()
    train, test = split_train_test(df, cfg)
    y_te = test[cfg.target_col].values

    # Build semua model
    all_results = []
    res_xgb_base = ckpt_load(cfg, "xgb_baseline")
    if res_xgb_base is None:
        res_xgb_base = run_xgboost_model("Baseline", AR, CAL_T, CAL_NT, EXO, train, test, cfg, fig_dir)
        ckpt_save(cfg, "xgb_baseline", res_xgb_base)

    res_xgb_full = ckpt_load(cfg, "xgb_full")
    if res_xgb_full is None:
        res_xgb_full = run_xgboost_model("Penuh", AR, CAL_T, CAL_NT, EXO, train, test, cfg, fig_dir)
        ckpt_save(cfg, "xgb_full", res_xgb_full)

    res_sarima = ckpt_load(cfg, "sarima")
    if res_sarima is None:
        res_sarima = run_sarimax_model("SARIMA", train, test, cfg, fig_dir, use_exog=False)
        ckpt_save(cfg, "sarima", res_sarima)

    res_sarimax = ckpt_load(cfg, "sarimax")
    if res_sarimax is None:
        res_sarimax = run_sarimax_model("SARIMAX", train, test, cfg, fig_dir, use_exog=True)
        ckpt_save(cfg, "sarimax", res_sarimax)

    res_ridge = ckpt_load(cfg, "ridge")
    if res_ridge is None:
        res_ridge = run_ridge_model(train, test, AR + CAL_NT + EXO, cfg)
        ckpt_save(cfg, "ridge", res_ridge)

    # Naive persistence
    naive_pred = test["pkb_lag_1"].values
    # FIX: sebelumnya membandingkan train[target][1:] vs train["pkb_lag_1"][:-1],
    # yang secara tidak sengaja menggeser SEKALI LAGI kolom yang sudah di-lag
    # (efektif membandingkan y_t vs y_{t-2}, bukan y_{t-1}). Karena "pkb_lag_1"
    # sudah merupakan y_{t-1} yang selaras baris-per-baris, cukup dibandingkan
    # langsung tanpa slicing tambahan.
    res_naive = {
        "name": "Naive (Persistence)",
        "pred_train": train["pkb_lag_1"].values,
        "pred_test": naive_pred,
        "metrics_train": compute_metrics(train[cfg.target_col].values, train["pkb_lag_1"].values),
        "metrics_test": compute_metrics(y_te, naive_pred),
        "residual_test": y_te - naive_pred,
    }

    all_results += [res_xgb_base, res_xgb_full, res_sarima, res_sarimax, res_ridge, res_naive]

    # Ensemble #1: lintas keluarga (SARIMAX + XGBoost Penuh)
    ensembles_a = run_ensemble(res_sarimax, res_xgb_full, y_te)
    all_results += ensembles_a

    # Ensemble #2: dua model individual terbaik by RMSE-CV (kausal)
    ranked_cv = sorted([r for r in all_results if "Ensemble" not in r["name"] and "cv_rmse" in r],
                       key=lambda r: r["cv_rmse"])
    if len(ranked_cv) >= 2:
        ensembles_b = run_ensemble(ranked_cv[0], ranked_cv[1], y_te)
        all_results += ensembles_b

    # Ensemble #3 (ORACLE eksploratif): dua model dgn RMSE uji terendah
    ranked_test = sorted([r for r in all_results if "Ensemble" not in r["name"]],
                         key=lambda r: r["metrics_test"]["RMSE"])
    if len(ranked_test) >= 2:
        if {ranked_test[0]["name"], ranked_test[1]["name"]} != {ranked_cv[0]["name"], ranked_cv[1]["name"]}:
            print(f"\n[ORACLE] Pasangan berbeda dari CV: {ranked_test[0]['name']} & {ranked_test[1]['name']}")
            ensembles_c = run_ensemble(ranked_test[0], ranked_test[1], y_te)
            for e in ensembles_c:
                e["name"] = "[ORACLE-EKSPLORATIF] " + e["name"]
            all_results += ensembles_c

    # Plot diagnostik SARIMA/SARIMAX (re-fit untuk plot)
    try:
        p1, p2 = res_sarima["order"], (*res_sarima["seasonal_order"], cfg.seasonal_period)
        y_tr_s = np.log(train[cfg.target_col]) if res_sarima["use_log"] else train[cfg.target_col]
        m1 = SARIMAX(y_tr_s, order=p1, seasonal_order=p2,
                     enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        plot_sarima_diagnostics(m1, "SARIMA", fig_dir)
    except Exception as e:
        print(f"[WARN] Diagnostik plot SARIMA gagal: {e}")

    try:
        p1, p2 = res_sarimax["order"], (*res_sarimax["seasonal_order"], cfg.seasonal_period)
        # FIX: pakai exog_cols HASIL ABLASI yang benar-benar dipakai model
        # final (bisa 'level' atau 'terdiferensiasi'), bukan hardcode level.
        exog_tr = train[res_sarimax["exog_cols"]]
        y_tr_s = np.log(train[cfg.target_col]) if res_sarimax["use_log"] else train[cfg.target_col]
        m2 = SARIMAX(y_tr_s, exog=exog_tr, order=p1, seasonal_order=p2,
                     enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        plot_sarima_diagnostics(m2, "SARIMAX", fig_dir)
    except Exception as e:
        print(f"[WARN] Diagnostik plot SARIMAX gagal: {e}")

    # Bootstrap CI
    print("\n" + "=" * 80)
    print("BOOTSTRAP CI (95%) - DATA UJI")
    print("=" * 80)
    for r in all_results:
        r["ci"] = block_bootstrap_ci(y_te, r["pred_test"], cfg)
        print(f"[{r['name']}] MAPE 95% CI: {r['ci']['MAPE_CI95']}")

    # Ringkasan akhir
    summary_rows = []
    for res in all_results:
        m_train = res.get("metrics_train", {})
        summary_rows.append({
            "Model": res["name"],
            "RMSE_CV_Latih": res.get("cv_rmse", np.nan),
            "RMSE_Latih": m_train.get("RMSE", np.nan),
            "MAPE_Latih (%)": m_train.get("MAPE (%)", np.nan),
            "RMSE_Uji": res["metrics_test"]["RMSE"],
            "MAE_Uji": res["metrics_test"]["MAE"],
            "MAPE_Uji (%)": res["metrics_test"]["MAPE (%)"],
            # FIX: kolom berikut sebelumnya TIDAK ADA, menyebabkan KeyError di
            # plot_metrics_with_ci() dan meng-crash pipeline SETELAH seluruh
            # komputasi berat (tuning, grid search, bootstrap) selesai --
            # seluruh plot & file sesudah titik ini (gain/cover, SHAP, model
            # tersimpan, metadata JSON) tidak akan pernah dibuat.
            "RMSE_CI95": res["ci"]["RMSE_CI95"],
            "MAPE_CI95": res["ci"]["MAPE_CI95"],
            "MAPE_CI95 (%)": f"[{res['ci']['MAPE_CI95'][0]:.2f}; {res['ci']['MAPE_CI95'][1]:.2f}]"
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE_Uji").reset_index(drop=True)
    summary_df.to_csv(tab_dir / "ringkasan_semua_model.csv", index=False)

    print("\n" + "=" * 120)
    print(" RINGKASAN AKHIR (diurutkan dari RMSE Uji terbaik) - Skala: Miliar Rupiah")
    print("=" * 120)
    # Format agar mudah dibaca di console
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}" if pd.notnull(x) else "-"))

    # Matriks DM
    res_dict = {r["name"]: r["residual_test"] for r in all_results}
    dm_stat, dm_pn, dm_pt = build_dm_matrix(res_dict)
    dm_stat.to_csv(tab_dir / "dm_matrix_statistic.csv")
    dm_pn.to_csv(tab_dir / "dm_matrix_pvalue_normal.csv")
    dm_pt.to_csv(tab_dir / "dm_matrix_pvalue_t.csv")

    # Visualisasi
    plot_all_models_forecast(all_results, test[cfg.date_col], y_te, fig_dir)
    plot_metrics_with_ci(summary_df, fig_dir)
    plot_dm_heatmap(dm_pt, fig_dir)

    # Gain/Cover + SHAP untuk XGBoost
    plot_gain_cover(res_xgb_base["model"], res_xgb_base["features"], "Baseline", fig_dir)
    plot_gain_cover(res_xgb_full["model"], res_xgb_full["features"], "Penuh", fig_dir)
    plot_shap_summary(res_xgb_base["model"], test[res_xgb_base["features"]], "Baseline", fig_dir)
    plot_shap_summary(res_xgb_full["model"], test[res_xgb_full["features"]], "Penuh", fig_dir)

    # Actual vs predicted untuk tiap model
    for r in all_results:
        try:
            plot_actual_vs_predicted(r, train, test, cfg, fig_dir)
        except Exception as e:
            print(f"[WARN] Plot actual-vs-predicted gagal untuk {r['name']}: {e}")

    # Simpan model XGBoost
    res_xgb_base["model"].save_model(str(mod_dir / "xgb_baseline.json"))
    res_xgb_full["model"].save_model(str(mod_dir / "xgb_penuh.json"))

    # Metadata JSON
    meta = {}
    for r in all_results:
        meta[r["name"]] = {
            k: r.get(k) for k in ["order", "seasonal_order", "use_log", "diagnostics",
                                   "exog_cols", "exog_ablation",
                                   "cv_rmse", "ablation", "chosen_variant", "best_params", "weights"]
            if k in r
        }
    with open(tab_dir / "metadata_final.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n[SELESAI] Seluruh output tersimpan di: {out.resolve()}")
    print(f"   - Tabel: {tab_dir}")
    print(f"   - Gambar: {fig_dir}")
    print(f"   - Model: {mod_dir}")

    return {
        "results": all_results,
        "summary": summary_df,
        "dm_stat": dm_stat, "dm_p_norm": dm_pn, "dm_p_t": dm_pt,
    }


if __name__ == "__main__":
    main()