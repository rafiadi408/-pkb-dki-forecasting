"""
REMEDIASI BAGIAN A — Detrending + XGBoost Penuh
=================================================
Mengatasi keterbatasan ekstrapolasi XGBoost dengan memodelkan tren linear
secara terpisah (OLS, dikontrol dummy pandemi), lalu XGBoost meramalkan
komponen residual/detrended. Prediksi akhir = tren ekstrapolasi + prediksi
XGBoost atas residual.

PRASYARAT:
  - pipeline_v4_final.py dan dataset_skripsi.csv berada di direktori yang sama
    (atau path di Config sudah disesuaikan).
  - Jalankan pipeline_v4_final.py (fungsi main()) minimal sekali sebelumnya
    agar checkpoint model original (xgb_full.pkl, sarima.pkl, dst) tersedia
    untuk keperluan perbandingan di akhir script.

OUTPUT:
  - outputs_final/checkpoints/xgb_detrended_final.pkl
  - outputs_final/tables/xgb_detrended_summary.csv
  - outputs_final/figures/xgb_detrended_diagnostik.png
"""

import sys
import pickle
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# --- Reuse fungsi & Config dari pipeline utama ---
try:
    # Jika sel pipeline_v4_final.py sudah dijalankan di notebook yang sama,
    # fungsi-fungsi berikut sudah ada di namespace global.
    load_raw_data, engineer_features, prepare_modeling_sample
    split_train_test, get_xgb_feature_sets, tune_xgb
    _fit_with_inner_early_stopping, compute_metrics
    Config, CFG
except NameError:
    import pipeline_v4_final as P
    load_raw_data = P.load_raw_data
    engineer_features = P.engineer_features
    prepare_modeling_sample = P.prepare_modeling_sample
    split_train_test = P.split_train_test
    get_xgb_feature_sets = P.get_xgb_feature_sets
    tune_xgb = P.tune_xgb
    _fit_with_inner_early_stopping = P._fit_with_inner_early_stopping
    compute_metrics = P.compute_metrics
    Config = P.Config
    CFG = P.CFG

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = CFG.output_dir
CK = Path(BASE) / "checkpoints"
TB = Path(BASE) / "tables"
FG = Path(BASE) / "figures"
for p in (CK, TB, FG):
    p.mkdir(parents=True, exist_ok=True)

MAX_TRAIN = 983.24  # nilai maksimum data latih level (Des 2022) — untuk diagnosis ekstrapolasi


# ==============================================================================
# LANGKAH 1 — Estimasi & Ekstrapolasi Tren Linear (pada data latih saja)
# ==============================================================================
print("=" * 80)
print("LANGKAH 1: Estimasi Tren Linear (OLS, dikontrol dummy pandemi)")
print("=" * 80)

raw = load_raw_data(CFG)
raw["t_idx"] = np.arange(len(raw))

mask_train = (raw["tanggal"] >= "2015-01-01") & (raw["tanggal"] <= CFG.train_end_date)
X_trend = sm.add_constant(raw.loc[mask_train, ["t_idx", "covid19"]])
y_trend = raw.loc[mask_train, "pkb"]
trend_model = sm.OLS(y_trend, X_trend).fit()
print(trend_model.summary().tables[1])

a = float(trend_model.params["const"])
b = float(trend_model.params["t_idx"])
print(f"\n[TREN] pkb = {a:.3f} + {b:.4f} * t_idx  (komponen covid dikeluarkan dari tren)")

# Ekstrapolasi tren ke SELURUH periode (burn-in 2014, data latih, data uji)
raw["trend"] = a + b * raw["t_idx"]
raw["pkb_level"] = raw["pkb"]
raw["pkb_detrend"] = raw["pkb"] - raw["trend"]

print(f"[INFO] Rentang trend  : {raw['trend'].min():.1f} - {raw['trend'].max():.1f}")
print(f"[INFO] Rentang detrend: {raw['pkb_detrend'].min():.1f} - {raw['pkb_detrend'].max():.1f}")


# ==============================================================================
# LANGKAH 2 — Rekayasa Fitur dari Deret Detrended
# ==============================================================================
print("\n" + "=" * 80)
print("LANGKAH 2: Rekayasa Fitur (basis deret detrended)")
print("=" * 80)

df_dt = raw.copy()
df_dt["pkb"] = df_dt["pkb_detrend"]  # target sementara diganti detrended utk engineer_features

df_dt_feat = engineer_features(df_dt, CFG)
df_dt_final = prepare_modeling_sample(df_dt_feat, CFG)
train_dt, test_dt = split_train_test(df_dt_final, CFG)

print(f"[INFO] Rentang target latih (detrended): {train_dt['pkb'].min():.1f} - {train_dt['pkb'].max():.1f}")
print(f"[INFO] Rentang target uji   (detrended): {test_dt['pkb'].min():.1f} - {test_dt['pkb'].max():.1f}")


# ==============================================================================
# LANGKAH 3 — Ablasi Fitur `tahun` pada Target Detrended (konsisten prinsip pipeline utama)
# ==============================================================================
print("\n" + "=" * 80)
print("LANGKAH 3: Ablasi Fitur `tahun` + Tuning Penuh (100 trial Optuna)")
print("=" * 80)

AR, CAL_T, CAL_NT, EXO = get_xgb_feature_sets()
candidates = {
    "dengan_tahun": AR + CAL_T + EXO,
    "tanpa_tahun": AR + CAL_NT + EXO,
}

results = {}
for tag, feats in candidates.items():
    print(f"\n[ABLASI] XGBoost Detrended ({tag}) — {len(feats)} fitur ...")
    X_tr, y_tr = train_dt[feats], train_dt[CFG.target_col]
    t0 = time.time()
    params, cv_rmse = tune_xgb(X_tr, y_tr, CFG, f"Detrended_{tag}")
    print(f"[ABLASI] CV-RMSE ({tag}, skala detrended) = {cv_rmse:,.2f}  |  waktu={time.time()-t0:.1f}s")
    results[tag] = {"feats": feats, "params": params, "cv_rmse": cv_rmse}

best_tag = min(results, key=lambda t: results[t]["cv_rmse"])
feats = results[best_tag]["feats"]
params = results[best_tag]["params"]
print(f"\n[SELECT] Varian terpilih: {best_tag}  (CV-RMSE={results[best_tag]['cv_rmse']:.2f})")


# ==============================================================================
# LANGKAH 4 — Fit Model Final & Rekonstruksi Prediksi ke Skala Level
# ==============================================================================
print("\n" + "=" * 80)
print("LANGKAH 4: Fit Model Final & Rekonstruksi Skala Level")
print("=" * 80)

X_tr_final = train_dt[feats]
y_tr_final = train_dt[CFG.target_col]

probe = _fit_with_inner_early_stopping(X_tr_final, y_tr_final, params, CFG)
best_n_trees = probe.best_iteration + 1
final_model = xgb.XGBRegressor(
    n_estimators=best_n_trees, objective="reg:squarederror",
    random_state=CFG.random_state, n_jobs=-1, **params,
)
final_model.fit(X_tr_final, y_tr_final, verbose=False)
print(f"[FINAL] Jumlah pohon: {best_n_trees}")

pred_train_dt = final_model.predict(X_tr_final)
pred_test_dt = final_model.predict(test_dt[feats])

# Rekonstruksi: prediksi_detrend + tren ekstrapolasi = prediksi level
train_dt["trend_recon"] = a + b * train_dt["t_idx"]
test_dt["trend_recon"] = a + b * test_dt["t_idx"]

pred_train_level = pred_train_dt + train_dt["trend_recon"].values
pred_test_level = pred_test_dt + test_dt["trend_recon"].values

y_train_level = train_dt["pkb_level"].values
y_test_level = test_dt["pkb_level"].values

metrics_train = compute_metrics(y_train_level, pred_train_level)
metrics_test = compute_metrics(y_test_level, pred_test_level)

print(f"\n[METRIK] Train — RMSE={metrics_train['RMSE']:.2f} | MAPE={metrics_train['MAPE (%)']:.2f}%")
print(f"[METRIK] Test  — RMSE={metrics_test['RMSE']:.2f} | MAPE={metrics_test['MAPE (%)']:.2f}%")
print(f"[INFO] Range prediksi uji (level): {pred_test_level.min():.2f} - {pred_test_level.max():.2f}")
print(f"[INFO] Range aktual uji   (level): {y_test_level.min():.2f} - {y_test_level.max():.2f}")
print(f"[INFO] Max data latih (level)    : {MAX_TRAIN:.2f}")


# ==============================================================================
# LANGKAH 5 — Simpan Checkpoint & Bandingkan dengan Model Original
# ==============================================================================
result = {
    "name": "XGBoost Penuh (Detrended)",
    "features": feats,
    "chosen_variant": best_tag,
    "ablation": {t: r["cv_rmse"] for t, r in results.items()},
    "best_params": params,
    "n_trees": best_n_trees,
    "cv_rmse_detrended_scale": results[best_tag]["cv_rmse"],
    "trend_a": a,
    "trend_b": b,
    "pred_train_level": pred_train_level,
    "pred_test_level": pred_test_level,
    "y_train_level": y_train_level,
    "y_test_level": y_test_level,
    "metrics_train": metrics_train,
    "metrics_test": metrics_test,
    "residual_test": y_test_level - pred_test_level,
    "dates_test": test_dt["tanggal"].values,
}
with open(CK / "xgb_detrended_final.pkl", "wb") as f:
    pickle.dump(result, f)
print(f"\n[SAVED] {CK / 'xgb_detrended_final.pkl'}")

# --- Bandingkan dengan XGBoost Penuh original bila checkpoint tersedia ---
orig_path = CK / "xgb_full.pkl"
if orig_path.exists():
    with open(orig_path, "rb") as f:
        xgb_orig = pickle.load(f)

    print("\n" + "=" * 80)
    print("PERBANDINGAN: XGBOOST PENUH ORIGINAL vs DETRENDED")
    print("=" * 80)
    print(f"Original  — RMSE={xgb_orig['metrics_test']['RMSE']:.2f} | "
          f"MAPE={xgb_orig['metrics_test']['MAPE (%)']:.2f}% | "
          f"Range={xgb_orig['pred_test'].min():.0f}-{xgb_orig['pred_test'].max():.0f}")
    print(f"Detrended — RMSE={metrics_test['RMSE']:.2f} | "
          f"MAPE={metrics_test['MAPE (%)']:.2f}% | "
          f"Range={pred_test_level.min():.0f}-{pred_test_level.max():.0f}")

    summary = pd.DataFrame([
        {"Model": "XGBoost Penuh (Original)",
         "RMSE_Uji": xgb_orig["metrics_test"]["RMSE"],
         "MAE_Uji": xgb_orig["metrics_test"]["MAE"],
         "MAPE_Uji (%)": xgb_orig["metrics_test"]["MAPE (%)"],
         "Range_Pred_Uji": f"{xgb_orig['pred_test'].min():.0f}-{xgb_orig['pred_test'].max():.0f}"},
        {"Model": "XGBoost Penuh (Detrended)",
         "RMSE_Uji": metrics_test["RMSE"],
         "MAE_Uji": metrics_test["MAE"],
         "MAPE_Uji (%)": metrics_test["MAPE (%)"],
         "Range_Pred_Uji": f"{pred_test_level.min():.0f}-{pred_test_level.max():.0f}"},
    ])
    summary.to_csv(TB / "xgb_detrended_summary.csv", index=False)
    print(f"\n[SAVED] {TB / 'xgb_detrended_summary.csv'}")

    # --- Plot perbandingan ---
    test_dates = [pd.Timestamp(d) for d in test_dt["tanggal"].values]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(test_dates, y_test_level, "o-", color="#2d3748", lw=2, ms=5, label="Aktual", zorder=5)
    ax.plot(test_dates, xgb_orig["pred_test"], "s--", color="#e53e3e", lw=1.8, ms=5,
            label=f"Original (RMSE={xgb_orig['metrics_test']['RMSE']:.1f})", alpha=0.85)
    ax.plot(test_dates, pred_test_level, "^--", color="#38a169", lw=1.8, ms=5,
            label=f"Detrended (RMSE={metrics_test['RMSE']:.1f})", alpha=0.85)
    ax.axhline(MAX_TRAIN, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"Max data latih ({MAX_TRAIN:.0f} M)")
    ax.set_title("XGBoost Penuh: Original vs Detrended\n(100 trial Optuna, data uji)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Periode"); ax.set_ylabel("PKB (Miliar Rupiah)")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.tight_layout()
    plt.savefig(FG / "xgb_detrended_diagnostik.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {FG / 'xgb_detrended_diagnostik.png'}")
else:
    print(f"\n[INFO] Checkpoint original tidak ditemukan di {orig_path} — lewati perbandingan.")
    print("[INFO] Jalankan main() pada pipeline_v4_final.py terlebih dahulu jika ingin perbandingan otomatis.")

print("\n[SELESAI] Bagian A (Detrending + XGBoost) selesai dijalankan.")
