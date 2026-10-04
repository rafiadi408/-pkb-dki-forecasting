"""
REMEDIASI BAGIAN B — SARIMAX dengan Eksogen Terdiferensiasi
=============================================================
Mengatasi over-extrapolation SARIMAX dengan memakai bentuk terdiferensiasi
untuk eksogen yang non-stasioner di level (BBNKB, IKK), alih-alih bentuk
level yang dipilih pada pipeline utama berdasarkan CV-RMSE data latih.

Rasional: bentuk level membuat koefisien regresi bertumpu pada hubungan
JANGKA PANJANG yang tidak stabil ketika eksogen terus bertumbuh melampaui
pola historisnya — menyebabkan prediksi SARIMAX "melambung" jauh di atas
nilai aktual pada periode uji. Bentuk terdiferensiasi mengikat model pada
LAJU PERUBAHAN eksogen (lebih stasioner, lebih terbatas), sehingga
seharusnya lebih tahan terhadap over-extrapolation meski CV-RMSE latihnya
sedikit lebih buruk.

PRASYARAT:
  - pipeline_v4_final.py dan dataset_skripsi.csv berada di direktori yang sama.
  - Disarankan menjalankan main() pipeline utama terlebih dahulu agar
    checkpoint grid search/ablasi eksogen (jika ada) bisa dipakai ulang
    secara otomatis oleh mekanisme ckpt_load/ckpt_save bawaan pipeline
    (mempercepat proses, opsional — script ini tetap berjalan dari nol
    bila checkpoint belum ada).

OUTPUT:
  - outputs_final/checkpoints/sarimax_diff_final.pkl
  - outputs_final/tables/sarimax_diff_summary.csv
  - outputs_final/figures/sarimax_diff_diagnostik.png
"""

import pickle
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

try:
    load_and_engineer_features, split_train_test, get_sarimax_exog_cols_diff
    grid_search_sarimax_order, cv_validate_sarima, compute_metrics
    SARIMAX, Config, CFG
except NameError:
    import pipeline_v4_final as P
    load_and_engineer_features = P.load_and_engineer_features
    split_train_test = P.split_train_test
    get_sarimax_exog_cols_diff = P.get_sarimax_exog_cols_diff
    grid_search_sarimax_order = P.grid_search_sarimax_order
    cv_validate_sarima = P.cv_validate_sarima
    compute_metrics = P.compute_metrics
    SARIMAX = P.SARIMAX
    Config = P.Config
    CFG = P.CFG

BASE = CFG.output_dir
CK = Path(BASE) / "checkpoints"
TB = Path(BASE) / "tables"
FG = Path(BASE) / "figures"
for p in (CK, TB, FG):
    p.mkdir(parents=True, exist_ok=True)

MAX_TRAIN = 983.24


# ==============================================================================
# LANGKAH 1 — Muat Data & Bentuk Eksogen Terdiferensiasi
# ==============================================================================
print("=" * 80)
print("LANGKAH 1: Muat Data & Siapkan Eksogen Terdiferensiasi")
print("=" * 80)

df = load_and_engineer_features(CFG)
train, test = split_train_test(df, CFG)

y_train = train[CFG.target_col]
y_test = test[CFG.target_col]

exog_cols = get_sarimax_exog_cols_diff()  # ['bbnkb_lag_1_diff','ikk_lag_1_diff','inflasi_yoy_lag_1','covid19','pemutihan']
print(f"[INFO] Kolom eksogen (terdiferensiasi): {exog_cols}")

exog_train = train[exog_cols]
exog_test = test[exog_cols]


# ==============================================================================
# LANGKAH 2 — Grid Search + TSCV untuk Orde SARIMAX (memakai checkpoint bila ada)
# ==============================================================================
print("\n" + "=" * 80)
print("LANGKAH 2: Grid Search + TSCV — Orde SARIMAX (Eksogen Terdiferensiasi)")
print("=" * 80)

candidates = grid_search_sarimax_order(
    y_train, CFG, exog=exog_train,
    label="SARIMAX_terdiferensiasi",
    ckpt_key="grid_sarimax_terdiferensiasi",
)
best_order6, best_cv_rmse = cv_validate_sarima(
    y_train, candidates, CFG, exog=exog_train,
    label="SARIMAX_terdiferensiasi",
)
print(f"\n[TERPILIH] Orde: {best_order6[:3]} x {best_order6[3:]}  |  CV-RMSE={best_cv_rmse:.2f}")


# ==============================================================================
# LANGKAH 3 — Fit Model Final pada Seluruh Data Latih
# ==============================================================================
print("\n" + "=" * 80)
print("LANGKAH 3: Fit Model Final & Diagnostik Residual")
print("=" * 80)

mod = SARIMAX(
    y_train, exog=exog_train, order=best_order6[:3],
    seasonal_order=(*best_order6[3:], CFG.seasonal_period),
    enforce_stationarity=True, enforce_invertibility=True,
)
res = mod.fit(disp=False, maxiter=200)

lb = acorr_ljungbox(res.resid, lags=[12], return_df=True)
lb_p = float(lb["lb_pvalue"].iloc[0])
jb_p = float(jarque_bera(res.resid).pvalue)
het = res.test_heteroskedasticity(method="breakvar")[0]
het_p = float(het[0, 1] if het.ndim == 2 else het[1])

print(f"[DIAG] Ljung-Box(12) p={lb_p:.4f} | Jarque-Bera p={jb_p:.4f} | Breakvar p={het_p:.4f}")


# ==============================================================================
# LANGKAH 4 — Forecast ke Data Uji & Evaluasi
# ==============================================================================
print("\n" + "=" * 80)
print("LANGKAH 4: Forecast & Evaluasi Data Uji")
print("=" * 80)

fc = res.get_forecast(steps=len(y_test), exog=exog_test)
pred_test = fc.predicted_mean.values

metrics_test = compute_metrics(y_test.values, pred_test)
print(f"[METRIK] RMSE={metrics_test['RMSE']:.2f} | MAE={metrics_test['MAE']:.2f} | "
      f"MAPE={metrics_test['MAPE (%)']:.2f}%")
print(f"[INFO] Range prediksi uji: {pred_test.min():.2f} - {pred_test.max():.2f}")
print(f"[INFO] Range aktual uji  : {y_test.min():.2f} - {y_test.max():.2f}")
print(f"[INFO] Max data latih    : {MAX_TRAIN:.2f}")


# ==============================================================================
# LANGKAH 5 — Simpan Checkpoint & Bandingkan dengan Model Original (Level)
# ==============================================================================
result = {
    "name": "SARIMAX (Eksogen Terdiferensiasi)",
    "order": best_order6[:3],
    "seasonal_order": best_order6[3:],
    "exog_cols": exog_cols,
    "cv_rmse": best_cv_rmse,
    "pred_test": pred_test,
    "residual_test": y_test.values - pred_test,
    "metrics_test": metrics_test,
    "diagnostics": {"ljung_box_p": lb_p, "jarque_bera_p": jb_p, "breakvar_p": het_p},
    "dates_test": test["tanggal"].values,
}
with open(CK / "sarimax_diff_final.pkl", "wb") as f:
    pickle.dump(result, f)
print(f"\n[SAVED] {CK / 'sarimax_diff_final.pkl'}")

# --- Bandingkan dengan SARIMAX level (original) bila checkpoint tersedia ---
orig_path = CK / "sarimax.pkl"
if orig_path.exists():
    with open(orig_path, "rb") as f:
        sarimax_level = pickle.load(f)

    print("\n" + "=" * 80)
    print("PERBANDINGAN: SARIMAX LEVEL (ORIGINAL) vs TERDIFERENSIASI")
    print("=" * 80)
    print(f"Level          — RMSE={sarimax_level['metrics_test']['RMSE']:.2f} | "
          f"MAPE={sarimax_level['metrics_test']['MAPE (%)']:.2f}% | "
          f"Range={sarimax_level['pred_test'].min():.0f}-{sarimax_level['pred_test'].max():.0f}")
    print(f"Terdiferensiasi — RMSE={metrics_test['RMSE']:.2f} | "
          f"MAPE={metrics_test['MAPE (%)']:.2f}% | "
          f"Range={pred_test.min():.0f}-{pred_test.max():.0f}")

    summary = pd.DataFrame([
        {"Model": "SARIMAX (Eksogen Level, Original)",
         "Orde": f"{sarimax_level['order']}x{sarimax_level['seasonal_order']}",
         "RMSE_Uji": sarimax_level["metrics_test"]["RMSE"],
         "MAE_Uji": sarimax_level["metrics_test"]["MAE"],
         "MAPE_Uji (%)": sarimax_level["metrics_test"]["MAPE (%)"],
         "Range_Pred_Uji": f"{sarimax_level['pred_test'].min():.0f}-{sarimax_level['pred_test'].max():.0f}"},
        {"Model": "SARIMAX (Eksogen Terdiferensiasi)",
         "Orde": f"{best_order6[:3]}x{best_order6[3:]}",
         "RMSE_Uji": metrics_test["RMSE"],
         "MAE_Uji": metrics_test["MAE"],
         "MAPE_Uji (%)": metrics_test["MAPE (%)"],
         "Range_Pred_Uji": f"{pred_test.min():.0f}-{pred_test.max():.0f}"},
    ])
    summary.to_csv(TB / "sarimax_diff_summary.csv", index=False)
    print(f"\n[SAVED] {TB / 'sarimax_diff_summary.csv'}")

    # --- Plot perbandingan ---
    test_dates = [pd.Timestamp(d) for d in test["tanggal"].values]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(test_dates, y_test.values, "o-", color="#2d3748", lw=2, ms=5, label="Aktual", zorder=5)
    ax.plot(test_dates, sarimax_level["pred_test"], "s--", color="#e53e3e", lw=1.8, ms=5,
            label=f"Level (RMSE={sarimax_level['metrics_test']['RMSE']:.1f})", alpha=0.85)
    ax.plot(test_dates, pred_test, "^--", color="#38a169", lw=1.8, ms=5,
            label=f"Terdiferensiasi (RMSE={metrics_test['RMSE']:.1f})", alpha=0.85)
    ax.axhline(MAX_TRAIN, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"Max data latih ({MAX_TRAIN:.0f} M)")
    ax.set_title("SARIMAX: Eksogen Level vs Terdiferensiasi\n(data uji)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Periode"); ax.set_ylabel("PKB (Miliar Rupiah)")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.tight_layout()
    plt.savefig(FG / "sarimax_diff_diagnostik.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {FG / 'sarimax_diff_diagnostik.png'}")
else:
    print(f"\n[INFO] Checkpoint original tidak ditemukan di {orig_path} — lewati perbandingan.")
    print("[INFO] Jalankan main() pada pipeline_v4_final.py terlebih dahulu jika ingin perbandingan otomatis.")

print("\n[SELESAI] Bagian B (SARIMAX Terdiferensiasi) selesai dijalankan.")
