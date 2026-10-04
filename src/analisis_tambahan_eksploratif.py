"""
ANALISIS TAMBAHAN EKSPLORATIF
================================
Kumpulan script pendukung yang digunakan untuk melengkapi Bab IV:

  1. Uji Stasioneritas (ADF + KPSS) lengkap dengan nilai statistik & nilai
     kritis — untuk Tabel uji stasioneritas.
  2. ACF & PACF deret PKB (level dan diff-1) — untuk identifikasi lag
     autoregresif dan orde SARIMA sebelum grid search.
  3. Cross-Correlation Function (CCF) Pearson biasa, PKB vs BBNKB/IKK/
     Inflasi YoY, lag 0-12 — analisis awal sebelum prewhitening.
  4. VIF seluruh 21 fitur pool XGBoost Penuh (versi non-prewhitened,
     sebagai pembanding tabel VIF pada remediasi_v2).
  5. Analisis Zona Ekstrapolasi — klasifikasi observasi data uji ke dalam
     zona normal / sparse / hard-extrapolation relatif terhadap rentang
     nilai pada data latih, beserta visualisasi residual.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

OUT_FIG = Path("outputs_final/figures")
OUT_TAB = Path("outputs_final/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("dataset_skripsi.csv", parse_dates=["tanggal"])
df = df.sort_values("tanggal").reset_index(drop=True)
df["pkb"] = df["pkb"] / 1e9
df["bbnkb"] = df["bbnkb"] / 1e9

train = df[(df["tanggal"] >= "2015-01-01") & (df["tanggal"] <= "2023-09-01")].copy().reset_index(drop=True)
test = df[df["tanggal"] >= "2023-10-01"].head(27).copy().reset_index(drop=True)

y = train["pkb"]
y_diff = y.diff().dropna()
MAX_TRAIN = train["pkb"].max()  # 983.24 (Des 2022)
SPARSE_LOW = 900.0

# ==============================================================================
# 1. UJI STASIONERITAS (ADF + KPSS) — level, diff(1), diff(12), diff(1)+diff(12)
# ==============================================================================
print("=" * 80); print("1. UJI STASIONERITAS (ADF + KPSS)"); print("=" * 80)

variants = {
    "Level": y.values,
    "Diff(1)": np.diff(y.values, n=1),
    "Diff(12)": np.diff(y.values, n=12),
    "Diff(1)+Diff(12)": np.diff(np.diff(y.values, n=1), n=12),
}
rows = []
for name, series in variants.items():
    adf_res = adfuller(series, autolag="AIC")
    adf_stat, adf_p, adf_cv5 = adf_res[0], adf_res[1], adf_res[4]["5%"]
    kpss_res = kpss(series, regression="c", nlags="auto")
    kpss_stat, kpss_p, kpss_cv5 = kpss_res[0], kpss_res[1], kpss_res[3]["5%"]
    adf_stasioner = adf_p < 0.05
    kpss_stasioner = kpss_p > 0.05
    kesimpulan = "Stasioner" if (adf_stasioner and kpss_stasioner) else "Tidak Stasioner"
    rows.append({
        "Varian Data": name, "Statistik ADF": round(adf_stat, 4),
        "Nilai Kritis ADF 5%": round(adf_cv5, 4), "p-value ADF": round(adf_p, 4),
        "Statistik KPSS": round(kpss_stat, 4), "Nilai Kritis KPSS 5%": round(kpss_cv5, 4),
        "p-value KPSS": ">0.10" if kpss_p >= 0.10 else ("<0.01" if kpss_p <= 0.01 else round(kpss_p, 4)),
        "Kesimpulan": kesimpulan,
    })
    print(f"{name}: ADF stat={adf_stat:.4f}, p={adf_p:.4f} | KPSS stat={kpss_stat:.4f}, p={kpss_p:.4f} | {kesimpulan}")

pd.DataFrame(rows).to_csv(OUT_TAB / "uji_stasioneritas.csv", index=False)
print(f"[SAVED] {OUT_TAB / 'uji_stasioneritas.csv'}")


# ==============================================================================
# 2. ACF & PACF — Identifikasi Lag Autoregresif (level & diff-1)
# ==============================================================================
print("\n" + "=" * 80); print("2. ACF & PACF — PKB Level dan Diff(1)"); print("=" * 80)

n = len(y_diff)
ci95 = 1.96 / np.sqrt(n)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
plot_acf(y, lags=36, ax=axes[0, 0], title="ACF — PKB (level)")
plot_pacf(y, lags=36, ax=axes[0, 1], title="PACF — PKB (level)", method="ywm")
plot_acf(y_diff, lags=36, ax=axes[1, 0], title="ACF — PKB diff(1)")
plot_pacf(y_diff, lags=36, ax=axes[1, 1], title="PACF — PKB diff(1)", method="ywm")
plt.tight_layout()
plt.savefig(OUT_FIG / "acf_pacf_pkb.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[PLOT] {OUT_FIG / 'acf_pacf_pkb.png'}")

# Lag signifikan PACF diff(1) — untuk justifikasi lag fitur AR XGBoost
pv = pacf(y_diff, nlags=24, method="ywm")
sig_lags = [i for i in range(1, 25) if abs(pv[i]) > ci95]
print("PACF signifikan di lag:", sig_lags)


# ==============================================================================
# 3. CCF PEARSON (non-prewhitened) — PKB vs BBNKB/IKK/Inflasi, lag 0-12
# ==============================================================================
print("\n" + "=" * 80); print("3. CCF Pearson — PKB vs Eksogen (lag 0-12)"); print("=" * 80)

eksogen_list = [("bbnkb", "BBNKB"), ("ikk", "IKK"), ("inflasi_yoy", "Inflasi YoY")]
n_train = len(y)
ci_train = 1.96 / np.sqrt(n_train)
lag_range = list(range(0, 13))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (col, label) in zip(axes, eksogen_list):
    x = train[col].values
    y_vals = y.values
    ccf_vals = []
    for lag in lag_range:
        c = np.corrcoef(y_vals, x)[0, 1] if lag == 0 else np.corrcoef(y_vals[lag:], x[:-lag])[0, 1]
        ccf_vals.append(c)
    colors = ["#e53e3e" if abs(v) > ci_train else "#90cdf4" for v in ccf_vals]
    ax.bar(lag_range, ccf_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(ci_train, color="gray", ls="--", lw=1)
    ax.axhline(-ci_train, color="gray", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"CCF: PKB vs {label}\n(merah = signifikan \u03b1=5%)", fontsize=10)
    ax.set_xlabel("Lag (bulan)"); ax.set_ylabel("Korelasi Pearson")
    ax.set_xticks(lag_range); ax.grid(alpha=0.2)
    print(f"\nCCF PKB vs {label}:")
    for lag, val in zip(lag_range, ccf_vals):
        sig = "** SIGNIFIKAN" if abs(val) > ci_train else ""
        print(f"  lag-{lag:2d}: {val:+.3f}  {sig}")
plt.suptitle("Cross-Correlation Function (CCF): PKB vs Variabel Eksogen", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_FIG / "ccf_lag_selection.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[PLOT] {OUT_FIG / 'ccf_lag_selection.png'}")


# ==============================================================================
# 4. VIF — Pool 21 Fitur XGBoost Penuh (rekonstruksi manual dari raw features)
# ==============================================================================
print("\n" + "=" * 80); print("4. VIF — Pool Fitur XGBoost Penuh"); print("=" * 80)

feat = pd.DataFrame()
feat["pkb_lag_1"] = y.shift(1); feat["pkb_lag_2"] = y.shift(2); feat["pkb_lag_3"] = y.shift(3)
feat["pkb_lag_6"] = y.shift(6); feat["pkb_lag_12"] = y.shift(12)
feat["roll_mean_3"] = y.shift(1).rolling(3).mean(); feat["roll_mean_6"] = y.shift(1).rolling(6).mean()
feat["roll_mean_12"] = y.shift(1).rolling(12).mean()
feat["roll_std_3"] = y.shift(1).rolling(3).std(); feat["roll_std_6"] = y.shift(1).rolling(6).std()
feat["ewma_3"] = y.shift(1).ewm(span=3).mean(); feat["ewma_6"] = y.shift(1).ewm(span=6).mean()
feat["ewma_12"] = y.shift(1).ewm(span=12).mean()
feat["diff_1"] = y.shift(1).diff(1); feat["diff_12"] = y.shift(1).diff(12)
feat["bulan_sin"] = np.sin(2 * np.pi * train["tanggal"].dt.month / 12)
feat["bulan_cos"] = np.cos(2 * np.pi * train["tanggal"].dt.month / 12)
feat["kuartal"] = train["tanggal"].dt.quarter
feat["bbnkb_lag_1"] = train["bbnkb"].shift(1); feat["bbnkb_lag_2"] = train["bbnkb"].shift(2)
feat["ikk_lag_1"] = train["ikk"].shift(1); feat["ikk_lag_2"] = train["ikk"].shift(2)
feat["inflasi_lag_1"] = train["inflasi_yoy"].shift(1); feat["inflasi_lag_2"] = train["inflasi_yoy"].shift(2)
feat["covid19"] = train["covid19"]; feat["pemutihan"] = train["pemutihan"]
feat = feat.dropna().reset_index(drop=True)

vif_rows = []
X = feat.values
for i, col in enumerate(feat.columns):
    try:
        v = variance_inflation_factor(X, i)
    except Exception:
        v = np.nan
    vif_rows.append({"Fitur": col, "VIF": round(v, 2)})
vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif_df.to_csv(OUT_TAB / "vif_fitur_xgboost.csv", index=False)
print(vif_df.to_string(index=False))
print(f"[SAVED] {OUT_TAB / 'vif_fitur_xgboost.csv'}")


# ==============================================================================
# 5. ANALISIS ZONA EKSTRAPOLASI — Residual XGBoost Penuh vs Rentang Data Latih
# ==============================================================================
print("\n" + "=" * 80); print("5. Analisis Zona Ekstrapolasi (XGBoost Penuh)"); print("=" * 80)

with open("outputs_final/checkpoints/xgb_full.pkl", "rb") as f:
    xgb_full = pickle.load(f)

pred_test = xgb_full["pred_test"]
residual_test = xgb_full["residual_test"]
actual_test = pred_test + residual_test
test_dates = [pd.Timestamp(t) for t in test["tanggal"].values]

def classify(a):
    if a > MAX_TRAIN:
        return "hard"
    elif a >= SPARSE_LOW:
        return "sparse"
    return "normal"

classes = [classify(a) for a in actual_test]
color_map = {"normal": "#3182ce", "sparse": "#dd6b20", "hard": "#e53e3e"}
size_map = {"normal": 70, "sparse": 110, "hard": 130}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax1 = axes[0]
for cls, label in [("normal", "Dalam rentang latih"), ("sparse", "Zona sparse (900-983 M)"),
                    ("hard", "Di luar rentang latih (>983 M)")]:
    mask = [c == cls for c in classes]
    ax1.scatter(actual_test[mask], residual_test[mask], c=color_map[cls], s=size_map[cls],
                label=label, edgecolors="white", linewidths=0.8)
ax1.axhline(0, color="black", lw=1.2)
ax1.axvline(MAX_TRAIN, color="#e53e3e", lw=1.5, ls="--", alpha=0.9)
ax1.axvspan(SPARSE_LOW, MAX_TRAIN, alpha=0.07, color="#dd6b20")
ax1.axvline(SPARSE_LOW, color="#dd6b20", lw=1.2, ls=":", alpha=0.8)
ax1.set_xlabel("Nilai Aktual PKB (Miliar Rupiah)"); ax1.set_ylabel("Residual = Aktual - Prediksi")
ax1.set_title("Residual vs Nilai Aktual — XGBoost Penuh (Data Uji)", fontsize=10.5, fontweight="bold")
ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.2)

ax2 = axes[1]
ax2.plot(test_dates, actual_test, "o-", color="#2d3748", lw=2, ms=5, label="Aktual", zorder=5)
ax2.plot(test_dates, pred_test, "s--", color="#3182ce", lw=1.8, ms=5, label="Prediksi", alpha=0.85)
ax2.axhline(MAX_TRAIN, color="#e53e3e", lw=1.5, ls="--", alpha=0.85)
ax2.axhline(SPARSE_LOW, color="#dd6b20", lw=1.2, ls=":", alpha=0.8)
ax2.set_xlabel("Periode"); ax2.set_ylabel("PKB (Miliar Rupiah)")
ax2.set_title("Aktual vs Prediksi — Periode Uji", fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.suptitle("Analisis Zona Ekstrapolasi XGBoost Penuh", fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_FIG / "xgb_penuh_extrapolation_analysis.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[PLOT] {OUT_FIG / 'xgb_penuh_extrapolation_analysis.png'}")
print("\n[SELESAI] Seluruh analisis tambahan eksploratif selesai dijalankan.")
