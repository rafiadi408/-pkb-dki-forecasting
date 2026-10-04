# Dokumentasi Teknis Metodologi

Dokumen ini menjelaskan keputusan desain utama dalam pipeline pemodelan, untuk memudahkan reproduksi dan ekstensi penelitian.

---

## 1. Kerangka Anti-Leakage

Pipeline ini dirancang untuk menghindari kebocoran informasi (*data leakage*) dalam tiga lapisan:

### Lapisan 1 — Fitur Turunan Target
Semua fitur yang diturunkan dari variabel target (`pkb`) dihitung dari deret yang sudah digeser satu periode (`y.shift(1)`):

```python
feat['pkb_lag_1']    = y.shift(1)           # BUKAN y langsung
feat['roll_mean_3']  = y.shift(1).rolling(3).mean()  # Rolling dari y_{t-1}
feat['ewma_3']       = y.shift(1).ewm(span=3).mean() # EWMA dari y_{t-1}
```

Ini memastikan fitur momentum dan autoregresif hanya memanfaatkan informasi yang benar-benar tersedia sebelum periode berjalan.

### Lapisan 2 — Variabel Eksogen (Desain Ex-Ante)

| Variabel | Lag | Justifikasi |
|---|---|---|
| BBNKB | t−1 | Dikompilasi bersamaan dengan PKB; baru tersedia setelah periode t berakhir |
| IKK | t−1 | Mengikuti jadwal publikasi sumber IKK yang digunakan dalam penelitian |
| Inflasi YoY | t−1 | Mengikuti jadwal publikasi sumber resmi yang digunakan |
| Dummy Pandemi | t | Tanggal PSBB diumumkan sebelum berlaku (kalender kebijakan) |
| Dummy Pemutihan | t | Kalender program pemutihan diketahui di muka |

### Lapisan 3 — Isolasi Data Uji
Semua keputusan pemodelan (hyperparameter, orde ARIMA, ablasi fitur) dibuat eksklusif di data latih via TSCV kausal. Data uji hanya disentuh **sekali** pada evaluasi akhir.

---

## 2. Prosedur Grid Search SARIMA/SARIMAX

**Ruang pencarian:** p,q ∈ {0,1,2,3}, P,Q ∈ {0,1,2}, d=1 (dikunci dari uji ADF+KPSS), D=1 (musiman) → 576 kombinasi total.

**Filter validitas:**
1. Konvergensi estimasi MLE (model yang gagal konvergen dibuang)
2. Solusi degenerate (log-likelihood ≈ 0 atau SE tidak terhingga)
3. Parsimoni: p+q+P+Q ≤ 6

**Seleksi dua tahap:**
1. **AIC** → shortlist 8 kandidat terbaik
2. **TSCV 5-fold out-of-sample RMSE** → keputusan final

Pemilihan berdasarkan CV-RMSE (bukan AIC) karena AIC mengukur goodness-of-fit in-sample sedangkan CV-RMSE mengukur kemampuan generalisasi yang relevan untuk tujuan peramalan.

---

## 3. Bayesian Optimization XGBoost (Nested TSCV)

**Arsitektur dua lapis:**

```
Outer loop: Optuna TPE, 100 trial
  ↓ setiap trial
  Inner loop: TimeSeriesSplit(n_splits=5)
    ↓ setiap fold
    Fit pada 85% fold latih + early stopping pada 15% terakhir
    Evaluasi RMSE pada fold validasi
  ↓
  Rata-rata RMSE 5 fold → objective Optuna
↓
Trial terbaik → refit pada seluruh data latih → model final
```

**Ruang pencarian hyperparameter:**

| Parameter | Rentang |
|---|---|
| `max_depth` | [2, 6] |
| `learning_rate` | [0.01, 0.3] log-uniform |
| `subsample` | [0.6, 1.0] |
| `colsample_bytree` | [0.6, 1.0] |
| `min_child_weight` | [1, 20] |
| `gamma` | [0, 5] |
| `reg_alpha` | [1e-8, 10] log-uniform |
| `reg_lambda` | [1e-8, 10] log-uniform |

---

## 4. Prosedur Ablasi

Semua ablasi mengikuti protokol yang sama:
1. Aturan keputusan ditetapkan di muka (threshold, bukan berdasarkan hasil)
2. Hyperparameter dikunci dari run sebelumnya jika ada
3. Keputusan berdasarkan CV-RMSE data latih saja

| Ablasi | Threshold | Keputusan |
|---|---|---|
| Fitur `tahun` | Pilih varian dengan CV-RMSE lebih rendah | `tanpa_tahun` menang di kedua XGBoost |
| Bentuk eksogen SARIMAX | Pilih varian dengan CV-RMSE lebih rendah | `level` menang (114,99 vs 127,65) |
| Blok lag-2 eksogen | Perbaikan ≥ 0,5% → pertahankan lag-2 | CV-RMSE tanpa lag-2 lebih rendah pada varian hyperparameter final → buang lag-2 |

---

## 5. Evaluasi Model

### Metrik Titik
- **RMSE** (Root Mean Squared Error) — sensitif terhadap outlier, skala miliar Rupiah
- **MAE** (Mean Absolute Error) — robust terhadap outlier
- **MAPE** (Mean Absolute Percentage Error) — interpretasi persentase, pembanding antar skala

### Bootstrap CI (95%)
Moving block bootstrap dengan block=3 dan B=5000 iterasi untuk menghasilkan confidence interval yang menghormati dependensi temporal pada residual deret waktu.

### Uji Diebold-Mariano
Menggunakan distribusi-t dengan df=n−1=26 (bukan distribusi normal), tepat untuk sampel kecil (Harvey et al., 1997). Loss function: kuadrat galat (squared error).

---

## 6. Analisis CCF Prewhitened

CCF biasa (Pearson) rentan korelasi spurious akibat tren deterministik bersama. Prosedur prewhitening:

1. Fit ARIMA pada deret PKB (kandidat: (1,1,1), (2,1,1), (3,1,1), (2,1,2)) → pilih AIC terendah dengan residual white noise (Ljung-Box p > 0)
2. Terapkan filter ARIMA yang sama pada deret eksogen
3. Hitung korelasi silang antara residual yang sudah di-filter

**Hasil:** Hanya BBNKB lag-1 (r=−0,202) dan lag-12 (r=+0,222) yang signifikan (band ±0,192, α=5%). IKK dan Inflasi tidak signifikan di lag manapun setelah prewhitening.

---

## 7. SHAP Berkelompok

Karena multikolinearitas tinggi di antara fitur turunan target (VIF rank-deficient pada pkb_lag_1, pkb_lag_2, pkb_lag_3, roll_mean_3, diff_1), SHAP per-fitur individual tidak stabil. Solusi: agregasi per kelompok fitur dengan confidence interval bootstrap (B=500).

**Kelompok fitur:**

| Kelompok | Fitur | Share SHAP |
|---|---|---|
| AR & Momentum | pkb_lag_*, roll_*, ewma_*, diff_* | 54,6% [51,8; 56,8] |
| Kalender | bulan_sin, bulan_cos, kuartal | 14,7% [13,7; 15,9] |
| Eksogen Ex-Ante | bbnkb_*, ikk_*, inflasi_*, covid19, pemutihan | 30,7% [28,7; 33,3] |

---

## 8. Hyperparameter Final

### XGBoost Baseline
```json
{
  "max_depth": 2,
  "learning_rate": 0.2363,
  "subsample": 0.8045,
  "colsample_bytree": 0.7812,
  "min_child_weight": 8,
  "gamma": 0.6697,
  "reg_alpha": 1.8334,
  "reg_lambda": 0.0081,
  "n_estimators": (dari best_iteration pipeline)
}
```

### XGBoost Penuh
```json
{
  "max_depth": 2,
  "learning_rate": 0.2428,
  "subsample": 0.9808,
  "colsample_bytree": 0.9516,
  "min_child_weight": 7,
  "gamma": 1.3928,
  "reg_alpha": 2.5589,
  "reg_lambda": 0.0015,
  "n_estimators": (dari best_iteration pipeline)
}
```

### SARIMA Final
- Orde: **(0,1,1)(0,1,2)₁₂**
- Ljung-Box(12) p = 0,626 ✓ | Jarque-Bera p < 0,001 | Breakvar p = 0,323 ✓

### SARIMAX Final
- Orde: **(1,1,2)(0,1,1)₁₂**, eksogen level: BBNKB_lag1, IKK_lag1, Inflasi_lag1, covid19, pemutihan
- Ljung-Box(12) p = 0,754 ✓ | Jarque-Bera p < 0,001 | Breakvar p = 0,136 ✓

---

## 9. Eksperimen Remediasi (Analisis Sensitivitas)

Dua eksperimen tambahan dilakukan atas masukan dosen pembimbing, bersifat post-hoc konfirmatori:

### A. XGBoost Detrended
Memisahkan komponen tren linear (OLS, dikontrol dummy pandemi) sebelum pemodelan XGBoost pada residual. Prediksi akhir = prediksi residual + tren ekstrapolasi.
- **Hasil:** RMSE uji 87,25 M (vs 99,80 M original), MAE 68,98 M, MAPE 8,70%; rentang prediksi uji 736–996 M (vs 587–896 M pada model original).

### B. SARIMAX Eksogen Terdiferensiasi
Menggunakan diferensiasi pertama untuk BBNKB dan IKK (non-stasioner di level) alih-alih nilai level.
- **Hasil:** RMSE uji 93,05 M (vs 117,48 M original), MAE 77,78 M, MAPE 9,81%; rentang prediksi 694–994 M (vs 663–1.087 M pada model original).

**Catatan metodologis:** Kedua eksperimen dievaluasi setelah data uji tersentuh model utama, sehingga statusnya adalah analisis konfirmatori/deskriptif dan **tidak digunakan** untuk memilih ulang spesifikasi utama.
