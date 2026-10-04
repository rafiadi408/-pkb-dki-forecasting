# Peramalan Penerimaan PKB DKI Jakarta

**Evaluasi Performa Model XGBoost dan SARIMAX dalam Peramalan Penerimaan Pajak Kendaraan Bermotor (PKB) di Provinsi DKI Jakarta dengan Integrasi Variabel Eksogen**

> Skripsi — Muhammad Rafi Adipratama (NPM 4131220149)  
> Politeknik Keuangan Negara STAN, 2026

---

## Ringkasan Penelitian

Penelitian ini membandingkan dua keluarga model peramalan deret waktu — **SARIMA/SARIMAX** (statistik) dan **XGBoost** (machine learning) — dalam meramalkan penerimaan PKB bulanan Provinsi DKI Jakarta pada periode Oktober 2023–Desember 2025 (n=27 observasi uji). Fokus penelitian adalah mengevaluasi apakah integrasi variabel eksogen yang tersedia secara **ex-ante** memberikan nilai tambah prediktif.

### Hasil Utama

| Model | RMSE Uji (M Rp) | MAPE Uji (%) | CI 95% MAPE |
|---|---:|---:|---|
| Ensemble Berbobot SARIMAX+XGBoost Penuh | **79,04** | **6,94** | [4,74; 9,18] |
| SARIMA(0,1,1)(0,1,2)₁₂ | 79,51 | 7,93 | [5,93; 10,04] |
| Ridge Regression | 81,31 | 7,01 | [5,06; 9,12] |
| XGBoost Penuh | 99,80 | 9,46 | [7,04; 11,93] |
| XGBoost Baseline | 111,93 | 9,51 | [6,69; 12,64] |
| SARIMAX(1,1,2)(0,1,1)₁₂ | 117,48 | 12,54 | [9,10; 16,86] |
| Naive (Persistence) | 138,38 | 12,94 | [9,09; 17,61] |

**Catatan:** Ensemble dicantumkan sebagai analisis eksploratif. MAPE 6,94% merupakan error point-estimate terendah pada tabel, tetapi hasil uji Diebold-Mariano tidak menunjukkan perbedaan yang signifikan secara statistik terhadap model pembanding utama.

### Temuan Kunci

1. **Kontribusi variabel eksogen bersifat asimetris.** Pada keluarga linear, SARIMAX dengan variabel eksogen ex-ante menghasilkan MAPE 12,54%, lebih tinggi daripada SARIMA 7,93%. Pada keluarga machine learning, XGBoost Penuh menghasilkan MAPE 9,46%, sedikit lebih rendah daripada XGBoost Baseline 9,51%.

2. **Sinyal temporal eksogen terbatas.** CCF pre-whitened menunjukkan sinyal signifikan pada BBNKB lag-1 (r=−0,202) dan lag-12 (r=+0,222), sedangkan IKK dan inflasi tidak menunjukkan sinyal signifikan pada lag yang diuji. Temuan CCF ini merupakan bukti hubungan temporal/statistik, bukan bukti kausalitas.

3. **XGBoost menghadapi keterbatasan ekstrapolasi.** Galat terbesar terkonsentrasi pada periode ketika penerimaan PKB berada di luar atau mendekati batas dukungan data pelatihan. Model berbasis pohon tidak melakukan ekstrapolasi linear di luar pola nilai yang dipelajari dari data latih.

4. **Model linear tetap kompetitif.** SARIMA dan Ridge Regression menghasilkan performa yang secara statistik tidak berbeda signifikan pada data uji (uji Diebold-Mariano SARIMA vs Ridge: p=0,862). Dengan demikian, model yang lebih kompleks tidak otomatis memberikan keunggulan statistik pada sampel uji yang relatif kecil.

5. **Atribusi SHAP menunjukkan peran eksogen yang tidak dominan tetapi material.** Pada XGBoost Penuh, kelompok Eksogen Ex-Ante menyumbang 30,7% dari total atribusi absolut SHAP, dibandingkan 54,6% untuk AR & Momentum dan 14,7% untuk fitur Kalender.

---

## Ketersediaan Data

Dataset bulanan PKB dan BBNKB yang digunakan dalam penelitian diperoleh melalui permohonan resmi kepada Bapenda Provinsi DKI Jakarta untuk kepentingan penelitian akademik. Berdasarkan konfirmasi tertulis dari Bapenda, data tersebut **tidak diperbolehkan untuk disebarkan secara publik**.

Karena itu, data mentah tidak disertakan dalam repository ini. Peneliti yang memerlukan data perlu mengajukan permohonan melalui mekanisme resmi kepada pemilik data.

File hasil yang disertakan dalam repository dibatasi pada dokumentasi, kode, visualisasi agregat tertentu, dan tabel hasil analisis yang tidak memuat observasi mentah.

## Struktur Repository

```text
pkb-dki-forecasting/
├── data/
│   └── README.md
├── src/
│   ├── pipeline_v4_final.py
│   ├── analisis_tambahan_eksploratif.py
│   ├── remediasi_v2_ccf_prewhitened.py
│   ├── remediasi_A_xgboost_detrended.py
│   └── remediasi_B_sarimax_diff.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── docs/
│   └── METODOLOGI.md
├── notebooks/
│   └── eksplorasi_awal.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Folder `outputs/checkpoints/` **tidak disertakan dalam paket publik** karena berisi artefak intermediate `.pkl` yang dapat diregenerasi oleh pipeline. Saat pipeline dijalankan, checkpoint akan dibuat secara lokal di `outputs_final/checkpoints/`.

---

## Cara Menjalankan

### 1. Persiapan Lingkungan

```bash
git clone https://github.com/<username>/pkb-dki-forecasting.git
cd pkb-dki-forecasting
pip install -r requirements.txt
```

Pipeline utama dikembangkan dan diuji pada Python 3.x. Untuk reproduksi yang paling konsisten, gunakan lingkungan Python yang sama dengan lingkungan pengembangan penelitian.

### 2. Menjalankan Pipeline Utama

```bash
python src/pipeline_v4_final.py
```

Pipeline utama menyimpan hasil komputasi di folder lokal `outputs_final/`, termasuk checkpoint, tabel, gambar, dan hasil pemodelan. Folder `outputs_final/` diabaikan oleh Git melalui `.gitignore`.

Pipeline mendukung *checkpointing*. Jika proses terhenti, pipeline dapat dijalankan kembali untuk melanjutkan tahap yang hasilnya sudah tersimpan.

> Estimasi waktu komputasi sekitar 2–4 jam pada CPU standar, bergantung pada lingkungan dan beban pencarian model.

### 3. Menjalankan Analisis Tambahan

Analisis berikut dijalankan setelah pipeline utama selesai karena membaca checkpoint dan hasil intermediate dari `outputs_final/`:

```bash
python src/analisis_tambahan_eksploratif.py
python src/remediasi_v2_ccf_prewhitened.py
```

### 4. Eksperimen Remediasi/Sensitivitas

```bash
python src/remediasi_A_xgboost_detrended.py
python src/remediasi_B_sarimax_diff.py
```

Kedua eksperimen ini bersifat analisis sensitivitas/konfirmatori dan tidak digunakan untuk memilih ulang spesifikasi utama setelah data uji dievaluasi.

### 5. Memuat Model XGBoost Tersimpan

```python
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("outputs/models/xgb_penuh.json")

baseline = xgb.XGBRegressor()
baseline.load_model("outputs/models/xgb_baseline.json")
```

---

## Data

**Dataset mentah:** tidak disertakan dalam repository publik.

| Kolom | Satuan | Sumber |
|---|---|---|
| `tanggal` | YYYY-MM-DD | — |
| `pkb` | Rupiah | Bapenda DKI Jakarta |
| `bbnkb` | Rupiah | Bapenda DKI Jakarta |
| `ikk` | Indeks | Bank Indonesia |
| `inflasi_yoy` | % | Badan Pusat Statistik |
| `covid19` | 0/1 dummy | Pemerintah Pusat / kebijakan pandemi |
| `pemutihan` | 0/1 dummy | Pemerintah Provinsi DKI Jakarta |

**Cakupan:** Januari 2014 – Desember 2025 (144 observasi)  
**Sampel pemodelan:** Januari 2015 – Desember 2025 (132 observasi; 12 observasi 2014 digunakan sebagai *burn-in* untuk fitur lag-12)  
**Split:** Latih Januari 2015–September 2023 (105 observasi) | Uji Oktober 2023–Desember 2025 (27 observasi)

> **Catatan distribusi data:** Data PKB dan BBNKB bulanan diperoleh melalui permohonan resmi kepada Bapenda Provinsi DKI Jakarta untuk kepentingan penelitian. Berdasarkan konfirmasi tertulis dari Bapenda, data yang diberikan hanya untuk kepentingan penelitian dan **tidak diperbolehkan disebarkan secara publik**. Oleh karena itu, dataset mentah tidak disertakan dalam repository ini. Repository hanya menyediakan kode, metodologi, dokumentasi, serta hasil agregat penelitian yang tidak memuat observasi mentah.

---

## Outputs Tersedia

### Tabel (`outputs/tables/`)

| File | Isi |
|---|---|
| `ringkasan_semua_model.csv` | Metrik lengkap model, termasuk RMSE, MAE, MAPE, dan CI bootstrap |
| `metadata_final.json` | Hyperparameter terpilih dan diagnostik model |
| `dm_matrix_pvalue_t.csv` | Matriks p-value uji Diebold-Mariano distribusi-t |
| `dm_matrix_pvalue_normal.csv` | Matriks p-value uji DM distribusi normal sebagai pembanding |
| `dm_matrix_statistic.csv` | Matriks statistik DM |
| `tabel_ccf_prewhitened.csv` | Hasil CCF pre-whitened PKB vs variabel eksogen pada lag 1–12 |
| `tabel_vif_pool.csv` | VIF fitur pada ruang fitur XGBoost |
| `tabel_ablasi_lag2.csv` | Hasil ablasi blok lag-2 eksogen |
| `tabel_shap_grouped.csv` | SHAP berkelompok dan CI bootstrap |
| `tabel_audit_kebocoran.csv` | Audit kebocoran fitur |
| `sarimax_diff_summary.csv` | Perbandingan SARIMAX level vs eksogen terdiferensiasi |
| `xgb_detrended_summary.csv` | Perbandingan XGBoost original vs detrended |

### Gambar (`outputs/figures/`)

Gambar mencakup plot aktual vs prediksi, diagnostik residual SARIMA/SARIMAX, feature importance dan SHAP XGBoost, heatmap uji DM, serta analisis zona ekstrapolasi.

---

## Kerangka Metodologis

Penelitian menerapkan beberapa prinsip metodologis utama:

- **Anti-leakage berlapis** — fitur turunan target dihitung dari informasi historis, sedangkan variabel kontinu eksogen digunakan dengan lag-1 sesuai desain ex-ante; dummy kebijakan menggunakan informasi kalender yang diketahui sebelum periode berjalan.
- **Ablasi berbasis validasi silang** — keputusan fitur dan spesifikasi ditentukan menggunakan data latih dan validasi silang kronologis.
- **Bootstrap block** — confidence interval dihitung menggunakan moving block bootstrap untuk mempertimbangkan dependensi temporal.
- **Uji Diebold-Mariano** — digunakan untuk menguji perbedaan akurasi prediksi pada data uji dengan sampel n=27.
- **CCF pre-whitened** — digunakan untuk mengurangi risiko korelasi spurious akibat tren bersama sebelum mengevaluasi sinyal temporal antara PKB dan variabel eksogen.

Dokumentasi teknis lengkap tersedia di [`docs/METODOLOGI.md`](docs/METODOLOGI.md).

---

## Sitasi

Jika menggunakan kode atau data dari repositori ini, mohon sitasi:

```text
Adipratama, M. R. (2026). Evaluasi Performa Model XGBoost dan SARIMAX dalam
Peramalan Penerimaan Pajak Kendaraan Bermotor (PKB) di Provinsi DKI Jakarta
dengan Integrasi Variabel Eksogen. Skripsi. Politeknik Keuangan Negara STAN.
```

---

## Lisensi

Kode dalam repositori ini dirilis di bawah lisensi [MIT](LICENSE). Untuk data, ketentuan distribusi mengikuti hak dan ketentuan sumber data masing-masing. Lisensi kode tidak secara otomatis memberikan hak redistribusi terhadap data pihak ketiga.
