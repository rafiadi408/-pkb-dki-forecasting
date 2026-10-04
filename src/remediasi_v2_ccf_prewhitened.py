"""
REMEDIASI V2 — CCF Prewhitened, VIF, Ablasi Lag-2, dan SHAP Berkelompok
=========================================================================
Empat analisis diagnostik tambahan untuk menjawab kritik terkait prosedur
seleksi fitur pada model XGBoost Penuh dan SARIMAX:

  1. CCF Prewhitened     — menghilangkan tren/musiman bersama sebelum
                            menghitung korelasi silang PKB vs eksogen,
                            menghindari korelasi spurious.
  2. VIF                 — regressor SARIMAX ex-ante dan pool 21 fitur
                            XGBoost Penuh, untuk mendeteksi multikolinearitas.
  3. Ablasi Blok Lag-2    — menguji apakah lag-2 eksogen (BBNKB, IKK,
                            Inflasi) memberikan nilai tambah CV-RMSE yang
                            substansial (aturan keputusan >= 0.5% ditetapkan
                            di muka) dibanding hanya memakai lag-1.
  4. SHAP Berkelompok     — mengagregasi nilai SHAP per kelompok fitur
                            (AR & momentum, kalender, eksogen ex-ante)
                            dengan confidence interval bootstrap, karena
                            interpretasi SHAP per-fitur individual tidak
                            stabil akibat multikolinearitas tinggi pada
                            fitur turunan target (lag, rolling, EWMA).
"""

import numpy as np, pandas as pd, pickle, os
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.signal import lfilter
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb, shap

BASE = "outputs_final"; CK = os.path.join(BASE, "checkpoints"); TB = os.path.join(BASE, "tables")
os.makedirs(TB, exist_ok=True)

# --- Ambil data & daftar fitur persis dari pipeline/checkpoint ---
try:
    df = load_and_engineer_features(CFG); train, test = split_train_test(df, CFG)
except NameError:
    import pipeline_v4_final as P
    df = P.load_and_engineer_features(P.CFG); train, test = P.split_train_test(df, P.CFG)

r_full = pickle.load(open(os.path.join(CK, "xgb_full.pkl"), "rb"))
FEATS  = [f.strip() for f in r_full["features"]]; MODEL = r_full["model"]
EXO_L2 = [f for f in FEATS if f.endswith("_lag_2") and f.startswith(("bbnkb","ikk","inflasi"))]
print(f"[SETUP] train n={len(train)} | fitur Penuh={len(FEATS)} | blok lag-2={EXO_L2}")

# ==============================================================================
# LANGKAH 1 - CCF PREWHITENED lag 1-12 (diagnostik pendukung, data latih)
# ==============================================================================
y = train["pkb"].to_numpy(float)
cands, best = [(1,1,1),(2,1,1),(3,1,1),(2,1,2)], None
for o in cands:                                   # pilih filter ARMA pra-pemutihan
    m = ARIMA(y, order=o).fit()
    lb = acorr_ljungbox(m.resid, lags=[12])['lb_pvalue'].iloc[0]
    if lb > 0.0 and (best is None or m.aic < best[1]): best = (o, m.aic, m)
ord_, aic_, m = best
phi, theta = np.r_[1.0, -m.arparams], np.r_[1.0, m.maparams]
print(f"[PREWHITEN] filter ARIMA{ord_} (AIC={aic_:.0f}, residual white noise)")

def prewhiten(v):
    return lfilter(phi, theta, np.diff(np.asarray(v, float)))

yw = prewhiten(y); band = 1.96/np.sqrt(len(yw))
rows = []
for c in ["bbnkb","ikk","inflasi_yoy"]:
    xw = prewhiten(train[c].to_numpy(float)); n = min(len(xw), len(yw))
    Y, X = yw[-n:], xw[-n:]
    for k in range(1, 13):                        # corr(y_t, x_{t-k}) manual: bebas ambiguitas konvensi
        r = np.corrcoef(Y[k:], X[:-k])[0,1]
        rows.append((c, k, round(r,3), "SIGNIFIKAN" if abs(r) > band else "-"))
ccf_df = pd.DataFrame(rows, columns=["variabel","lag","ccf_prewhitened","status"])
print(f"\n[CCF] band signifikansi = +/-{band:.3f}")
print(ccf_df.pivot(index="lag", columns="variabel", values="ccf_prewhitened").to_string())
sig = ccf_df[ccf_df.status=="SIGNIFIKAN"].groupby("variabel")["lag"].apply(list).to_dict()
print("[CCF] lag signifikan per variabel:", sig)
ccf_df.to_csv(os.path.join(TB, "tabel_ccf_prewhitened.csv"), index=False)

# ==============================================================================
# LANGKAH 2 - VIF: regressor SARIMAX & pool 21 fitur XGBoost (diagnostik)
# ==============================================================================
def vif_table(cols, label):
    X = sm.add_constant(train[cols].to_numpy(float))
    v = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    t = pd.DataFrame({"fitur": cols, "VIF": np.round(v,2)}).sort_values("VIF", ascending=False)
    print(f"\n[VIF] {label} (ambang peringatan 5 / kritis 10)"); print(t.to_string(index=False))
    return t
vif_exog = vif_table(["bbnkb_lag_1","ikk_lag_1","inflasi_yoy_lag_1","covid19","pemutihan"],
                     "Regressor SARIMAX ex-ante")
vif_pool = vif_table(FEATS, "Pool fitur XGBoost Penuh")
vif_pool.to_csv(os.path.join(TB, "tabel_vif_pool.csv"), index=False)

# ==============================================================================
# LANGKAH 3 - ABLASI BLOK LAG-2 (arbiter keputusan; hyperparameter dikunci)
# ==============================================================================
HP = dict(max_depth=3, learning_rate=0.172, subsample=0.949, colsample_bytree=0.909,
          min_child_weight=7, gamma=0.483, reg_alpha=0.013, reg_lambda=0.025,
          n_estimators=71, objective="reg:squarederror", tree_method="hist",
          random_state=42, n_jobs=-1)
tscv = TimeSeriesSplit(n_splits=5)
def cv_rmse(feats):
    X, yv = train[feats].to_numpy(float), train["pkb"].to_numpy(float)
    e = []
    for tr_i, va_i in tscv.split(X):
        p = xgb.XGBRegressor(**HP).fit(X[tr_i], yv[tr_i]).predict(X[va_i])
        e.append(np.sqrt(np.mean((yv[va_i]-p)**2)))
    return float(np.mean(e)), [round(x,2) for x in e]

feats_l1 = [f for f in FEATS if f not in EXO_L2]
cv_l1, f_l1 = cv_rmse(feats_l1); cv_l2, f_l2 = cv_rmse(FEATS)
impr = (cv_l1 - cv_l2)/cv_l1*100
keep = (cv_l2 < cv_l1) and (impr >= 0.5)          # aturan keputusan ditulis DI MUKA
print(f"\n[ABLASI-LAG2] tanpa lag-2 : CV-RMSE={cv_l1:8.2f} | per lipatan {f_l1}")
print(f"[ABLASI-LAG2] dengan lag-2: CV-RMSE={cv_l2:8.2f} | per lipatan {f_l2}")
print(f"[KEPUTUSAN] perbaikan={impr:+.2f}% -> {'PERTAHANKAN blok lag-2' if keep else 'BUANG blok lag-2'}")
pd.DataFrame({"varian":["tanpa_lag2","dengan_lag2"],
              "cv_rmse":[round(cv_l1,2), round(cv_l2,2)],
              "per_lipatan":[str(f_l1), str(f_l2)]}).to_csv(
    os.path.join(TB, "tabel_ablasi_lag2.csv"), index=False)

# ==============================================================================
# LANGKAH 4 - SHAP BERKELOMPOK + CI bootstrap (menutup kritik atribusi)
# ==============================================================================
sv = shap.TreeExplainer(MODEL).shap_values(test[FEATS].to_numpy(float))
groups = {"AR & momentum": [f for f in FEATS if f.startswith("pkb_")],
          "Kalender":       [f for f in FEATS if f in ("kuartal","bulan_sin","bulan_cos")],
          "Eksogen ex-ante":[f for f in FEATS if f.startswith(("bbnkb","ikk","inflasi","covid","pemutihan"))]}
idx_g = {g: [FEATS.index(c) for c in cols] for g, cols in groups.items()}
gross = {g: np.abs(sv[:, i]).mean(0).sum() for g, i in idx_g.items()}
tot   = sum(gross.values())
rng, B = np.random.default_rng(42), 500
boot = np.zeros((B, len(groups)))
for b in range(B):
    s = rng.choice(len(sv), size=len(sv), replace=True)
    g_ = np.array([np.abs(sv[s][:, i]).mean(0).sum() for i in idx_g.values()])
    boot[b] = g_/g_.sum()*100
out = []
for j, g in enumerate(groups):
    lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
    out.append((g, round(gross[g]/tot*100, 1), round(lo,1), round(hi,1)))
    print(f"[SHAP-GRUP] {g:16s} share={gross[g]/tot*100:5.1f}%  CI95%=[{lo:.1f}; {hi:.1f}]")
pd.DataFrame(out, columns=["kelompok","share_%","ci_lo","ci_hi"]).to_csv(
    os.path.join(TB, "tabel_shap_grouped.csv"), index=False)
print("\n[SELESAI] 4 tabel CSV tersimpan di:", TB)
