# =============================================================
# File : train_model.py  (Synced with app.py v3)
# Fungsi: Melatih ulang model XGBoost menggunakan dataset bersih.
#
# Alur Pelatihan (2 Tahap):
# 1. EVALUATION: Dilatih pada 80% data (tanpa bobot tambahan) 
#    untuk menguji akurasi model pada data baru (20%).
# 2. PRODUCTION: Dilatih pada 100% data menggunakan 'Weighted Training'
#    agar model siap dipakai untuk memprediksi bulan depan.
# =============================================================

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────
# KONSTANTA FITUR (Harus sinkron dengan pipeline app.py)
# ──────────────────────────────────────────────────────────────
BASE_COLS = [
    'New_Employee', 'Intern_Count', 'Resigned_Employee', 'Broken_Device',
    'Refresh_Cycle', 'Device_Out', 'Spare_Pool', 'Device_In',
]

FEATURES = (
    ['Month', 'Quarter', 'Avg_Recruitment_3Mos', 'Avg_Broken_3Mos',
     'Stock_Momentum', 'Urgency_Ratio', 'Gap_To_Safety']
    + [f'Lag_{x}' for x in BASE_COLS]
)

def train_engine():
    print("=" * 60)
    print(" PRODUCTION TRAINING — Synced with app.py v3")
    print("=" * 60)

    file_path = "data_cleaned/data_final_cleaned.csv"

    if not os.path.exists(file_path):
        print("❌ ERROR: data_final_cleaned.csv tidak ditemukan.")
        print("   Jalankan preprocessing.py terlebih dahulu.")
        return

    # ── Load Dataset ────────────────────────────────────────
    df = pd.read_csv(file_path)
    print(f"✔ Dataset dimuat: {len(df)} baris")

    # ── Validasi Integritas Kolom ───────────────────────────
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Kolom hilang: {missing_cols}")
        return

    X = df[FEATURES]
    y = df['Device_In']

    # ── Pembagian Data Berurutan (Sequential Split 80:20) ───
    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"   Data Latih (80%) : {len(X_train)} baris")
    print(f"   Data Uji  (20%) : {len(X_test)} baris")

    # ────────────────────────────────────────────────────────
    # TAHAP 1: EVALUATION MODEL
    # Menghasilkan metrik untuk justifikasi kelayakan sistem.
    # ────────────────────────────────────────────────────────
    print("\n[TAHAP 1] Melatih Evaluation Model...")

    eval_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    eval_model.fit(X_train, y_train)

    # Prediksi dibulatkan ke Integer (karena objek fisik/laptop)
    pred_eval = np.maximum(0, np.round(eval_model.predict(X_test)))

    mae  = mean_absolute_error(y_test, pred_eval)
    rmse = np.sqrt(mean_squared_error(y_test, pred_eval))
    r2   = r2_score(y_test, pred_eval)

    print("\n  Evaluation Result (data uji 20%):")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    # ────────────────────────────────────────────────────────
    # TAHAP 2: PRODUCTION MODEL
    # Dilatih pada seluruh data (100%) menggunakan Weighted Training.
    # 
    # Logika Bisnis (Proporsional Bobot):
    # Model memberikan bobot lebih berat pada bulan yang memiliki 
    # volume pengadaan tinggi (misal: saat musim rekrutmen magang).
    # Tujuannya agar AI lebih sensitif mencegah risiko defisit stok.
    # ────────────────────────────────────────────────────────
    print("\n[TAHAP 2] Melatih Production Model (Weighted Training)...")

    # Kalkulasi bobot proporsional (y / mean)
    weights = (y / y.mean()).values

    final_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    final_model.fit(X, y, sample_weight=weights)

    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_model.pkl'
    joblib.dump(final_model, model_path)

    print(f"✔ Model produksi siap digunakan di: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_engine()