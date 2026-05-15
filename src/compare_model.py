# File: src/compare_model.py

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compare():
    print("=" * 60)
    print("EVALUASI DAN KOMPARASI MODEL PENGADAAN ASET TI")
    print("=" * 60)

    file_path = "data_cleaned/data_final_cleaned.csv"

    if not os.path.exists(file_path):
        print("ERROR: data_final_cleaned.csv tidak ditemukan.")
        print("Harap lakukan proses training melalui dashboard web terlebih dahulu.")
        return

    # Muat dataset yang telah diproses oleh app.py
    df = pd.read_csv(file_path)

    base_features = [
        "New_Employee", "Intern_Count", "Resigned_Employee", "Broken_Device",
        "Refresh_Cycle", "Device_Out", "Spare_Pool", "Device_In"
    ]

    features = [
        "Month", "Quarter", "Avg_Recruitment_3Mos", "Avg_Broken_3Mos",
        "Stock_Momentum", "Urgency_Ratio", "Gap_To_Safety"
    ] + [f"Lag_{col}" for col in base_features]

    X = df[features]
    y = df["Device_In"]

    # Sequential Split 80:20 (Identik dengan logika app.py)
    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 1. Pelatihan dan Prediksi Model XGBoost (Model Usulan)
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = np.maximum(0, np.round(xgb.predict(X_test)))

    # 2. Pelatihan dan Prediksi Model Regresi Linear Berganda (Baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = np.maximum(0, np.round(lr.predict(X_test)))

    # 3. Perhitungan Metrik
    mae_xgb = mean_absolute_error(y_test, pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    r2_xgb = r2_score(y_test, pred_xgb)

    mae_lr = mean_absolute_error(y_test, pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
    r2_lr = r2_score(y_test, pred_lr)

    # 4. Tampilkan Hasil Komparasi
    print(f"Total baris data  : {len(df)} baris")
    print(f"Data Latih (80%)  : {len(X_train)} baris")
    print(f"Data Uji (20%)    : {len(X_test)} baris")
    print("\n[HASIL KOMPARASI METRIK]")
    print(f"{'Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
    print("-" * 55)
    print(f"{'XGBoost (Usulan)':<22} | {mae_xgb:<8.2f} | {rmse_xgb:<8.2f} | {r2_xgb:<8.4f}")
    print(f"{'Linear Reg. (Baseline)':<22} | {mae_lr:<8.2f} | {rmse_lr:<8.2f} | {r2_lr:<8.4f}")
    print("=" * 60)

if __name__ == "__main__":
    compare()