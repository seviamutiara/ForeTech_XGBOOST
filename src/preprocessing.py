# =============================================================
# File : preprocessing.py
# Fungsi: Pipeline Autonomous Cleansing & Feature Engineering.
# Deskripsi: Skrip mandiri untuk membersihkan data mentah, menangani 
#            missing values, dan membentuk fitur prediktor canggih 
#            (lagging, rasio) sebelum dikonsumsi oleh XGBoost.
# =============================================================

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
# KONSTANTA FITUR
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

REQUIRED_INPUT_COLS = [
    'Year', 'Month', 'New_Employee', 'Intern_Count', 'Resigned_Employee',
    'Broken_Device', 'Refresh_Cycle', 'Device_Out', 'Spare_Pool', 'Device_In',
]


def clean_dataset(input_path: str = 'data_raw/foretech_dataset_raw.xlsx'):
    """
    Menjalankan pipeline preprocessing data secara otomatis.
    
    Tahapan Operasional:
      1. Validasi integritas kolom input.
      2. Imputasi nilai hilang (interpolasi, median, ffill).
      3. Rekayasa Fitur Teknis (Lagging, Rasio Kegawatan, Momentum).
      4. Pemotongan baris awal (akibat perhitungan rolling window).
      5. Ekspor dataset siap latih.
    """
    print("=" * 60)
    print(" PREPROCESSING (AUTONOMOUS CLEANSING ENGINE)")
    print("=" * 60)

    # ── Load Dataset Mentah ─────────────────────────────────
    if input_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path, encoding='utf-8-sig')

    print(f"✔ Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")

    # ── Validasi Integritas Data ─────────────────────────────
    missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing_cols:
        print(f"\n❌ Upload gagal — {len(missing_cols)} kolom tidak ditemukan:")
        print(f"   {missing_cols}")
        print("   Gunakan template resmi ForeTech (.xlsx).")
        return

    if len(df) < 10:
        print(f"\n❌ Dataset terlalu kecil ({len(df)} baris).")
        print("   Dibutuhkan minimal 10 data bulanan berurutan.")
        return

    # ── Preprocessing (Penanganan Nilai Hilang) ──────────────
    # - Rekrutmen: Interpolasi untuk menjaga tren kesinambungan.
    # - Kerusakan: Median untuk menghindari anomali data (outlier).
    # - Stok: Forward fill (ffill) untuk meneruskan sisa stok gudang.
    
    df['New_Employee']  = df['New_Employee'].interpolate().fillna(0)
    df['Broken_Device'] = df['Broken_Device'].fillna(df['Broken_Device'].median())
    for c in ['Intern_Count', 'Resigned_Employee',
              'Refresh_Cycle', 'Device_Out', 'Device_In']:
        df[c] = df[c].fillna(0)
    df['Spare_Pool'] = df['Spare_Pool'].ffill().fillna(20).clip(lower=0)

    print("✔ Imputasi selesai (interpolasi, median, ffill).")
    
    # Catatan: Format angka dipertahankan sebagai float untuk menjaga presisi 
    # matematis saat perhitungan rasio dan rata-rata di tahap selanjutnya.

    # ── Feature Engineering (Rekayasa Fitur Pintar) ──────────
    # Membentuk variabel "ingatan" jangka pendek untuk XGBoost
    
    # 1. Fitur Lag (Bulan Sebelumnya)
    for col in BASE_COLS:
        df[f'Lag_{col}'] = df[col].shift(1)

    # 2. Fitur Tren Kebutuhan
    df['Avg_Recruitment_3Mos'] = df['New_Employee'].shift(1).rolling(3).mean()
    df['Avg_Broken_3Mos']      = df['Broken_Device'].shift(1).rolling(3).mean()
    
    # 3. Fitur Peringatan Dini (Early Warning)
    df['Stock_Momentum']       = df['Spare_Pool'].shift(1) - df['Spare_Pool'].shift(2)
    df['Urgency_Ratio'] = (
        (df['New_Employee'].shift(1) + df['Intern_Count'].shift(1)
         + df['Broken_Device'].shift(1))
        / df['Spare_Pool'].shift(1).clip(lower=1)
    )
    df['Gap_To_Safety'] = 20 - df['Spare_Pool'].shift(1)
    df['Quarter']       = ((df['Month'] - 1) // 3) + 1

    # Membuang baris awal yang kosong akibat proses shift() dan rolling()
    df = df.dropna().reset_index(drop=True)

    print(f"✔ Feature engineering selesai. Baris valid yang tersisa: {len(df)}")

    if len(df) < 6:
        print("\n❌ Setelah preprocessing, baris yang tersisa terlalu sedikit.")
        print("   Sediakan dataset dengan rentang waktu yang lebih panjang.")
        return

    # ── Ekspor Data Bersih ───────────────────────────────────
    os.makedirs('data_cleaned', exist_ok=True)
    df.to_csv('data_cleaned/data_final_cleaned.csv', index=False)

    print("✔ data_final_cleaned.csv berhasil disimpan.")
    print(f"  Kolom siap latih: {list(df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    clean_dataset()