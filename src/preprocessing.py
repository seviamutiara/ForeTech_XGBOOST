import pandas as pd
import numpy as np
import os

def preprocess_data():
    print("--- Memulai Proses Preprocessing Data ForeTech (Final Check) ---")
    
    # Path folder
    base_path = r'D:\ForeTech_XGBOOST'
    input_path = os.path.join(base_path, 'data_raw', 'foretech_dataset_raw.csv')
    output_dir = os.path.join(base_path, 'data_cleaned')
    output_path = os.path.join(output_dir, 'data_final_cleaned.csv')

    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"ERROR: File sumber tidak ditemukan di: {input_path}")
        return
    df = pd.read_csv(input_path)
    
    # 2. Missing Values 
    np.random.seed(42)
    df.loc[10:12, 'X1_NewEmployee'] = np.nan
    df.loc[25, 'X4_DeviceBroken'] = np.nan
    
    # 3. Handling Missing Values
    # Lakukan Interpolasi lalu bulatkan ke angka terdekat
    df['X1_NewEmployee'] = df['X1_NewEmployee'].interpolate(method='linear')
    df['X4_DeviceBroken'] = df['X4_DeviceBroken'].fillna(df['X4_DeviceBroken'].median())
    
    # 4. Feature Engineering
    df['Lag_Y_DeviceIn'] = df['Y_DeviceIn'].shift(1).fillna(0)
    df['Rolling_Avg_Hiring'] = df['X1_NewEmployee'].rolling(window=3).mean().fillna(df['X1_NewEmployee'])

    # Mengubah semua kolom angka menjadi Integer (Bilangan Bulat)
    cols_to_fix = [
        'X1_NewEmployee', 'X2_InternCount', 'X3_ResignCount', 
        'X4_DeviceBroken', 'X5_RefreshCycle', 'X6_DeviceOut', 
        'X7_SparePool', 'Y_DeviceIn', 'TotalLaptopActive',
        'Lag_Y_DeviceIn', 'Rolling_Avg_Hiring'
    ]
    
    for col in cols_to_fix:
        # Rounding dulu baru diubah ke int agar 12.7 jadi 13, bukan 12
        df[col] = df[col].round(0).astype(int)

    # 5. Simpan Hasil
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_path, index=False)
    
    print(f"SUKSES! Data sudah dibersihkan dan semua angka sudah dibulatkan (Integer).")
    print(f"File tersimpan di: {output_path}")
    print("\nContoh Baris 10-13 (Cek apakah sudah bulat):")
    print(df.loc[10:13, ['X1_NewEmployee', 'Rolling_Avg_Hiring']])

if __name__ == "__main__":
    preprocess_data()