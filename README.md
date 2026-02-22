# ForeTech: IT Asset Forecasting System

## Deskripsi Proyek
Proyek ini adalah implementasi Machine Learning untuk memprediksi kebutuhan pengadaan aset TI (Laptop) menggunakan algoritma **XGBoost Regressor**. Sistem ini dirancang untuk meminimalkan risiko surplus dan defisit stok perangkat pada operasional perusahaan digital.

## Tahapan Hasil dan Pembahasan (Bab IV)
1. **Pengumpulan Data**: Mengolah data simulasi historis periode 2019–2024.
2. **Preprocessing Data**: Penanganan nilai hilang, pemeriksaan outlier, dan rekayasa fitur (lag features).
3. **Pengolahan Data**: Pelatihan model XGBoost dengan pembagian dataset 80% training dan 20% testing.
4. **Integrasi ke Aplikasi**: Pengembangan dashboard berbasis web untuk otomatisasi alur kerja prediksi.
5. **Hasil Evaluasi**: Pengujian akurasi menggunakan metrik MAE, RMSE, dan R².

## Variabel Input (Prediktor)
* **X1 (NewEmployee)**: Jumlah karyawan baru.
* **X2 (InternCount)**: Jumlah peserta magang.
* **X3 (ResignCount)**: Jumlah karyawan keluar.
* **X4 (DeviceBroken)**: Jumlah perangkat rusak.
* **X5 (RefreshCycle)**: Perangkat dalam siklus penggantian rutin.
* **X6 (DeviceOut)**: Perangkat keluar dari pool utama.
* **X7 (SparePool)**: Stok cadangan di gudang.

## Teknologi
* Python 3.11.9
* XGBoost & Scikit-learn
* Pandas & NumPy
* Flask Web Framework