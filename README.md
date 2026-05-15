# ForeTech - IT Asset & Budget Intelligence System

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

**ForeTech** adalah sistem pendukung keputusan berbasis kecerdasan buatan (*AI-driven Decision Support System*) yang dirancang untuk memprediksi kuantitas pengadaan aset TI (khususnya laptop). Sistem ini bertujuan mengoptimalkan perencanaan anggaran dengan meminimalkan risiko surplus stok maupun defisit perangkat melalui pemodelan prediktif yang akurat terhadap dinamika rekrutmen dan operasional.

## 🚀 Fitur Utama
- **Autonomous Cleansing:** Pipeline pembersihan data otomatis (Interpolasi, Median, & ffill) untuk menangani *missing values*.
- **Feature Engineering Engine:** Pembentukan fitur prediktor canggih secara temporal, seperti *Urgency Ratio*, *Stock Momentum*, *Gap to Safety*, dan *Lagging Transformation*.
- **Production-Grade Training:** Implementasi *Weighted Training* proporsional pada algoritma XGBoost untuk meningkatkan sensitivitas model terhadap lonjakan kebutuhan ekstrem (contoh: musim penerimaan peserta magang).
- **Executive Dashboard:** Visualisasi tren *Estimation Horizon* dan analisis faktor penentu (*Feature Importance*) secara *real-time*.
- **Justification Reporter:** Ekspor laporan operasional otomatis (PDF) sebagai dokumen pendukung dan justifikasi pengambilan keputusan anggaran.

## 📊 Hasil Evaluasi & Interpretasi Bisnis

### 1. Komparasi Algoritma (*Technical Benchmark*)
Pengujian dilakukan dengan membandingkan model **XGBoost Regressor** (model usulan) terhadap **Linear Regression** sebagai *baseline*. Hasil pengujian menunjukkan bahwa arsitektur XGBoost berbasis *decision tree* mampu menangani fluktuasi data operasional non-linear jauh lebih baik:

| Metrik Evaluasi | Baseline (Linear Regression) | Model Akhir (XGBoost) | Signifikansi |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 12.79 Units | **8.21 Units** | Error prediksi menyusut 35,81% |
| **Root Mean Squared Error (RMSE)** | 17.38 Units | **10.17 Units** | Sensitivitas terhadap outlier membaik 41,48% |
| **R-Squared (R²)** | 0.1545 | **0.7107** | Keandalan sistem meningkat drastis |

### 2. Interpretasi Operasional (*Business Impact*)
Angka evaluasi teknis di atas merepresentasikan keandalan aplikasi saat digunakan langsung di lapangan untuk memproyeksikan anggaran:

* **R² (0.7107):** Sistem secara akurat mampu mengidentifikasi dan menjelaskan **71,07%** pola fluktuasi kebutuhan aset TI berdasarkan data riwayat HR, menjadikannya metrik yang sangat reliabel untuk pengambilan keputusan.
* **MAE (8.21):** Tingkat presisi AI sangat tinggi, di mana rata-rata selisih tebakan sistem dengan kebutuhan riil lapangan hanya terpaut **± 8 unit perangkat**.
* **RMSE (10.17):** Selisih yang kecil antara nilai RMSE dan MAE membuktikan bahwa model beroperasi dengan stabil dan sangat jarang menghasilkan prediksi ekstrem (kalkulasi *error* yang terlampau jauh) yang berpotensi merugikan anggaran perusahaan.

## 📁 Struktur Organisasi Sistem
Proyek ini mengadopsi *Source Layout* untuk menjaga skalabilitas (*maintainability*) dengan memisahkan logika inti AI dan *interface* aplikasi web:

```text
ForeTech_XGBOOST/
├── src/                    # Logika Inti Kecerdasan Buatan
│   ├── preprocessing.py    # Pipeline pembersihan data otomatis
│   ├── train_model.py      # Skrip pelatihan model produksi
│   └── compare_model.py    # Skrip pengujian komparatif algoritma
├── data_raw/               # Folder dataset historis mentah (2019-2024)
├── data_cleaned/           # Folder dataset hasil transformasi preprocessing
├── models/                 # Penyimpanan biner model (.pkl)
├── static/ & templates/    # Aset antarmuka UI/UX (Flask Templates)
└── app.py                  # Mesin orkestrasi & routing sistem utama