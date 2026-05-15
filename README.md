# ForeTech - IT Asset & Budget Intelligence System

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

**ForeTech** adalah sistem pendukung keputusan berbasis kecerdasan buatan (*AI-driven Decision Support System*) yang dirancang untuk memprediksi kuantitas pengadaan aset TI (khususnya laptop). Sistem ini bertujuan mengoptimalkan perencanaan anggaran dengan meminimalkan risiko surplus stok maupun defisit perangkat melalui pemodelan prediktif yang akurat.

## 🚀 Fitur Utama
- **Autonomous Cleansing:** Pipeline pembersihan data otomatis (Interpolasi, Median, & ffill) untuk menangani *missing values*.
- **Feature Engineering Engine:** Pembentukan fitur prediktor canggih seperti *Urgency Ratio*, *Stock Momentum*, *Gap to Safety*, dan *Lagging Transformation*.
- **Production-Grade Training:** Implementasi *Weighted Training* pada algoritma XGBoost untuk meningkatkan sensitivitas model terhadap lonjakan kebutuhan (musim rekrutmen/magang).
- **Executive Dashboard:** Visualisasi tren *Estimation Horizon* dan analisis faktor penentu (*Feature Importance*).
- **Justification Reporter:** Ekspor laporan otomatis (PDF) sebagai dokumen pendukung pengambilan keputusan anggaran.

## 📊 Hasil Evaluasi Model
Berdasarkan pengujian komparasi terhadap model *baseline* (Linear Regression), algoritma **XGBoost** menunjukkan keunggulan signifikan:

| Metrik Evaluasi | Hasil (XGBoost) | Peningkatan Efisiensi |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **8.21 Units** | Penurunan Error 35,81% |
| **Root Mean Squared Error (RMSE)** | **10.17 Units** | Penurunan Error 41,48% |
| **R-Squared (R²)** | **0.7107** | Akurasi meningkat 360% |

## 📁 Struktur Proyek
Proyek ini menggunakan *Source Layout* untuk memisahkan logika inti AI dengan *interface* aplikasi:
```text
ForeTech_XGBOOST/
├── src/                    # Logika Inti Kecerdasan Buatan
│   ├── preprocessing.py    # Pipeline pembersihan data otomatis
│   ├── train_model.py      # Script pelatihan model produksi
│   └── compare_model.py    # Script pengujian komparatif algoritma
├── data_raw/               # Dataset historis mentah (2019-2024)
├── data_cleaned/           # Dataset hasil transformasi preprocessing
├── models/                 # Penyimpanan biner model (.pkl)
├── static/ & templates/    # Aset antarmuka web (Flask UI)
└── app.py                  # Mesin orkestrasi sistem utama