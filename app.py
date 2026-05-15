import gc
import io
import os
import base64
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import (Flask, flash, redirect, render_template,
                   request, send_file, session, url_for)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- INISIALISASI FLASK & DIREKTORI ---
app = Flask(__name__)
app.secret_key = os.environ.get('FORETECH_SECRET_KEY', os.urandom(24))

os.makedirs('data_cleaned', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- KONSTANTA FITUR & LABEL BISNIS ---
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

LABEL_BISNIS = {
    'Month':                 'Month',
    'Quarter':               'Quarter',
    'Avg_Recruitment_3Mos':  'Recruitment Trend',     
    'Avg_Broken_3Mos':       'Damage Trend',          
    'Stock_Momentum':        'Stock Momentum',
    'Urgency_Ratio':         'Demand vs Supply Ratio',
    'Gap_To_Safety':         'Safety Stock Deficit',
    'Lag_New_Employee':      'New Employee',
    'Lag_Intern_Count':      'Intern Count',
    'Lag_Resigned_Employee': 'Resigned Employee',
    'Lag_Broken_Device':     'Broken Device',
    'Lag_Refresh_Cycle':     'Refresh Cycle',
    'Lag_Device_Out':        'Asset Disposal',       
    'Lag_Spare_Pool':        'Spare Pool',
    'Lag_Device_In':         'Previous Procurement',
}

# --- 1. FUNGSI INFERENSI: MENYIAPKAN DATA BULAN DEPAN (t+1) ---
def prepare_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    l      = df.iloc[-1]
    m_next = int(l['Month'] + 1) if l['Month'] < 12 else 1
    y_next = int(l['Year'])      if l['Month'] < 12 else int(l['Year'] + 1)
    session['target_period'] = f"Month {m_next} / {y_next}"

    # Kalkulasi fitur memori jangka pendek (Rolling Average & Momentum)
    if len(df) >= 4:
        avg_recruit = df['New_Employee'].iloc[-4:-1].mean()
        avg_broken  = df['Broken_Device'].iloc[-4:-1].mean()
    elif len(df) >= 2:
        avg_recruit = df['New_Employee'].iloc[:-1].mean()
        avg_broken  = df['Broken_Device'].iloc[:-1].mean()
    else:
        avg_recruit = float(l['New_Employee'])
        avg_broken  = float(l['Broken_Device'])

    stock_momentum = (
        float(l['Spare_Pool']) - float(df.iloc[-2]['Spare_Pool'])
        if len(df) > 1 else 0.0
    )

    row = {
        'Month':                 m_next,
        'Quarter':               int(((m_next - 1) // 3) + 1),
        'Avg_Recruitment_3Mos':  avg_recruit,
        'Avg_Broken_3Mos':       avg_broken,
        'Stock_Momentum':        stock_momentum,
        'Urgency_Ratio': (
            (float(l['New_Employee']) + float(l['Intern_Count'])
             + float(l['Broken_Device']))
            / max(1.0, float(l['Spare_Pool']))
        ),
        'Gap_To_Safety':         20.0 - float(l['Spare_Pool']),
        'Lag_New_Employee':      float(l['New_Employee']),
        'Lag_Intern_Count':      float(l['Intern_Count']),
        'Lag_Resigned_Employee': float(l['Resigned_Employee']),
        'Lag_Broken_Device':     float(l['Broken_Device']),
        'Lag_Refresh_Cycle':     float(l['Refresh_Cycle']),
        'Lag_Device_Out':        float(l['Device_Out']),
        'Lag_Spare_Pool':        float(l['Spare_Pool']),
        'Lag_Device_In':         float(l['Device_In']),
    }
    return pd.DataFrame([row])[FEATURES]

# --- 2. FUNGSI VISUALISASI GRAFIK (STATELESS / IN-MEMORY) ---
def generate_plots(df, prediction_val, model, feature_names):
    plt.style.use('seaborn-v0_8-whitegrid')

    # Pembuatan Grafik 1: Horizon Prediksi
    df_recent = df.tail(12).reset_index(drop=True)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    xr = range(len(df_recent))
    ax1.fill_between(xr, df_recent['Device_In'], color='#1e3a8a', alpha=0.08)
    ax1.plot(df_recent['Device_In'], label='Historical Data', marker='o', color='#1e3a8a', linewidth=3, markersize=8)
    ax1.plot([len(df_recent)-1, len(df_recent)], [df_recent['Device_In'].iloc[-1], prediction_val], color='#f59e0b', linestyle='--', linewidth=2.5)
    ax1.scatter(len(df_recent), prediction_val, color='#f59e0b', s=200, zorder=5, edgecolor='white', linewidth=2, label='AI Recommendation')
    ax1.annotate(f'{int(prediction_val)} Units', xy=(len(df_recent), prediction_val), xytext=(0, 15), textcoords='offset points', ha='center', va='bottom', fontsize=13, fontweight='900', color='#f59e0b', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#f59e0b', alpha=0.9))
    
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)
    ax1.spines['left'].set_color('#cbd5e1')
    ax1.spines['bottom'].set_color('#cbd5e1')
    ax1.set_title('Procurement Estimation Horizon', fontsize=16, fontweight='900', color='#0f172a', pad=20)
    ax1.tick_params(axis='both', labelsize=11, colors='#475569')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', fontsize=11)
    
    plt.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=140, transparent=True)
    plt.close('all')
    trend_url = base64.b64encode(buf1.getvalue()).decode()
    buf1.close()

    # Pembuatan Grafik 2: Feature Importance (Analisis Pengaruh)
    importances    = model.feature_importances_
    friendly_names = [LABEL_BISNIS.get(c, c) for c in feature_names]
    feat_imp = pd.Series(importances, index=friendly_names)
    feat_imp = feat_imp.drop(labels=[x for x in ['Month', 'Quarter'] if x in feat_imp.index], errors='ignore')
    feat_imp = feat_imp.sort_values(ascending=True).tail(7)

    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(feat_imp)))
    bars   = ax2.barh(feat_imp.index, feat_imp.values, color=colors, height=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w*100:.1f}%', ha='left', va='center', fontsize=11, fontweight='800', color='#1e3a8a')
    
    ax2.set_xticks([])
    for sp in ['top', 'right', 'bottom']:
        ax2.spines[sp].set_visible(False)
    ax2.spines['left'].set_color('#cbd5e1')
    ax2.set_title('Primary Decisional Factors (%)', fontsize=15, fontweight='900', color='#0f172a', pad=15)
    ax2.tick_params(axis='y', labelsize=11, colors='#475569')
    
    plt.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=140, transparent=True)
    plt.close('all')
    imp_url = base64.b64encode(buf2.getvalue()).decode()
    buf2.close()

    gc.collect()
    return trend_url, imp_url

# --- 3. RUTE APLIKASI WEB ---

@app.route('/')
def index():
    # Render dashboard utama & jalankan prediksi jika model sudah ada
    if session.get('is_trained') and os.path.exists('models/xgboost_model.pkl'):
        df    = pd.read_csv('data_cleaned/data_final_cleaned.csv')
        model = joblib.load('models/xgboost_model.pkl')
        in_df = prepare_inference_data(df)
        res   = int(np.maximum(0, np.round(model.predict(in_df)[0])))
        trend_plot, imp_plot = generate_plots(df, res, model, in_df.columns)
        return render_template(
            'index.html', active_tab='dashboard',
            prediction_text=res, plot_url=trend_plot,
            importance_plot_url=imp_plot)
    return render_template('index.html', active_tab='dashboard')

@app.route('/model')
def model_page():
    # Halaman untuk upload data dan mengecek metrik evaluasi
    return render_template(
        'index.html', active_tab='model',
        show_eval=session.get('is_trained'),
        eval_metrics=session.get('eval_metrics'))

@app.route('/train', methods=['POST'])
def train():
    # PIPELINE UTAMA: Validasi -> Cleansing -> Pelatihan AI
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected. Please choose an Excel file.', 'error')
        return redirect(url_for('model_page'))
    try:
        # A. Proses Ekstraksi & Validasi File
        if file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, encoding='utf-8-sig')

        missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
        if missing_cols:
            flash(f"Upload failed — {len(missing_cols)} required column(s) not found: [ {', '.join(missing_cols)} ]. Please use the official ForeTech Template (.xlsx).", 'error')
            return redirect(url_for('model_page'))

        if len(df) < 10:
            flash(f"Dataset too small ({len(df)} rows detected). A minimum of 10 consecutive monthly records is required.", 'error')
            return redirect(url_for('model_page'))

        # B. Imputasi Nilai Kosong (Missing Values)
        df['New_Employee']  = df['New_Employee'].interpolate().fillna(0)
        df['Broken_Device'] = df['Broken_Device'].fillna(df['Broken_Device'].median())
        for c in ['Intern_Count', 'Resigned_Employee', 'Refresh_Cycle', 'Device_Out', 'Device_In']:
            df[c] = df[c].fillna(0)
        df['Spare_Pool'] = df['Spare_Pool'].ffill().fillna(20).clip(lower=0)

        # C. Feature Engineering (Logika Temporal & Rasio)
        for col in BASE_COLS:
            df[f'Lag_{col}'] = df[col].shift(1)
        df['Avg_Recruitment_3Mos'] = df['New_Employee'].shift(1).rolling(3).mean()
        df['Avg_Broken_3Mos']      = df['Broken_Device'].shift(1).rolling(3).mean()
        df['Stock_Momentum']       = df['Spare_Pool'].shift(1) - df['Spare_Pool'].shift(2)
        df['Urgency_Ratio'] = ((df['New_Employee'].shift(1) + df['Intern_Count'].shift(1) + df['Broken_Device'].shift(1)) / df['Spare_Pool'].shift(1).clip(lower=1))
        df['Gap_To_Safety'] = 20 - df['Spare_Pool'].shift(1)
        df['Quarter']       = ((df['Month'] - 1) // 3) + 1
        df = df.dropna().reset_index(drop=True)

        if len(df) < 6:
            flash('After preprocessing, not enough rows remain. Please provide at least 6+ consecutive monthly records.', 'error')
            return redirect(url_for('model_page'))

        df.to_csv('data_cleaned/data_final_cleaned.csv', index=False)

        X, y  = df[FEATURES], df['Device_In']
        split = int(len(df) * 0.8)

        # D. Pelatihan Tahap 1: Evaluation Model (Uji Metrik Akurasi 80:20)
        eval_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
        eval_model.fit(X.iloc[:split], y.iloc[:split])
        p_eval = np.maximum(0, np.round(eval_model.predict(X.iloc[split:])))
        session['eval_metrics'] = {
            'mae':  round(mean_absolute_error(y.iloc[split:], p_eval), 2),
            'rmse': round(np.sqrt(mean_squared_error(y.iloc[split:], p_eval)), 2),
            'r2':   round(r2_score(y.iloc[split:], p_eval), 4),
        }

        # E. Pelatihan Tahap 2: Production Model (Menggunakan Weighted Training)
        weights = (y / y.mean()).values
        final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
        final_model.fit(X, y, sample_weight=weights)
        joblib.dump(final_model, 'models/xgboost_model.pkl')

        session['is_trained'] = True
        flash('Model trained successfully!', 'success')
        return redirect(url_for('model_page'))

    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}', 'error')
        return redirect(url_for('model_page'))

@app.route('/export_report')
def export_report():
    # Render berkas laporan PDF / HTML berdasarkan data inference terbaru
    if not session.get('is_trained') or not os.path.exists('models/xgboost_model.pkl'):
        return redirect(url_for('index'))
    df    = pd.read_csv('data_cleaned/data_final_cleaned.csv')
    model = joblib.load('models/xgboost_model.pkl')
    in_df = prepare_inference_data(df)
    res   = int(np.maximum(0, np.round(model.predict(in_df)[0])))
    trend_plot, imp_plot = generate_plots(df, res, model, in_df.columns)
    return render_template('report.html', data={
        'prediction':          res,
        'plot_url':            trend_plot,
        'importance_plot_url': imp_plot,
        'eval_metrics':        session.get('eval_metrics', {'r2':0,'mae':0,'rmse':0}),
        'target_period':       session.get('target_period', ''),
        'date':                datetime.now().strftime('%B %d, %Y'),
    })

@app.route('/download_template')
def download_template():
    # Menghasilkan template file Excel kosong secara dinamis
    buf = io.BytesIO()
    pd.DataFrame(columns=REQUIRED_INPUT_COLS).to_excel(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='ForeTech_Template.xlsx')

@app.route('/reset')
def reset():
    # Membersihkan sesi / reset dashboard
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)