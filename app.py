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

from flask import (
    Flask, flash, redirect, render_template,
    request, send_file, session, url_for
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = os.environ.get('FORETECH_SECRET_KEY', os.urandom(24))

os.makedirs('data_cleaned', exist_ok=True)
os.makedirs('models', exist_ok=True)

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
    'Month':                 'Bulan',
    'Quarter':               'Kuartal',
    'Avg_Recruitment_3Mos':  'Tren Karyawan Masuk',
    'Avg_Broken_3Mos':       'Tren Kerusakan',
    'Stock_Momentum':        'Pergerakan Stok Gudang',
    'Urgency_Ratio':         'Rasio Kebutuhan Mendesak',
    'Gap_To_Safety':         'Jarak ke Batas Aman Stok',
    'Lag_New_Employee':      'Karyawan Baru',
    'Lag_Intern_Count':      'Anak Magang',
    'Lag_Resigned_Employee': 'Karyawan Resign',
    'Lag_Broken_Device':     'Perangkat Rusak',
    'Lag_Refresh_Cycle':     'Pembaruan Aset Usang',
    'Lag_Device_Out':        'Aset Ditarik/Hilang',
    'Lag_Spare_Pool':        'Sisa Stok Gudang',
    'Lag_Device_In':         'Riwayat Belanja Sebelumnya',
}

CORE_OPERATIONAL = [
    'Lag_New_Employee', 'Lag_Intern_Count', 'Lag_Resigned_Employee',
    'Lag_Broken_Device', 'Lag_Refresh_Cycle', 'Lag_Device_Out', 'Lag_Spare_Pool',
]

def prepare_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    l = df.iloc[-1]
    m_next = int(l['Month'] + 1) if l['Month'] < 12 else 1
    y_next = int(l['Year']) if l['Month'] < 12 else int(l['Year'] + 1)
    
    session['target_period'] = f"Bulan {m_next} / {y_next}"

    if len(df) >= 4:
        avg_recruit = df['New_Employee'].iloc[-4:-1].mean()
        avg_broken  = df['Broken_Device'].iloc[-4:-1].mean()
    elif len(df) >= 2:
        avg_recruit = df['New_Employee'].iloc[:-1].mean()
        avg_broken  = df['Broken_Device'].iloc[:-1].mean()
    else:
        avg_recruit = float(l['New_Employee'])
        avg_broken  = float(l['Broken_Device'])

    stock_momentum = float(l['Spare_Pool']) - float(df.iloc[-2]['Spare_Pool']) if len(df) > 1 else 0.0

    row = {
        'Month': m_next,
        'Quarter': int(((m_next - 1) // 3) + 1),
        'Avg_Recruitment_3Mos': avg_recruit,
        'Avg_Broken_3Mos': avg_broken,
        'Stock_Momentum': stock_momentum,
        'Urgency_Ratio': (
            (float(l['New_Employee']) + float(l['Intern_Count']) + float(l['Broken_Device'])) 
            / max(1.0, float(l['Spare_Pool']))
        ),
        'Gap_To_Safety': 20.0 - float(l['Spare_Pool']),
        'Lag_New_Employee': float(l['New_Employee']),
        'Lag_Intern_Count': float(l['Intern_Count']),
        'Lag_Resigned_Employee': float(l['Resigned_Employee']),
        'Lag_Broken_Device': float(l['Broken_Device']),
        'Lag_Refresh_Cycle': float(l['Refresh_Cycle']),
        'Lag_Device_Out': float(l['Device_Out']),
        'Lag_Spare_Pool': float(l['Spare_Pool']),
        'Lag_Device_In': float(l['Device_In']),
    }
    return pd.DataFrame([row])[FEATURES]

def generate_plots(df, prediction_val, model, feature_names):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')

        df_recent = df.tail(12).reset_index(drop=True)
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        xr = range(len(df_recent))
        
        ax1.fill_between(xr, df_recent['Device_In'], color='#1e3a8a', alpha=0.08)
        ax1.plot(df_recent['Device_In'], label='Realisasi Pengadaan', 
                 marker='o', color='#1e3a8a', linewidth=3, markersize=8)
        
        ax1.plot([len(df_recent)-1, len(df_recent)], 
                 [df_recent['Device_In'].iloc[-1], prediction_val], 
                 color='#f59e0b', linestyle='--', linewidth=2.5)
                 
        ax1.scatter(len(df_recent), prediction_val, color='#f59e0b', s=200, 
                    zorder=5, edgecolor='white', linewidth=2, label='Proyeksi Sistem')
                    
        ax1.annotate(
            f'{int(prediction_val)} Unit', 
            xy=(len(df_recent), prediction_val), 
            xytext=(0, 15), textcoords='offset points', 
            ha='center', va='bottom', fontsize=12, fontweight='900', color='#f59e0b', 
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#f59e0b', alpha=0.9)
        )
        
        for sp in ['top', 'right']: ax1.spines[sp].set_visible(False)
        ax1.spines['left'].set_color('#cbd5e1')
        ax1.spines['bottom'].set_color('#cbd5e1')
        ax1.set_title('Proyeksi Pengadaan Aset Periode Berikutnya', fontsize=14, fontweight='900', color='#0f172a', pad=15)
        ax1.tick_params(axis='both', labelsize=10, colors='#475569')
        ax1.legend(loc='upper left', frameon=True, facecolor='white', fontsize=10)
        
        plt.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=140, transparent=True)
        plt.close('all')
        trend_url = base64.b64encode(buf1.getvalue()).decode()
        buf1.close()

        try:
            trained_names = list(model.feature_names_in_)
        except AttributeError:
            trained_names = list(feature_names)

        importances = model.feature_importances_
        all_feat_imp = pd.Series(importances, index=trained_names)
        core_imp = all_feat_imp[all_feat_imp.index.isin(CORE_OPERATIONAL)]

        if core_imp.empty:
            core_imp = all_feat_imp.drop(labels=[x for x in ['Month', 'Quarter'] if x in all_feat_imp.index], errors='ignore').nlargest(7)

        core_imp_normalized = core_imp / core_imp.sum()
        core_imp_normalized.index = [LABEL_BISNIS.get(c, c) for c in core_imp_normalized.index]
        core_imp_normalized = core_imp_normalized.sort_values(ascending=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(core_imp_normalized)))
        bars = ax2.barh(core_imp_normalized.index, core_imp_normalized.values, color=colors, height=0.6)

        for bar in bars:
            w = bar.get_width()
            ax2.text(
                w + 0.005, bar.get_y() + bar.get_height() / 2, 
                f'{w * 100:.1f}%', ha='left', va='center', 
                fontsize=10, fontweight='800', color='#1e3a8a'
            )

        ax2.set_xticks([])
        for sp in ['top', 'right', 'bottom']: ax2.spines[sp].set_visible(False)
        ax2.spines['left'].set_color('#cbd5e1')
        ax2.set_title('Kontribusi Faktor Pemicu Pengadaan (%)', fontsize=13, fontweight='900', color='#0f172a', pad=15)
        ax2.tick_params(axis='y', labelsize=10, colors='#475569')
        
        plt.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=140, transparent=True)
        plt.close('all')
        imp_url = base64.b64encode(buf2.getvalue()).decode()
        buf2.close()

        gc.collect()
        return trend_url, imp_url
    except Exception as e:
        plt.close('all')
        gc.collect()
        return None, None

@app.route('/')
def index():
    if session.get('is_trained') and os.path.exists('models/xgboost_model.pkl'):
        df = pd.read_csv('data_cleaned/data_final_cleaned.csv')
        model = joblib.load('models/xgboost_model.pkl')
        in_df = prepare_inference_data(df)
        
        # ERROR PREVENTION: Validasi Prediksi Minus
        raw_prediction = model.predict(in_df)[0]
        if raw_prediction < 0:
            res = 0
            flash('Validasi Sistem: Kalkulasi negatif terdeteksi akibat surplus stok. Rekomendasi pengadaan otomatis dibatasi menjadi 0 Unit.', 'error')
        else:
            res = int(np.round(raw_prediction))
            
        trend_plot, imp_plot = generate_plots(df, res, model, in_df.columns)
        
        return render_template(
            'index.html', 
            active_tab='dashboard', 
            prediction_text=res, 
            plot_url=trend_plot, 
            importance_plot_url=imp_plot
        )
            
    return render_template('index.html', active_tab='dashboard')

@app.route('/model')
def model_page():
    return render_template(
        'index.html', 
        active_tab='model', 
        show_eval=session.get('is_trained'), 
        eval_metrics=session.get('eval_metrics')
    )

@app.route('/train', methods=['POST'])
def train():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('Inisialisasi dibatalkan: Tidak ada berkas dataset yang dipilih.', 'error')
        return redirect(url_for('model_page'))
        
    try:
        if file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, encoding='utf-8-sig')

        missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
        if missing_cols:
            flash(f"Validasi struktur gagal. Kolom wajib tidak ditemukan: {', '.join(missing_cols)}.", 'error')
            return redirect(url_for('model_page'))

        # ERROR PREVENTION: Validasi Dataset Minimum
        if len(df) < 12:
            flash(f"Validasi gagal: Volume data ({len(df)} observasi) tidak memenuhi batas minimum pelatihan (12 bulan historis).", 'error')
            return redirect(url_for('model_page'))

        df['New_Employee'] = df['New_Employee'].interpolate().fillna(0)
        df['Broken_Device'] = df['Broken_Device'].fillna(df['Broken_Device'].median())
        for c in ['Intern_Count', 'Resigned_Employee', 'Refresh_Cycle', 'Device_Out', 'Device_In']:
            df[c] = df[c].fillna(0)
        df['Spare_Pool'] = df['Spare_Pool'].ffill().fillna(20).clip(lower=0)

        for col in BASE_COLS: 
            df[f'Lag_{col}'] = df[col].shift(1)
            
        df['Avg_Recruitment_3Mos'] = df['New_Employee'].shift(1).rolling(3).mean()
        df['Avg_Broken_3Mos'] = df['Broken_Device'].shift(1).rolling(3).mean()
        df['Stock_Momentum'] = df['Spare_Pool'].shift(1) - df['Spare_Pool'].shift(2)
        df['Urgency_Ratio'] = ((df['New_Employee'].shift(1) + df['Intern_Count'].shift(1) + df['Broken_Device'].shift(1)) / df['Spare_Pool'].shift(1).clip(lower=1))
        df['Gap_To_Safety'] = 20 - df['Spare_Pool'].shift(1)
        df['Quarter'] = ((df['Month'] - 1) // 3) + 1
        
        df = df.dropna().reset_index(drop=True)

        if len(df) < 6:
            flash('Ekstraksi fitur gagal: Baris observasi pasca-pembersihan tidak mencukupi untuk memvalidasi model.', 'error')
            return redirect(url_for('model_page'))

        df.to_csv('data_cleaned/data_final_cleaned.csv', index=False)

        X, y = df[FEATURES], df['Device_In']
        split = int(len(df) * 0.8)

        eval_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
        eval_model.fit(X.iloc[:split], y.iloc[:split])
        p_eval = np.maximum(0, np.round(eval_model.predict(X.iloc[split:])))
        
        r2_value = round(r2_score(y.iloc[split:], p_eval), 4)
        
        session['eval_metrics'] = {
            'mae': round(mean_absolute_error(y.iloc[split:], p_eval), 2),
            'rmse': round(np.sqrt(mean_squared_error(y.iloc[split:], p_eval)), 2),
            'r2': r2_value,
        }

        # ERROR PREVENTION: Validasi R-Squared Kelayakan
        if r2_value < 0.50:
            session['is_trained'] = False
            flash(f"Pelatihan dihentikan: Nilai reliabilitas model (R² = {r2_value}) di bawah ambang batas minimum kelayakan (0.50).", 'error')
            return redirect(url_for('model_page'))

        weights = (y / y.mean()).values
        final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
        final_model.fit(X, y, sample_weight=weights)
        joblib.dump(final_model, 'models/xgboost_model.pkl')

        session['is_trained'] = True
        flash('Pelatihan berhasil: Model prediktif telah diverifikasi dan siap dioperasikan.', 'success')
        return redirect(url_for('model_page'))

    except Exception as e:
        flash(f'Terjadi kesalahan internal sistem: {str(e)}', 'error')
        return redirect(url_for('model_page'))

@app.route('/export_report')
def export_report():
    if not session.get('is_trained') or not os.path.exists('models/xgboost_model.pkl'):
        return redirect(url_for('index'))
        
    df = pd.read_csv('data_cleaned/data_final_cleaned.csv')
    model = joblib.load('models/xgboost_model.pkl')
    in_df = prepare_inference_data(df)
    
    raw_prediction = model.predict(in_df)[0]
    res = 0 if raw_prediction < 0 else int(np.round(raw_prediction))
        
    trend_plot, imp_plot = generate_plots(df, res, model, in_df.columns)
    
    return render_template('report.html', data={
        'prediction': res,
        'plot_url': trend_plot,
        'importance_plot_url': imp_plot,
        'eval_metrics': session.get('eval_metrics', {'r2':0,'mae':0,'rmse':0}),
        'target_period': session.get('target_period', ''),
        'date': datetime.now().strftime('%d %B %Y'),
    })

@app.route('/download_template')
def download_template():
    buf = io.BytesIO()
    pd.DataFrame(columns=REQUIRED_INPUT_COLS).to_excel(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='Template_ForeTech.xlsx')

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)