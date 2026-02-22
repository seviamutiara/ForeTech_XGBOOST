import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

def train_xgboost():
    data_path = r'D:\ForeTech_XGBOOST\data_cleaned\data_final_cleaned.csv'
    if not os.path.exists(data_path):
        return

    df = pd.read_csv(data_path)

    df['Next_Month_NewEmp'] = df['X1_NewEmployee'].shift(-1).fillna(df['X1_NewEmployee'].mean())
    df['Next_Month_Intern'] = df['X2_InternCount'].shift(-1).fillna(0)
    df['Quarter'] = ((df['Bulan'] - 1) // 3) + 1

    df = df.iloc[3:-1].reset_index(drop=True)

    features = [
        'Bulan', 'Quarter', 'X1_NewEmployee', 'X2_InternCount', 
        'X4_DeviceBroken', 'X5_RefreshCycle', 'X7_SparePool', 
        'Lag_Y_DeviceIn', 'Rolling_Avg_Hiring', 
        'Next_Month_NewEmp', 'Next_Month_Intern'
    ]
    target = 'Y_DeviceIn'

    X = df[features]
    y = df[target]

    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions_rounded = np.maximum(0, np.round(predictions)).astype(int)
    
    mae = mean_absolute_error(y_test, predictions_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_rounded))
    r2 = r2_score(y_test, predictions_rounded)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-Squared: {r2:.4f}")

    model_dir = r'D:\ForeTech_XGBOOST\models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, 'xgboost_model.pkl'))

if __name__ == "__main__":
    train_xgboost()