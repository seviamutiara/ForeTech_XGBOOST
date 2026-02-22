import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def compare_baseline():
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

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    predictions = model_lr.predict(X_test)
    predictions_rounded = np.maximum(0, np.round(predictions)).astype(int)
    
    mae = mean_absolute_error(y_test, predictions_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_rounded))
    r2 = r2_score(y_test, predictions_rounded)

    print(f"MAE Linear Regression: {mae:.2f}")
    print(f"RMSE Linear Regression: {rmse:.2f}")
    print(f"R-Squared Linear Regression: {r2:.4f}")

if __name__ == "__main__":
    compare_baseline()