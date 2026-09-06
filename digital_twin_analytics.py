import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def generate_telemetry_file():
    np.random.seed(42)
    n_rows = 30000
    time = np.linspace(0, 3000, n_rows)
    
    map_press = 30.0 + 5.0 * np.sin(time / 200.0) + np.random.normal(0, 0.2, n_rows)
    fuel_flow = 0.0025 + 0.0005 * np.sin(time / 150.0) + np.random.normal(0, 0.0001, n_rows)
    cht = 85.0 + 20.0 * (time / 3000.0) + np.random.normal(0, 0.5, n_rows)
    egt = 650.0 + 100.0 * np.sin(time / 100.0) + np.random.normal(0, 2.0, n_rows)
    t_oil = 75.0 + 15.0 * (time / 3000.0) + np.random.normal(0, 0.3, n_rows)
    p_oil = 4.5 - 0.5 * (time / 3000.0) + np.random.normal(0, 0.05, n_rows)
    power = 75.0 + 10.0 * np.sin(time / 250.0) + np.random.normal(0, 0.5, n_rows)
    v_bus = 14.2 + np.random.normal(0, 0.1, n_rows)
    
    cht[25000:] += np.linspace(0, 60, 5000)

    df_raw = pd.DataFrame({
        0: time, 1: map_press, 2: fuel_flow, 3: cht, 
        4: egt, 5: t_oil, 6: p_oil, 7: power, 8: v_bus
    })
    df_raw.to_csv('uav_engine_telemetry.csv', index=False, header=False)

def process_digital_twin_analytics():
    generate_telemetry_file()
    df_raw = pd.read_csv('uav_engine_telemetry.csv', header=None)
    
    df = pd.DataFrame()
    df['Time'] = df_raw.iloc[:, 0]
    df['MAP'] = df_raw.iloc[:, 1]
    df['Fuel_Flow'] = df_raw.iloc[:, 2]
    df['CHT'] = df_raw.iloc[:, 3]
    df['EGT'] = df_raw.iloc[:, 4]
    df['T_oil'] = df_raw.iloc[:, 5]
    df['P_oil'] = df_raw.iloc[:, 6]
    df['Power'] = df_raw.iloc[:, 7]
    df['V_bus'] = df_raw.iloc[:, 8]
    
    # ✅ CORRECTED PHYSICS SCALING: Map Power to realistic UAV engine bounds (4000 - 5500 RPM)
    df['RPM'] = 3000.0 + (df['Power'] / 100.0) * 2500.0 + np.random.normal(0, 15, len(df))
    # ✅ CORRECTED VIBRATION SCALING: Keep nominal vibration under 2.5 g's
    df['Vibration'] = 0.8 + 1.2 * (df['RPM'] / 5800.0)**2 + np.random.normal(0, 0.05, len(df))

    df['BSFC'] = (df['Fuel_Flow'] * 3600) / np.maximum(df['Power'], 0.1)
    df['Thermal_Stress'] = (df['CHT'] * df['EGT']) / 1000.0

    features = ['RPM', 'MAP', 'Fuel_Flow', 'CHT', 'EGT', 'T_oil', 'P_oil', 'Power', 'Vibration', 'V_bus', 'BSFC']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    df['Anomaly_Flag'] = iso_forest.fit_predict(X_scaled)
    df['Anomaly_Score'] = -iso_forest.score_samples(X_scaled)

    def isolate_faults(row):
        faults = []
        if row['CHT'] > 135.0: faults.append("High CHT / Overheating")
        if row['EGT'] > 880.0: faults.append("High EGT")
        if row['Fuel_Flow'] < 0.0012 and row['EGT'] > 880.0: faults.append("Injector Abnormality / Misfire")
        if row['P_oil'] < 2.0: faults.append("Low Oil Pressure")
        if row['T_oil'] > 110.0: faults.append("High Oil Temperature")
        if row['Vibration'] > 4.5: faults.append("Abnormal Vibration / Instability")
        if row['V_bus'] < 12.2 and row['RPM'] > 2000: faults.append("Alternator / Battery Fault")
        if row['CHT'] <= 0.0 or row['CHT'] > 250.0: faults.append("CHT Sensor Fault")
        if row['EGT'] <= 0.0 or row['EGT'] > 1200.0: faults.append("EGT Sensor Fault")
        if row['P_oil'] <= 0.1 and row['RPM'] > 2000: faults.append("Oil Pressure Sensor Fault")
        if row['T_oil'] <= -20.0 or row['T_oil'] > 180.0: faults.append("Oil Temp Sensor Fault")
        if row['Vibration'] < 0.01: faults.append("Vibration Sensor Fault")
        
        if not faults:
            return "Nominal / Operational Envelope" if row['Anomaly_Flag'] == 1 else "Multi-Sensor Operational Drift"
        return " + ".join(faults)

    df['Fault_Diagnosis'] = df.apply(isolate_faults, axis=1)

    hi_cht = np.maximum(0, 100 * (1 - (df['CHT'] - 25.0) / 110.0))
    hi_oil = np.maximum(0, 100 * (1 - (df['T_oil'] - 20.0) / 90.0))
    df['Health_Index'] = np.clip(0.5 * hi_cht + 0.5 * hi_oil, 0, 100)

    dCHT_dt = np.gradient(df['CHT'], df['Time'])
    df['RUL_Seconds'] = np.where(dCHT_dt > 0.005, (135.0 - df['CHT']) / np.maximum(dCHT_dt, 1e-3), 9999.0)
    df['RUL_Seconds'] = np.clip(df['RUL_Seconds'], -999, 9999)

    return df

if __name__ == "__main__":
    df_proc = process_digital_twin_analytics()
    df_proc.to_csv('digital_twin_processed_analytics.csv', index=False)
    print(f"✅ Physics-Corrected Analytics Complete. Processed {len(df_proc)} rows.")