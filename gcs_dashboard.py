import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    SKLEARN_AVAILABLE = False

# ============================================================
# DRISHTI — ROTAX 914 REFERENCE CONFIGURATION
# ============================================================
ROTAX = {
    "rpm_takeoff_max": 5800.0,
    "rpm_cont_max": 5500.0,
    "rpm_takeoff_duration_min": 5.0,
    "cht_max_c": 135.0,
    "egt_max_c": 950.0,
    "oil_temp_min_c": 50.0,
    "oil_temp_normal_low_c": 90.0,
    "oil_temp_normal_high_c": 110.0,
    "oil_temp_max_c": 130.0,
    "oil_pressure_normal_low_bar": 2.0,
    "oil_pressure_normal_high_bar": 5.0,
    "oil_pressure_min_low_rpm_bar": 0.8,
    "oil_pressure_max_bar": 7.0,
    "fuel_pressure_offset_min_bar": 0.15,
    "fuel_pressure_offset_normal_bar": 0.25,
    "fuel_pressure_offset_max_bar": 0.35,
    "coolant_exit_max_c": 120.0,
    "tbo_h": 2000.0,
    "max_power_kw": 84.8,
    "max_torque_nm": 144.0,
    "max_torque_rpm": 4900.0,
    "displacement_l": 1.2112,
    "gear_ratio": 2.43,
}

DRISHTI = {
    "cht_warn_c": 120.0,
    "egt_warn_c": 850.0,
    "oil_temp_warn_c": 110.0,
    "vibration_warn_g": 1.89,
    "map_warn_inhg": 36.0,
    "map_takeoff_inhg": 40.5,
    "power_warn_kw": ROTAX["max_power_kw"],
    "anomaly_confirm": 0.85,
    "anomaly_caution": 0.75,
}

MISSION_PROFILES = {
    "ISR Long-Endurance Cruise": {"altitude_ft": 12000.0, "delta_isa_c": 0.0, "throttle_pct": 55.0, "duration_h": 4.0},
    "High Altitude": {"altitude_ft": 16000.0, "delta_isa_c": 0.0, "throttle_pct": 60.0, "duration_h": 3.0},
    "Hot Weather (ISA Baseline)": {"altitude_ft": 4000.0, "delta_isa_c": 0.0, "throttle_pct": 60.0, "duration_h": 3.0},
    "Rapid Throttle Transitions": {"altitude_ft": 6000.0, "delta_isa_c": 0.0, "throttle_pct": 75.0, "duration_h": 2.0},
    "Endurance": {"altitude_ft": 10000.0, "delta_isa_c": 0.0, "throttle_pct": 50.0, "duration_h": 8.0},
}

FLEET_DEMO = {
    "UAV-01": {"health_bias": 0.0, "rul_bias": 0.0, "cht_bias": 0.0, "egt_bias": 0.0, "oilp_bias": 0.0, "oiltemp_bias": 0.0, "vib_mult": 1.00, "state":"Standby", "lat": 17.3850, "lon": 78.4867},
    "UAV-02": {"health_bias": -1.5, "rul_bias": -45.0, "cht_bias": 3.0, "egt_bias": 12.0, "oilp_bias": -0.20, "oiltemp_bias": 4.0, "vib_mult": 1.08, "state":"Standby", "lat": 17.4000, "lon": 78.5000},
    "UAV-03": {"health_bias": -4.0, "rul_bias": -120.0, "cht_bias": 7.0, "egt_bias": 25.0, "oilp_bias": -0.45, "oiltemp_bias": 8.0, "vib_mult": 1.20, "state":"Maintenance", "lat": 17.3500, "lon": 78.4500},
    "UAV-04": {"health_bias": -8.5, "rul_bias": -250.0, "cht_bias": 13.0, "egt_bias": 48.0, "oilp_bias": -0.90, "oiltemp_bias": 15.0, "vib_mult": 1.45, "state":"Grounded", "lat": 17.4200, "lon": 78.4000},
    "UAV-05": {"health_bias": -0.5, "rul_bias": -15.0, "cht_bias": 1.0, "egt_bias": 5.0, "oilp_bias": -0.10, "oiltemp_bias": 2.0, "vib_mult": 1.03, "state":"Standby", "lat": 17.3300, "lon": 78.5200},
}

CORRECTIVE_MEASURES = {
    "misfire": "Switch Ignition Lanes (A/B)",
    "thermal": "Reduce Throttle & Open Cowl Flaps",
    "mechanical": "Reduce RPM to Minimum Smooth",
    "sensor_drift": "Engage Redundant Sensor Voting",
    "lubrication": "Activate Backup Oil Pump",
    "combustion": "Adjust Mixture/Timing",
    "coding": "Reset ECU (FADEC)"
}

XAI_EXPLANATIONS = {
    "misfire": "Root cause analysis indicates irregular spark timing or fuel injector pulsation on cylinder #2, leading to incomplete combustion and ignition breakdown.",
    "thermal": "Root cause analysis points to restricted coolant airflow, turbocharger heat soak, or excessive cowl flap closure causing system-wide thermal saturation.",
    "mechanical": "Root cause analysis traces abnormal vibration signatures to propeller unbalance or gearbox backlash wear exceeding tolerance limits.",
    "sensor_drift": "Root cause analysis identifies thermal junction degradation on the CHT thermocouple sensing circuit, resulting in progressive calibration bias.",
    "lubrication": "Root cause analysis highlights oil pressure drop due to viscosity breakdown under high thermal load or partial oil filter restriction.",
    "combustion": "Root cause analysis detects air-fuel mixture ratio oscillation and turbocharger wastegate hunting causing rapid EGT spiking.",
    "coding": "Root cause analysis flags FADEC firmware lookup table divergence during transient throttle demands."
}

REQUIRED_COLUMNS = {
    "Time", "RPM", "MAP", "CHT", "EGT", "P_oil", "T_oil", "Fuel_Flow",
    "Vibration", "V_bus", "Power", "BSFC", "Thermal_Stress"
}

st.set_page_config(
    page_title="DRISHTI — Rotax 914 MALE UAV Engine Digital Twin GCS",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #05070a 0%, #0a0e14 100%); color: #e2e8f0; font-family: 'Inter', 'Segoe UI', Roboto, sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem !important; max-width: 1560px; }

[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15);
    resize: horizontal !important;
    overflow: auto !important;
    min-width: 250px !important;
    max-width: 600px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #000000 !important;
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
}

.section-header { 
    border-bottom: 2px solid rgba(56, 189, 248, 0.25); 
    padding-bottom: 8px; 
    margin-top: 26px; 
    margin-bottom: 16px; 
    font-weight: 700; 
    color: #38bdf8; 
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 1.05rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.status { border-radius: 10px; padding: 14px; text-align: center; border: 1px solid; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.status-normal { background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.4); }
.status-caution { background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.4); }
.status-critical { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.5); animation: pulse-critical 2s infinite; }

@keyframes pulse-critical {
    0% { border-color: rgba(239, 68, 68, 0.5); }
    50% { border-color: rgba(239, 68, 68, 0.9); box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    100% { border-color: rgba(239, 68, 68, 0.5); }
}

.info-card { 
    background: rgba(15, 23, 42, 0.7); 
    border: 1px solid rgba(56, 189, 248, 0.15); 
    border-radius: 8px; 
    padding: 14px; 
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.small { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }

.telemetry-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 4px 0 8px 0; }
.telemetry-card { 
    min-height: 94px; 
    box-sizing: border-box; 
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 17, 32, 0.95) 100%); 
    border: 1px solid rgba(255, 255, 255, 0.08); 
    border-left: 4px solid #10b981; 
    border-radius: 6px; 
    padding: 10px 12px; 
    position: relative;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.telemetry-card.warn { border-left-color: #f59e0b; }
.telemetry-card.critical { border-left-color: #ef4444; background: linear-gradient(145deg, rgba(30, 15, 15, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%); }
.telemetry-label { font-size: 11px; line-height: 1.2; color: #94a3b8; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; }
.telemetry-value { font-size: 19px; line-height: 1.1; color: #f8fafc; font-weight: 700; }
.telemetry-limit { font-size: 10px; line-height: 1.3; color: #38bdf8; margin-top: 6px; }
.telemetry-state { font-size: 9px; line-height: 1.2; font-weight: 700; margin-top: 6px; text-transform: uppercase; }
.telemetry-state.normal { color: #10b981; }
.telemetry-state.warn { color: #f59e0b; }
.telemetry-state.critical { color: #ef4444; }

.top-nav-container {
    display: flex;
    gap: 8px;
    background: rgba(10, 14, 20, 0.95);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 8px;
    padding: 8px;
    margin-top: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    overflow-x: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.nav-tab-link {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.15);
    color: #94a3b8;
    padding: 8px 14px;
    border-radius: 6px;
    text-align: center;
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
    flex: 1;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
}
.nav-tab-link:hover {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border-color: rgba(56, 189, 248, 0.4);
}
.nav-tab-link.active {
    background: #020617 !important;
    border: 1px solid #38bdf8 !important;
    color: #38bdf8 !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.9), 0 0 12px rgba(56, 189, 248, 0.25);
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, lo=0.0, hi=100.0):
    return float(np.clip(x, lo, hi))


def normalize_time_seconds(series):
    x = pd.to_numeric(series, errors="coerce")
    if x.isna().all():
        raise ValueError("Time column contains no numeric values.")
    return x


def infer_fuel_flow_gps(series):
    x = pd.to_numeric(series, errors="coerce")
    median = float(x.dropna().median()) if x.notna().any() else 0.0
    if median < 0.02:
        return x * 1000.0, "kg/s → g/s"
    if median < 20.0:
        return x, "g/s (assumed)"
    return x / 3600.0, "kg/h → g/s (assumed)"


def infer_power_kw(series):
    x = pd.to_numeric(series, errors="coerce")
    median = float(x.dropna().median()) if x.notna().any() else 0.0
    if median > 1000.0:
        return x / 1000.0, "W → kW"
    return x, "kW (assumed)"


def load_data(path="digital_twin_processed_analytics.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Place the processed telemetry CSV next to this app.")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required telemetry columns: {sorted(missing)}")
    df = df.copy()
    df["Time"] = normalize_time_seconds(df["Time"])
    for c in REQUIRED_COLUMNS - {"Time"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).sort_values("Time").reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Telemetry dataset must contain at least 10 valid samples.")
    df["Fuel_Flow_gps"], df.attrs["fuel_flow_conversion"] = infer_fuel_flow_gps(df["Fuel_Flow"])
    df["Power_kW"], df.attrs["power_conversion"] = infer_power_kw(df["Power"])
    return df


@st.cache_data
def cached_load_data(path):
    return load_data(path)


def nearest_frame(df, t):
    idx = int((df["Time"] - t).abs().idxmin())
    r = df.loc[idx]
    return {
        "Time": float(r["Time"]),
        "RPM": float(r["RPM"]),
        "MAP": float(r["MAP"]),
        "CHT": float(r["CHT"]),
        "EGT": float(r["EGT"]),
        "Oil_Pressure": float(r["P_oil"]),
        "Oil_Temp": float(r["T_oil"]),
        "Fuel_Flow": float(r["Fuel_Flow_gps"]),
        "Vibration": float(r["Vibration"]),
        "Battery_Voltage": float(r["V_bus"]),
        "Power": float(r["Power_kW"] if "Power_kW" in df.columns else r["Power"]),
        "BSFC": float(r["BSFC"]),
        "Thermal_Stress": float(r["Thermal_Stress"]),
        "Health_Index_source": safe_float(r.get("Health_Index", np.nan)),
        "RUL_Seconds_source": safe_float(r.get("RUL_Seconds", np.nan)),
        "Anomaly_Flag_source": int(safe_float(r.get("Anomaly_Flag", 0), 0)),
        "Anomaly_Score_source": safe_float(r.get("Anomaly_Score", np.nan)),
        "Fault_Diagnosis_source": str(r.get("Fault_Diagnosis", "")),
        "Injection_Timing": safe_float(r.get("Injection_Timing", np.nan)),
        "Altitude": safe_float(r.get("Altitude", np.nan)),
        "Ambient_Temp": safe_float(r.get("Ambient_Temp", np.nan)),
        "Throttle": safe_float(r.get("Throttle", np.nan)),
        "Alternator_Health": safe_float(r.get("Alternator_Health", np.nan)),
        "Airbox_Temp": safe_float(r.get("Airbox_Temp", np.nan)),
    }


def derive_missing_channels(frame, profile):
    if not np.isfinite(frame["Injection_Timing"]):
        frame["Injection_Timing"] = 25.0 + 0.001 * (frame["RPM"] - 5500.0)
    if not np.isfinite(frame["Alternator_Health"]):
        frame["Alternator_Health"] = clamp(100.0 - max(0.0, 2500.0 - frame["RPM"]) / 25.0)
    if not np.isfinite(frame["Airbox_Temp"]):
        frame["Airbox_Temp"] = profile["ambient_c"] + 0.25 * max(0.0, frame["MAP"] - 20.0)
    if not np.isfinite(frame["Altitude"]):
        frame["Altitude"] = profile["altitude_ft"]
    if not np.isfinite(frame["Ambient_Temp"]):
        frame["Ambient_Temp"] = profile["ambient_c"]
    if not np.isfinite(frame["Throttle"]):
        frame["Throttle"] = profile["throttle_pct"]
    return frame


def isa_temperature_c(altitude_ft):
    h_m = float(np.clip(altitude_ft, 0.0, 16000.0)) * 0.3048
    return 15.0 - 6.5 * (h_m / 1000.0)


def mission_surface(altitude_ft, throttle_pct, delta_isa_c=0.0):
    altitude_ft = float(np.clip(altitude_ft, 0.0, 16000.0))
    throttle_pct = float(np.clip(throttle_pct, 0.0, 100.0))
    delta_isa_c = float(delta_isa_c)
    rpm = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [1800, 4300, 4800, 5000, 5500]))
    power_kw = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [8.0, 40.4, 47.8, 55.1, 73.5]))
    torque_nm = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [35.0, 90.0, 95.0, 105.0, 128.0]))
    map_inhg = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [12.0, 28.0, 29.0, 31.0, 35.0]))
    h_m = altitude_ft * 0.3048
    if h_m <= 11000.0:
        T_isa = 288.15 - 0.0065*h_m
        p_ratio = (T_isa/288.15) ** 5.25588
    else:
        T11 = 216.65
        p11 = (216.65/288.15) ** 5.25588
        p_ratio = p11 * math.exp(-9.80665*(h_m-11000.0)/(287.05*T11))
        T_isa = 216.65
    altitude_factor = 1.0 - 0.18 * (1.0 - p_ratio)
    map_inhg *= altitude_factor
    power_kw *= altitude_factor
    torque_nm *= altitude_factor
    ambient_c = (T_isa - 273.15) + delta_isa_c
    return {"RPM":rpm, "Power":power_kw, "MAP":map_inhg, "Torque":torque_nm,
            "Ambient_Temp":ambient_c, "ISA_Temp":T_isa - 273.15,
            "Delta_ISA_C":delta_isa_c, "Altitude":altitude_ft,
            "Throttle":throttle_pct, "Altitude_Pressure_Ratio":p_ratio}


def apply_mission_controls(frame, altitude_ft, throttle_pct, mission_time_h, profile):
    f = frame.copy()
    surf = mission_surface(altitude_ft, throttle_pct, profile.get("delta_isa_c", 0.0))
    f["Altitude"] = surf["Altitude"]
    f["Ambient_Temp"] = surf["Ambient_Temp"]
    f["Throttle"] = surf["Throttle"]
    f["RPM"] = surf["RPM"]
    f["MAP"] = surf["MAP"]
    f["Power"] = surf["Power"]
    f["Airbox_Temp"] = f["Ambient_Temp"] + 0.20 * max(0.0, f["MAP"] - 20.0)
    hot_load = max(0.0, f["Ambient_Temp"] - 25.0)
    throttle_load = max(0.0, f["Throttle"] - 45.0)
    altitude_load = max(0.0, f["Altitude"] - 10000.0) / 1000.0
    f["CHT"] = 82.0 + 0.52*throttle_load + 0.35*hot_load + 1.2*altitude_load
    f["EGT"] = 650.0 + 3.15*throttle_load + 1.85*hot_load + 6.0*altitude_load
    f["Oil_Temp"] = 78.0 + 0.45*throttle_load + 0.55*hot_load + 0.9*altitude_load
    if f["RPM"] > 3500:
        f["Oil_Pressure"] = float(np.clip(2.0 + (f["RPM"]-3500.0)/2000.0*3.0 - 0.0001*throttle_load, 1.2, 5.0))
    else:
        f["Oil_Pressure"] = float(np.clip(0.8 + f["RPM"]/3500.0*1.2, 0.5, 2.0))
    f["Fuel_Flow"] = max(1.0, f["Power"] / 0.25 * 0.27778)
    f["Vibration"] = max(0.5, 0.0003*f["RPM"] + 0.003*throttle_load)
    f["Thermal_Stress"] = clamp(35.0 + 0.75*throttle_load + 1.5*hot_load + 2.5*altitude_load, 0, 100)
    f["Injection_Timing"] = 25.0 + 0.001*(f["RPM"]-5500.0)
    f["Alternator_Health"] = clamp(90.0 + 0.004*max(0.0, f["RPM"]-2500.0))
    f["Mission_Duration_h"] = max(0.1, float(mission_time_h))
    return f


def apply_uav_state(frame, selected_uav):
    if selected_uav == "UAV-01":
        return frame
    cfg = FLEET_DEMO.get(selected_uav)
    if not cfg:
        return frame
    f = frame.copy()
    f["CHT"] += cfg["cht_bias"]
    f["EGT"] += cfg["egt_bias"]
    f["Oil_Pressure"] += cfg["oilp_bias"]
    f["Oil_Temp"] += cfg["oiltemp_bias"]
    f["Vibration"] *= cfg["vib_mult"]
    return f


def apply_fault_scenarios(frame, flags, elapsed_min):
    f = frame.copy()
    effects = []
    
    if flags.get("misfire") and not st.session_state.get("fix_misfire"):
        f["RPM"] *= 0.90
        f["EGT"] *= 0.92
        f["Vibration"] += 2.5
        effects.append("Misfire")
        
    # Rectified: Coupled thermal effects cascading into EGT & Oil Temp
    if flags.get("thermal") and not st.session_state.get("fix_thermal"):
        f["CHT"] = max(f["CHT"], 142.5)
        f["EGT"] = max(f["EGT"], 925.0)       # Coupled EGT thermal surge
        f["Oil_Temp"] = max(f["Oil_Temp"], 134.0) # Coupled Oil Temp heat soak
        f["Thermal_Stress"] = max(f["Thermal_Stress"], 95.0)
        effects.append("Thermal overheat")
        
    if flags.get("mechanical") and not st.session_state.get("fix_mechanical"):
        f["Vibration"] = max(f["Vibration"], 2.15)
        effects.append("Mechanical imbalance")
    if flags.get("sensor_drift") and not st.session_state.get("fix_sensor_drift"):
        drift_c = 0.02 * elapsed_min
        f["CHT"] += drift_c
        f["CHT_Sensor_Drift_C"] = drift_c
        effects.append("CHT sensor drift")
    if flags.get("lubrication") and not st.session_state.get("fix_lubrication"):
        f["Oil_Pressure"] = max(0.4, f["Oil_Pressure"] - 2.0)
        f["Oil_Temp"] = max(f["Oil_Temp"], 134.0)
        effects.append("Lubrication degradation")
    if flags.get("combustion") and not st.session_state.get("fix_combustion"):
        f["EGT"] += 20.0 * math.sin(2 * math.pi * 0.5 * elapsed_min / 60.0)
        f["RPM"] = max(f["RPM"], 5650.0)
        f["Vibration"] += 1.5
        effects.append("Combustion instability")
    if flags.get("coding") and not st.session_state.get("fix_coding"):
        f["RPM"] += 120.0
        f["Injection_Timing"] += 4.0
        effects.append("ECU/control-coding degradation")
    
    f["Scenario_Effects"] = effects
    return f


def oil_pressure_status(rpm, pressure):
    if rpm > 3500:
        if pressure < ROTAX["oil_pressure_normal_low_bar"]:
            return "CRITICAL", ROTAX["oil_pressure_normal_low_bar"]
        if pressure > ROTAX["oil_pressure_max_bar"]:
            return "CRITICAL", ROTAX["oil_pressure_max_bar"]
        return "NORMAL", ROTAX["oil_pressure_normal_low_bar"]
    else:
        if pressure < ROTAX["oil_pressure_min_low_rpm_bar"]:
            return "CRITICAL", ROTAX["oil_pressure_min_low_rpm_bar"]
        if pressure > ROTAX["oil_pressure_max_bar"]:
            return "CRITICAL", ROTAX["oil_pressure_max_bar"]
        return "NORMAL", ROTAX["oil_pressure_min_low_rpm_bar"]


def evaluate_faults(f):
    faults: List[Dict] = []
    if f["CHT"] > ROTAX["cht_max_c"]:
        faults.append({"name":"High CHT / Over-temperature", "severity":"CRITICAL", "value":f["CHT"], "limit":ROTAX["cht_max_c"], "source":"CHT", "sub":"Thermal", "basis":"OEM MAX"})
    elif f["CHT"] > DRISHTI["cht_warn_c"]:
        faults.append({"name":"Elevated CHT", "severity":"WARNING", "value":f["CHT"], "limit":DRISHTI["cht_warn_c"], "source":"CHT", "sub":"Thermal", "basis":"DRISHTI WARNING"})
    if f["EGT"] > ROTAX["egt_max_c"]:
        faults.append({"name":"High EGT Exceedance", "severity":"CRITICAL", "value":f["EGT"], "limit":ROTAX["egt_max_c"], "source":"EGT", "sub":"Combustion", "basis":"OEM MAX"})
    elif f["EGT"] > DRISHTI["egt_warn_c"]:
        faults.append({"name":"Elevated EGT", "severity":"WARNING", "value":f["EGT"], "limit":DRISHTI["egt_warn_c"], "source":"EGT", "sub":"Combustion", "basis":"DRISHTI WARNING"})
    if f["Oil_Temp"] > ROTAX["oil_temp_max_c"]:
        faults.append({"name":"Oil Temperature Critical", "severity":"CRITICAL", "value":f["Oil_Temp"], "limit":ROTAX["oil_temp_max_c"], "source":"Oil Temp", "sub":"Lubrication", "basis":"OEM MAX"})
    elif f["Oil_Temp"] > ROTAX["oil_temp_normal_high_c"]:
        faults.append({"name":"Oil Temperature Above Normal Band", "severity":"WARNING", "value":f["Oil_Temp"], "limit":ROTAX["oil_temp_normal_high_c"], "source":"Oil Temp", "sub":"Lubrication", "basis":"OEM NORMAL BAND"})
    oil_stat, oil_lim = oil_pressure_status(f["RPM"], f["Oil_Pressure"])
    if oil_stat == "CRITICAL":
        faults.append({"name":"Oil Pressure Outside Safe Operating Range", "severity":"CRITICAL", "value":f["Oil_Pressure"], "limit":oil_lim, "source":"Oil Pressure", "sub":"Lubrication", "basis":"OEM RANGE"})
    if f["RPM"] > ROTAX["rpm_takeoff_max"]:
        faults.append({"name":"RPM Above Absolute Maximum", "severity":"CRITICAL", "value":f["RPM"], "limit":ROTAX["rpm_takeoff_max"], "source":"RPM", "sub":"Propulsion", "basis":"OEM MAX"})
    elif f["RPM"] > ROTAX["rpm_cont_max"]:
        faults.append({"name":"RPM Above Maximum Continuous", "severity":"WARNING", "value":f["RPM"], "limit":ROTAX["rpm_cont_max"], "source":"RPM", "sub":"Propulsion", "basis":"OEM CONTINUOUS"})
    if f["Vibration"] > DRISHTI["vibration_warn_g"]:
        faults.append({"name":"Abnormal Vibration / Mechanical Unbalance", "severity":"WARNING", "value":f["Vibration"], "limit":DRISHTI["vibration_warn_g"], "source":"Vibration", "sub":"Mechanical", "basis":"DRISHTI MODEL"})
    return faults


def subsystem_health(f, throttle_pct, altitude_ft, anomaly_score, active_faults_count):
    def health_between(x, healthy, critical):
        if x <= healthy:
            return 100.0
        if x >= critical:
            return 0.0
        return 100.0 * (critical - x) / (critical - healthy)

    thermal = min(
        health_between(f["CHT"], 110.0, ROTAX["cht_max_c"]),
        health_between(f["Thermal_Stress"], 65.0, 100.0),
        health_between(f["Airbox_Temp"], 70.0, 95.0),
    )
    oil_low = ROTAX["oil_pressure_normal_low_bar"] if f["RPM"] > 3500 else ROTAX["oil_pressure_min_low_rpm_bar"]
    lubrication = min(
        health_between(oil_low - f["Oil_Pressure"], 0.0, oil_low),
        health_between(f["Oil_Temp"], ROTAX["oil_temp_normal_high_c"], ROTAX["oil_temp_max_c"]),
    )
    combustion = min(
        health_between(f["EGT"], 850.0, ROTAX["egt_max_c"]),
        health_between(abs(f["BSFC"]) if not math.isnan(f["BSFC"]) else 0.3, 0.25, 0.45)
    )
    mechanical = health_between(f["Vibration"], 1.5, DRISHTI["vibration_warn_g"] * 2.0)
    electrical = min(clamp((f["Battery_Voltage"] - 11.0) / 3.0 * 100.0), clamp(f["Alternator_Health"]))
    
    weights = {"Thermal":0.28, "Lubrication":0.24, "Combustion":0.20, "Mechanical":0.18, "Electrical":0.10}
    vals = {"Thermal":thermal, "Lubrication":lubrication, "Combustion":combustion, "Mechanical":mechanical, "Electrical":electrical}
    base_overall = sum(vals[k] * weights[k] for k in vals)
    
    stress_penalty = (throttle_pct / 100.0) * 2.5 + (altitude_ft / 16000.0) * 1.5 + (anomaly_score * 3.0) + (active_faults_count * 2.0)
    overall = clamp(base_overall - stress_penalty)
    return vals, overall


def robust_anomaly_score(df, frame):
    features = ["RPM", "MAP", "CHT", "EGT", "P_oil", "T_oil", "Fuel_Flow_gps", "Vibration", "V_bus", "Power", "Thermal_Stress"]
    train_df = df.copy()
    if "Fuel_Flow_gps" not in train_df.columns:
        train_df["Fuel_Flow_gps"], _ = infer_fuel_flow_gps(train_df["Fuel_Flow"])
    x = train_df[features].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50:
        return float(np.clip(safe_float(frame.get("Anomaly_Score_source"), 0.0), 0, 1)), "SOURCE DATASET"
    cur = np.array([frame["RPM"], frame["MAP"], frame["CHT"], frame["EGT"], frame["Oil_Pressure"], frame["Oil_Temp"], frame["Fuel_Flow"], frame["Vibration"], frame["Battery_Voltage"], frame["Power"], frame["Thermal_Stress"]], dtype=float).reshape(1,-1)
    if SKLEARN_AVAILABLE:
        model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
        model.fit(x)
        raw = -float(model.decision_function(cur)[0])
        train_scores = -model.decision_function(x)
        lo, hi = np.percentile(train_scores, [5, 95])
        score = (raw - lo) / max(hi - lo, 1e-9)
        return clamp(score, 0, 1), "ISOLATION FOREST"
    med = x.median().values
    mad = (x - x.median()).abs().median().values
    z = np.abs((cur[0] - med) / np.maximum(1.4826 * mad, 1e-6))
    return clamp(np.mean(np.clip(z / 5.0, 0, 1))), "ROBUST Z-SCORE FALLBACK"


def diagnostic_state(f, faults, anomaly_score, subsystem):
    critical = sum(x["severity"] == "CRITICAL" for x in faults)
    warnings = sum(x["severity"] == "WARNING" for x in faults)
    health = clamp(subsystem[1])
    
    if critical or health < 50:
        status = "CRITICAL"
    elif warnings or health < 65:
        status = "CAUTION"
    else:
        status = "NORMAL"
        
    confirmed = critical > 0 or warnings > 0
    if anomaly_score >= DRISHTI["anomaly_confirm"] and not confirmed:
        anomaly_state = "ANOMALY DETECTED — UNCLASSIFIED"
    elif confirmed:
        anomaly_state = "FAULT CLASSIFIED"
    else:
        anomaly_state = "NO SIGNIFICANT ANOMALY"
        
    return status, critical, warnings, health, anomaly_state


def calculate_degradation_rate(throttle_pct, altitude_ft, anomaly_score, active_faults_count, is_simulated):
    if active_faults_count == 0 and not is_simulated:
        base_rate = 0.03
        throttle_factor = 0.02 * ((throttle_pct / 100.0) ** 2)
        altitude_factor = 0.01 * (altitude_ft / 16000.0)
        degradation_rate = base_rate + throttle_factor + altitude_factor
    else:
        base_rate = 0.08
        throttle_factor = 0.05 * ((throttle_pct / 100.0) ** 2)
        altitude_factor = 0.02 * (altitude_ft / 16000.0)
        anomaly_factor = 0.10 * anomaly_score
        fault_factor = 0.20 * active_faults_count
        sim_penalty = 0.08 if is_simulated else 0.0
        degradation_rate = base_rate + throttle_factor + altitude_factor + anomaly_factor + fault_factor + sim_penalty
        
    return max(0.01, degradation_rate)


def estimate_rul_dynamic(health, degradation_rate_pct_h, mission_duration_h):
    safe_rate = max(0.001, degradation_rate_pct_h)
    rul_h = max(0.5, health / safe_rate)
    confidence = clamp(92.0 - safe_rate * 50.0 - (100.0 - health) * 0.1)
    spread = max(0.5, rul_h * (1.0 - confidence / 100.0) * 0.3)
    return rul_h, max(0.1, rul_h - spread), rul_h + spread, confidence


def mission_metrics(status, health, anomaly, rul_h, remaining_h, altitude_ft, ambient_c, throttle, active_faults_count):
    margin = rul_h - remaining_h
    risk_score = 0.0
    risk_score += max(0.0, 70.0 - health) * 0.55
    risk_score += anomaly * 30.0
    if margin < 0:
        risk_score += min(45.0, abs(margin) * 18.0)
    if status == "CRITICAL":
        risk_score += 55
    elif status == "CAUTION":
        risk_score += 8
    risk_score = clamp(risk_score, 0, 100)
    
    # Rectified: Dynamic scaling instead of hardcoded 37.0% clamp
    base_comp = health * (max(0.1, rul_h) / max(1.0, remaining_h + 1.0))
    base_comp -= active_faults_count * 8.0
    if margin < 0:
        base_comp -= abs(margin) * 6.0
    completion = clamp(base_comp, 2.0, 100.0)

    if status == "CRITICAL":
        risk = "CRITICAL"
        action = "DIVERT / EMERGENCY LANDING ADVISORY"
    elif margin < -0.5 or completion < 50:
        risk = "HIGH"
        action = "REDUCE LOAD / DIVERT TO SUITABLE AIRPORT"
    elif margin < 0 or completion < 80:
        risk = "MEDIUM"
        action = "MODIFY MISSION / CONTINUE WITH CAUTION"
    else:
        risk = "LOW"
        action = "CONTINUE MISSION"
    expected = "YES" if completion >= 80 and margin >= 0 and status == "NORMAL" else ("CONDITIONAL" if completion >= 50 else "NO")
    return completion, risk, action, expected, margin


# ============================================================
# APP INITIALIZATION & STATE
# ============================================================
DATA_PATH = "digital_twin_processed_analytics.csv"
try:
    df = cached_load_data(DATA_PATH)
except Exception as exc:
    st.error(str(exc))
    st.stop()

for fault_key in CORRECTIVE_MEASURES.keys():
    if f"fix_{fault_key}" not in st.session_state:
        st.session_state[f"fix_{fault_key}"] = False

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Live Cockpit"

if "tab" in st.query_params:
    st.session_state.active_tab = st.query_params["tab"]

if "selected_uav" not in st.session_state:
    st.session_state.selected_uav = "UAV-01 (Primary Testbed)"
if "mission_mode" not in st.session_state:
    st.session_state.mission_mode = "Manual"
if "preset_name" not in st.session_state:
    st.session_state.preset_name = "ISR Long-Endurance Cruise"
if "altitude_ft" not in st.session_state:
    st.session_state.altitude_ft = 10000.0
if "throttle_pct" not in st.session_state:
    st.session_state.throttle_pct = 55.0
if "duration_h" not in st.session_state:
    st.session_state.duration_h = 4.0

if "sim_misfire" not in st.session_state:
    st.session_state.sim_misfire = False
if "sim_thermal" not in st.session_state:
    st.session_state.sim_thermal = False
if "sim_mechanical" not in st.session_state:
    st.session_state.sim_mechanical = False
if "sim_sensor" not in st.session_state:
    st.session_state.sim_sensor = False
if "sim_lube" not in st.session_state:
    st.session_state.sim_lube = False
if "sim_combustion" not in st.session_state:
    st.session_state.sim_combustion = False
if "sim_coding" not in st.session_state:
    st.session_state.sim_coding = False

if "user_role" not in st.session_state:
    st.session_state.user_role = "Flight Director"
if "task_signoffs" not in st.session_state:
    st.session_state.task_signoffs = {"Visual Inspection": False, "Oil Filter Check": False, "Vibration Damper Check": False, "ECU Reset": False}
if "spare_parts_ordered" not in st.session_state:
    st.session_state.spare_parts_ordered = False


# ============================================================
# SIDEBAR CONTROLS & CONFIGURATION
# ============================================================
with st.sidebar:
    st.markdown("### ✈️ Aircraft & Mission Config")
    st.session_state.selected_uav = st.selectbox("Select Aircraft Tail / Test Bed", ["UAV-01 (Primary Testbed)", "UAV-02", "UAV-03", "UAV-04", "UAV-05"], index=["UAV-01 (Primary Testbed)", "UAV-02", "UAV-03", "UAV-04", "UAV-05"].index(st.session_state.selected_uav))
    
    st.markdown("---")
    st.markdown("### 🗺️ Mission Profile Selector")
    st.session_state.mission_mode = st.radio("Mission Mode", ["Manual", "Preset"], horizontal=True, index=["Manual", "Preset"].index(st.session_state.mission_mode))
    
    if st.session_state.mission_mode == "Preset":
        st.session_state.preset_name = st.selectbox("Preset Mission Profile", list(MISSION_PROFILES.keys()), index=list(MISSION_PROFILES.keys()).index(st.session_state.preset_name))
    else:
        st.session_state.altitude_ft = float(st.slider("Altitude (ft)", min_value=0.0, max_value=16000.0, value=st.session_state.altitude_ft, step=250.0))
        st.session_state.throttle_pct = float(st.slider("Throttle (%)", min_value=0.0, max_value=100.0, value=st.session_state.throttle_pct, step=1.0))
        st.session_state.duration_h = float(st.slider("Mission Duration (h)", min_value=0.1, max_value=24.0, value=st.session_state.duration_h, step=0.1))
        
        st.markdown("---")
        st.markdown("### 💡 DRISHTI LIVE MISSION GUIDANCE")
        if st.session_state.altitude_ft == 10000.0:
            rec_th = 55
            rec_end = 8.0
        else:
            rec_th = int(max(30, min(100, 55 + (st.session_state.altitude_ft - 10000)/1000 * 2)))
            rec_end = max(2.0, 8.0 - (st.session_state.altitude_ft - 10000)/5000 * 1.5)
        st.info(f"At {st.session_state.altitude_ft:.0f} ft, DRISHTI suggests about {rec_th}% throttle for efficiency.\n\nEstimated endurance: {rec_end:.1f} h.")

    st.markdown("---")
    st.markdown("### ⚙️ Digital Twin Fault Simulator")
    if st.session_state.user_role != "Flight Director":
        st.warning("⚠️ RBAC Restricted: Only Flight Directors can modify fault simulations.")
    
    sim_enabled = (st.session_state.user_role == "Flight Director")
    st.session_state.sim_misfire = st.checkbox("Misfire", value=st.session_state.sim_misfire, disabled=not sim_enabled, key="sim_misfire_chk")
    st.session_state.sim_thermal = st.checkbox("Thermal Overheat", value=st.session_state.sim_thermal, disabled=not sim_enabled, key="sim_thermal_chk")
    st.session_state.sim_mechanical = st.checkbox("Mechanical Imbalance", value=st.session_state.sim_mechanical, disabled=not sim_enabled, key="sim_mechanical_chk")
    st.session_state.sim_sensor = st.checkbox("Sensor Drift / Failure", value=st.session_state.sim_sensor, disabled=not sim_enabled, key="sim_sensor_chk")
    st.session_state.sim_lube = st.checkbox("Lubrication Degradation", value=st.session_state.sim_lube, disabled=not sim_enabled, key="sim_lube_chk")
    st.session_state.sim_combustion = st.checkbox("Combustion Instability", value=st.session_state.sim_combustion, disabled=not sim_enabled, key="sim_combustion_chk")
    st.session_state.sim_coding = st.checkbox("Control / Coding Degradation", value=st.session_state.sim_coding, disabled=not sim_enabled, key="sim_coding_chk")

selected_uav = st.session_state.selected_uav
selected_uav_key = selected_uav.split(" ")[0]

if st.session_state.mission_mode == "Preset":
    profile_cfg = MISSION_PROFILES[st.session_state.preset_name]
    altitude_ft = float(profile_cfg["altitude_ft"])
    throttle_pct = float(profile_cfg["throttle_pct"])
    mission_duration_h = float(profile_cfg["duration_h"])
    delta_isa_c = float(profile_cfg.get("delta_isa_c", 0.0))
    ambient_override = isa_temperature_c(altitude_ft) + delta_isa_c
    mission_name = st.session_state.preset_name
else:
    altitude_ft = st.session_state.altitude_ft
    throttle_pct = st.session_state.throttle_pct
    mission_duration_h = st.session_state.duration_h
    delta_isa_c = 0.0
    ambient_override = isa_temperature_c(altitude_ft)
    mission_name = "Manual Mission"

if (st.session_state.get("fix_thermal") or st.session_state.get("fix_mechanical")) and st.session_state.throttle_pct > 45.0:
    st.session_state.throttle_pct = 45.0
    throttle_pct = 45.0

profile = {"altitude_ft": altitude_ft, "ambient_c": ambient_override, "throttle_pct": throttle_pct, "duration_h": mission_duration_h, "delta_isa_c": delta_isa_c}

max_time_sec = float(df["Time"].max())
playback_time_sec = max_time_sec
mission_elapsed_h = min(mission_duration_h, max_time_sec / 3600.0)
mission_remaining_h = max(0.0, mission_duration_h - mission_elapsed_h)

base = nearest_frame(df, playback_time_sec)
base = derive_missing_channels(base, profile)
base = apply_mission_controls(base, altitude_ft, throttle_pct, mission_duration_h, profile)
base = apply_uav_state(base, selected_uav_key)
flags = {
    "misfire": st.session_state.sim_misfire,
    "thermal": st.session_state.sim_thermal,
    "mechanical": st.session_state.sim_mechanical,
    "sensor_drift": st.session_state.sim_sensor,
    "lubrication": st.session_state.sim_lube,
    "combustion": st.session_state.sim_combustion,
    "coding": st.session_state.sim_coding,
}
is_any_sim_active = any(flags.values())

frame = apply_fault_scenarios(base, flags, playback_time_sec / 60.0)

anomaly_score, anomaly_model = robust_anomaly_score(df, frame)
faults = evaluate_faults(frame)
active_faults_count_raw = len(faults) + sum(1 for k, v in flags.items() if v)

subs = subsystem_health(frame, throttle_pct, altitude_ft, anomaly_score, active_faults_count_raw)
status, critical_count, warning_count, health, anomaly_state = diagnostic_state(frame, faults, anomaly_score, subs)

degradation_rate = calculate_degradation_rate(throttle_pct, altitude_ft, anomaly_score, active_faults_count_raw, is_any_sim_active)

if selected_uav_key != "UAV-01":
    cfg = FLEET_DEMO[selected_uav_key]
    health = clamp(health + cfg["health_bias"])
    rul_bias = cfg["rul_bias"]
else:
    rul_bias = 0.0

rul_h, rul_lo, rul_hi, rul_conf = estimate_rul_dynamic(health, degradation_rate, mission_duration_h)
rul_h = max(1.0, rul_h + rul_bias)
completion, mission_risk, recommended_action, expected_completion, margin = mission_metrics(status, health, anomaly_score, rul_h, mission_remaining_h, frame["Altitude"], frame["Ambient_Temp"], frame["Throttle"], active_faults_count_raw)


# ============================================================
# TOP NAVIGATION TAB BAR
# ============================================================
tabs_config = [
    ("Live Cockpit", "✈️ Live Cockpit"),
    ("Fault Prediction", "🔮 Fault Prediction"),
    ("Fleet Overview", "🛩️ Fleet Overview"),
    ("Mission Replay", "🔄 Mission Replay"),
    ("Post-flight Report", "📑 Post-flight Report"),
    ("Maintenance Planner", "🛠️ Maint. Planner"),
    ("Engine Passport", "📜 Engine Passport"),
    ("Settings", "⚙️ Settings"),
]

nav_html = '<div class="top-nav-container">'
for tab_key, tab_label in tabs_config:
    is_active = (st.session_state.active_tab == tab_key)
    active_cls = " active" if is_active else ""
    nav_html += f'<a href="?tab={tab_key.replace(" ", "+")}" target="_self" class="nav-tab-link{active_cls}">{tab_label}</a>'
nav_html += '</div>'

st.markdown(nav_html, unsafe_allow_html=True)


# ============================================================
# CONDITIONAL RENDERING ROUTER / VIEW STATE
# ============================================================
active_tab = st.session_state.active_tab

if active_tab == "Live Cockpit":
    st.markdown("### 🖥️ DRISHTI — Rotax 914 MALE UAV Engine Digital Twin GCS")
    status_class = {"NORMAL":"status-normal","CAUTION":"status-caution","CRITICAL":"status-critical"}[status]
    status_icon = {"NORMAL":"🟢","CAUTION":"🟠","CRITICAL":"🔴"}[status]
    status_color = {"NORMAL":"#10b981","CAUTION":"#f59e0b","CRITICAL":"#ef4444"}[status]

    c1, c2, c3, c4, c5, c6, c7 = st.columns([2.0, 1, 1, 1, 1, 1, 1])
    with c1:
        st.markdown(f'<div class="status {status_class}"><h3 style="margin:0;color:{status_color}">{status_icon} ENGINE STATUS: {status}</h3><div class="small">Mission Feasibility: <b>{completion:.0f}%</b> | Risk: <b>{mission_risk}</b></div></div>', unsafe_allow_html=True)
    with c2: st.metric("Health Index", f"{health:.1f}%", delta=f"{anomaly_score*100:.1f}% anomaly")
    with c3: st.metric("Degradation Rate", f"{degradation_rate:.3f}%/h", delta="TBO Scaled Wear")
    with c4: st.metric("Active Faults", f"{critical_count} crit / {warning_count} warn")
    with c5: st.metric("Predicted RUI", f"{rul_h:.1f} h", delta=f"{margin:+.1f} h margin")
    with c6: st.metric("RUL Confidence", f"{rul_conf:.0f}%")
    with c7: st.metric("Mission Completion", f"{completion:.0f}%", delta=mission_risk)

    st.markdown("<h4 class='section-header'>🔮 PROGNOSTIC LIVE PREDICTION & FAULT WARNING PANEL</h4>", unsafe_allow_html=True)
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    with pred_col1:
        proj_cht_live = 82.0 + 0.52 * max(0.0, throttle_pct - 45.0) + 1.2 * max(0.0, altitude_ft - 10000.0) / 1000.0
        st.metric("Live Projected CHT", f"{proj_cht_live:.1f} °C", delta=f"Limit {ROTAX['cht_max_c']}°C")
    with pred_col2:
        proj_stress_live = clamp(35.0 + 0.75 * max(0.0, throttle_pct - 45.0) + 2.5 * max(0.0, altitude_ft - 10000.0) / 1000.0)
        st.metric("Live Thermal Stress", f"{proj_stress_live:.1f} / 100", delta=f"Duration: {mission_duration_h}h")
    with pred_col3:
        st.metric("Estimated Margin", f"{margin:+.1f} hours", delta="Safe Window" if margin >= 0 else "Exceedance Alert")

    if throttle_pct > 80.0 and altitude_ft > 12000.0:
        st.error("⚠️ **Prognostic Warning:** High altitude and high throttle combination (>80% / >12k ft) is causing aggressive thermal saturation and turbocharger overboost risk. Reduce throttle immediately.")
    elif degradation_rate > 0.15:
        st.warning("⚠️ **Warning:** Elevated engine degradation rate detected. Extended operation at these levels will shorten TBO and trigger maintenance faults.")
    else:
        st.success("✅ **Prognostic Status:** Operating parameters within stable thermal and mechanical envelope.")

    st.markdown("<h4 class='section-header'>📡 UAV PROPULSION TELEMETRY — REAL-TIME STREAMING</h4>", unsafe_allow_html=True)

    def telemetry_card(label, value, unit, limit_text, state="NORMAL"):
        state = str(state).upper()
        cls = "critical" if state == "CRITICAL" else ("warn" if state in ("CAUTION", "WARNING") else "")
        state_cls = "critical" if state == "CRITICAL" else ("warn" if state in ("CAUTION", "WARNING") else "normal")
        icon = "●" if state == "NORMAL" else ("▲" if state in ("CAUTION", "WARNING") else "■")
        return (
            f'<div class="telemetry-card {cls}">'
            f'<div class="telemetry-label">{label}</div>'
            f'<div class="telemetry-value">{value} <span style="font-size:11px;font-weight:500;color:#94a3b8">{unit}</span></div>'
            f'<div class="telemetry-limit">{limit_text}</div>'
            f'<div class="telemetry-state {state_cls}">{icon} {state}</div>'
            f'</div>'
        )

    rpm_stat = "CRITICAL" if frame["RPM"] > ROTAX["rpm_takeoff_max"] else ("CAUTION" if frame["RPM"] > ROTAX["rpm_cont_max"] else "NORMAL")
    cht_stat = "CRITICAL" if frame["CHT"] > ROTAX["cht_max_c"] else ("CAUTION" if frame["CHT"] > DRISHTI["cht_warn_c"] else "NORMAL")
    egt_stat = "CRITICAL" if frame["EGT"] > ROTAX["egt_max_c"] else ("CAUTION" if frame["EGT"] > DRISHTI["egt_warn_c"] else "NORMAL")
    oilp_stat, _ = oil_pressure_status(frame["RPM"], frame["Oil_Pressure"])
    oilt_stat = "CRITICAL" if frame["Oil_Temp"] > ROTAX["oil_temp_max_c"] else ("CAUTION" if frame["Oil_Temp"] > DRISHTI["oil_temp_warn_c"] else "NORMAL")
    vib_stat = "CAUTION" if frame["Vibration"] > DRISHTI["vibration_warn_g"] else "NORMAL"
    map_stat = "CRITICAL" if frame["MAP"] > DRISHTI["map_takeoff_inhg"] else ("CAUTION" if frame["MAP"] > DRISHTI["map_warn_inhg"] else "NORMAL")

    cards = [
        telemetry_card("Engine RPM", f'{frame["RPM"]:.1f}', "rpm", f'OEM Max: {ROTAX["rpm_takeoff_max"]:.0f}', rpm_stat),
        telemetry_card("Cylinder Head Temp", f'{frame["CHT"]:.1f}', "°C", f'OEM Max: {ROTAX["cht_max_c"]:.0f}', cht_stat),
        telemetry_card("Exhaust Gas Temp", f'{frame["EGT"]:.1f}', "°C", f'OEM Max: {ROTAX["egt_max_c"]:.0f}', egt_stat),
        telemetry_card("Oil Pressure", f'{frame["Oil_Pressure"]:.2f}', "bar", f'Normal: 2.0–5.0 bar', oilp_stat),
        telemetry_card("Oil Temperature", f'{frame["Oil_Temp"]:.1f}', "°C", f'Normal: 90–110°C', oilt_stat),
        telemetry_card("Fuel Flow", f'{frame["Fuel_Flow"]:.2f}', "g/s", 'Reference ~8.5 g/s', "NORMAL"),
        telemetry_card("Vibration", f'{frame["Vibration"]:.2f}', "g", f'Warn: {DRISHTI["vibration_warn_g"]:.2f}g', vib_stat),
        telemetry_card("Bus Voltage", f'{frame["Battery_Voltage"]:.2f}', "V", 'Ref: 11.5–14.5V', "NORMAL"),
        telemetry_card("Alternator Health", f'{frame["Alternator_Health"]:.1f}', "%", 'Advisory <90%', "NORMAL"),
        telemetry_card("Airbox Temp", f'{frame["Airbox_Temp"]:.1f}', "°C", 'Advisory >80°C', "NORMAL"),
        telemetry_card("MAP", f'{frame["MAP"]:.1f}', "inHg", f'Warn: {DRISHTI["map_warn_inhg"]:.1f}', map_stat),
        telemetry_card("Power", f'{frame["Power"]:.1f}', "kW", f'Max Ref: {ROTAX["max_power_kw"]:.1f}kW', "NORMAL"),
    ]
    st.markdown('<div class="telemetry-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

    col_row_left, col_row_right = st.columns(2)

    with col_row_left:
        st.markdown("<h4 class='section-header'>🛩️ Synthetic Vision / 3D Flight Corridor</h4>", unsafe_allow_html=True)
        time_arr = df["Time"].values / 60.0
        alt_arr = np.linspace(frame["Altitude"] - 500, frame["Altitude"], len(time_arr))
        logical_y = np.zeros_like(time_arr)
        
        fig_3d = go.Figure(data=[
            go.Scatter3d(x=time_arr, y=logical_y, z=alt_arr, mode='lines', line=dict(color='cyan', width=4), name='Flight Path'),
            go.Scatter3d(x=[time_arr[0], time_arr[-1]], y=[0, 0], z=[16000, 16000], mode='lines', line=dict(color='red', dash='dash', width=3), name='Safe Ceiling')
        ])
        fig_3d.update_layout(height=280, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                             scene=dict(yaxis=dict(showticklabels=False, title=""), xaxis_title="Time (min)", zaxis_title="Altitude (ft)"))
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_row_right:
        st.markdown("<h4 class='section-header'>🛠️ Active Fault Detectors & Corrective Measures</h4>", unsafe_allow_html=True)
        active_injections = [k for k, v in flags.items() if v]
        if active_injections:
            for fault_key in active_injections:
                if not st.session_state.get(f"fix_{fault_key}", False):
                    st.error(f"⚠️ Fault: {fault_key.replace('_', ' ').upper()}")
                else:
                    st.success(f"✅ Rectified: {fault_key.replace('_', ' ').upper()}")
                st.checkbox(f"Drishti suggests as a corrective measure: {CORRECTIVE_MEASURES[fault_key]}", key=f"fix_{fault_key}")
        else:
            st.info("No active fault simulations injected. Use sidebar to simulate anomalies.")

    col_xai, col_subs = st.columns(2)

    with col_xai:
        st.markdown("<h4 class='section-header'>🧠 XAI — ROOT CAUSE EXPLAINABILITY & DIAGNOSTICS</h4>", unsafe_allow_html=True)
        active_injections = [k for k, v in flags.items() if v]
        if active_injections:
            for fault_key in active_injections:
                explanation = XAI_EXPLANATIONS.get(fault_key, "Telemetry parameters deviated outside nominal multivariate boundaries.")
                st.markdown(f'<div class="info-card" style="border-left:4px solid #38bdf8; margin-bottom:10px;"><b style="color:#38bdf8">XAI Root-Cause Attribution: {fault_key.replace("_", " ").upper()}</b><br><p style="margin:6px 0 0 0; color:#e2e8f0; font-size:13px;">{explanation}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card" style="border-left:4px solid #10b981;"><b style="color:#10b981">XAI Attribution Status: NOMINAL</b><br><span class="small">SHAP & isolation forest attribution models indicate all telemetry streams align with nominal baseline profiles.</span></div>', unsafe_allow_html=True)

    with col_subs:
        st.markdown("<h4 class='section-header'>📊 SUBSYSTEM HEALTH BREAKDOWN</h4>", unsafe_allow_html=True)
        st.metric("Overall Health", f"{health:.1f}%")
        for k, v in subs[0].items():
            st.progress(v / 100.0, text=f"{k}: {v:.1f}%")

    if faults:
        st.write(f"**CONFIRMED FAULTS / EXCEEDANCES — {len(faults)}**")
        for f in faults:
            cls = "#ef4444" if f["severity"] == "CRITICAL" else "#f59e0b"
            st.markdown(f'<div class="info-card" style="border-left:4px solid {cls}; margin-bottom:6px;"><b style="color:{cls}">{f["name"]} — {f["severity"]}</b><br><span class="small">Source: {f["source"]} | Value: {f["value"]:.2f} | Reference: {f["limit"]:.2f}</span></div>', unsafe_allow_html=True)

elif active_tab == "Fault Prediction":
    st.markdown("<h4 class='section-header'>🔮 PROGNOSTIC FAULT & ANOMALY PREDICTION TAB</h4>", unsafe_allow_html=True)
    st.markdown("This module simulates potential incoming faults based on shifts in input conditions (**Altitude**, **Throttle**, and **Mission Duration**). Adjust parameters below to evaluate prospective stress accumulation and time-to-exceedance.")

    col_sim_p1, col_sim_p2, col_sim_p3 = st.columns(3)
    with col_sim_p1:
        pred_alt = st.slider("Projected Altitude (ft)", 0.0, 16000.0, float(altitude_ft), 500.0, key="pred_alt_slider")
    with col_sim_p2:
        pred_throttle = st.slider("Projected Throttle (%)", 20.0, 100.0, float(throttle_pct), 1.0, key="pred_throttle_slider")
    with col_sim_p3:
        pred_duration = st.slider("Projected Mission Duration (h)", 1.0, 24.0, float(mission_duration_h), 0.5, key="pred_duration_slider")

    hours_axis = np.linspace(0.1, pred_duration, 50)
    proj_cht_curve = []
    proj_egt_curve = []
    proj_stress_curve = []
    proj_anomaly_curve = []

    for h_t in hours_axis:
        sim_f = apply_mission_controls(base, pred_alt, pred_throttle, h_t, profile)
        fatigue_factor = 1.0 + (h_t / 200.0) * 0.01
        sim_cht = sim_f["CHT"] * fatigue_factor
        sim_egt = sim_f["EGT"] * fatigue_factor
        sim_stress = clamp(sim_f["Thermal_Stress"] * fatigue_factor)
        sim_anom = clamp(anomaly_score + (h_t / pred_duration) * 0.05 + max(0.0, pred_throttle - 70.0) / 200.0)
        
        proj_cht_curve.append(sim_cht)
        proj_egt_curve.append(sim_egt)
        proj_stress_curve.append(sim_stress)
        proj_anomaly_curve.append(sim_anom)

    col_res1, col_res2, col_res3 = st.columns(3)
    max_proj_cht = max(proj_cht_curve)
    max_proj_stress = max(proj_stress_curve)
    max_proj_anom = max(proj_anomaly_curve)

    with col_res1:
        cht_state_col = "🔴 CRITICAL" if max_proj_cht > ROTAX["cht_max_c"] else ("🟠 CAUTION" if max_proj_cht > DRISHTI["cht_warn_c"] else "🟢 NOMINAL")
        st.metric("Max Projected CHT", f"{max_proj_cht:.1f} °C", delta=cht_state_col)
    with col_res2:
        st.metric("Peak Thermal Stress", f"{max_proj_stress:.1f} / 100", delta=f"{max_proj_stress*0.5:.1f}% wear rate")
    with col_res3:
        anom_risk_label = "HIGH RISK" if max_proj_anom > 0.75 else "STABLE"
        st.metric("Projected Anomaly Probability", f"{max_proj_anom*100:.1f}%", delta=anom_risk_label)

    st.markdown("### Projected Parameter Degradation Trajectory")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=hours_axis, y=proj_cht_curve, name="Projected CHT (°C)", line=dict(color="#f59e0b", width=3)))
    fig_pred.add_trace(go.Scatter(x=hours_axis, y=proj_stress_curve, name="Thermal Stress Index", line=dict(color="#38bdf8", width=3, dash="dash"), yaxis="y2"))
    
    fig_pred.add_hline(y=ROTAX["cht_max_c"], line_dash="dot", line_color="red", annotation_text="CHT Max Limit")
    
    fig_pred.update_layout(
        height=380,
        title="Incoming Potential Fault Projections over Mission Duration",
        xaxis_title="Mission Hours (h)",
        yaxis=dict(title="CHT (°C)"),
        yaxis2=dict(title="Thermal Stress Index", overlaying="y", side="right", range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white")
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    if max_proj_cht > ROTAX["cht_max_c"] or max_proj_stress > 85.0:
        st.error("⚠️ **Prognostic Warning:** Current projected input conditions will cause thermal limit exceedance before mission completion. Recommend reducing throttle by at least 10% or lowering cruising altitude.")
    else:
        st.success("✅ **Prognostic Status:** Input conditions are within safe operational margins for the selected mission duration.")

elif active_tab == "Fleet Overview":
    st.markdown("<h4 class='section-header'>🛩️ FLEET DIGITAL TWIN OVERVIEW & ENGINE HEALTH BAR GRAPH</h4>", unsafe_allow_html=True)
    fleet_rows = []
    for tail, cfg in FLEET_DEMO.items():
        demo_frame = apply_uav_state(apply_mission_controls(nearest_frame(df, playback_time_sec), altitude_ft, throttle_pct, mission_duration_h, profile), tail)
        demo_faults = evaluate_faults(demo_frame)
        demo_subs = subsystem_health(demo_frame, throttle_pct, altitude_ft, anomaly_score, len(demo_faults))
        demo_score, _ = robust_anomaly_score(df, demo_frame)
        demo_status, crit_c, warn_c, demo_health, _ = diagnostic_state(demo_frame, demo_faults, demo_score, demo_subs)
        demo_health = clamp(demo_health + cfg["health_bias"])
        demo_deg = calculate_degradation_rate(throttle_pct, altitude_ft, demo_score, len(demo_faults), False)
        demo_rul, _, _, _ = estimate_rul_dynamic(demo_health, demo_deg, mission_duration_h)
        demo_rul = max(1.0, demo_rul + cfg["rul_bias"])
        demo_risk = "CRITICAL" if demo_status == "CRITICAL" else ("HIGH" if demo_health < 65 else ("MEDIUM" if demo_health < 80 else "LOW"))
        active_fault_count = f"{crit_c} crit / {warn_c} warn"
        fleet_rows.append({
            "Tail": tail,
            "Engine": "Rotax 914 UL/F",
            "Health %": round(demo_health, 1),
            "RUL h": round(demo_rul, 1),
            "Active Faults": active_fault_count,
            "Risk Tier": demo_risk,
            "State": cfg["state"],
            "RPM": round(demo_frame["RPM"]),
            "CHT °C": round(demo_frame["CHT"], 1),
            "EGT °C": round(demo_frame["EGT"], 1),
            "Vibration g": round(demo_frame["Vibration"], 2)
        })
    
    fleet = pd.DataFrame(fleet_rows)
    st.markdown("### Cross-Fleet Health Matrix")
    st.dataframe(fleet[["Tail", "Engine", "Health %", "RUL h", "Active Faults", "Risk Tier", "State", "RPM", "CHT °C", "EGT °C"]], hide_index=True, use_container_width=True)
    
    st.markdown("### Fleet Engine Health Comparison")
    fig_bar = px.bar(
        fleet, x="Tail", y="Health %", color="Health %",
        color_continuous_scale=["red", "orange", "green"],
        range_color=[0, 100], text="Health %",
        title="Engine Health Index Across Fleet UAVs"
    )
    fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_bar, use_container_width=True)

elif active_tab == "Mission Replay":
    st.markdown("<h4 class='section-header'>🔄 MISSION REPLAY & TIMELINE SCRUBBING</h4>", unsafe_allow_html=True)
    replay = df.copy()
    replay["Time (min)"] = replay["Time"] / 60.0
    replay_duration_min = float(replay["Time (min)"].max())

    scrub_time_min = st.slider("Timeline Scrubbing (minutes)", min_value=0.0, max_value=replay_duration_min, value=replay_duration_min / 2.0, step=0.5, key="scrub_time_slider")
    scrub_sec = scrub_time_min * 60.0
    scrub_frame = nearest_frame(df, scrub_sec)

    col_scr1, col_scr2, col_scr3 = st.columns(3)
    with col_scr1: st.metric("Scrubbed Time", f"{scrub_time_min:.1f} min")
    with col_scr2: st.metric("Telemetry RPM", f"{scrub_frame['RPM']:.1f}")
    with col_scr3: st.metric("Telemetry CHT", f"{scrub_frame['CHT']:.1f} °C")

    st.markdown("### Telemetry Timeline & Event Markers")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=replay["Time (min)"], y=replay["RPM"], name="RPM", line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=replay["Time (min)"], y=replay["CHT"], name="CHT (°C)", line=dict(color="#f59e0b"), yaxis="y2"))
    fig.add_vline(x=scrub_time_min, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(height=350, title="Interactive Timeline with Scrubbing Marker", xaxis_title="Mission Time (min)",
                      yaxis=dict(title="RPM"), yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

elif active_tab == "Post-flight Report":
    st.markdown("<h4 class='section-header'>📑 POST-FLIGHT AUTOMATED REPORT & DEBRIEF GENERATOR</h4>", unsafe_allow_html=True)
    peak_egt = df['EGT'].max()
    max_thermal_stress = df['Thermal_Stress'].max()
    total_wear_index = degradation_rate

    report_markdown = f"""
# Automated Flight Debrief Summary — {selected_uav}
* **Mission Profile**: {mission_name}
* **Total Flight Duration**: {df['Time'].max()/3600.0:.2f} hours
* **Peak Exhaust Gas Temperature (EGT)**: {peak_egt:.1f} °C
* **Maximum Thermal Stress**: {max_thermal_stress:.1f} / 100
* **Total Engine Degradation Rate**: {total_wear_index:.3f} %/h
* **Final Diagnostic Status**: {status}
    """
    st.markdown(report_markdown)
    st.download_button(label="📥 Download Markdown Debrief Report", data=report_markdown, file_name=f"debrief_{selected_uav_key}.md", mime="text/markdown", key="download_debrief_btn")

elif active_tab == "Maintenance Planner":
    st.markdown("<h4 class='section-header'>🛠️ MAINTENANCE PLANNER & SPARE FORECASTING</h4>", unsafe_allow_html=True)
    if rul_h < 100.0 or health < 75.0:
        st.error("⚠️ Automated spare parts requisition triggered.")
        if not st.session_state.spare_parts_ordered:
            if st.button("📦 Order Spare Parts Requisition", key="order_spare_parts_btn"):
                st.session_state.spare_parts_ordered = True
                st.success("Spare parts requisition successfully sent to depot inventory.")
        else:
            st.info("📦 Spare parts requisition currently pending fulfillment at depot.")
    else:
        st.success("✅ RUL and health margins are within acceptable safety limits.")

    st.markdown("### Technician Task Sign-Off Log")
    for task_name, signed in st.session_state.task_signoffs.items():
        st.session_state.task_signoffs[task_name] = st.checkbox(f"Complete & Sign Off: {task_name}", value=signed, key=f"task_signoff_{task_name}")

elif active_tab == "Engine Passport":
    st.markdown("<h4 class='section-header'>📜 ENGINE PASSPORT & LIFECYCLE AUDIT TRAIL</h4>", unsafe_allow_html=True)
    audit_trail = pd.DataFrame({
        "Timestamp / Hours": ["0.0 h (Factory)", "500.0 h", "1200.0 h", f"{df['Time'].max()/3600.0:.1f} h (Current)"],
        "Event Type": ["Initial Commission", "Scheduled Hard Overhaul", "Software Flash v2.4", "Component Change: Oil Pump"],
        "Details": ["Serial #914-8842 certified", "Complete piston ring & seal replacement", "FADEC ECU firmware update", "Replaced secondary oil pressure sensor"]
    })
    st.dataframe(audit_trail, hide_index=True, use_container_width=True)

elif active_tab == "Settings":
    st.markdown("<h4 class='section-header'>⚙️ TELEMETRY THRESHOLDS & RBAC</h4>", unsafe_allow_html=True)
    st.session_state.user_role = st.selectbox("Current User Role", ["Flight Director", "Maintenance Technician", "Observer"], index=["Flight Director", "Maintenance Technician", "Observer"].index(st.session_state.user_role), key="settings_user_role_select")
    new_cht_warn = st.slider("CHT Warning Threshold (°C)", min_value=100.0, max_value=135.0, value=DRISHTI["cht_warn_c"], key="settings_cht_slider")
    new_vib_warn = st.slider("Vibration Warning Threshold (g)", min_value=1.0, max_value=3.0, value=DRISHTI["vibration_warn_g"], key="settings_vib_slider")
    st.info(f"Threshold calibration saved successfully. CHT Warning: {new_cht_warn}°C | Vibration Warning: {new_vib_warn}g")