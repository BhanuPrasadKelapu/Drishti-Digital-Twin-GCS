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
# IMPORTANT: Rotax values below are reference limits, not MALE
# UAV-wide limits. The MALE mission envelope is configurable.
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

# Conservative DRISHTI warnings are deliberately below OEM maxima.
DRISHTI = {
    "cht_warn_c": 120.0,
    "egt_warn_c": 850.0,
    "oil_temp_warn_c": 110.0,
    "vibration_warn_g": 1.89,  # engineering/model threshold; NOT Rotax OEM
    "map_warn_inhg": 36.0,      # Rotax max-continuous nominal airbox pressure reference
    "map_takeoff_inhg": 40.5,   # Rotax takeoff nominal airbox pressure reference
    "power_warn_kw": ROTAX["max_power_kw"],
    "anomaly_confirm": 0.70,
    "anomaly_caution": 0.30,
}

# Configurable MALE-class prototype mission envelope.
# These are scenario assumptions, NOT aircraft certification limits.
MISSION_PROFILES = {
    # All presets use ISA-compliant ambient conditions (delta_isa_c = 0).
    # If a hot/cold-day deviation is needed later, model it explicitly as Delta ISA.
    "ISR Long-Endurance Cruise": {"altitude_ft": 12000.0, "delta_isa_c": 0.0, "throttle_pct": 55.0, "duration_h": 4.0},
    "High Altitude": {"altitude_ft": 16000.0, "delta_isa_c": 0.0, "throttle_pct": 60.0, "duration_h": 3.0},
    "Hot Weather (ISA Baseline)": {"altitude_ft": 4000.0, "delta_isa_c": 0.0, "throttle_pct": 60.0, "duration_h": 3.0},
    "Rapid Throttle Transitions": {"altitude_ft": 6000.0, "delta_isa_c": 0.0, "throttle_pct": 75.0, "duration_h": 2.0},
    "Endurance": {"altitude_ft": 10000.0, "delta_isa_c": 0.0, "throttle_pct": 50.0, "duration_h": 8.0},
}

# Reference operating points from the Rotax 914 operator data used elsewhere in DRISHTI.
# These points are used only as a prototype interpolation surface for mission what-if analysis.
ROTAX_MAP = pd.DataFrame([
    [4300.0, 40.4, 90.0, 28.0, 59.0],
    [4800.0, 47.8, 95.0, 29.0, 64.0],
    [5000.0, 55.1, 105.0, 31.0, 67.0],
    [5500.0, 73.5, 128.0, 35.0, 100.0],
    [5800.0, 84.5, 139.0, 39.0, 115.0],
], columns=["rpm","power_kw","torque_nm","map_inhg","throttle_pct"])

# Demonstration fleet health states. These are intentionally separate from UAV-01's live replay.
FLEET_DEMO = {
    "UAV-01": {"health_bias": 0.0, "rul_bias": 0.0, "cht_bias": 0.0, "egt_bias": 0.0, "oilp_bias": 0.0, "oiltemp_bias": 0.0, "vib_mult": 1.00, "state":"Standby"},
    "UAV-02": {"health_bias": -8.0, "rul_bias": -46.0, "cht_bias": 3.0, "egt_bias": 12.0, "oilp_bias": -0.20, "oiltemp_bias": 4.0, "vib_mult": 1.08, "state":"Standby"},
    "UAV-03": {"health_bias": -22.0, "rul_bias": -119.0, "cht_bias": 7.0, "egt_bias": 25.0, "oilp_bias": -0.45, "oiltemp_bias": 8.0, "vib_mult": 1.20, "state":"Maintenance"},
    "UAV-04": {"health_bias": -46.0, "rul_bias": -182.0, "cht_bias": 13.0, "egt_bias": 48.0, "oilp_bias": -0.90, "oiltemp_bias": 15.0, "vib_mult": 1.45, "state":"Grounded"},
    "UAV-05": {"health_bias": -3.0, "rul_bias": -15.0, "cht_bias": 1.0, "egt_bias": 5.0, "oilp_bias": -0.10, "oiltemp_bias": 2.0, "vib_mult": 1.03, "state":"Standby"},
}


REQUIRED_COLUMNS = {
    "Time", "RPM", "MAP", "CHT", "EGT", "P_oil", "T_oil", "Fuel_Flow",
    "Vibration", "V_bus", "Power", "BSFC", "Thermal_Stress"
}
OPTIONAL_COLUMNS = {"Health_Index", "RUL_Seconds", "Anomaly_Flag", "Anomaly_Score", "Fault_Diagnosis", "Injection_Timing", "Altitude", "Ambient_Temp", "Throttle", "Alternator_Health", "Airbox_Temp"}

st.set_page_config(
    page_title="DRISHTI — Rotax 914 MALE UAV Engine Digital Twin GCS",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#06090c;--panel:#0d1318;--line:#27333d;--text:#eef2f4;--muted:#8997a3;--green:#62dc55;--amber:#f0b72d;--red:#e34242;--blue:#65b8ef;--lcd:#baff58;}
html,body,[class*="css"],.stApp{font-family:'Inter','Segoe UI',Arial,sans-serif!important}.stApp{background:#06090c!important;color:var(--text)!important}.main .block-container{max-width:1540px;padding:.45rem 1rem 3rem!important}header[data-testid="stHeader"]{background:rgba(6,9,12,.94)!important}[data-testid="stToolbar"],footer{display:none!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0f14,#070a0e)!important;border-right:1px solid #2a343d!important;min-width:252px!important;max-width:252px!important}section[data-testid="stSidebar"]>div{padding:.65rem .62rem .9rem!important}section[data-testid="stSidebar"] *{font-family:'Inter','Segoe UI',Arial,sans-serif!important}section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] .stCaption{font-size:9px!important}.gcs-panel{position:relative;background:linear-gradient(145deg,#161d23,#080c10 72%);border:1px solid #303b45;border-radius:7px;padding:9px 10px 8px;margin:0 0 7px;box-shadow:inset 0 1px rgba(255,255,255,.04),0 8px 18px rgba(0,0,0,.25)}.gcs-title{font-size:11px!important;font-weight:800;letter-spacing:.7px;color:#edf2f5;margin-bottom:3px}.gcs-subtitle{font-size:7px!important;color:#73808c;letter-spacing:.55px}.side-section{font-size:8px!important;font-weight:800;letter-spacing:.85px;color:#cbd4da;text-transform:uppercase;margin:10px 2px 5px}.side-led{font-size:8px;color:#54dc68;margin:3px 2px 7px}.side-led:before{content:"●";font-size:10px;margin-right:5px}section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]{min-height:30px!important;background:#090e13!important;border:1px solid #303b44!important;border-radius:5px!important}section[data-testid="stSidebar"] .stRadio>div{gap:3px!important}section[data-testid="stSidebar"] .stRadio label{padding:2px 0!important}section[data-testid="stSidebar"] .stSlider [role="slider"]{background:#e34c4c!important;border-color:#e34c4c!important}section[data-testid="stSidebar"] .stCheckbox{padding:1px 0!important}section[data-testid="stSidebar"] .stButton>button{border-radius:5px!important;font-size:9px!important}
.sky-window{height:132px;border:1px solid #33434d;border-radius:7px 7px 0 0;overflow:hidden;position:relative;background:radial-gradient(ellipse at 17% 28%,rgba(255,255,255,.82) 0 2%,transparent 3.5%),radial-gradient(ellipse at 22% 25%,rgba(255,255,255,.58) 0 4%,transparent 5.5%),radial-gradient(ellipse at 76% 30%,rgba(255,255,255,.55) 0 3%,transparent 4.5%),linear-gradient(180deg,#79c8f0 0%,#b5e1f5 40%,#699db9 72%,#3a5b6a 100%);box-shadow:inset 0 -40px 60px rgba(0,0,0,.25),0 8px 20px rgba(0,0,0,.35)}.sky-window:before{content:"";position:absolute;left:0;right:0;bottom:0;height:38%;background:linear-gradient(180deg,transparent,#294653 72%,#142229);opacity:.78}.sky-window:after{content:"";position:absolute;left:49.3%;top:-20px;width:6px;height:175px;background:linear-gradient(90deg,#0d1519,#43515a,#10171b);box-shadow:30px 0 0 rgba(20,27,31,.45);transform:rotate(1deg)}.sky-hud{position:absolute;z-index:3;top:11px;left:16px;color:#f7fafb;text-shadow:0 1px 5px #142029}.sky-hud .name{font-size:15px;font-weight:800;letter-spacing:.6px}.sky-hud .sub{font-size:7px;letter-spacing:1px;opacity:.82;margin-top:2px}.sky-badges{position:absolute;right:12px;top:9px;display:flex;gap:7px;z-index:3}.sky-badge{background:rgba(7,14,19,.78);border:1px solid rgba(255,255,255,.17);border-radius:5px;padding:6px 9px;min-width:92px}.sky-badge .k{font-size:6.5px;color:#aeb8c0;letter-spacing:.55px}.sky-badge .v{font-size:9px;font-weight:700;color:#72e25d;margin-top:2px}
.cockpit-section-title{margin-top:10px!important}.real-cockpit{position:relative;margin-top:0;padding:13px 14px 12px;border:1px solid #3a454d;border-radius:0 0 9px 9px;background:radial-gradient(circle at 13px 13px,#8a9398 0 1.7px,#22292e 2.3px 3.6px,transparent 4px) 0 0/52px 52px,radial-gradient(circle at 76% 36%,rgba(255,255,255,.025),transparent 32%),linear-gradient(135deg,#1d2429,#0b1014 32%,#171e23 63%,#080c10);box-shadow:inset 0 1px rgba(255,255,255,.06),inset 0 -20px 40px rgba(0,0,0,.32),0 14px 28px rgba(0,0,0,.42)}.real-cockpit:before{content:"";position:absolute;inset:7px;border:1px solid rgba(255,255,255,.045);border-radius:6px;pointer-events:none}.panel-screws i{position:absolute;width:6px;height:6px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#b6bcc0,#394148 45%,#101418 70%);box-shadow:0 0 0 1px #050708;z-index:2}.panel-screws i:nth-child(1){left:9px;top:9px}.panel-screws i:nth-child(2){right:9px;top:9px}.panel-screws i:nth-child(3){left:9px;bottom:9px}.panel-screws i:nth-child(4){right:9px;bottom:9px}.annunciator-row{display:flex;justify-content:center;align-items:center;gap:9px;margin:0 0 10px}.ann{min-width:72px;text-align:center;padding:5px 9px;border:1px solid #39434a;border-radius:4px;background:linear-gradient(#171d21,#090d10);box-shadow:inset 0 0 10px #000;color:#aab2b7;font-size:7px;letter-spacing:.5px}.ann b{display:block;font-size:8px;line-height:1.05}.ann.red{color:#ff4747}.ann.amber{color:#ffc62e}.ann.green{color:#64df58}
.instrument-layout{display:grid;grid-template-columns:minmax(0,1fr) 205px;grid-template-rows:auto auto;gap:8px 11px;align-items:center}.primary-grid{display:grid;grid-template-columns:1.16fr 1fr 1fr 1fr;grid-template-rows:auto auto;gap:3px;align-items:center;justify-items:center}.dial-wrap{position:relative;display:flex;align-items:center;justify-content:center;flex-direction:column;min-width:0}.dial-wrap.xl{width:178px;height:178px}.dial-wrap.lg{width:148px;height:148px}.dial-wrap.md{width:116px;height:116px}.dial-wrap.sm{width:102px;height:102px}.dial-title{position:absolute;top:4px;z-index:3;font-size:8px;font-weight:800;letter-spacing:.9px;color:#eef0eb;text-align:center;white-space:nowrap}.dial-unit{position:absolute;top:16px;z-index:3;font-size:6.5px;color:#9ea8ad;text-align:center;white-space:nowrap}.dial-face{position:relative;width:100%;height:100%}.dial-svg{position:absolute;inset:0;width:100%;height:100%}.dial-digital{position:absolute;left:22%;right:22%;bottom:12%;height:19%;display:flex;align-items:center;justify-content:center;background:linear-gradient(#0c1309,#101b0c);border:1px solid #263725;border-radius:2px;color:var(--lcd);font:600 15px 'Consolas','Courier New',monospace;text-shadow:0 0 7px rgba(186,255,88,.35);box-shadow:inset 0 0 7px #000}.dial-wrap.xl .dial-digital{font-size:17px}.dial-wrap.md .dial-digital{font-size:12px}.dial-sub{position:absolute;bottom:5%;left:5%;right:5%;text-align:center;font-size:5.5px;color:#8e999f;z-index:3;white-space:nowrap}.quantity-stack{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;align-items:center}.v-instrument{height:150px;background:linear-gradient(180deg,#151b20,#090d10);border:1px solid #39434a;border-radius:5px;box-shadow:inset 0 0 10px #000,0 4px 9px rgba(0,0,0,.35);position:relative;padding:7px 4px;text-align:center}.v-title{font-size:6.5px;font-weight:800;color:#e8ecee;letter-spacing:.5px}.v-unit{font-size:5.5px;color:#8e999f;margin-top:1px}.v-scale{position:absolute;left:4px;top:36px;bottom:32px;display:flex;flex-direction:column;justify-content:space-between;font-size:5.5px;color:#76828a}.v-track{position:absolute;left:50%;transform:translateX(-50%);top:30px;bottom:31px;width:14px;background:#070a0c;border:1px solid #505b62;border-radius:2px;box-shadow:inset 0 0 6px #000}.v-fill{position:absolute;left:2px;right:2px;bottom:2px;border-radius:1px;background:linear-gradient(180deg,#d7ec55,#83ca3d 55%,#3c9b4c);box-shadow:0 0 7px rgba(125,220,70,.32)}.v-fill.amber{background:linear-gradient(180deg,#f0d44e,#a9bc32 60%,#6f8d2b)}.v-digital{position:absolute;bottom:6px;left:0;right:0;color:var(--lcd);font:600 11px 'Consolas',monospace;text-shadow:0 0 5px rgba(186,255,88,.28)}.command-stack{grid-column:1/2;display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-left:174px;margin-right:5px}.command-instrument{background:linear-gradient(#11181c,#080c0f);border:1px solid #303b42;border-radius:5px;padding:7px 10px;box-shadow:inset 0 0 9px #000}.command-head{font-size:6.5px;color:#aab4bb;letter-spacing:.65px;text-align:center}.command-head b{float:right;color:var(--lcd);font:600 12px 'Consolas',monospace}.command-scale{display:flex;justify-content:space-between;color:#7a858d;font-size:5px;margin:7px 0 2px}.command-track{height:7px;background:#050708;border:1px solid #465159;border-radius:8px;position:relative}.command-needle{position:absolute;top:-5px;width:9px;height:16px;background:#d7d0bd;border-radius:2px;transform:translateX(-50%);box-shadow:0 1px 5px #000}.alt-health-dial{position:absolute;right:12px;bottom:11px;width:116px;height:116px}.cockpit-caption{font-size:6.5px;letter-spacing:.75px;color:#73808a;margin:4px 0 0 3px}
.section-header{border-bottom:1px solid #26313a;padding:8px 0 7px 9px;margin-top:14px;margin-bottom:8px;font-weight:800;color:#dce5ec;letter-spacing:.65px;font-size:11px;background:linear-gradient(90deg,rgba(80,155,220,.07),transparent);border-left:3px solid #4da6e8}.info-card{background:linear-gradient(180deg,#11181e,#0a0e12);border:1px solid #27323a;border-radius:6px;padding:9px;box-shadow:inset 0 1px rgba(255,255,255,.025),0 5px 14px rgba(0,0,0,.22)}.small{font-size:8px;color:#8996a3}.status{border-radius:6px;padding:9px;text-align:center;border:1px solid;box-shadow:0 6px 16px rgba(0,0,0,.25)}.status h3{font-size:13px!important;letter-spacing:.5px}.stMetric{font-family:'Inter',sans-serif!important}.stMetric label{font-size:8px!important;color:#7e8b97!important;text-transform:uppercase!important;letter-spacing:.45px!important}.stMetric [data-testid="stMetricValue"]{font-size:22px!important;line-height:1.05!important}.stMetric [data-testid="stMetricDelta"]{font-size:8px!important}[data-testid="stDataFrame"]{border:1px solid #25313a!important;border-radius:6px!important;overflow:hidden!important}.js-plotly-plot .plotly .modebar{display:none!important}
@media(max-width:1250px){.instrument-layout{grid-template-columns:1fr}.quantity-stack{grid-template-columns:repeat(4,1fr);max-width:620px;margin:auto}.command-stack{grid-column:auto;margin:0 0 0 174px}.alt-health-dial{right:9px;bottom:10px}}@media(max-width:900px){.primary-grid{grid-template-columns:repeat(2,1fr)}.quantity-stack{grid-template-columns:repeat(2,1fr)}.command-stack{margin:0}.alt-health-dial{position:relative;right:auto;bottom:auto;margin:auto}}
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
    """Convert common CSV fuel-flow representations to g/s without assuming a unit silently."""
    x = pd.to_numeric(series, errors="coerce")
    # Heuristic only; keep the raw value in the source dataset and label the displayed conversion.
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
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the processed telemetry CSV next to this app."
        )
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
    frame = {
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
    return frame


def derive_missing_channels(frame, profile):
    # Missing values are derived and explicitly marked as derived.
    if not np.isfinite(frame["Injection_Timing"]):
        # Prototype placeholder only; do not claim OEM timing.
        frame["Injection_Timing"] = 25.0 + 0.001 * (frame["RPM"] - 5500.0)
        frame["Injection_Timing_source"] = "DERIVED prototype"
    else:
        frame["Injection_Timing_source"] = "SOURCE CSV"
    if not np.isfinite(frame["Alternator_Health"]):
        frame["Alternator_Health"] = clamp(100.0 - max(0.0, 2500.0 - frame["RPM"]) / 25.0)
        frame["Alternator_Health_source"] = "DERIVED prototype"
    else:
        frame["Alternator_Health_source"] = "SOURCE CSV"
    if not np.isfinite(frame["Airbox_Temp"]):
        ambient = profile["ambient_c"]
        frame["Airbox_Temp"] = ambient + 0.25 * max(0.0, frame["MAP"] - 20.0)
        frame["Airbox_Temp_source"] = "DERIVED prototype"
    else:
        frame["Airbox_Temp_source"] = "SOURCE CSV"
    if not np.isfinite(frame["Altitude"]):
        frame["Altitude"] = profile["altitude_ft"]
        frame["Altitude_source"] = "MISSION PROFILE"
    else:
        frame["Altitude_source"] = "SOURCE CSV"
    if not np.isfinite(frame["Ambient_Temp"]):
        frame["Ambient_Temp"] = profile["ambient_c"]
        frame["Ambient_Temp_source"] = "MISSION PROFILE"
    else:
        frame["Ambient_Temp_source"] = "SOURCE CSV"
    if not np.isfinite(frame["Throttle"]):
        frame["Throttle"] = profile["throttle_pct"]
        frame["Throttle_source"] = "MISSION PROFILE"
    else:
        frame["Throttle_source"] = "SOURCE CSV"
    return frame



def isa_temperature_c(altitude_ft):
    """ISA temperature for the 0–16,000 ft DRISHTI mission envelope."""
    h_m = float(np.clip(altitude_ft, 0.0, 16000.0)) * 0.3048
    return 15.0 - 6.5 * (h_m / 1000.0)


def mission_surface(altitude_ft, throttle_pct, delta_isa_c=0.0):
    """Generate a responsive prototype engine operating point.
    Ambient temperature = ISA temperature + explicit ISA deviation.
    Pure ISA operation uses delta_isa_c=0.0.
    """
    altitude_ft = float(np.clip(altitude_ft, 0.0, 16000.0))
    throttle_pct = float(np.clip(throttle_pct, 0.0, 100.0))
    delta_isa_c = float(delta_isa_c)
    # Extend the reference throttle range down to idle and cap at max-continuous for normal mission operation.
    rpm = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [1800, 4300, 4800, 5000, 5500]))
    power_kw = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [8.0, 40.4, 47.8, 55.1, 73.5]))
    torque_nm = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [35.0, 90.0, 95.0, 105.0, 128.0]))
    map_inhg = float(np.interp(throttle_pct, [0, 55, 64, 67, 100], [12.0, 28.0, 29.0, 31.0, 35.0]))
    # ISA pressure ratio, limited to the prototype MALE mission envelope.
    h_m = altitude_ft * 0.3048
    if h_m <= 11000.0:
        T_isa = 288.15 - 0.0065*h_m
        p_ratio = (T_isa/288.15) ** 5.25588
    else:
        T11 = 216.65
        p11 = (216.65/288.15) ** 5.25588
        p_ratio = p11 * math.exp(-9.80665*(h_m-11000.0)/(287.05*T11))
        T_isa = 216.65
    # Turbocharging maintains MAP substantially better than a naturally aspirated engine;
    # apply a modest high-altitude derating rather than assuming full pressure recovery.
    altitude_factor = 1.0 - 0.18 * (1.0 - p_ratio)
    map_inhg *= altitude_factor
    power_kw *= altitude_factor
    torque_nm *= altitude_factor
    # Thermal behavior: higher throttle, altitude and hot ambient raise thermal load.
    ambient_c = (T_isa - 273.15) + delta_isa_c
    return {"RPM":rpm, "Power":power_kw, "MAP":map_inhg, "Torque":torque_nm,
            "Ambient_Temp":ambient_c, "ISA_Temp":T_isa - 273.15,
            "Delta_ISA_C":delta_isa_c, "Altitude":altitude_ft,
            "Throttle":throttle_pct, "Altitude_Pressure_Ratio":p_ratio}


def apply_mission_controls(frame, altitude_ft, throttle_pct, mission_time_h):
    """Make manual/preset mission controls actually drive the displayed engine state."""
    f = frame.copy()
    surf = mission_surface(altitude_ft, throttle_pct, profile.get("delta_isa_c", 0.0))
    # Preserve fault/scenario inputs later; this function creates the healthy mission baseline.
    f["Altitude"] = surf["Altitude"]
    f["Ambient_Temp"] = surf["Ambient_Temp"]
    f["Throttle"] = surf["Throttle"]
    f["RPM"] = surf["RPM"]
    f["MAP"] = surf["MAP"]
    f["Power"] = surf["Power"]
    f["Airbox_Temp"] = f["Ambient_Temp"] + 0.20 * max(0.0, f["MAP"] - 20.0)
    # Simple but internally consistent mission trends. Coefficients are prototype assumptions.
    hot_load = max(0.0, f["Ambient_Temp"] - 25.0)
    throttle_load = max(0.0, f["Throttle"] - 45.0)
    altitude_load = max(0.0, f["Altitude"] - 10000.0) / 1000.0
    f["CHT"] = 82.0 + 0.32*throttle_load + 0.22*hot_load + 0.9*altitude_load
    f["EGT"] = 650.0 + 2.15*throttle_load + 1.25*hot_load + 5.0*altitude_load
    f["Oil_Temp"] = 78.0 + 0.34*throttle_load + 0.45*hot_load + 0.7*altitude_load
    # Context-dependent oil pressure follows RPM; low pressure remains physically possible for fault injection.
    if f["RPM"] > 3500:
        f["Oil_Pressure"] = float(np.clip(2.0 + (f["RPM"]-3500.0)/2000.0*3.0, 2.0, 5.0))
    else:
        f["Oil_Pressure"] = float(np.clip(0.8 + f["RPM"]/3500.0*1.2, 0.8, 2.0))
    f["Fuel_Flow"] = max(1.0, f["Power"] / 0.25 * 0.27778)  # prototype fuel-energy proxy, g/s
    f["Vibration"] = max(0.5, 0.00025*f["RPM"] + 0.002*throttle_load)
    f["Thermal_Stress"] = clamp(35.0 + 0.55*throttle_load + 1.2*hot_load + 2.0*altitude_load, 0, 100)
    f["Injection_Timing"] = 25.0 + 0.001*(f["RPM"]-5500.0)
    f["Alternator_Health"] = clamp(90.0 + 0.004*max(0.0, f["RPM"]-2500.0))
    f["Mission_Duration_h"] = max(0.1, float(mission_time_h))
    f["Mission_Surface"] = surf
    return f


def apply_uav_state(frame, selected_uav):
    """Map the selected fleet demo aircraft into the same telemetry/intelligence pipeline."""
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
    f["Health_Bias"] = cfg["health_bias"]
    f["RUL_Bias"] = cfg["rul_bias"]
    f["Fleet_State"] = cfg["state"]
    return f


def mission_recommendations(focus, altitude_ft, throttle_pct, duration_h):
    """Give actionable mission-setting guidance based on the selected primary parameter."""
    alt = float(altitude_ft); thr = float(throttle_pct); dur = float(duration_h)
    if focus == "Altitude":
        # Mid-altitude cruise is the prototype efficiency sweet spot; avoid treating it as an OEM optimum.
        suggested_thr = float(np.clip(np.interp(alt, [0, 6000, 10000, 14000, 16000], [62, 58, 55, 60, 65]), 45, 70))
        eff = max(0.1, 1.0 - abs(suggested_thr-55.0)/100.0 - max(0, alt-12000)/50000)
        endurance = max(0.5, 8.0 * eff * (55.0/max(suggested_thr, 1.0)))
        return suggested_thr, endurance, f"At {alt:.0f} ft, target about **{suggested_thr:.0f}% throttle** for the best prototype efficiency balance. Estimated endurance at that setting is **{endurance:.1f} h**."
    if focus == "Throttle":
        suggested_alt = float(np.clip(np.interp(thr, [45, 55, 65, 75, 90], [12000, 10000, 8000, 6000, 4000]), 0, 16000))
        eff = max(0.1, 1.0 - max(0, thr-55)/100.0)
        endurance = max(0.5, 8.0 * eff)
        return suggested_alt, endurance, f"At {thr:.0f}% throttle, a prototype efficiency-oriented cruise altitude is about **{suggested_alt:.0f} ft**. Estimated endurance is **{endurance:.1f} h** if thermal and mission limits remain satisfied."
    # Mission time as the primary constraint: choose a lower-power setting that can cover it.
    target_thr = float(np.clip(55.0 - max(0, dur-4.0)*2.0, 45, 60))
    target_alt = 10000.0 if dur <= 6 else 8000.0
    feasible_endurance = max(0.5, 8.0*(55.0/max(target_thr,1.0))*0.92)
    return target_thr, feasible_endurance, f"For a planned **{dur:.1f} h** mission, use about **{target_thr:.0f}% throttle at {target_alt:.0f} ft** as a prototype endurance-oriented starting point. Estimated endurance is **{feasible_endurance:.1f} h**, leaving **{max(0, feasible_endurance-dur):.1f} h** margin."

def apply_fault_scenarios(frame, flags, elapsed_min):
    f = frame.copy()
    effects = []
    if flags.get("injector"):
        f["Fuel_Flow"] *= 1.15
        f["RPM"] *= 0.95
        f["CHT"] += 8.0
        f["EGT"] += 30.0
        effects.append("Injector abnormality")
    if flags.get("misfire"):
        f["RPM"] *= 0.90
        f["EGT"] *= 0.92
        f["Vibration"] += 2.5
        effects.append("Misfire")
    if flags.get("thermal"):
        f["CHT"] = max(f["CHT"], 142.5)
        f["Thermal_Stress"] = max(f["Thermal_Stress"], 95.0)
        effects.append("Thermal overheat")
    if flags.get("mechanical"):
        f["Vibration"] = max(f["Vibration"], 2.15)
        effects.append("Mechanical imbalance")
    if flags.get("sensor_drift"):
        f["CHT"] += 0.02 * elapsed_min
        f["Anomaly_Score_source"] = max(0.88, safe_float(f["Anomaly_Score_source"], 0.0))
        effects.append("CHT sensor drift")
    if flags.get("lubrication"):
        f["Oil_Pressure"] = max(0.4, f["Oil_Pressure"] - 1.5)
        f["Oil_Temp"] = max(f["Oil_Temp"], 134.0)
        effects.append("Lubrication degradation")
    if flags.get("combustion"):
        f["EGT"] += 20.0 * math.sin(2 * math.pi * 0.5 * elapsed_min / 60.0)
        f["RPM"] = max(f["RPM"], 5650.0)
        f["Vibration"] += 1.5
        effects.append("Combustion instability")
    if flags.get("coding"):
        # PS wording says "coding degradation". Treat as ECU/control-logic degradation scenario.
        f["RPM"] += 120.0
        f["Injection_Timing"] += 4.0
        f["Anomaly_Score_source"] = max(0.82, safe_float(f["Anomaly_Score_source"], 0.0))
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
    # OEM limits / conservative DRISHTI warnings
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


def subsystem_health(f):
    # Transparent engineering score: 100 at/inside healthy region, declines toward hard limit.
    def health_between(x, healthy, critical):
        if x <= healthy:
            return 100.0
        if x >= critical:
            return 0.0
        return 100.0 * (critical - x) / (critical - healthy)

    thermal = min(
        health_between(f["CHT"], 100.0, ROTAX["cht_max_c"]),
        health_between(f["Thermal_Stress"], 50.0, 100.0),
        health_between(f["Airbox_Temp"], 60.0, 88.0),
    )
    oil_low = ROTAX["oil_pressure_normal_low_bar"] if f["RPM"] > 3500 else ROTAX["oil_pressure_min_low_rpm_bar"]
    lubrication = min(
        health_between(oil_low - f["Oil_Pressure"], 0.0, oil_low),
        health_between(f["Oil_Temp"], ROTAX["oil_temp_normal_high_c"], ROTAX["oil_temp_max_c"]),
    )
    combustion = min(health_between(f["EGT"], 750.0, ROTAX["egt_max_c"]), health_between(abs(f["BSFC"]), 0.11, 0.25))
    mechanical = health_between(f["Vibration"], 1.3, DRISHTI["vibration_warn_g"] * 2.0)
    electrical = min(clamp((f["Battery_Voltage"] - 11.0) / 3.0 * 100.0), clamp(f["Alternator_Health"]))
    weights = {"Thermal":0.28, "Lubrication":0.24, "Combustion":0.20, "Mechanical":0.18, "Electrical":0.10}
    vals = {"Thermal":thermal, "Lubrication":lubrication, "Combustion":combustion, "Mechanical":mechanical, "Electrical":electrical}
    overall = sum(vals[k] * weights[k] for k in vals)
    return vals, overall


def robust_anomaly_score(df, frame):
    """Data-driven anomaly score. Uses IsolationForest when enough healthy samples exist; fallback is robust z-score."""
    features = ["RPM", "MAP", "CHT", "EGT", "P_oil", "T_oil", "Fuel_Flow_gps", "Vibration", "V_bus", "Power", "Thermal_Stress"]
    train_df = df.copy()
    if "Fuel_Flow_gps" not in train_df.columns:
        train_df["Fuel_Flow_gps"], _ = infer_fuel_flow_gps(train_df["Fuel_Flow"])
    x = train_df[["RPM", "MAP", "CHT", "EGT", "P_oil", "T_oil", "Fuel_Flow_gps", "Vibration", "V_bus", "Power", "Thermal_Stress"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50:
        return float(np.clip(safe_float(frame.get("Anomaly_Score_source"), 0.0), 0, 1)), "SOURCE DATASET"
    # Prefer samples not already marked anomalous/faulted when available.
    if "Anomaly_Flag" in df.columns:
        healthy = x.loc[train_df.loc[x.index, "Anomaly_Flag"].fillna(0).astype(int) == 0]
        if len(healthy) >= 30:
            x = healthy
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


def mission_aware_anomaly_score(df, frame, scenario_active=False, nav_mode="Live Diagnostic Cockpit"):
    """Prevent false AI anomaly inflation on synthetic mission operating points.
    Replayed data uses the data-driven model; healthy synthetic mission states are
    evaluated against explicit limits/fault scenarios instead of mismatched CSV distributions.
    """
    if nav_mode == "Mission Replay & Analytics":
        return robust_anomaly_score(df, frame)
    if scenario_active:
        score = 0.0
        score += 0.35 if frame["CHT"] > DRISHTI["cht_warn_c"] else 0.0
        score += 0.35 if frame["EGT"] > DRISHTI["egt_warn_c"] else 0.0
        score += 0.35 if frame["Vibration"] > DRISHTI["vibration_warn_g"] else 0.0
        score += 0.35 if oil_pressure_status(frame["RPM"], frame["Oil_Pressure"])[0] == "CRITICAL" else 0.0
        return clamp(score, 0, 1), "SCENARIO + LIMIT HYBRID"
    limit_hits = 0
    limit_hits += int(frame["CHT"] > DRISHTI["cht_warn_c"])
    limit_hits += int(frame["EGT"] > DRISHTI["egt_warn_c"])
    limit_hits += int(frame["Oil_Temp"] > ROTAX["oil_temp_normal_high_c"])
    limit_hits += int(oil_pressure_status(frame["RPM"], frame["Oil_Pressure"])[0] == "CRITICAL")
    limit_hits += int(frame["Vibration"] > DRISHTI["vibration_warn_g"])
    return clamp(limit_hits * 0.15, 0, 1), "MISSION LIMIT MONITOR"


def diagnostic_state(f, faults, anomaly_score, subsystem):
    critical = sum(x["severity"] == "CRITICAL" for x in faults)
    warnings = sum(x["severity"] == "WARNING" for x in faults)
    health = clamp(subsystem[1])
    if critical or health < 50:
        status = "CRITICAL"
    elif warnings or anomaly_score >= DRISHTI["anomaly_caution"] or health < 75:
        status = "CAUTION"
    else:
        status = "NORMAL"
    confirmed = critical > 0 or warnings > 0
    if anomaly_score >= DRISHTI["anomaly_confirm"] and not confirmed:
        anomaly_state = "ANOMALY DETECTED — UNCLASSIFIED"
    elif anomaly_score >= DRISHTI["anomaly_caution"] and not confirmed:
        anomaly_state = "EARLY DEVIATION — MONITOR"
    elif confirmed:
        anomaly_state = "FAULT CLASSIFIED"
    else:
        anomaly_state = "NO SIGNIFICANT ANOMALY"
    return status, critical, warnings, health, anomaly_state


def trend_slope(df, col, t_now, window=120):
    h = df[(df["Time"] >= t_now - window) & (df["Time"] <= t_now)][["Time", col]].dropna()
    if len(h) < 5 or h["Time"].nunique() < 2:
        return np.nan
    return float(np.polyfit(h["Time"].values, h[col].values, 1)[0] * 60.0)  # units/min


def time_to_limit(value, slope_per_min, limit, direction="up"):
    if not np.isfinite(slope_per_min):
        return np.inf
    if direction == "up":
        if slope_per_min <= 0 or value >= limit:
            return np.inf if value < limit else 0.0
        return max(0.0, (limit - value) / slope_per_min)
    if slope_per_min >= 0 or value <= limit:
        return np.inf if value > limit else 0.0
    return max(0.0, (value - limit) / abs(slope_per_min))


def physics_consistency(frame):
    """First-principles consistency check for a 4-stroke piston engine.
    Assumptions are prototype constants and are displayed as such.
    MAP is converted from inHg to Pa.
    """
    Vd=0.0012112  # m^3, Rotax 914 displacement reference
    eta_v=0.85
    R=287.05
    afr=14.7
    lhv=43e6
    eta_b=0.30
    map_pa=frame["MAP"]*3386.389
    t_k=max(250.0, frame["Airbox_Temp"]+273.15)
    mdot_air=eta_v*map_pa*Vd*(frame["RPM"]/60.0)/(2.0*R*t_k)
    mdot_fuel=mdot_air/afr
    expected_power_kw=mdot_fuel*lhv*eta_b/1000.0
    fuel_residual_gps=frame["Fuel_Flow"]-mdot_fuel*1000.0
    power_residual_kw=frame["Power"]-expected_power_kw
    return {"mdot_air_gps":mdot_air*1000.0,"expected_fuel_gps":mdot_fuel*1000.0,"expected_power_kw":expected_power_kw,"fuel_residual_gps":fuel_residual_gps,"power_residual_kw":power_residual_kw}


def xai_local_contributions(df, frame, faults=None):
    """Fine-grained local explainability using healthy-baseline deviation + limit proximity.
    This is deliberately transparent and is not mislabeled as SHAP.
    """
    specs = [
        ("CHT", "CHT", 100.0, ROTAX["cht_max_c"], "Thermal load", "Reduce throttle; check cooling"),
        ("EGT", "EGT", 750.0, ROTAX["egt_max_c"], "Combustion temperature", "Reduce load; inspect mixture/injection/ignition"),
        ("Oil Pressure", "P_oil", ROTAX["oil_pressure_normal_low_bar"], ROTAX["oil_pressure_max_bar"], "Lubrication margin", "Reduce load and inspect oil circuit"),
        ("Oil Temperature", "T_oil", ROTAX["oil_temp_normal_high_c"], ROTAX["oil_temp_max_c"], "Lubrication thermal load", "Reduce load; verify cooling/oil system"),
        ("Vibration", "Vibration", 1.3, DRISHTI["vibration_warn_g"], "Mechanical condition", "Reduce load and inspect rotating assembly"),
        ("MAP", "MAP", 31.0, DRISHTI["map_takeoff_inhg"], "Engine loading", "Reduce throttle if MAP remains high"),
        ("RPM", "RPM", ROTAX["rpm_cont_max"], ROTAX["rpm_takeoff_max"], "Speed margin", "Reduce throttle immediately if above continuous limit"),
        ("Bus Voltage", "V_bus", 12.0, 14.5, "Electrical stability", "Inspect alternator/bus if voltage is abnormal"),
        ("Thermal Stress", "Thermal_Stress", 50.0, 100.0, "Accumulated thermal load", "Reduce thermal loading"),
    ]
    rows=[]
    for label,col,healthy,critical,why,action in specs:
        s=pd.to_numeric(df[col],errors="coerce").dropna()
        if len(s)<10: continue
        med=float(s.median())
        mad=float((s-med).abs().median())
        scale=max(1.4826*mad,float(s.std())*0.25,1e-6)
        value=float(frame.get({"P_oil":"Oil_Pressure","T_oil":"Oil_Temp","V_bus":"Battery_Voltage"}.get(col,col),np.nan))
        if not np.isfinite(value): continue
        deviation=min(1.0,abs(value-med)/(5.0*scale))
        # Proximity score is based on the relevant side of the safety band.
        if label == "Oil Pressure":
            proximity=min(1.0,max(0.0,(healthy-value)/max(healthy,1e-6))) if value < healthy else min(1.0,max(0.0,(value-critical)/max(critical,1e-6)))
        elif label in ("RPM","MAP"):
            proximity=min(1.0,max(0.0,(value-healthy)/max(critical-healthy,1e-6)))
        else:
            proximity=min(1.0,max(0.0,(value-healthy)/max(critical-healthy,1e-6)))
        contribution=0.55*deviation+0.45*proximity
        rows.append((label,contribution,deviation,proximity,why,action,value,med))
    out=pd.DataFrame(rows,columns=["Feature","Contribution","Deviation","Limit_Proximity","Why","Corrective Action","Value","Healthy Baseline"])
    return out.sort_values("Contribution",ascending=False) if not out.empty else out


def live_corrective_suggestions(frame, faults, health, mission_risk, altitude_ft, throttle_pct, mission_remaining_h):
    suggestions=[]
    if frame["CHT"] > DRISHTI["cht_warn_c"]:
        suggestions.append(("THERMAL", "Reduce throttle by 5–10 percentage points and monitor CHT; inspect cooling if the rise persists."))
    if frame["EGT"] > DRISHTI["egt_warn_c"]:
        suggestions.append(("COMBUSTION", "Reduce engine load and check injector/mixture/ignition behavior."))
    oil_stat,_=oil_pressure_status(frame["RPM"],frame["Oil_Pressure"])
    if oil_stat == "CRITICAL":
        suggestions.append(("LUBRICATION", "Oil pressure is outside the reference operating range — reduce load and investigate before continuing."))
    if frame["Oil_Temp"] > ROTAX["oil_temp_normal_high_c"]:
        suggestions.append(("OIL TEMPERATURE", "Lower engine load and monitor oil temperature toward the 90–110°C normal band."))
    if frame["Vibration"] > DRISHTI["vibration_warn_g"]:
        suggestions.append(("VIBRATION", "Reduce load and inspect the mechanical/rotating assembly for imbalance."))
    if frame["RPM"] > ROTAX["rpm_cont_max"]:
        suggestions.append(("RPM", "Reduce throttle — operation above 5500 rpm is limited to the takeoff-duration allowance."))
    if mission_risk in ("HIGH","CRITICAL"):
        suggestions.append(("MISSION", "Mission risk is elevated. Shorten the remaining mission or move to a lower-load operating point."))
    if mission_remaining_h > 6.0 and throttle_pct > 60:
        suggestions.append(("ENDURANCE", "For a long remaining mission, reduce throttle and use an efficiency-oriented cruise altitude."))
    if not suggestions:
        suggestions.append(("MONITOR", "No immediate corrective action required. Continue trend monitoring and preserve positive mission margin."))
    return suggestions


def mission_metrics(status, health, anomaly, rul_h, mission_remaining_h, altitude_ft, ambient_c, throttle):
    margin = rul_h - mission_remaining_h
    risk_score = 0.0
    risk_score += max(0.0, 70.0 - health) * 0.55
    risk_score += anomaly * 30.0
    if margin < 0:
        risk_score += min(45.0, abs(margin) * 18.0)
    if ambient_c > 40:
        risk_score += 4
    if altitude_ft > 12000:
        risk_score += 3
    if throttle > 75:
        risk_score += 4
    if status == "CRITICAL":
        risk_score += 55
    elif status == "CAUTION":
        risk_score += 8
    risk_score = clamp(risk_score, 0, 100)
    completion = clamp(100.0 - risk_score)
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


def estimate_rul(frame, health, anomaly, df):
    """Prototype RUL: prefer an existing model RUL, but apply explicit scenario penalties.
    This is NOT a certified prognostic model and must be labeled prototype.
    """
    src = frame["RUL_Seconds_source"]
    if np.isfinite(src) and src > 0:
        base_h = src / 3600.0
    else:
        # No run-to-failure data => do not invent a precise life; use TBO as a reference envelope only.
        base_h = ROTAX["tbo_h"] * health / 100.0
    # Degradation penalty based on anomaly and health; transparent engineering heuristic.
    factor = clamp(1.0 - 0.45 * anomaly - 0.35 * max(0.0, 70.0 - health) / 70.0, 0.15, 1.0)
    rul = max(0.05, base_h * factor)
    # Confidence is independent from RUL magnitude.
    healthy_samples = len(df)
    confidence = clamp(45.0 + min(35.0, healthy_samples / 1000.0 * 10.0) - anomaly * 30.0)
    spread = max(0.1, rul * (1.0 - confidence / 100.0) * 0.8)
    return rul, max(0.01, rul - spread), rul + spread, confidence


def counterfactuals(frame, status, health, anomaly, rul_h, remaining_h, profile):
    base = mission_metrics(status, health, anomaly, rul_h, remaining_h, profile["altitude_ft"], profile["ambient_c"], frame["Throttle"])[0]
    scenarios = [("Current", frame["Throttle"], profile["altitude_ft"])]
    scenarios.append(("Reduce throttle 10%", max(0, frame["Throttle"] * 0.90), profile["altitude_ft"]))
    scenarios.append(("Lower altitude 20%", frame["Throttle"], profile["altitude_ft"] * 0.80))
    scenarios.append(("Combined mitigation", max(0, frame["Throttle"] * 0.90), profile["altitude_ft"] * 0.80))
    out = []
    for name, thr, alt in scenarios:
        thermal_penalty = max(0.0, (thr - 55.0) * 0.15) + max(0.0, (alt - 10000.0) / 1000.0) * 1.5 + max(0.0, profile["ambient_c"] - 35.0) * 0.4
        adjusted_health = clamp(health - thermal_penalty + max(0.0, profile["altitude_ft"] - alt) / 1000.0 * 1.0)
        adjusted_anomaly = clamp(anomaly + thermal_penalty / 100.0, 0, 1)
        completion = mission_metrics(status, adjusted_health, adjusted_anomaly, rul_h, remaining_h, alt, profile["ambient_c"], thr)[0]
        out.append((name, completion, thr, alt))
    return out


def mission_safety_advisory(status, mission_risk, mission_elapsed_h, mission_duration_h,
                            mission_remaining_h, health, completion,
                            home_distance_nm=None, nearest_airport_distance_nm=None,
                            cruise_speed_kt=85.0):
    """Prototype mission-level diversion advisory for the hackathon demonstrator.
    Navigation, airport choice and ATC messaging are simulated, not live commands.
    """
    elapsed_fraction = mission_elapsed_h / max(mission_duration_h, 1e-6)
    if home_distance_nm is None:
        home_distance_nm = 6.0 + 32.0 * min(1.0, 2.0 * elapsed_fraction)
        if elapsed_fraction > 0.5:
            home_distance_nm = max(6.0, 38.0 - 28.0 * (elapsed_fraction - 0.5) / 0.5)
    if nearest_airport_distance_nm is None:
        nearest_airport_distance_nm = max(8.0, home_distance_nm * 0.55)

    if mission_risk == "LOW" and status == "NORMAL":
        return {"decision":"CONTINUE TO DESTINATION",
                "reason":"Engine state and remaining mission margin support completion under the prototype model.",
                "atc":"No emergency communication required.",
                "home_distance_nm":home_distance_nm,
                "nearest_airport_distance_nm":nearest_airport_distance_nm,
                "eta_min":None,"severity":"NORMAL"}
    if mission_risk == "MEDIUM":
        return {"decision":"CONTINUE WITH CAUTION / MONITOR",
                "reason":"Mission remains conditionally feasible; reduce load if trends worsen.",
                "atc":"No emergency call required unless the condition escalates.",
                "home_distance_nm":home_distance_nm,
                "nearest_airport_distance_nm":nearest_airport_distance_nm,
                "eta_min":None,"severity":"ADVISORY"}

    early_and_home_close = elapsed_fraction <= 0.30 and home_distance_nm <= nearest_airport_distance_nm + 5.0
    if early_and_home_close:
        decision = "RETURN TO HOME AIRPORT"
        target = "home airport"
        target_distance = home_distance_nm
    else:
        decision = "DIVERT TO NEAREST SUITABLE AIRPORT"
        target = "nearest suitable airport"
        target_distance = nearest_airport_distance_nm
    eta_min = target_distance / max(cruise_speed_kt, 1.0) * 60.0
    reason = (f"{mission_risk} mission risk. Estimated route to {target}: {target_distance:.1f} NM, "
              f"ETA ≈ {eta_min:.0f} min. Navigation/airport selection is simulated for the demonstrator.")
    atc = ("DISTRESS / PRIORITY LANDING MESSAGE: SIMULATED TRANSMISSION SENT NOW to the destination airport ATC; "
           "request priority handling and emergency landing clearance before arrival. SIMULATION ONLY — no real radio/ATC message is transmitted.")
    return {"decision":decision,"reason":reason,"atc":atc,
            "home_distance_nm":home_distance_nm,
            "nearest_airport_distance_nm":nearest_airport_distance_nm,
            "eta_min":eta_min,"severity":"URGENT"}


def maintenance_recommendation(status, faults, subsystem, sensor_health):
    if status == "CRITICAL":
        priority = "P1 — HIGH"
    elif status == "CAUTION":
        priority = "P2 — MEDIUM"
    else:
        priority = "P3 — LOW"
    items = []
    if any(f["sub"] == "Thermal" for f in faults):
        items += ["Inspect cooling system", "Review CHT/EGT thermal trend"]
    if any(f["sub"] == "Lubrication" for f in faults):
        items += ["Check oil level/pressure circuit", "Inspect lubrication system"]
    if any(f["sub"] == "Mechanical" for f in faults):
        items += ["Inspect vibration source / rotating assembly"]
    if any(f["sub"] == "Combustion" for f in faults):
        items += ["Inspect injector / mixture / ignition system"]
    if sensor_health < 95:
        items.append("Verify sensor calibration and wiring")
    if not items:
        items = ["Continue trend monitoring", "Review post-flight health record"]
    return priority, items


def sensor_integrity(df, frame, sensor_drift_active=False):
    results = []
    for col, label, unit in [("CHT","CHT Sensor","°C"),("EGT","EGT Sensor","°C"),("P_oil","Oil Pressure Sensor","bar"),("T_oil","Oil Temperature Sensor","°C"),("Vibration","Vibration Sensor","g")]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            rel = 90.0
        else:
            recent = s.tail(min(500, len(s)))
            drift = abs(float(recent.iloc[-1] - recent.iloc[0])) / max(len(recent), 1)
            rel = clamp(100.0 - drift * 10.0)
        if sensor_drift_active and label == "CHT Sensor":
            rel = min(rel, 88.0)
            drift_display = 0.02 * max(0.0, frame["Time"] / 60.0)
        else:
            drift_display = 0.0
        results.append((label, drift_display, rel, "WARNING" if rel < 95 else "HEALTHY", unit))
    return results


def build_event_log(df):
    events=[]
    if "Anomaly_Flag" in df.columns:
        a=pd.to_numeric(df["Anomaly_Flag"],errors="coerce").fillna(0)
        starts=df.index[(a>0) & (a.shift(1,fill_value=0)<=0)]
        for i in starts:
            r=df.loc[i]
            label=str(r.get("Fault_Diagnosis","Anomaly detected"))
            events.append({"Time (s)":float(r["Time"]),"Type":"Anomaly/Fault","Description":label})
    if "Health_Index" in df.columns:
        h=pd.to_numeric(df["Health_Index"],errors="coerce")
        if h.notna().any():
            i=int(h.idxmin()); r=df.loc[i]
            events.append({"Time (s)":float(r["Time"]),"Type":"Degradation","Description":f"Minimum recorded source Health Index: {float(r['Health_Index']):.1f}%"})
    return pd.DataFrame(events).sort_values("Time (s)") if events else pd.DataFrame(columns=["Time (s)","Type","Description"])


def render_faults(faults, anomaly_state):
    if faults:
        st.write(f"**CONFIRMED / LIMIT-BASED CONDITIONS — {len(faults)}**")
        for f in faults:
            cls = "#da3633" if f["severity"] == "CRITICAL" else "#d97706"
            st.markdown(f'<div class="info-card" style="border-left:4px solid {cls}; margin-bottom:6px;"><b style="color:{cls}">{f["name"]} — {f["severity"]}</b><br><span class="small">Source: {f["source"]} | Value: {f["value"]:.2f} | Reference: {f["limit"]:.2f} | Basis: {f["basis"]}</span></div>', unsafe_allow_html=True)
    else:
        st.success("No confirmed fault or reference-limit exceedance in the current frame.")
    st.info(f"Diagnostic state: **{anomaly_state}**")


# ============================================================
# APPLICATION
# ============================================================
DATA_PATH = "digital_twin_processed_analytics.csv"
try:
    df = cached_load_data(DATA_PATH)
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ========================= SIDEBAR / GCS CONSOLE =========================
st.sidebar.markdown("""
<div class='gcs-panel'>
  <div style='display:flex;align-items:center;gap:8px'>
    <div style='font-size:24px;filter:grayscale(.2)'>✦</div>
    <div><div class='gcs-title'>DRISHTI GCS</div><div class='gcs-subtitle'>ROTAX 914 · MISSION DIGITAL TWIN</div></div>
  </div>
</div>
<div class='side-led'>ACTIVE TELEMETRY LINK</div>
<div class='side-section'>Aircraft / Test Bed</div>
""", unsafe_allow_html=True)
selected_uav = st.sidebar.selectbox("Aircraft Tail / Test Bed", ["UAV-01 (Primary Testbed)", "UAV-02", "UAV-03", "UAV-04", "UAV-05"])
selected_uav_key = selected_uav.split(" ")[0]

st.sidebar.markdown("<div class='side-section'>Navigation View</div>", unsafe_allow_html=True)
nav_mode = st.sidebar.radio("Navigation View", ["Live Diagnostic Cockpit", "Fleet Digital Twin", "Mission Replay & Analytics", "Post-Flight Report Generator"], label_visibility="collapsed")

st.sidebar.markdown("<div class='side-section'>Mission Control</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='gcs-panel'>", unsafe_allow_html=True)
mission_mode = st.sidebar.radio("Mission Profile Mode", ["Manual", "Preset"], horizontal=True)
if mission_mode == "Preset":
    mission_name = st.sidebar.selectbox("Preset Mission", list(MISSION_PROFILES.keys()), index=0)
    profile = MISSION_PROFILES[mission_name]
    altitude_ft = float(profile["altitude_ft"])
    throttle_pct = float(profile["throttle_pct"])
    mission_duration_h = float(profile["duration_h"])
    delta_isa_c = float(profile.get("delta_isa_c", 0.0))
    ambient_override = isa_temperature_c(altitude_ft) + delta_isa_c
    st.sidebar.markdown(f"""
    <div class='info-card' style='padding:7px;margin-top:6px'>
      <div style='font-size:8px;color:#82909b;letter-spacing:.7px'>PRESET VALUES</div>
      <div style='font-size:10px;margin-top:4px'>ALT <b>{altitude_ft:,.0f} ft</b> &nbsp; THR <b>{throttle_pct:.0f}%</b></div>
      <div style='font-size:10px'>TIME <b>{mission_duration_h:.1f} h</b> &nbsp; ISA <b>{ambient_override:.1f}°C</b></div>
      <div style='font-size:7px;color:#73808b;margin-top:3px'>ISA deviation {delta_isa_c:+.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
else:
    mission_name = "Manual Mission"
    focus = st.sidebar.selectbox("Primary Parameter to Optimize", ["Altitude", "Throttle", "Mission time"])
    altitude_ft = float(st.sidebar.slider("Altitude (ft)", min_value=0, max_value=16000, value=10000, step=250))
    throttle_pct = float(st.sidebar.slider("Throttle (%)", min_value=0, max_value=100, value=55, step=1))
    mission_duration_h = float(st.sidebar.slider("Mission time (h)", min_value=0.1, max_value=24.0, value=4.0, step=0.1))
    delta_isa_c = 0.0
    ambient_override = isa_temperature_c(altitude_ft)
    rec_a, rec_b, rec_text = mission_recommendations(focus, altitude_ft, throttle_pct, mission_duration_h)
    st.sidebar.markdown("<div class='side-section' style='margin-top:8px'>DRISHTI Guidance</div>", unsafe_allow_html=True)
    if focus == "Altitude":
        st.sidebar.caption(f"At **{altitude_ft:.0f} ft**, suggested throttle ≈ **{rec_a:.0f}%**.")
        st.sidebar.caption(f"Estimated endurance: **{rec_b:.1f} h**.")
    elif focus == "Throttle":
        st.sidebar.caption(f"At **{throttle_pct:.0f}%**, suggested altitude ≈ **{rec_a:.0f} ft**.")
        st.sidebar.caption(f"Estimated endurance: **{rec_b:.1f} h**.")
    else:
        st.sidebar.caption(f"For **{mission_duration_h:.1f} h**, suggested throttle ≈ **{rec_a:.0f}%**.")
        st.sidebar.caption(f"Estimated endurance: **{rec_b:.1f} h**.")
    st.sidebar.markdown(f"<div class='info-card' style='padding:7px;color:#73bfff'>ISA at {altitude_ft:,.0f} ft: <b>{ambient_override:.1f} °C</b></div>", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Replay is isolated to the Mission Replay page; live cockpit remains on latest telemetry.
max_time_sec = int(df["Time"].max())
playback_time_sec = max_time_sec
mission_elapsed_h = 0.0
mission_remaining_h = mission_duration_h
if nav_mode == "Mission Replay & Analytics":
    st.sidebar.markdown("<div class='side-section'>Mission Replay</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='gcs-panel'>", unsafe_allow_html=True)
    playback_time_sec = st.sidebar.slider("Replay position", 0, max_time_sec, max_time_sec, step=max(1, min(10, max_time_sec)))
    mission_elapsed_h = min(mission_duration_h, playback_time_sec / 3600.0)
    mission_remaining_h = max(0.0, mission_duration_h - mission_elapsed_h)
    st.sidebar.caption(f"Elapsed **{mission_elapsed_h:.2f} h** · Remaining **{mission_remaining_h:.2f} h**")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='side-section'>Fault Injection · Simulated</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='gcs-panel'>", unsafe_allow_html=True)
f_injector = st.sidebar.checkbox("Fuel Injector Abnormality", key="fault_injector")
f_misfire = st.sidebar.checkbox("Misfire", key="fault_misfire")
f_cht = st.sidebar.checkbox("Thermal Overheat", key="fault_thermal")
f_vib = st.sidebar.checkbox("Mechanical Imbalance", key="fault_mechanical")
f_sensor = st.sidebar.checkbox("Sensor Drift / Failure", key="fault_sensor_drift")
f_lube = st.sidebar.checkbox("Lubrication Degradation", key="fault_lubrication")
f_comb = st.sidebar.checkbox("Combustion Instability", key="fault_combustion")
f_coding = st.sidebar.checkbox("Control / Coding Degradation", key="fault_coding")
scenario_active = any([f_injector, f_misfire, f_cht, f_vib, f_sensor, f_lube, f_comb, f_coding])
if scenario_active:
    st.sidebar.warning("SIMULATION MODE ACTIVE")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

profile = {"altitude_ft": altitude_ft, "ambient_c": ambient_override, "throttle_pct": throttle_pct, "duration_h": mission_duration_h, "delta_isa_c": delta_isa_c}

# -------------------- Mission-responsive telemetry pipeline --------------------
base = nearest_frame(df, playback_time_sec)
base = derive_missing_channels(base, profile)
base = apply_mission_controls(base, altitude_ft, throttle_pct, mission_duration_h)
base = apply_uav_state(base, selected_uav_key)
flags = {"injector":f_injector,"misfire":f_misfire,"thermal":f_cht,"mechanical":f_vib,"sensor_drift":f_sensor,"lubrication":f_lube,"combustion":f_comb,"coding":f_coding}
frame = apply_fault_scenarios(base, flags, playback_time_sec / 60.0)

# Selected UAV demo state must flow through the exact same diagnostic pipeline.
anomaly_score, anomaly_model = mission_aware_anomaly_score(df, frame, scenario_active, nav_mode)
faults = evaluate_faults(frame)
subs = subsystem_health(frame)
status, critical_count, warning_count, health, anomaly_state = diagnostic_state(frame, faults, anomaly_score, subs)
# Fleet demo aircraft carry a transparent health/RUL bias on top of their telemetry state.
if selected_uav_key != "UAV-01":
    cfg = FLEET_DEMO[selected_uav_key]
    health = clamp(health + cfg["health_bias"])
    rul_bias = cfg["rul_bias"]
else:
    rul_bias = 0.0
rul_h, rul_lo, rul_hi, rul_conf = estimate_rul(frame, health, anomaly_score, df)
rul_h = max(0.05, rul_h + rul_bias)
rul_lo = max(0.01, min(rul_lo + rul_bias, rul_h))
rul_hi = max(rul_h, rul_hi + rul_bias)
completion, mission_risk, recommended_action, expected_completion, margin = mission_metrics(status, health, anomaly_score, rul_h, mission_remaining_h, frame["Altitude"], frame["Ambient_Temp"], frame["Throttle"])
safety_advisory = mission_safety_advisory(status, mission_risk, mission_elapsed_h, mission_duration_h, mission_remaining_h, health, completion)
priority, maintenance_items = maintenance_recommendation(status, faults, subs, 98.0)

sensor_results = sensor_integrity(df, frame, f_sensor)
sensor_health = float(np.mean([x[2] for x in sensor_results]))
cht_slope = trend_slope(df, "CHT", frame["Time"])
egt_slope = trend_slope(df, "EGT", frame["Time"])
oil_slope = trend_slope(df, "T_oil", frame["Time"])
vib_slope = trend_slope(df, "Vibration", frame["Time"])
cht_ttl = time_to_limit(frame["CHT"], cht_slope, ROTAX["cht_max_c"])
egt_ttl = time_to_limit(frame["EGT"], egt_slope, ROTAX["egt_max_c"])
oil_ttl = time_to_limit(frame["Oil_Temp"], oil_slope, ROTAX["oil_temp_max_c"])
vib_ttl = time_to_limit(frame["Vibration"], vib_slope, DRISHTI["vibration_warn_g"])

# ========================= TOP-LEVEL COCKPIT =========================
status_class = {"NORMAL":"status-normal","CAUTION":"status-caution","CRITICAL":"status-critical"}[status]
status_icon = {"NORMAL":"🟢","CAUTION":"🟠","CRITICAL":"🔴"}[status]
status_color = {"NORMAL":"#63df5c","CAUTION":"#f0b52b","CRITICAL":"#e34b4b"}[status]

st.markdown(f"""
<div class='sky-window'>
  <div class='sky-hud'><div class='name'>DRISHTI GCS</div><div class='sub'>ROTAX 914 · MALE UAV ENGINE DIGITAL TWIN</div></div>
  <div class='sky-badges'>
    <div class='sky-badge'><div class='k'>AIRCRAFT / TEST BED</div><div class='v'>{selected_uav}</div></div>
    <div class='sky-badge'><div class='k'>FLIGHT STATUS</div><div class='v'>{'SIMULATION' if scenario_active else 'IN FLIGHT'}</div></div>
  </div>
</div>
<div class='cockpit-shell'>
  <div class='cockpit-strip'>
    <div class='cockpit-lamp lamp-red'>MASTER WARNING</div>
    <div class='cockpit-lamp lamp-amber'>CAUTION</div>
    <div class='cockpit-lamp'>ANNUN TEST</div>
    <div class='cockpit-lamp lamp-green'>SYSTEM OK</div>
  </div>
</div>
""", unsafe_allow_html=True)

if nav_mode == "Live Diagnostic Cockpit":
    st.markdown("<h4 class='section-header cockpit-section-title'>LIVE PROPULSION INSTRUMENT PANEL · UAV TELEMETRY — RECEIVED</h4>", unsafe_allow_html=True)

    # Cosmetic cockpit renderer only. It reads the already-computed `frame` values.
    def _dial(title, value, unit, vmin, vmax, thresholds=None, decimals=1, size="md", sub=""):
        val = safe_float(value, vmin)
        if not np.isfinite(val):
            val = float(vmin)
        val = float(np.clip(val, vmin, vmax))
        span = max(float(vmax) - float(vmin), 1e-9)
        frac = (val - float(vmin)) / span
        angle = -135.0 + 270.0 * frac
        rad = math.radians(angle)
        cx = cy = 100.0
        r = 76.0

        def arc(a0, a1, color, width=8):
            pts = []
            for j in range(40):
                aa = math.radians(a0 + (a1-a0)*j/39)
                pts.append(f"{cx+r*math.sin(aa):.2f},{cy-r*math.cos(aa):.2f}")
            return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'

        svg = [
            '<svg class="dial-svg" viewBox="0 0 200 200" preserveAspectRatio="xMidYMid meet">',
            '<defs><radialGradient id="dialFace" cx="48%" cy="36%"><stop offset="0" stop-color="#30363a"/><stop offset=".58" stop-color="#151a1e"/><stop offset="1" stop-color="#07090b"/></radialGradient>',
            '<linearGradient id="dialNeedle" x1="0" x2="1"><stop offset="0" stop-color="#f5f1df"/><stop offset=".72" stop-color="#cfc7b2"/><stop offset="1" stop-color="#777065"/></linearGradient></defs>',
            '<circle cx="100" cy="100" r="96" fill="#090b0d" stroke="#4b555c" stroke-width="2"/>',
            '<circle cx="100" cy="100" r="91" fill="url(#dialFace)" stroke="#111519" stroke-width="5"/>',
            '<circle cx="100" cy="100" r="86" fill="none" stroke="#687179" stroke-width="1" opacity=".65"/>',
            arc(-135, 135, "#214b2f", 8),
        ]
        if thresholds:
            prev = float(vmin)
            for bound, color in thresholds:
                bound = float(bound)
                if bound > prev:
                    a0 = -135 + 270*((prev-vmin)/span)
                    a1 = -135 + 270*((min(bound,vmax)-vmin)/span)
                    svg.append(arc(a0, a1, color, 8))
                prev = bound

        for i in range(41):
            a = -135 + 270*i/40
            ar = math.radians(a)
            major = (i % 5 == 0)
            outer = 83
            inner = 73 if major else 78
            x1 = cx + outer*math.sin(ar); y1 = cy - outer*math.cos(ar)
            x2 = cx + inner*math.sin(ar); y2 = cy - inner*math.cos(ar)
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#eee9d7" stroke-width="{2 if major else 1}" opacity="{1 if major else .72}"/>')

        for i in range(5):
            a = -135 + 270*i/4
            ar = math.radians(a)
            rr = 62
            x = cx + rr*math.sin(ar); y = cy - rr*math.cos(ar) + 3
            tick = float(vmin) + span*i/4
            fmt = f"{tick:.0f}" if decimals == 0 else f"{tick:.{decimals}f}"
            svg.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="#e4e0d2" font-size="8.5" font-family="Inter,Segoe UI,Arial,sans-serif" text-anchor="middle">{fmt}</text>')

        nx = cx + r*.68*math.sin(rad)
        ny = cy - r*.68*math.cos(rad)
        svg.append(f'<line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="url(#dialNeedle)" stroke-width="3.2" stroke-linecap="round"/>')
        svg.append('<circle cx="100" cy="100" r="8" fill="#20262a" stroke="#d7d0bd" stroke-width="2"/>')
        svg.append('<circle cx="100" cy="100" r="3" fill="#8c8779"/>')
        svg.append('</svg>')

        return f"""
        <div class="dial-wrap {size}">
          <div class="dial-title">{title}</div>
          <div class="dial-unit">{unit}</div>
          <div class="dial-face">{''.join(svg)}
            <div class="dial-digital">{val:.{decimals}f}</div>
            <div class="dial-sub">{sub}</div>
          </div>
        </div>
        """

    def _level(title, value, vmin, vmax, unit, decimals=1, accent="green"):
        val = safe_float(value, vmin)
        if not np.isfinite(val):
            val = float(vmin)
        val = float(np.clip(val, vmin, vmax))
        pct = (val-vmin)/max(vmax-vmin,1e-9)*100
        return f"""
        <div class="v-instrument">
          <div class="v-title">{title}</div><div class="v-unit">{unit}</div>
          <div class="v-scale"><span>{vmax:.0f}</span><span>{(vmin+vmax)/2:.0f}</span><span>{vmin:.0f}</span></div>
          <div class="v-track"><div class="v-fill {accent}" style="height:{pct:.1f}%"></div></div>
          <div class="v-digital">{val:.{decimals}f}</div>
        </div>
        """

    def _command(title, value, min_v, max_v, unit="", decimals=0):
        val = float(np.clip(safe_float(value, min_v), min_v, max_v))
        pct = (val-min_v)/max(max_v-min_v,1e-9)*100
        return f"""
        <div class="command-instrument">
          <div class="command-head"><span>{title}</span><b>{val:.{decimals}f}{unit}</b></div>
          <div class="command-scale"><span>{min_v:g}</span><span>{(min_v+max_v)/2:g}</span><span>{max_v:g}</span></div>
          <div class="command-track"><div class="command-needle" style="left:{pct:.2f}%"></div></div>
        </div>
        """

    cockpit_html = f"""
    <div class="real-cockpit">
      <div class="panel-screws"><i></i><i></i><i></i><i></i></div>
      <div class="annunciator-row">
        <div class="ann red"><span>MASTER</span><b>WARNING</b></div>
        <div class="ann amber"><span>CAUTION</span></div>
        <div class="ann"><span>ANNUN</span><b>TEST</b></div>
        <div class="ann green"><span>SYSTEM</span><b>OK</b></div>
      </div>
      <div class="instrument-layout">
        <div class="primary-grid">
          {_dial("RPM", frame["RPM"], "x1000", 0, 6000, [(5500,"#e9a927"),(5800,"#dc3c3c")], 0, "xl", "CONT 5500 · MAX 5800")}
          {_dial("MAP", frame["MAP"], "inHg", 0, 42, [(35,"#e9a927"),(39,"#dc3c3c")], 1, "lg", "AIRBOX PRESSURE")}
          {_dial("CHT", frame["CHT"], "°C", 40, 145, [(120,"#e9a927"),(135,"#dc3c3c")], 1, "lg", "OEM MAX 135°C")}
          {_dial("EGT", frame["EGT"], "°C", 400, 1000, [(850,"#e9a927"),(950,"#dc3c3c")], 0, "lg", "OEM MAX 950°C")}
          {_dial("OIL PRESSURE", frame["Oil_Pressure"], "bar", 0, 7.5, [(0.8,"#dc3c3c"),(2,"#e9a927"),(5,"#214b2f"),(7,"#dc3c3c")], 2, "md", "REF 2–5 bar")}
          {_dial("OIL TEMP", frame["Oil_Temp"], "°C", 40, 140, [(110,"#e9a927"),(130,"#dc3c3c")], 1, "md", "NORMAL 90–110°C")}
          {_dial("VIBRATION", frame["Vibration"], "g", 0, 4, [(DRISHTI["vibration_warn_g"],"#e9a927")], 2, "md", "DRISHTI MODEL LIMIT")}
          {_dial("ENGINE POWER", frame["Power"], "kW", 0, 90, [(ROTAX["max_power_kw"],"#e9a927")], 1, "md", "84.8 kW REF")}
        </div>
        <div class="quantity-stack">
          {_level("FUEL FLOW", frame["Fuel_Flow"], 0, max(20, frame["Fuel_Flow"]*1.35), "g/s", 1, "green")}
          {_level("BUS VOLTAGE", frame["Battery_Voltage"], 10, 15, "V", 1, "amber")}
          {_level("ALT HEALTH", frame["Alternator_Health"], 0, 100, "%", 0, "green")}
          {_level("FUEL QTY", 62, 0, 100, "%", 0, "amber")}
        </div>
        <div class="command-stack">
          {_command("THROTTLE", frame["Throttle"], 0, 100, "%", 0)}
          {_command("INJECTION TIMING", frame.get("Injection_Timing",0), 0, 40, "° BTDC", 1)}
        </div>
        <div class="alt-health-dial">
          {_dial("ALTERNATOR", frame["Alternator_Health"], "%", 0, 100, [(50,"#dc3c3c"),(75,"#e9a927")], 0, "md", "ELECTRICAL")}
        </div>
      </div>
    </div>
    """
    st.markdown(cockpit_html, unsafe_allow_html=True)
    st.markdown("<div class='cockpit-caption'>RECEIVED TELEMETRY · ENGINE / ECU / FADEC CHANNELS · INSTRUMENTS ARE VISUALIZATION ONLY</div>", unsafe_allow_html=True)

    st.markdown("<h4 class='section-header'>DRISHTI INTELLIGENCE SUMMARY</h4>", unsafe_allow_html=True)
    st.markdown(f"""<div class='info-card' style='padding:7px 10px;margin-bottom:7px'>
      <span style='color:{status_color};font-weight:800;font-size:10px'>{status_icon} ENGINE STATUS: {status}</span>
      <span style='float:right;color:#8996a3;font-size:7px'>MISSION FEASIBILITY {completion:.0f}% · RISK {mission_risk} · COMPLETION {expected_completion}</span>
    </div>""", unsafe_allow_html=True)
    d1,d2,d3,d4,d5,d6 = st.columns(6)
    with d1: st.metric("Engine Health", f"{health:.0f}%")
    with d2: st.metric("Active Faults", f"{len(faults)}")
    with d3: st.metric("Prototype RUL", f"{rul_h:.1f} h")
    with d4: st.metric("Mission Risk", mission_risk)
    with d5: st.metric("Completion Probability", f"{completion:.0f}%")
    with d6: st.metric("Twin Confidence", f"{rul_conf:.0f}%")
    st.caption("Raw/processed telemetry is received from the UAV/ECU/FADEC path; DRISHTI derives health, diagnosis, degradation, RUL, mission risk and maintenance intelligence.")

    st.markdown("<h4 class='section-header'>📚 ROTAX 914 REFERENCE OPERATING MAP</h4>", unsafe_allow_html=True)
    perf = pd.DataFrame([
        [5800, 84.5, 139, 39.0, "TAKEOFF · ≤5 min"],
        [5500, 73.5, 128, 35.0, "MAX CONTINUOUS"],
        [5000, 55.1, 105, 31.0, "75%"],
        [4800, 47.8, 95, 29.0, "65%"],
        [4300, 40.4, 90, 28.0, "55%"],
    ], columns=["RPM","Power kW","Torque Nm","Airbox Pressure inHg","Reference Mode"])
    st.dataframe(perf, hide_index=True, use_container_width=True)
    st.caption("Reference operating points from Rotax 914 operator documentation; exact performance varies with configuration and conditions. These are reference points, not a universal MALE-UAV operating envelope.")

    st.markdown("<h4 class='section-header'>🧠 HYBRID PHYSICS / BASELINE + AI DIAGNOSTICS</h4>", unsafe_allow_html=True)
    a,b = st.columns(2)
    # Healthy-baseline residuals: not mislabeled as OEM physics equations.
    baseline = df[df.get("Anomaly_Flag", pd.Series(0,index=df.index)).fillna(0).astype(int)==0].tail(1000)
    if len(baseline) < 20: baseline = df.tail(min(1000,len(df)))
    exp_cht = float(baseline["CHT"].median())
    exp_egt = float(baseline["EGT"].median())
    exp_map = float(baseline["MAP"].median())
    residuals = {"CHT":frame["CHT"]-exp_cht,"EGT":frame["EGT"]-exp_egt,"MAP":frame["MAP"]-exp_map}
    with a:
        st.markdown(f'<div class="info-card"><b>Healthy Baseline Residuals</b><br>CHT: {residuals["CHT"]:+.1f} °C (baseline {exp_cht:.1f})<br>EGT: {residuals["EGT"]:+.1f} °C (baseline {exp_egt:.1f})<br>MAP: {residuals["MAP"]:+.2f} inHg (baseline {exp_map:.2f})<br>AI anomaly score: <b>{anomaly_score*100:.1f}%</b> ({anomaly_model})<br>Assessment: <b>{anomaly_state}</b></div>', unsafe_allow_html=True)
    with b:
        st.markdown("**Diagnostic Evidence**")
        evidence = []
        evidence.append(("CHT", "CRITICAL" if frame["CHT"]>135 else ("WARNING" if frame["CHT"]>120 else "OK")))
        evidence.append(("EGT", "CRITICAL" if frame["EGT"]>950 else ("WARNING" if frame["EGT"]>850 else "OK")))
        evidence.append(("Oil pressure", "CRITICAL" if oilp_stat=="CRITICAL" else "OK"))
        evidence.append(("Oil temperature", "CRITICAL" if frame["Oil_Temp"]>130 else ("WARNING" if frame["Oil_Temp"]>110 else "OK")))
        evidence.append(("Vibration", "WARNING" if frame["Vibration"]>1.89 else "OK"))
        for label, ev in evidence:
            icon = "🔴" if ev=="CRITICAL" else ("🟠" if ev=="WARNING" else "🟢")
            st.write(f"{icon} {label}: **{ev}**")

    phys = physics_consistency(frame)
    st.markdown(f'<div class="info-card"><b>First-Principles Consistency Check</b><br>Estimated air mass flow: {phys["mdot_air_gps"]:.2f} g/s<br>Expected fuel flow: {phys["expected_fuel_gps"]:.2f} g/s · Actual: {frame["Fuel_Flow"]:.2f} g/s · Residual: {phys["fuel_residual_gps"]:+.2f} g/s<br>Expected brake power: {phys["expected_power_kw"]:.2f} kW · Actual: {frame["Power"]:.2f} kW · Residual: {phys["power_residual_kw"]:+.2f} kW<br><span class="small">Prototype assumptions: ηv=0.85, AFR=14.7, LHV=43 MJ/kg, ηb=0.30.</span></div>', unsafe_allow_html=True)

    st.markdown("<h4 class='section-header'>🧠 EXPLAINABLE AI — WHY IS THE ENGINE IN THIS STATE?</h4>", unsafe_allow_html=True)
    xai_df = xai_local_contributions(df, frame, faults)
    if not xai_df.empty:
        plot_df = xai_df.head(6).sort_values("Contribution", ascending=True)
        fig_xai = px.bar(plot_df, x="Contribution", y="Feature", orientation="h", title="Local risk attribution — baseline deviation + limit proximity")
        fig_xai.update_layout(height=240, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), xaxis=dict(range=[0,1]))
        st.plotly_chart(fig_xai, use_container_width=True)
        top = xai_df.iloc[0]
        st.warning(f"**Primary driver:** {top['Feature']} — contribution {top['Contribution']*100:.0f}%. Current value {top['Value']:.2f}; healthy baseline {top['Healthy Baseline']:.2f}. **Why:** {top['Why']}. **Action:** {top['Corrective Action']}.")
        st.dataframe(xai_df.head(6)[["Feature","Value","Healthy Baseline","Deviation","Limit_Proximity","Contribution","Corrective Action"]], hide_index=True, use_container_width=True)
    else:
        st.info("No sufficient baseline data for local explainability.")

    st.markdown("<h4 class='section-header'>🛠️ LIVE CORRECTIVE SUGGESTIONS</h4>", unsafe_allow_html=True)
    for level, text in live_corrective_suggestions(frame, faults, health, mission_risk, frame["Altitude"], frame["Throttle"], mission_remaining_h):
        icon = "🔴" if level in ("MISSION","LUBRICATION") and mission_risk in ("HIGH","CRITICAL") else "🟠" if level != "MONITOR" else "🟢"
        st.markdown(f'<div class="info-card" style="margin-bottom:6px;"><b>{icon} {level}</b><br>{text}</div>', unsafe_allow_html=True)

    st.markdown("<h4 class='section-header'>❤️ SUBSYSTEM HEALTH & TRACEABLE CONTRIBUTORS</h4>", unsafe_allow_html=True)
    s1,s2 = st.columns([1,2])
    with s1: st.metric("Overall Health", f"{health:.0f}%")
    with s2:
        for k,v in subs[0].items(): st.progress(v/100.0, text=f"{k}: {v:.0f}%")
    top_sub = sorted(subs[0].items(), key=lambda x:x[1])[0]
    st.caption(f"Primary health constraint: {top_sub[0]} subsystem ({top_sub[1]:.0f}%).")

    st.markdown("<h4 class='section-header'>⏳ PROTOTYPE RUL & TIME-TO-LIMIT</h4>", unsafe_allow_html=True)
    r1,r2,r3 = st.columns(3)
    with r1: st.metric("Prototype RUL", f"{rul_h:.1f} h", delta=f"Range {rul_lo:.1f}–{rul_hi:.1f} h")
    with r2: st.metric("Mission Margin", f"{margin:+.1f} h", delta="ENDURANCE RISK" if margin<0 else "POSITIVE MARGIN")
    with r3: st.metric("RUL Confidence", f"{rul_conf:.0f}%", delta=conf_label)
    ttl_rows = [("CHT",cht_ttl),("EGT",egt_ttl),("Oil Temp",oil_ttl),("Vibration",vib_ttl)]
    ttl_df = pd.DataFrame({"Parameter":[x[0] for x in ttl_rows],"Minutes to critical/model limit":[None if not np.isfinite(x[1]) else round(x[1],1) for x in ttl_rows]})
    st.dataframe(ttl_df, hide_index=True, use_container_width=True)

    render_faults(faults, anomaly_state)

    st.markdown("<h4 class='section-header'>📈 Processed Telemetry History & Limit Markers</h4>", unsafe_allow_html=True)
    hist = df[df["Time"] <= frame["Time"]].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Time"], y=hist["CHT"], name="CHT"))
    fig.add_trace(go.Scatter(x=hist["Time"], y=hist["T_oil"], name="Oil Temp"))
    fig.add_hline(y=135, line_dash="dash", annotation_text="OEM CHT MAX 135°C")
    fig.add_vline(x=frame["Time"], line_dash="dot", annotation_text="Current")
    if scenario_active: fig.add_vline(x=frame["Time"], line_dash="solid", annotation_text="Simulation Active")
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

    if "Health_Index" in df.columns:
        st.markdown("<h4 class='section-header'>📉 HEALTH / DEGRADATION TRAJECTORY</h4>", unsafe_allow_html=True)
        hd = df[df["Time"] <= frame["Time"]][["Time","Health_Index"]].dropna()
        if not hd.empty:
            fig_h = px.line(hd, x="Time", y="Health_Index", title="Source Health Index over mission timeline")
            fig_h.add_hline(y=70, line_dash="dash", annotation_text="Degraded reference")
            fig_h.update_layout(height=210, margin=dict(l=0,r=0,t=25,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_h, use_container_width=True)
        ev = build_event_log(df[df["Time"] <= frame["Time"]])
        if not ev.empty:
            st.dataframe(ev.tail(10), hide_index=True, use_container_width=True)

    st.markdown("<h4 class='section-header'>⚖️ Before vs After Scenario</h4>", unsafe_allow_html=True)
    base_faults = evaluate_faults(base)
    base_subs = subsystem_health(base)
    base_score,_ = robust_anomaly_score(df, base)
    base_status,*_ = diagnostic_state(base, base_faults, base_score, base_subs)
    base_rul,_,_,_ = estimate_rul(base, base_subs[1], base_score, df)
    base_comp,*_ = mission_metrics(base_status, base_subs[1], base_score, base_rul, mission_remaining_h, base["Altitude"], base["Ambient_Temp"], base["Throttle"])
    comp_df = pd.DataFrame([
        ["Health Index", round(base_subs[1],1), round(health,1), round(health-base_subs[1],1)],
        ["Anomaly Probability", round(base_score*100,1), round(anomaly_score*100,1), round((anomaly_score-base_score)*100,1)],
        ["Mission Completion %", round(base_comp,1), round(completion,1), round(completion-base_comp,1)],
        ["RUL (h)", round(base_rul,2), round(rul_h,2), round(rul_h-base_rul,2)],
    ], columns=["Metric","Baseline","Scenario","Delta"])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("<h4 class='section-header'>🔄 Counterfactual What-If Mission Simulator</h4>", unsafe_allow_html=True)
    cf = counterfactuals(frame,status,health,anomaly_score,rul_h,mission_remaining_h,profile)
    cf_df = pd.DataFrame([[x[0],round(x[1],1),round(x[2],1),round(x[3],0)] for x in cf], columns=["Scenario","Mission Completion %","Throttle %","Altitude ft"])
    st.dataframe(cf_df, hide_index=True, use_container_width=True)
    best = max(cf, key=lambda x:x[1])
    st.info(f"🧠 **DRISHTI recommendation:** {best[0]} → estimated completion probability **{best[1]:.1f}%**. This is a prototype counterfactual model, not a flight-control command.")

    st.markdown("<h4 class='section-header'>📊 ENGINE EFFICIENCY TREND</h4>", unsafe_allow_html=True)
    eff = df[df["Time"] <= frame["Time"]].copy()
    eff["Power_kW"] = eff["Power"] / 1000.0 if eff["Power"].median() > 1000 else eff["Power"]
    # BSFC is displayed as supplied by the dataset; no OEM optimum is asserted.
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(x=eff["Time"], y=eff["BSFC"], name="BSFC"))
    fig_eff.update_layout(height=210, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), yaxis_title="BSFC (dataset units)" )
    st.plotly_chart(fig_eff, use_container_width=True)

    st.markdown("<h4 class='section-header'>🚀 MISSION INTELLIGENCE & OPERATIONAL DECISION</h4>", unsafe_allow_html=True)
    mi1,mi2 = st.columns(2)
    with mi1:
        st.write(f"**Mission:** {mission_name}")
        st.write(f"**Altitude:** {frame['Altitude']:.0f} ft · **ISA ambient:** {isa_temperature_c(frame['Altitude']):.1f} °C · **Actual ambient:** {frame['Ambient_Temp']:.1f} °C · **Throttle:** {frame['Throttle']:.0f}%")
        st.write(f"**Elapsed:** {frame['Time']/3600:.2f} h · **Remaining:** {mission_remaining_h:.1f} h")
        st.write(f"**RUL:** {rul_h:.1f} h · **Mission Margin:** {margin:+.1f} h")
        st.write(f"**Completion Probability:** {completion:.1f}% · **Risk:** {mission_risk}")
    with mi2:
        st.markdown(f'<div class="info-card"><div class="small">DRISHTI OPERATIONAL DECISION</div><h3 style="margin:4px 0">{recommended_action}</h3><div>Expected completion: <b>{expected_completion}</b></div><div>Maintenance priority: <b>{priority}</b></div></div>', unsafe_allow_html=True)

    st.markdown("<h4 class='section-header'>🛬 MISSION SAFETY & DIVERSION ADVISORY</h4>", unsafe_allow_html=True)
    adv = safety_advisory
    advisory_html = (
        '<div class="info-card"><div class="small">DRISHTI OPERATIONAL DECISION — PROTOTYPE</div>'
        f'<h3 style="margin:4px 0">{adv["decision"]}</h3>'
        f'<div>{adv["reason"]}</div><br>'
        f'<div><b>Home distance:</b> {adv["home_distance_nm"]:.1f} NM &nbsp; | &nbsp; '
        f'<b>Nearest suitable airport:</b> {adv["nearest_airport_distance_nm"]:.1f} NM</div>'
        f'<div style="margin-top:6px"><b>ATC / DISTRESS ACTION:</b> {adv["atc"]}</div></div>'
    )
    st.markdown(advisory_html, unsafe_allow_html=True)
    if adv["eta_min"] is not None:
        st.info(f"🧭 Navigation engine: calculating diversion ETA ≈ **{adv['eta_min']:.0f} min**. Airport routing/ATC transmission are simulated in this prototype.")

    st.markdown("<h4 class='section-header'>📡 SENSOR INTEGRITY & DRIFT</h4>", unsafe_allow_html=True)
    for label,drift,rel,stat,unit in sensor_results:
        st.write(f"{label}: drift {drift:+.3f}{unit} · reliability {rel:.1f}% · **{stat}**")

    st.markdown("<h4 class='section-header'>⚙️ DIGITAL TWIN SYNCHRONIZATION</h4>", unsafe_allow_html=True)
    st.write(f"Telemetry point: **{frame['Time']:.0f} s** · Acquisition path: **UAV / ECU / FADEC telemetry**")
    st.write(f"Baseline model: **ACTIVE** · AI anomaly model: **{anomaly_model}** · Fault logic: **ACTIVE** · Mission model: **ACTIVE**")
    st.write(f"Sensor integrity: **{sensor_health:.1f}%** · Scenario mode: **{'ACTIVE' if scenario_active else 'OFF'}**")

    st.markdown("<h4 class='section-header'>📋 DRISHTI MAINTENANCE ACTION PLANNER</h4>", unsafe_allow_html=True)
    st.write(f"**Priority:** {priority}")
    for item in maintenance_items: st.write(f"• {item}")

    st.markdown("<h4 class='section-header'>📜 ENGINE HEALTH PASSPORT</h4>", unsafe_allow_html=True)
    passport = pd.DataFrame({"Metric":["Engine","Tail","Dataset records","TBO reference","Current health","Current diagnostic state","Current RUL estimate","RUL confidence","Scenario"],"Value":["Rotax 914 UL/F",selected_uav,f"{len(df):,}","2000 h reference",f"{health:.1f}%",anomaly_state,f"{rul_h:.2f} h",f"{rul_conf:.0f}%",mission_name]})
    st.dataframe(passport, hide_index=True, use_container_width=True)

elif nav_mode == "Fleet Digital Twin":
    st.markdown("<h4 class='section-header'>🛩️ FLEET DIGITAL TWIN</h4>", unsafe_allow_html=True)
    fleet_rows = []
    # UAV-01 is the primary live/demo test bed; other rows are demonstration fleet states.
    for tail, cfg in [("UAV-01", FLEET_DEMO["UAV-01"]), ("UAV-02", FLEET_DEMO["UAV-02"]), ("UAV-03", FLEET_DEMO["UAV-03"]), ("UAV-04", FLEET_DEMO["UAV-04"]), ("UAV-05", FLEET_DEMO["UAV-05"])]:
        demo_frame = apply_uav_state(apply_mission_controls(nearest_frame(df, playback_time_sec), altitude_ft, throttle_pct, mission_duration_h), tail)
        demo_faults = evaluate_faults(demo_frame)
        demo_subs = subsystem_health(demo_frame)
        demo_score,_ = robust_anomaly_score(df, demo_frame)
        demo_status,*_ = diagnostic_state(demo_frame, demo_faults, demo_score, demo_subs)
        demo_health = clamp(demo_subs[1] + cfg["health_bias"])
        demo_rul,_,_,_ = estimate_rul(demo_frame, demo_health, demo_score, df)
        demo_rul = max(0.05, demo_rul + cfg["rul_bias"])
        demo_risk = "CRITICAL" if demo_status == "CRITICAL" else ("HIGH" if demo_health < 65 else ("MEDIUM" if demo_health < 80 else "LOW"))
        fleet_rows.append([tail,"Rotax 914 UL/F",round(demo_health,1),round(demo_rul,1),demo_risk,cfg["state"],round(demo_frame["RPM"]),round(demo_frame["CHT"],1),round(demo_frame["EGT"],1),round(demo_frame["Vibration"],2)])
    fleet = pd.DataFrame(fleet_rows, columns=["Tail","Engine","Health %","RUL h","Risk","State","RPM","CHT °C","EGT °C","Vibration g"])
    st.dataframe(fleet, hide_index=True, use_container_width=True)
    st.caption("UAV-01 is the DRISHTI primary test bed. Other fleet records are demonstration states used to show fleet-level monitoring and maintenance prioritization.")
    fig = px.bar(fleet, x="Tail", y="Health %", color="Risk", title="Fleet Health")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="white"))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Selected Aircraft Telemetry")
    st.write(f"**{selected_uav_key}** is selected in Aircraft Trail. The cockpit values above are computed from that aircraft's telemetry state, not copied from UAV-01.")
    st.write(f"RPM {frame['RPM']:.0f} · CHT {frame['CHT']:.1f} °C · EGT {frame['EGT']:.1f} °C · Oil pressure {frame['Oil_Pressure']:.2f} bar · Vibration {frame['Vibration']:.2f} g · Health {health:.1f}% · RUL {rul_h:.1f} h")

elif nav_mode == "Mission Replay & Analytics":
    st.markdown("<h4 class='section-header'>🔄 MISSION REPLAY & ANALYTICS</h4>", unsafe_allow_html=True)
    t = playback_time_sec
    r = derive_missing_channels(nearest_frame(df,t),profile)
    r = apply_mission_controls(r, altitude_ft, throttle_pct, mission_duration_h)
    r = apply_uav_state(r, selected_uav_key)
    st.json({k:v for k,v in r.items() if k not in ("Mission_Surface",)})
    hist=df[df["Time"]<=t]
    fig=px.line(hist,x="Time",y=["RPM","CHT","EGT","P_oil","T_oil"],title="Mission Replay Telemetry")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="white"))
    st.plotly_chart(fig,use_container_width=True)

else:
    st.markdown("<h4 class='section-header'>📑 POST-FLIGHT REPORT GENERATOR</h4>", unsafe_allow_html=True)
    report = {
        "aircraft": selected_uav,
        "engine": "Rotax 914 UL/F reference engine",
        "mission_profile": mission_name,
        "dataset_records": int(len(df)),
        "peak_cht_c": float(df["CHT"].max()),
        "peak_egt_c": float(df["EGT"].max()),
        "minimum_oil_pressure_bar": float(df["P_oil"].min()),
        "peak_oil_temp_c": float(df["T_oil"].max()),
        "peak_vibration": float(df["Vibration"].max()),
        "current_health": float(health),
        "current_status": status,
        "current_rul_h_prototype": float(rul_h),
        "current_rul_confidence_pct": float(rul_conf),
        "mission_completion_pct": float(completion),
        "mission_risk": mission_risk,
        "recommendation": recommended_action,
        "mission_safety_advisory": safety_advisory,
        "confirmed_faults": faults,
    }
    st.json(report)
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    st.download_button("📥 Download Mission Health Report (JSON)", report_bytes, file_name=f"{selected_uav}_DRISHTI_report.json", mime="application/json")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Processed Telemetry (CSV)", csv_bytes, file_name=f"{selected_uav}_telemetry.csv", mime="text/csv")
