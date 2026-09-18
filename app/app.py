import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Chennai Flood Risk Prediction", page_icon="🌊", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_spatial.pkl"
MAP_PATH = ROOT / "outputs" / "chennai_flood_risk_map.html"

st.title("🌊 Chennai Flood Risk Prediction")
st.caption("AI-assisted risk estimate. This is a probability estimate, not a guarantee.")

try:
    model = joblib.load(MODEL_PATH)
    st.success("AI Flood Risk Model Loaded")
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

st.header("Enter Location & Environmental Factors")

col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input("Elevation (m)", value=10.0)
    slope = st.number_input("Slope (degrees)", value=2.0)
    river_distance = st.number_input("Distance from River (m)", value=500.0)

with col2:
    land_cover = st.selectbox("Land Cover", [10, 20, 30, 40, 50, 60, 80, 90], index=4)
    latitude = st.number_input("Latitude", value=13.05000, format="%.5f")
    longitude = st.number_input("Longitude", value=80.25000, format="%.5f")

if st.button("🔍 PREDICT FLOOD RISK", use_container_width=True):

    X = pd.DataFrame([{
        "elevation_m": elevation,
        "slope_deg": slope,
        "distance_to_river_m": river_distance,
        "land_cover": land_cover,
        "latitude": latitude,
        "longitude": longitude
    }])

    probabilities = model.predict_proba(X)[0]
    prediction = int(model.predict(X)[0])

    labels = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    risk = labels[prediction]
    high_probability = probabilities[2] * 100

    st.header("Prediction Result")

    a, b = st.columns(2)

    with a:
        st.metric("Estimated HIGH-Risk Probability", f"{high_probability:.1f}%")

    with b:
        st.metric("Risk Level", risk)

    if risk == "HIGH":
        st.error("🔴 HIGH FLOOD RISK")
    elif risk == "MODERATE":
        st.warning("🟡 MODERATE FLOOD RISK")
    else:
        st.success("🟢 LOW FLOOD RISK")

    st.subheader("Model Probabilities")

    st.write(f"🟢 LOW: {probabilities[0] * 100:.1f}%")
    st.write(f"🟡 MODERATE: {probabilities[1] * 100:.1f}%")
    st.write(f"🔴 HIGH: {probabilities[2] * 100:.1f}%")

    st.info("Factors used: elevation, slope, distance from river, land cover and geographic location.")

if MAP_PATH.exists():
    st.header("🗺️ Chennai Flood Risk Map")
    html = MAP_PATH.read_text(encoding="utf-8")
    components.html(html, height=650, scrolling=True)
