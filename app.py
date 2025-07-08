# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import requests

# ---------- 1. Load trained model ----------
MODEL_PATH = Path("ev_charger_predictor.joblib")
if not MODEL_PATH.exists():
    st.error(f"Model file not found. Train and export the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------- 2. Geocoding helper ----------
def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params, headers={"User-Agent": "streamlit-ev-app"})
    if response.status_code == 200 and response.json():
        result = response.json()[0]
        return float(result["lat"]), float(result["lon"])
    return None, None

# ---------- 3. Streamlit UI ----------
st.set_page_config(page_title="EV Charger Predictor", page_icon="⚡")
st.title("⚡ EV Charging Sessions Predictor")

st.markdown("""
Enter a **location address** and **type of area** to predict how many EV charging sessions this location might attract daily.
""")

address = st.text_input("📍 Address", "Meir 50, Antwerp, Belgium")
location_type = st.selectbox(
    "🏙️ Location type",
    [
        "retail store",
        "office building",
        "public parking",
        "residential area",
        "hospital",
        "recreational facility",
    ]
)

# ---------- 4. Run prediction ----------
if st.button("Predict"):
    with st.spinner("Geocoding address..."):
        lat, lon = geocode_address(address)

    if lat is None:
        st.error("Could not find the address. Please try a more specific one.")
    else:
        # Optional: show map
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

        X_new = pd.DataFrame({
            "evse_latitude": [lat],
            "evse_longitude": [lon],
            "Pool_SiteType": [location_type]
        })

        prediction = model.predict(X_new)[0]
        st.success(f"⚡ Estimated daily charging sessions: **{prediction:.1f}**")
