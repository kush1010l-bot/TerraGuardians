import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from twilio.rest import Client
from supabase import create_client, Client as SupabaseClient
import os

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Solar Edge-AI Irrigation & Landslide Risk Monitoring",
    page_icon="🌱",
    layout="wide"
)

# -----------------------
# LOAD SECRETS
# -----------------------
TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]
FARMER_PHONE_NUMBER = st.secrets.get("FARMER_PHONE_NUMBER", "+9779861044678")
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SMS_PASSWORD = st.secrets.get("SMS_PASSWORD", "terraguardian123")

# -----------------------
# INITIALIZE CLIENTS
# -----------------------
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# SIMULATED AI MODEL CLASS
# -----------------------
class LandslideRiskModel:
    def __init__(self):
        self.model_loaded = True
        self.training_data_size = 12500
        self.accuracy = 0.87

    def predict_risk(self, soil_moisture, rainfall_24h, slope_angle,
                     antecedent_rain_7d=None, soil_type=None):
        saturation_ratio = soil_moisture / 50.0
        if antecedent_rain_7d is None:
            antecedent_rain_7d = rainfall_24h * 3.5

        risk_score = (
            0.35 * saturation_ratio +
            0.30 * min(1.0, rainfall_24h / 50) +
            0.25 * min(1.0, slope_angle / 40) +
            0.10 * min(1.0, antecedent_rain_7d / 100)
        )
        interaction_term = saturation_ratio * (slope_angle / 40) * 0.2
        risk_score += interaction_term
        risk_score += np.random.normal(0, 0.02)
        return np.clip(risk_score, 0, 1)

    def predict_irrigation_need(self, soil_moisture, temp, crop_type="rice",
                                growth_stage="vegetative"):
        crop_coefficients = {
            "rice": {"initial": 1.1, "vegetative": 1.2, "reproductive": 1.3, "mature": 0.9},
            "maize": {"initial": 0.8, "vegetative": 1.1, "reproductive": 1.2, "mature": 0.7},
            "vegetables": {"initial": 0.7, "vegetative": 1.0, "reproductive": 1.05, "mature": 0.8}
        }
        kc = crop_coefficients.get(crop_type, crop_coefficients["rice"]).get(growth_stage, 1.0)
        temp_factor = 0.05 * (temp - 15) if temp > 15 else 0.02 * temp
        moisture_deficit = max(0, (30 - soil_moisture) / 30)
        irrigation_mm = kc * 5.0 * (1 + temp_factor) * (1 + moisture_deficit)
        return round(irrigation_mm, 1)

# -----------------------
# LOAD MODEL (cached)
# -----------------------
@st.cache_resource
def load_model():
    return LandslideRiskModel()

model = load_model()

# -----------------------
# UI HEADER
# -----------------------
st.title("🌱 Solar Edge-AI Irrigation & Landslide Risk \nMonitoring")
st.markdown("**Inspired by SOTER Nepal soil database & historical landslide records**")

# -----------------------
# SIDEBAR FOR INPUTS
# -----------------------
st.sidebar.header("Sensor Inputs (Simulated)")
soil = st.sidebar.slider("Soil Moisture (%)", 0, 50, 20)
rain = st.sidebar.slider("Rainfall last 24h (mm)", 0, 20, 2)
temp = st.sidebar.slider("Temperature (°C)", 10, 40, 25)
slope = st.sidebar.slider("Slope Angle (°)", 0, 40, 20)
crop_type = st.sidebar.selectbox("Crop Type", ["rice", "maize", "vegetables"])
growth_stage = st.sidebar.selectbox("Growth Stage", ["initial", "vegetative", "reproductive", "mature"])

with st.sidebar.expander("Advanced"):
    antecedent_rain = st.number_input("Antecedent Rain (7-day, mm)", min_value=0, max_value=200, value=50)
    soil_type = st.selectbox("Soil Type", ["silty loam", "clay loam", "sandy loam", "loam"])

# SMS recipients
st.sidebar.markdown("---")
st.sidebar.subheader("📱 SMS Recipients")
recipient_numbers = st.sidebar.text_input(
    "Enter phone number(s) with country code",
    value=FARMER_PHONE_NUMBER,
    help="For multiple numbers, separate with commas (e.g., +97798..., +97798...)"
)

# -----------------------
# MAIN PANEL
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Sensor Readings")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soil,
        title={'text': "Soil Moisture (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 20], 'color': "lightgray"},
                   {'range': [20, 35], 'color': "gray"},
                   {'range': [35, 50], 'color': "darkgray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75,
                             'value': 35}}))
    st.plotly_chart(gauge_fig, use_container_width=True)

with col2:
    st.subheader("Run Analysis")
    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.spinner("AI model running inference..."):
            # Predictions
            risk_score = model.predict_risk(
                soil_moisture=soil,
                rainfall_24h=rain,
                slope_angle=slope,
                antecedent_rain_7d=antecedent_rain,
                soil_type=soil_type
            )
            irrigation_mm = model.predict_irrigation_need(
                soil_moisture=soil,
                temp=temp,
                crop_type=crop_type,
                growth_stage=growth_stage
            )

            # Determine messages based on risk
            if risk_score > 0.7:
                risk_level = "🔴 HIGH"
                landslide_msg = f"⚠️ HIGH landslide risk ({risk_score:.1%} probability). EVACUATE if on slope. DO NOT irrigate."
                irrigation_msg = "❌ DO NOT IRRIGATE – High landslide risk! Stay away from the Terrace."
            elif risk_score > 0.4:
                risk_level = "🟡 MODERATE"
                landslide_msg = f"⚠️ Moderate landslide risk ({risk_score:.1%} probability). Avoid irrigation, monitor closely."
                irrigation_msg = "⚠️ Irrigate with extreme caution or delay – moderate landslide risk."
            else:
                risk_level = "🟢 LOW"
                landslide_msg = f"✅ Low landslide risk ({risk_score:.1%} probability). Safe for normal irrigation."
                if irrigation_mm > 10:
                    irrigation_msg = f"✅ Irrigation recommended: {irrigation_mm} mm"
                elif irrigation_mm > 3:
                    irrigation_msg = f"⚠️ Light irrigation recommended: {irrigation_mm} mm"
                else:
                    irrigation_msg = "❌ No irrigation needed (sufficient soil moisture)"

            # Timestamp
            now = datetime.now().isoformat()

            # Prepare data for database
            record = {
                "timestamp": now,
                "soil_moisture": soil,
                "rainfall_24h": rain,
                "temperature": temp,
                "slope_angle": slope,
                "crop_type": crop_type,
                "growth_stage": growth_stage,
                "antecedent_rain": antecedent_rain,
                "soil_type": soil_type,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "irrigation_mm": irrigation_mm,
                "landslide_msg": landslide_msg,
                "irrigation_msg": irrigation_msg
            }

            # Save to Supabase
            try:
                supabase.table("advisories").insert(record).execute()
                st.success("✅ Analysis saved to database.")
            except Exception as e:
                st.warning(f"Could not save to database: {e}")

            # ----- FIXED: Display results with risk‑appropriate irrigation metric -----
            if risk_score > 0.7:
                display_irrigation = "0 mm (DO NOT IRRIGATE)"
            elif risk_score > 0.4:
                display_irrigation = f"{irrigation_mm} mm (irrigate only if essential – moderate risk)"
            else:
                display_irrigation = f"{irrigation_mm} mm"

            st.metric("Landslide Risk Score", f"{risk_score:.1%}", delta=None)
            st.metric("Recommended Irrigation", display_irrigation)
            st.info(landslide_msg)
            st.info(irrigation_msg)

            # Store data in session state for later SMS
            st.session_state['analysis_done'] = True
            st.session_state['full_message'] = (
                f"AI Advisory {now[:10]}\n"
                f"Soil:{soil}% Rain:{rain}mm Slope:{slope}°\n"
                f"{landslide_msg[:100]}\n{irrigation_msg}"
            )
            st.session_state['recipient_numbers'] = recipient_numbers

# -----------------------
# SMS SENDING SECTION (appears after analysis)
# -----------------------
if st.session_state.get('analysis_done'):
    st.markdown("---")
    st.subheader("📱 Send SMS with Terraguardian Key")

    with st.expander("Click to enter password and send SMS"):
        sms_password_input = st.text_input(
            "Enter Terraguardian Key",
            type="password",
            key="sms_password"
        )
        if st.button("Send SMS Now"):
            if sms_password_input == SMS_PASSWORD:
                # Parse recipients
                raw_numbers = st.session_state['recipient_numbers'].strip()
                if raw_numbers:
                    numbers = [num.strip() for num in raw_numbers.split(",") if num.strip()]
                else:
                    numbers = [FARMER_PHONE_NUMBER]

                if numbers:
                    success_count = 0
                    error_count = 0
                    for number in numbers:
                        try:
                            message = twilio_client.messages.create(
                                body=st.session_state['full_message'],
                                from_=TWILIO_PHONE_NUMBER,
                                to=number
                            )
                            success_count += 1
                        except Exception as e:
                            error_count += 1
                            st.warning(f"Failed to send to {number}: {e}")
                    if success_count > 0:
                        st.success(f"✅ SMS sent to {success_count} recipient(s).")
                    if error_count > 0:
                        st.error(f"❌ Failed to send to {error_count} recipient(s).")
                else:
                    st.warning("No valid phone numbers provided.")
            else:
                st.error("Incorrect Terraguardian Key. SMS not sent.")

# -----------------------
# HISTORY SECTION
# -----------------------
st.subheader("📊 Past Advisories")
try:
    response = supabase.table("advisories").select("*").order("timestamp", desc=True).limit(10).execute()
    if response.data:
        df = pd.DataFrame(response.data)
        display_cols = ["timestamp", "soil_moisture", "rainfall_24h", "slope_angle", "risk_level", "irrigation_mm"]
        st.dataframe(df[display_cols])
    else:
        st.info("No historical data yet. Run an analysis to see history.")
except Exception as e:
    st.warning(f"Could not load history: {e}")

# -----------------------
# MODEL INFO (with password explanation)
# -----------------------
with st.expander("About the AI Model (Demo Simulation)"):
    st.write("**Note:** This is a demonstration prototype. The AI logic is currently simulated using a rule‑based formula to illustrate the concept.")
    st.write("**Proposed Model for Future Implementation:** Ensemble combining XGBoost (for tabular sensor data) and LSTM (for time‑series patterns like antecedent rainfall).")
    st.write("**Proposed Training Data:** SOTER Nepal soil database (real, publicly available) + field‑collected data on rainfall, slope, and crop coefficients.")
    st.write("**Features used in concept:** Soil moisture, rainfall, slope angle, antecedent rainfall, soil type, crop coefficients.")
    st.write("**Current demo logic:** Weighted formula with a small random factor (not a trained ML model).")
    st.markdown("---")
    st.subheader("🔐 About SMS Password Protection")
    st.write("""
    To prevent accidental SMS costs, sending messages requires a **Terraguardian Key**. 
    After running an analysis, a separate "Send SMS" section appears where you can enter the key.
    This key is known only to team members. If you're a judge and wish to receive an SMS, 
    please ask a team member during the demo – we'll be happy to send one!
    """)
