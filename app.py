import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Smart Air Purification System", page_icon="🍃", layout="wide")

# Use a sleek dark theme via markdown
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .status-good { color: #00e676; font-weight: bold; font-size: 24px; }
    .status-moderate { color: #ffeb3b; font-weight: bold; font-size: 24px; }
    .status-poor { color: #ff9800; font-weight: bold; font-size: 24px; }
    .status-hazardous { color: #f44336; font-weight: bold; font-size: 24px; }
    .purifier-on {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        animation: pulse 2s infinite;
    }
    .purifier-off {
        background-color: #757575;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('ann_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error("Model files not found. Please train the model first by running `train_model.py`.")
        return None, None

model, scaler = load_model()

# AQI Classification Logic
def classify_aqi(aqi):
    if aqi <= 50: return "Good", "status-good"
    elif aqi <= 100: return "Moderate", "status-moderate"
    elif aqi <= 200: return "Poor", "status-poor"
    else: return "Hazardous", "status-hazardous"

# Initialize Session State for Historical Data
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'AQI'])

st.title("🍃 ANN Based Smart Air Purification System")
st.markdown("Monitor real-time environmental data, visualize historical trends, and forecast tomorrow's air quality.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🔴 Live Monitoring", "📊 Historical Data & Visualization", "🔮 Tomorrow's Forecast"])

# ----------------------------------------
# TAB 1: LIVE MONITORING
# ----------------------------------------
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎛️ Live Sensor Panel")
        st.markdown("Slide to simulate live sensor data streaming in.")
        
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 45.0, key='pm25_slider')
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 60.0, key='pm10_slider')
        temp = st.slider("Temperature (°C)", 10.0, 45.0, 25.0, key='temp_slider')
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 50.0, key='humidity_slider')
        co = st.slider("CO (ppm)", 0.0, 20.0, 1.5, key='co_slider')
        no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 30.0, key='no2_slider')
        
        # Store latest values in session state for forecasting tab
        st.session_state['latest_pm25'] = pm25
        st.session_state['latest_pm10'] = pm10
        st.session_state['latest_temp'] = temp
        st.session_state['latest_humidity'] = humidity
        st.session_state['latest_co'] = co
        st.session_state['latest_no2'] = no2
        
        simulate = st.checkbox("Auto-Simulate Live Data Updates")
        if simulate:
            pm25 = max(0, pm25 + np.random.normal(0, 2))
            pm10 = max(0, pm10 + np.random.normal(0, 3))
            temp = temp + np.random.normal(0, 0.5)
            humidity = max(10, min(100, humidity + np.random.normal(0, 1)))

    with col2:
        if model and scaler:
            input_data = pd.DataFrame([[pm25, pm10, temp, humidity, co, no2]], 
                                      columns=['PM2.5', 'PM10', 'Temperature', 'Humidity', 'CO', 'NO2'])
            scaled_input = scaler.transform(input_data)
            predicted_aqi = model.predict(scaled_input)[0]
            predicted_aqi = max(0, predicted_aqi) 
            
            status_text, status_class = classify_aqi(predicted_aqi)
            
            purifier_status = "ON" if predicted_aqi > 100 else "STANDBY"
            purifier_class = "purifier-on" if predicted_aqi > 100 else "purifier-off"
            
            current_time = time.strftime("%H:%M:%S")
            new_row = pd.DataFrame({'Time': [current_time], 'AQI': [predicted_aqi]})
            st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(30)
            
            st.markdown("### 📊 Live System Status")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'''
                    <div class="metric-card">
                        <h4>Predicted AQI</h4>
                        <h2 style="color: #2196F3;">{predicted_aqi:.1f}</h2>
                    </div>
                ''', unsafe_allow_html=True)
            with m2:
                st.markdown(f'''
                    <div class="metric-card">
                        <h4>Air Quality</h4>
                        <span class="{status_class}">{status_text}</span>
                    </div>
                ''', unsafe_allow_html=True)
            with m3:
                st.markdown(f'''
                    <div class="metric-card">
                        <h4>Air Purifier Status</h4>
                        <div class="{purifier_class}">{purifier_status}</div>
                    </div>
                ''', unsafe_allow_html=True)
                
            st.markdown("#### AQI Trend (Last 30 Readings)")
            if len(st.session_state.history) > 1:
                st.line_chart(st.session_state.history.set_index('Time'), use_container_width=True)
                
            if simulate:
                time.sleep(1.5)
                st.rerun()

# ----------------------------------------
# TAB 2: HISTORICAL DATA & VISUALIZATION
# ----------------------------------------
with tab2:
    st.header("📊 Data Visualizer")
    st.markdown("Upload a CSV or Excel file containing air purification data. The program will automatically visualize trends using charts.")
    
    uploaded_file = st.file_uploader("📂 Upload Data File", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"Successfully loaded '{uploaded_file.name}' with {len(df)} rows.")
            
            with st.expander("👀 View Raw Data (First 100 Rows)", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
                
            # Line Chart
            st.markdown("### 📈 Air Pollutant Trends")
            expected_pollutants = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'CO', 'NO2']
            available_cols = [col for col in expected_pollutants if col in df.columns]
            
            if available_cols:
                st.line_chart(df[available_cols], use_container_width=True)
            else:
                st.warning("Could not find standard pollutant columns (PM2.5, PM10, Temperature, Humidity, CO, NO2) in the uploaded dataset to plot trends.")
            
            # Pie Chart
            if 'AQI' in df.columns:
                st.markdown("### 🥧 AQI Classification Distribution")
                class_labels = []
                for aqi_val in df['AQI']:
                    lbl, _ = classify_aqi(aqi_val)
                    class_labels.append(lbl)
                    
                df['AQI_Class'] = class_labels
                class_counts = df['AQI_Class'].value_counts()
                
                # Setup colors matching the classes
                color_map = {
                    'Good': '#00e676',
                    'Moderate': '#ffeb3b',
                    'Poor': '#ff9800',
                    'Hazardous': '#f44336'
                }
                colors = [color_map.get(lbl, '#9e9e9e') for lbl in class_counts.index]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0e1117') # Streamlit dark mode backdrop
                ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', 
                       startangle=140, colors=colors, textprops={'color':"w", 'fontweight':'bold'})
                ax.axis('equal')
                
                colA, colB, colC = st.columns([1,2,1])
                with colB:
                    st.pyplot(fig)
            else:
                st.info("No 'AQI' column found in dataset. Unable to plot Air Quality Pie Chart.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ----------------------------------------
# TAB 3: FORECASTING (TOMORROW'S PREDICTION)
# ----------------------------------------
with tab3:
    st.header("🔮 Next-Day Forecasting")
    st.markdown("Based on today's averages, algorithmically project tomorrow's parameters and predict if the Air Purifier will need to run.")
    
    st.markdown("#### 1. Analyze Today's Baseline")
    col1, col2 = st.columns(2)
    
    # Let user define "Today's Averages" (Prefilling with the latest sensor values if available)
    today_pm25 = st.number_input("Today's Avg. PM2.5", min_value=0.0, value=st.session_state.get('latest_pm25', 45.0), step=1.0)
    today_pm10 = st.number_input("Today's Avg. PM10", min_value=0.0, value=st.session_state.get('latest_pm10', 60.0), step=1.0)
    today_temp = st.number_input("Today's Avg. Temperature", value=st.session_state.get('latest_temp', 25.0), step=1.0)
    today_hum = st.number_input("Today's Avg. Humidity (%)", value=st.session_state.get('latest_humidity', 50.0), step=1.0)
    today_co = st.number_input("Today's Avg. CO (ppm)", min_value=0.0, value=st.session_state.get('latest_co', 1.5), step=0.1)
    today_no2 = st.number_input("Today's Avg. NO2 (ppb)", min_value=0.0, value=st.session_state.get('latest_no2', 30.0), step=1.0)

    st.markdown("#### 2. Forecast Trend Calculation")
    st.info("The algorithm applies a simulated varying environmental penalty or improvement (based on assumed accumulation overnight or changing wind conditions) to predict tomorrow's factors.")

    if st.button("🚀 Run Tomorrow's Forecast", use_container_width=True):
        if model and scaler:
            # Algorithmic Projection:
            # Example heuristic: Temperatures fluctuate slightly, pollutants might accumulate (+10%) or disperse (-10%)
            trend_factor = np.random.uniform(0.90, 1.15) # Simulated day-to-day weather variance
            
            tomorrow_pm25 = max(0, today_pm25 * trend_factor)
            tomorrow_pm10 = max(0, today_pm10 * trend_factor)
            tomorrow_temp = today_temp + np.random.uniform(-3, 3) 
            tomorrow_hum = max(10, min(100, today_hum + np.random.uniform(-10, 10)))
            tomorrow_co = max(0, today_co * (trend_factor * 0.9)) # Gas might disperse differently
            tomorrow_no2 = max(0, today_no2 * trend_factor)
            
            # Predict
            tom_input = pd.DataFrame([[tomorrow_pm25, tomorrow_pm10, tomorrow_temp, tomorrow_hum, tomorrow_co, tomorrow_no2]], 
                                      columns=['PM2.5', 'PM10', 'Temperature', 'Humidity', 'CO', 'NO2'])
            scaled_tom = scaler.transform(tom_input)
            tom_predicted_aqi = model.predict(scaled_tom)[0]
            tom_predicted_aqi = max(0, tom_predicted_aqi) 
            
            tom_status, tom_class = classify_aqi(tom_predicted_aqi)
            tom_purifier = "ON" if tom_predicted_aqi > 100 else "STANDBY"
            purifier_color = "#4CAF50" if tom_predicted_aqi > 100 else "#757575"
            
            st.divider()
            st.markdown(f"### 📍 Tomorrow's Predicted Output")
            
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(f'''
                    <div class="metric-card">
                        <h4>Expected AQI</h4>
                        <h2 style="color: #2196F3;">{tom_predicted_aqi:.1f}</h2>
                    </div>
                ''', unsafe_allow_html=True)
            with fc2:
                st.markdown(f'''
                    <div class="metric-card">
                        <h4>Expected Air Quality</h4>
                        <span class="{tom_class}">{tom_status}</span>
                    </div>
                ''', unsafe_allow_html=True)
            with fc3:
                st.markdown(f'''
                    <div class="metric-card">
                        <h4>Purifier Prediction</h4>
                        <div style="background-color: {purifier_color}; color: white; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 20px;">{tom_purifier}</div>
                    </div>
                ''', unsafe_allow_html=True)
                
            st.markdown(f"**Insight:** Based on today's baseline, the system projects that tomorrow the average particulate matter (PM2.5) will be approximately **{tomorrow_pm25:.1f} µg/m³**, causing the expected AQI to be **{tom_predicted_aqi:.0f}**. The Air Purifier will likely be **{tom_purifier}**.", unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align: center; color: gray;'>Project: ANN Based Smart Air Purification System | Visualizer & Forecaster Enabled</div>", unsafe_allow_html=True)
