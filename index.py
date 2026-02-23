import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from datetime import datetime
from io import BytesIO

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ------------------------------------------------------------
#  PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Weather Prediction by City & Date",
    page_icon="☀️",
    layout="wide"
)

# ------------------------------------------------------------
#  CUSTOM CSS – SUNNY SKY + ANIMATED CLOUDS + MOVING SUN
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sunny sky gradient */
    .stApp {
        background: linear-gradient(180deg, #9fc5e8 0%, #fff2cc 80%);
        color: #1e2a3a;  /* dark text for contrast */
    }

    /* Floating clouds – white, semi‑transparent */
    .cloud {
        position: fixed;
        background: rgba(255, 255, 255, 0.35);
        border-radius: 1000px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        z-index: -1;
        filter: blur(6px);
        animation: drift linear infinite;
    }
    .cloud1 {
        width: 420px;
        height: 140px;
        top: 15%;
        left: -450px;
        animation-duration: 120s;
    }
    .cloud2 {
        width: 550px;
        height: 180px;
        top: 40%;
        left: -600px;
        animation-duration: 150s;
    }
    .cloud3 {
        width: 380px;
        height: 130px;
        top: 70%;
        left: -400px;
        animation-duration: 100s;
    }
    @keyframes drift {
        from { transform: translateX(0); }
        to { transform: translateX(180vw); }
    }

    /* Moving sun */
    .sun {
        position: fixed;
        top: 8%;
        left: -100px;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle at 30% 30%, #fff7b0, #ffdd55);
        border-radius: 50%;
        box-shadow: 0 0 60px #ffdd55;
        z-index: -1;
        animation: sunMove 90s linear infinite;
        filter: blur(2px);
    }
    @keyframes sunMove {
        from { transform: translateX(0); }
        to { transform: translateX(120vw); }
    }

    /* Style cards and widgets for light background */
    .stMetric {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.7);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: #1e2a3a;
    }
    h1, h2, h3 {
        font-weight: 300;
        letter-spacing: 0.5px;
        color: #1e3a5f;
    }
    .stButton>button {
        background: rgba(255, 255, 255, 0.6);
        color: #1e3a5f;
        border: 1px solid #5f9ea0;
        border-radius: 40px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #5f9ea0;
        color: white;
        border-color: #5f9ea0;
        transform: scale(1.02);
    }
    /* Dataframe text */
    .stDataFrame {
        color: #1e2a3a;
    }
    </style>

    <!-- Sun element -->
    <div class="sun"></div>

    <!-- Cloud elements -->
    <div class="cloud cloud1"></div>
    <div class="cloud cloud2"></div>
    <div class="cloud cloud3"></div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  GENERATE SYNTHETIC CLIMATE DATA (realistic monthly norms)
# ------------------------------------------------------------
@st.cache_data
def generate_climate_data():
    """Create a daily dataset for 5 cities based on approximate monthly norms."""
    cities = {
        "New York": {
            "monthly_temp": [1, 2, 7, 13, 19, 24, 27, 26, 22, 16, 10, 4],
            "monthly_rainfall": [90, 80, 110, 100, 110, 100, 110, 100, 100, 90, 100, 100],
            "monthly_humidity": [70, 68, 67, 64, 67, 69, 70, 71, 72, 70, 70, 70]
        },
        "London": {
            "monthly_temp": [5, 5, 8, 11, 15, 18, 21, 20, 17, 13, 9, 6],
            "monthly_rainfall": [55, 40, 55, 45, 50, 50, 45, 50, 50, 60, 60, 55],
            "monthly_humidity": [80, 78, 75, 72, 71, 70, 71, 73, 76, 79, 81, 81]
        },
        "Tokyo": {
            "monthly_temp": [6, 7, 10, 16, 21, 24, 28, 29, 25, 19, 14, 9],
            "monthly_rainfall": [50, 70, 110, 120, 130, 160, 150, 170, 190, 180, 90, 50],
            "monthly_humidity": [55, 56, 60, 66, 70, 75, 77, 76, 75, 70, 65, 59]
        },
        "Sydney": {
            "monthly_temp": [23, 23, 22, 19, 16, 14, 13, 15, 18, 20, 22, 24],
            "monthly_rainfall": [90, 110, 130, 130, 120, 130, 100, 80, 70, 80, 90, 80],
            "monthly_humidity": [70, 73, 74, 72, 70, 68, 65, 62, 63, 66, 68, 69]
        },
        "Cape Town": {
            "monthly_temp": [21, 21, 20, 18, 15, 13, 12, 13, 15, 17, 19, 21],
            "monthly_rainfall": [15, 15, 20, 50, 80, 110, 100, 90, 60, 35, 20, 15],
            "monthly_humidity": [70, 72, 73, 74, 76, 78, 78, 77, 75, 73, 71, 70]
        }
    }

    rows = []
    np.random.seed(42)  # for reproducibility
    for city, data in cities.items():
        for month in range(1, 13):
            days_in_month = calendar.monthrange(2024, month)[1]
            base_temp = data["monthly_temp"][month-1]
            base_rain = data["monthly_rainfall"][month-1]
            base_hum = data["monthly_humidity"][month-1]

            for day in range(1, days_in_month + 1):
                # Add realistic daily variation
                temp = base_temp + np.random.uniform(-2.5, 2.5)
                rain = max(0, base_rain + np.random.normal(0, base_rain * 0.15))
                hum = np.clip(base_hum + np.random.uniform(-8, 8), 20, 100)

                rows.append({
                    "city": city,
                    "month": month,
                    "day": day,
                    "temperature": round(temp, 1),
                    "rainfall": round(rain, 1),
                    "humidity": round(hum, 1)
                })

    return pd.DataFrame(rows)

# ------------------------------------------------------------
#  TRAIN MODELS (cached)
# ------------------------------------------------------------
@st.cache_resource
def train_models(df):
    """Train Random Forest regressors for temperature, rainfall, humidity."""
    # Encode city
    le_city = LabelEncoder()
    df["city_code"] = le_city.fit_transform(df["city"])

    # Features: city, month, day
    X = df[["city_code", "month", "day"]]
    y_temp = df["temperature"]
    y_rain = df["rainfall"]
    y_hum = df["humidity"]

    # Models
    model_temp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_rain = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_hum = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    model_temp.fit(X, y_temp)
    model_rain.fit(X, y_rain)
    model_hum.fit(X, y_hum)

    return model_temp, model_rain, model_hum, le_city

# ------------------------------------------------------------
#  HELPER: get weather condition from predictions
# ------------------------------------------------------------
def get_weather_condition(temp, rain, hum):
    """Classify weather based on predicted values."""
    if rain > 5.0:
        return "🌧️ Heavy Rain"
    elif rain > 1.0:
        return "🌦️ Rainy"
    elif rain > 0.1:
        return "☔ Drizzle"
    elif hum > 80:
        return "☁️ Cloudy"
    elif hum > 60:
        return "⛅ Partly Cloudy"
    else:
        return "☀️ Sunny"

# ------------------------------------------------------------
#  LOAD DATA & MODELS (once)
# ------------------------------------------------------------
df_climate = generate_climate_data()
model_temp, model_rain, model_hum, le_city = train_models(df_climate)

# Prepare city list for dropdown
cities = le_city.classes_.tolist()

# ------------------------------------------------------------
#  UI HEADER
# ------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; font-size: 3.2rem; font-weight: 300; margin-bottom: 0;'>
        🌤️ Weather Forecast by City & Date
    </h1>
    <p style='text-align: center; font-size: 1.2rem; color: #2c3e50; margin-top: 0;'>
        Select a city and a date – get instant temperature, rainfall, humidity & weather condition
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  INPUT SECTION
# ------------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1], gap="medium")

with col1:
    selected_city = st.selectbox("🌆 Choose a city", cities, index=0)

with col2:
    selected_date = st.date_input("📅 Pick a date", datetime(2024, 6, 15))

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Weather", type="primary")

# ------------------------------------------------------------
#  PREDICTION & VISUALIZATION (when button clicked)
# ------------------------------------------------------------
if predict_btn:
    # Encode inputs
    city_code = le_city.transform([selected_city])[0]
    month = selected_date.month
    day = selected_date.day

    # Create feature array
    X_input = np.array([[city_code, month, day]])

    # Predict
    pred_temp = model_temp.predict(X_input)[0]
    pred_rain = model_rain.predict(X_input)[0]
    pred_hum = model_hum.predict(X_input)[0]
    condition = get_weather_condition(pred_temp, pred_rain, pred_hum)

    # --------------------------------------------------------
    #  METRICS ROW
    # --------------------------------------------------------
    st.markdown("---")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("🌡️ Temperature", f"{pred_temp:.1f} °C")
    mcol2.metric("💧 Rainfall", f"{pred_rain:.1f} mm")
    mcol3.metric("💨 Humidity", f"{pred_hum:.0f} %")
    mcol4.metric("🌥️ Condition", condition)

    # --------------------------------------------------------
    #  VISUALIZATIONS (Actual vs Predicted for the selected city)
    # --------------------------------------------------------
    st.markdown("## 📊 Climate Profile & Model Performance")

    # Filter data for the selected city
    city_data = df_climate[df_climate["city"] == selected_city].copy()
    city_data["city_code"] = le_city.transform([selected_city])[0]

    # Compute actual monthly averages from the synthetic dataset
    monthly_actual = city_data.groupby("month")[["temperature", "rainfall"]].mean().reset_index()

    # Generate model predictions for each month (using the 15th day as representative)
    pred_rows = []
    for m in range(1, 13):
        X_pred = np.array([[city_code, m, 15]])
        t = model_temp.predict(X_pred)[0]
        r = model_rain.predict(X_pred)[0]
        pred_rows.append({"month": m, "temp_pred": t, "rain_pred": r})
    monthly_pred = pd.DataFrame(pred_rows)

    # Merge for plotting
    plot_df = monthly_actual.merge(monthly_pred, on="month")

    # Create two charts side by side
    colL, colR = st.columns(2)

    with colL:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(plot_df["month"], plot_df["temperature"], "o-", label="Actual (avg)", color="#1e6f9f", linewidth=2)
        ax1.plot(plot_df["month"], plot_df["temp_pred"], "s--", label="Predicted", color="#f39c12", linewidth=2)
        ax1.axvline(x=month, color="gray", linestyle=":", alpha=0.7, label="Selected month")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title(f"Temperature Profile – {selected_city}")
        ax1.legend()
        ax1.set_xticks(range(1,13))
        ax1.set_xticklabels([calendar.month_abbr[i] for i in range(1,13)])
        ax1.grid(alpha=0.2)
        st.pyplot(fig1)

    with colR:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(plot_df["month"] - 0.2, plot_df["rainfall"], width=0.4, label="Actual (avg)", color="#1e6f9f", alpha=0.8)
        ax2.bar(plot_df["month"] + 0.2, plot_df["rain_pred"], width=0.4, label="Predicted", color="#f39c12", alpha=0.8)
        ax2.axvline(x=month, color="gray", linestyle=":", alpha=0.7, label="Selected month")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Rainfall (mm)")
        ax2.set_title(f"Rainfall Profile – {selected_city}")
        ax2.legend()
        ax2.set_xticks(range(1,13))
        ax2.set_xticklabels([calendar.month_abbr[i] for i in range(1,13)])
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

    # --------------------------------------------------------
    #  DATA QUALITY / MODEL METRICS (like original, but adapted)
    # --------------------------------------------------------
    st.markdown("## 📋 Model Accuracy on Training Data")
    with st.expander("Click to view model performance metrics"):
        # Compute overall metrics for temperature and rainfall
        X_all = city_data[["city_code", "month", "day"]]
        y_temp_all = city_data["temperature"]
        y_rain_all = city_data["rainfall"]

        pred_temp_all = model_temp.predict(X_all)
        pred_rain_all = model_rain.predict(X_all)

        mae_temp = mean_absolute_error(y_temp_all, pred_temp_all)
        r2_temp = r2_score(y_temp_all, pred_temp_all)
        mae_rain = mean_absolute_error(y_rain_all, pred_rain_all)
        r2_rain = r2_score(y_rain_all, pred_rain_all)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("🌡️ Temp MAE", f"{mae_temp:.2f} °C")
        colB.metric("🌡️ Temp R²", f"{r2_temp:.3f}")
        colC.metric("💧 Rain MAE", f"{mae_rain:.1f} mm")
        colD.metric("💧 Rain R²", f"{r2_rain:.3f}")

        st.caption("*Metrics are computed on the synthetic training data for the selected city.*")

    # --------------------------------------------------------
    #  PDF REPORT (simplified, like original)
    # --------------------------------------------------------
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Weather Forecast for {selected_city}", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Date: {selected_date.strftime('%B %d, %Y')}", styles["Normal"]),
        Paragraph(f"Temperature: {pred_temp:.1f} °C", styles["Normal"]),
        Paragraph(f"Rainfall: {pred_rain:.1f} mm", styles["Normal"]),
        Paragraph(f"Humidity: {pred_hum:.0f} %", styles["Normal"]),
        Paragraph(f"Condition: {condition}", styles["Normal"]),
        Spacer(1, 24),
        Paragraph("Model Performance (on training data):", styles["Heading3"]),
        Paragraph(f"Temperature MAE: {mae_temp:.2f} °C, R²: {r2_temp:.3f}", styles["Normal"]),
        Paragraph(f"Rainfall MAE: {mae_rain:.1f} mm, R²: {r2_rain:.3f}", styles["Normal"]),
    ]
    doc.build(story)
    buffer.seek(0)

    st.download_button(
        label="📥 Download PDF Report",
        data=buffer,
        file_name=f"weather_{selected_city}_{selected_date}.pdf",
        mime="application/pdf"
    )

else:
    # Placeholder info when no prediction made yet
    st.info("👆 Select a city and date, then click **Predict Weather** to start.")
    # Show a preview of available cities
    st.markdown("### 🌍 Available cities")
    st.write(", ".join(cities))
