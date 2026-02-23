import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from datetime import datetime
from io import BytesIO
import hashlib

from countryinfo import CountryInfo
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ------------------------------------------------------------
#  PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Global Weather by City",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------------------------------------
#  CUSTOM CSS – SUNNY SKY + ANIMATED CLOUDS + MOVING SUN
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #9fc5e8 0%, #fff2cc 80%);
        color: #1e2a3a;
    }
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
    .weather-card {
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .weather-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.4);
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
    .stDataFrame {
        color: #1e2a3a;
    }
    </style>
    <div class="sun"></div>
    <div class="cloud cloud1"></div>
    <div class="cloud cloud2"></div>
    <div class="cloud cloud3"></div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  HELPER FUNCTIONS FOR CLIMATE ZONE AND PREDICTIONS
# ------------------------------------------------------------
@st.cache_data
def get_all_countries():
    """Return a sorted list of all country names using countryinfo."""
    try:
        all_countries = list(CountryInfo().all().keys())
        return sorted(all_countries)
    except:
        # Fallback to a reasonable list if countryinfo fails
        return ["United States", "Canada", "United Kingdom", "Germany", "France",
                "Italy", "Spain", "Australia", "Japan", "China", "India", "Brazil",
                "South Africa", "Egypt", "Russia"]

@st.cache_data
def get_country_lat(country_name):
    """Return approximate latitude of the country's center."""
    try:
        country = CountryInfo(country_name)
        latlng = country.info().get('latlng', [30, 0])
        return latlng[0]
    except:
        return 30.0

def get_climate_zone(lat):
    abs_lat = abs(lat)
    if abs_lat >= 60:
        return "polar"
    elif abs_lat >= 30:
        return "temperate"
    else:
        return "tropical"

def get_season_shift(lat):
    return 6 if lat < 0 else 0

def get_monthly_norms(zone):
    """
    Return baseline monthly temperature (°C) and monthly rainfall total (mm) for a given zone.
    Indices 0..11 correspond to months Jan..Dec in northern hemisphere.
    """
    if zone == "tropical":
        temp_base = [26, 27, 28, 28, 28, 27, 26, 26, 26, 27, 27, 26]
        rain_base = [50, 60, 80, 100, 150, 200, 250, 250, 200, 120, 70, 50]  # monthly totals
    elif zone == "temperate":
        temp_base = [2, 4, 8, 12, 17, 21, 24, 23, 19, 13, 7, 3]
        rain_base = [60, 50, 60, 50, 60, 70, 70, 70, 60, 60, 60, 60]
    elif zone == "polar":
        temp_base = [-20, -18, -15, -8, 0, 5, 8, 7, 2, -5, -12, -18]
        rain_base = [20, 15, 15, 10, 10, 15, 20, 25, 20, 15, 15, 20]
    else:
        temp_base = [2, 4, 8, 12, 17, 21, 24, 23, 19, 13, 7, 3]
        rain_base = [60, 50, 60, 50, 60, 70, 70, 70, 60, 60, 60, 60]
    return np.array(temp_base), np.array(rain_base)

def city_hash_offset(city_name, max_offset):
    """Deterministic offset between -max_offset and +max_offset from city name."""
    if not city_name:
        return 0.0
    hash_val = int(hashlib.md5(city_name.encode()).hexdigest()[:8], 16)
    return (hash_val / (16**8) * 2 - 1) * max_offset

def days_in_month(year, month):
    """Return number of days in given month (year is used for February leap years)."""
    if month == 2:
        # Simple leap year check
        leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        return 29 if leap else 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31

def predict_weather(country, city, date):
    """
    Predict temperature, rainfall, humidity, wind speed, and feels-like.
    Uses climate zone based on country latitude, seasonal shift, and city-specific variation.
    Rainfall is converted from monthly total to a realistic daily amount.
    """
    lat = get_country_lat(country)
    zone = get_climate_zone(lat)
    shift = get_season_shift(lat)

    month = date.month
    year = date.year
    # Adjust month for southern hemisphere
    adjusted_month = ((month - 1 + shift) % 12) + 1
    idx = adjusted_month - 1

    temp_base, rain_base = get_monthly_norms(zone)

    # Base monthly values
    base_temp = temp_base[idx]
    base_monthly_rain = rain_base[idx]

    # City-specific offsets
    temp_offset = city_hash_offset(city, 2.0)                     # ±2°C
    rain_factor = 0.5 + city_hash_offset(city + "_rf", 0.5)       # 0.0–1.0, to make daily rain variable
    hum_offset = city_hash_offset(city + "_hum", 8.0)             # ±8%
    wind_offset = city_hash_offset(city + "_wind", 5.0)           # ±5 km/h

    # Daily average rain from monthly total
    days = days_in_month(year, month)
    avg_daily_rain = base_monthly_rain / days

    # Daily rainfall amount: can be 0–2× average, but never negative
    # Use rain_factor to scale: if rain_factor < 0.2, it's a dry day; else rainy with amount.
    # This makes rainfall sporadic and more realistic.
    if rain_factor < 0.2:
        daily_rain = 0.0
    else:
        # Scale between 0.5× and 2× average
        daily_rain = avg_daily_rain * (0.5 + rain_factor)

    # Humidity – loosely based on rainfall
    if daily_rain > 5:
        base_hum = 85
    elif daily_rain > 1:
        base_hum = 75
    elif daily_rain > 0.1:
        base_hum = 70
    else:
        base_hum = 60
    # Add some variation from city
    hum = np.clip(base_hum + hum_offset, 20, 100)

    # Wind speed – base from zone plus city offset
    if zone == "polar":
        base_wind = 15
    elif zone == "temperate":
        base_wind = 12
    else:
        base_wind = 8
    wind_speed = max(0, base_wind + wind_offset)

    # Temperature
    temp = base_temp + temp_offset

    # Feels like – simplified heat index / wind chill
    if temp >= 27 and hum > 60:
        feels_like = temp + 0.1 * hum - 5
    elif temp < 10:
        feels_like = temp - 2
    else:
        feels_like = temp

    # Weather condition (more granular)
    if daily_rain > 10:
        condition = "🌧️ Heavy Rain"
    elif daily_rain > 2:
        condition = "🌦️ Rain"
    elif daily_rain > 0.1:
        condition = "☔ Drizzle"
    elif hum > 85:
        condition = "☁️ Cloudy"
    elif hum > 70:
        condition = "⛅ Partly Cloudy"
    elif temp > 30:
        condition = "☀️ Hot"
    else:
        condition = "☀️ Sunny"

    return {
        "temperature": round(temp, 1),
        "feels_like": round(feels_like, 1),
        "rainfall": round(daily_rain, 1),
        "humidity": round(hum, 1),
        "wind_speed": round(wind_speed, 1),
        "condition": condition,
        "zone": zone,
        "monthly_temp_base": temp_base,
        "monthly_rain_base": rain_base
    }

def get_country_monthly_profile(country):
    """Return typical monthly temperature and rainfall arrays for the country (shifted for hemisphere)."""
    lat = get_country_lat(country)
    zone = get_climate_zone(lat)
    shift = get_season_shift(lat)
    temp_base, rain_base = get_monthly_norms(zone)
    if shift != 0:
        temp_base = np.roll(temp_base, shift)
        rain_base = np.roll(rain_base, shift)
    return temp_base, rain_base

# ------------------------------------------------------------
#  UI HEADER
# ------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; font-size: 3.2rem; font-weight: 300; margin-bottom: 0;'>
        🌤️ Global Weather by City
    </h1>
    <p style='text-align: center; font-size: 1.2rem; color: #2c3e50; margin-top: 0;'>
        Any country, any city, any date – realistic forecast based on climate norms
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  INPUT SECTION
# ------------------------------------------------------------
all_countries = get_all_countries()

col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1], gap="medium")

with col1:
    selected_country = st.selectbox("🌎 Choose a country", all_countries, index=all_countries.index("United States") if "United States" in all_countries else 0)

with col2:
    city_name = st.text_input("🏙️ Enter city name", value="Mumbai")

with col3:
    selected_date = st.date_input("📅 Pick a date", datetime(2024, 6, 15))

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict", type="primary")

# ------------------------------------------------------------
#  PREDICTION & VISUALIZATION
# ------------------------------------------------------------
if predict_btn and city_name.strip():
    weather = predict_weather(selected_country, city_name.strip(), selected_date)

    # Metrics in styled cards
    st.markdown("---")
    st.markdown(
        f"""
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center;">
            <div class="weather-card" style="flex: 1; min-width: 140px;">
                <h3 style="margin:0;">🌡️ Temp</h3>
                <p style="font-size: 2.2rem; margin:0;">{weather['temperature']} °C</p>
            </div>
            <div class="weather-card" style="flex: 1; min-width: 140px;">
                <h3 style="margin:0;">🤔 Feels like</h3>
                <p style="font-size: 2.2rem; margin:0;">{weather['feels_like']} °C</p>
            </div>
            <div class="weather-card" style="flex: 1; min-width: 140px;">
                <h3 style="margin:0;">💧 Rain</h3>
                <p style="font-size: 2.2rem; margin:0;">{weather['rainfall']} mm</p>
            </div>
            <div class="weather-card" style="flex: 1; min-width: 140px;">
                <h3 style="margin:0;">💨 Humidity</h3>
                <p style="font-size: 2.2rem; margin:0;">{weather['humidity']} %</p>
            </div>
            <div class="weather-card" style="flex: 1; min-width: 140px;">
                <h3 style="margin:0;">🌬️ Wind</h3>
                <p style="font-size: 2.2rem; margin:0;">{weather['wind_speed']} km/h</p>
            </div>
            <div class="weather-card" style="flex: 1; min-width: 160px; background: rgba(255,240,200,0.4);">
                <h3 style="margin:0;">{weather['condition'].split()[0]}</h3>
                <p style="font-size: 1.8rem; margin:0;">{' '.join(weather['condition'].split()[1:])}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Climate profile chart
    st.markdown("## 📈 Typical Monthly Climate")
    temp_profile, rain_profile = get_country_monthly_profile(selected_country)

    months_abbr = [calendar.month_abbr[i] for i in range(1,13)]
    month_idx = selected_date.month - 1

    colL, colR = st.columns(2)

    with colL:
        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(range(1,13), temp_profile, 'o-', color="#1e6f9f", linewidth=2, label="Typical")
        ax1.plot(month_idx+1, weather['temperature'], 'ro', markersize=10, label=f"Predicted ({city_name})")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title(f"Temperature Profile – {selected_country}")
        ax1.legend()
        ax1.set_xticks(range(1,13))
        ax1.set_xticklabels(months_abbr)
        ax1.grid(alpha=0.2)
        st.pyplot(fig1)

    with colR:
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.bar(range(1,13), rain_profile, color="#1e6f9f", alpha=0.7, label="Typical monthly total")
        ax2.bar(month_idx+1, weather['rainfall'] * days_in_month(selected_date.year, selected_date.month),
                color="#f39c12", alpha=0.9, label=f"Predicted daily × days")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Rainfall (mm)")
        ax2.set_title(f"Rainfall Profile – {selected_country}")
        ax2.legend()
        ax2.set_xticks(range(1,13))
        ax2.set_xticklabels(months_abbr)
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

    # Climate Norms Table (replaces "Model Accuracy")
    st.markdown("## 📋 Climate Norms for " + selected_country)
    with st.expander("Click to view monthly averages (temperature & rainfall)"):
        norms_df = pd.DataFrame({
            "Month": months_abbr,
            "Avg Temp (°C)": temp_profile,
            "Monthly Rainfall (mm)": rain_profile
        })
        st.dataframe(norms_df, use_container_width=True)
        st.caption("*These are typical long-term averages used to generate daily predictions.*")

    # PDF Report
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Weather Forecast for {city_name}, {selected_country}", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Date: {selected_date.strftime('%B %d, %Y')}", styles["Normal"]),
        Paragraph(f"Temperature: {weather['temperature']} °C (feels like {weather['feels_like']} °C)", styles["Normal"]),
        Paragraph(f"Rainfall: {weather['rainfall']} mm", styles["Normal"]),
        Paragraph(f"Humidity: {weather['humidity']} %", styles["Normal"]),
        Paragraph(f"Wind Speed: {weather['wind_speed']} km/h", styles["Normal"]),
        Paragraph(f"Condition: {weather['condition']}", styles["Normal"]),
        Spacer(1, 24),
        Paragraph("Climate Zone: " + weather['zone'].capitalize(), styles["Normal"]),
    ]
    doc.build(story)
    buffer.seek(0)

    st.download_button(
        label="📥 Download PDF Report",
        data=buffer,
        file_name=f"weather_{city_name}_{selected_date}.pdf",
        mime="application/pdf"
    )

else:
    st.info("👆 Select a country, enter a city, pick a date, and click **Predict**.")
    st.markdown("### 🌍 How it works")
    st.markdown("""
    - The app uses the country's latitude to determine its **climate zone** (tropical, temperate, polar).
    - Monthly temperature and rainfall norms are shifted for the southern hemisphere.
    - Daily rainfall is derived from monthly totals with city‑specific variation, making it more realistic.
    - All predictions are **deterministic** – the same city and date always give the same result.
    """)
