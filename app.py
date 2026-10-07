import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import textwrap

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality Intelligence | India",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. MASTER HIGH-CONTRAST VISIBLE DESIGN SYSTEM (CSS)
# ============================================================

MASTER_CSS = textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

/* Canvas & Global Reset */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

#MainMenu, footer, header {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1440px;
}

/* Base Typography Defaults */
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
    color: #0f172a !important;
}

.secondary-text {
    color: #475569 !important;
}

/* Force Light Background & Dark Text on ALL Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
.stNumberInput input,
.stDateInput input,
.stSelectbox input,
.stTextInput input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    font-weight: 600 !important;
}

/* Dropdown Menu Popover Colors */
div[data-baseweb="menu"],
div[data-baseweb="popover"],
ul[role="listbox"],
li[role="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}

div[data-baseweb="select"] svg,
div[data-baseweb="select"] span {
    color: #0f172a !important;
    fill: #0f172a !important;
}

/* Number Input Buttons (+/-) */
.stNumberInput button {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

.stNumberInput button:hover {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
}

/* Widget Label Formatting */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] span {
    color: #1e293b !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    margin-bottom: 4px !important;
}

/* ALL BUTTON BASE STYLES */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* PRIMARY BUTTONS (Active Nav & Submit Predict Button) - WHITE TEXT ON DARK SLATE */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
div[data-testid="stFormSubmitButton"] > button {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #0f172a !important;
}

button[kind="primary"] *,
button[data-testid="stBaseButton-primary"] *,
div[data-testid="stFormSubmitButton"] > button * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-color: #1e293b !important;
}

/* SECONDARY BUTTONS (Inactive Nav Buttons) - DARK TEXT ON WHITE */
button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    background-color: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
}

button[kind="secondary"] *,
button[data-testid="stBaseButton-secondary"] * {
    color: #334155 !important;
    fill: #334155 !important;
}

button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #0f172a !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

button[kind="secondary"]:hover *,
button[data-testid="stBaseButton-secondary"]:hover * {
    color: #0f172a !important;
}

/* Header & Brand Title */
.brand-title {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -0.6px;
    line-height: 1.2;
}

.brand-subtitle {
    font-size: 12px;
    color: #64748b !important;
    font-weight: 600;
    margin-top: 2px;
}

/* Cards & Panel Styling */
.card-panel {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04) !important;
}

.card-header-label {
    font-size: 11px;
    font-weight: 800;
    color: #64748b !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.card-title-text {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #0f172a !important;
}

/* Metric Display Tiles */
.metric-tile {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
}

.metric-tile-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.metric-tile-val {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #0f172a !important;
    margin-top: 2px;
}

.metric-tile-sub {
    font-size: 11px;
    color: #64748b !important;
    font-weight: 500;
}

/* Form Container */
div[data-testid="stForm"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.03) !important;
}
</style>
""")

st.markdown(MASTER_CSS, unsafe_allow_html=True)

# ============================================================
# 3. DATA & MODEL LOADERS (CACHED)
# ============================================================

@st.cache_data
def load_master_data():
    df = pd.read_csv("Data/master_air_quality.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_test_predictions():
    try:
        df_pred = pd.read_csv("outputs/test_predictions.csv")
        df_pred["date"] = pd.to_datetime(df_pred["date"])
        return df_pred
    except Exception:
        return None

@st.cache_resource
def load_trained_model():
    return joblib.load("models/best_air_quality_model.pkl")

df = load_master_data()
df_test = load_test_predictions()
model = load_trained_model()

# ============================================================
# 4. AQI UTILITY FUNCTIONS
# ============================================================

def get_aqi_info(aqi):
    aqi_val = float(aqi)
    if aqi_val <= 50:
        return {
            "category": "Good",
            "color": "#10B981",
            "bg": "#ECFDF5",
            "text": "#047857",
            "desc": "Air quality is satisfactory and poses little or no health risk.",
            "dot": "🟢"
        }
    elif aqi_val <= 100:
        return {
            "category": "Satisfactory",
            "color": "#84CC16",
            "bg": "#F7FEE7",
            "text": "#4D7C0F",
            "desc": "Air quality is acceptable; minor breathing discomfort to sensitive individuals.",
            "dot": "🟡"
        }
    elif aqi_val <= 200:
        return {
            "category": "Moderate",
            "color": "#F59E0B",
            "bg": "#FEF3C7",
            "text": "#B45309",
            "desc": "Breathing discomfort to people with asthma, lung, and heart conditions.",
            "dot": "🟠"
        }
    elif aqi_val <= 300:
        return {
            "category": "Poor",
            "color": "#F97316",
            "bg": "#FFEDD5",
            "text": "#C2410C",
            "desc": "Breathing discomfort to most people on prolonged outdoor exposure.",
            "dot": "🔴"
        }
    elif aqi_val <= 400:
        return {
            "category": "Very Poor",
            "color": "#EF4444",
            "bg": "#FEF2F2",
            "text": "#B91C1C",
            "desc": "Respiratory illness on prolonged exposure. Significantly affects healthy people.",
            "dot": "🟣"
        }
    else:
        return {
            "category": "Severe",
            "color": "#881337",
            "bg": "#FFE4E6",
            "text": "#9F1239",
            "desc": "Seriously impacts healthy individuals and causes severe health impacts.",
            "dot": "🟤"
        }

# ============================================================
# 5. TOP NAVIGATION BAR
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Air Map"

nav_brand, nav_b1, nav_b2, nav_b3, nav_b4 = st.columns([2.5, 1, 1, 1.1, 1])

with nav_brand:
    st.markdown(textwrap.dedent("""
    <div style="padding-bottom: 4px;">
        <div class="brand-title">🍃 Air Quality Intelligence</div>
        <div class="brand-subtitle">India National Environmental Monitoring • 50 Cities Monitored</div>
    </div>
    """), unsafe_allow_html=True)

with nav_b1:
    if st.button("Air Map", use_container_width=True, type="primary" if st.session_state.page == "Air Map" else "secondary"):
        st.session_state.page = "Air Map"
        st.rerun()

with nav_b2:
    if st.button("Analytics", use_container_width=True, type="primary" if st.session_state.page == "Analytics" else "secondary"):
        st.session_state.page = "Analytics"
        st.rerun()

with nav_b3:
    if st.button("AI Prediction", use_container_width=True, type="primary" if st.session_state.page == "AI Prediction" else "secondary"):
        st.session_state.page = "AI Prediction"
        st.rerun()

with nav_b4:
    if st.button("Model", use_container_width=True, type="primary" if st.session_state.page == "Model" else "secondary"):
        st.session_state.page = "Model"
        st.rerun()

st.markdown("<div style='border-bottom: 1px solid #e2e8f0; margin-bottom: 18px;'></div>", unsafe_allow_html=True)

# ============================================================
# 6. SCREEN 1: AIR MAP (HERO SCREEN)
# ============================================================

if st.session_state.page == "Air Map":
    
    latest_available_date = df["date"].max()

    # --------------------------------------------------------
    # MAP FILTER STRIP
    # --------------------------------------------------------
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.4, 1.4, 1.2, 1.0])
    
    with f_col1:
        selected_date = st.date_input(
            "Monitoring Date",
            value=latest_available_date,
            min_value=df["date"].min().date(),
            max_value=latest_available_date.date()
        )
    
    date_dt = pd.to_datetime(selected_date)
    day_df = df[df["date"] == date_dt].copy()
    
    if day_df.empty:
        st.warning(f"No records for {selected_date}. Defaulting to latest: {latest_available_date.strftime('%Y-%m-%d')}")
        day_df = df[df["date"] == latest_available_date].copy()
        date_dt = latest_available_date

    with f_col2:
        all_cities = ["All Cities"] + sorted(day_df["city"].unique().tolist())
        selected_city_option = st.selectbox("Focus City", all_cities, index=0)

    with f_col3:
        indicator = st.selectbox("Pollutant Metric", ["AQI", "PM2.5", "PM10", "NO₂", "SO₂", "O₃", "CO"])

    with f_col4:
        st.markdown("<div style='margin-top: 26px;'></div>", unsafe_allow_html=True)
        show_heatmap = st.checkbox("AQI Heatmap Overlay", value=False)

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MAP RENDERING
    # --------------------------------------------------------
    if selected_city_option != "All Cities":
        focus_row = day_df[day_df["city"] == selected_city_option]
        if not focus_row.empty:
            map_center = [focus_row.iloc[0]["latitude"], focus_row.iloc[0]["longitude"]]
            map_zoom = 8
        else:
            map_center = [22.5, 78.9]
            map_zoom = 5
    else:
        map_center = [22.5, 78.9]
        map_zoom = 5

    m = folium.Map(
        location=map_center,
        zoom_start=map_zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    if show_heatmap:
        heat_data = [[row["latitude"], row["longitude"], float(row["india_aqi"])] for _, row in day_df.iterrows()]
        HeatMap(heat_data, radius=24, blur=18, min_opacity=0.35, max_val=400).add_to(m)

    for _, row in day_df.iterrows():
        aqi_val = float(row["india_aqi"])
        aqi_info = get_aqi_info(aqi_val)
        
        if indicator == "AQI":
            val_disp = f"{aqi_val:.0f}"
        elif indicator == "PM2.5":
            val_disp = f"{row['pm2_5_mean']:.1f} µg/m³"
        elif indicator == "PM10":
            val_disp = f"{row['pm10_mean']:.1f} µg/m³"
        elif indicator == "NO₂":
            val_disp = f"{row['no2_mean']:.1f} µg/m³"
        elif indicator == "SO₂":
            val_disp = f"{row['so2_mean']:.1f} µg/m³"
        elif indicator == "O₃":
            val_disp = f"{row['o3_8h_max']:.1f} µg/m³"
        else:
            val_disp = f"{row['co_8h_max']:.1f} µg/m³"

        circle_radius = max(7, min(20, 6 + (aqi_val / 24)))
        is_highlighted = (selected_city_option != "All Cities" and row["city"] == selected_city_option)
        stroke_color = "#000000" if is_highlighted else "#FFFFFF"
        stroke_weight = 3.5 if is_highlighted else 1.8

        popup_content = textwrap.dedent(f"""
        <div style="font-family: 'Inter', sans-serif; width: 240px; padding: 6px; color: #0f172a;">
            <div style="font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 2px;">{row['city']}</div>
            <div style="font-size: 11px; font-weight: 600; color: #64748b; margin-bottom: 10px;">{row['state']} • {row['region']}</div>
            
            <div style="background-color: {aqi_info['bg']}; color: {aqi_info['text']}; padding: 6px 12px; border-radius: 8px; text-align: center; font-weight: 800; font-size: 13px; margin-bottom: 12px; border: 1px solid {aqi_info['color']}40;">
                AQI {aqi_val:.0f} — {aqi_info['category']}
            </div>
            
            <table style="width: 100%; font-size: 12px; color: #0f172a; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0; color: #475569;"><b>PM2.5</b></td><td style="text-align: right; font-weight: 700; color: #0f172a;">{row['pm2_5_mean']:.1f} µg/m³</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0; color: #475569;"><b>PM10</b></td><td style="text-align: right; font-weight: 700; color: #0f172a;">{row['pm10_mean']:.1f} µg/m³</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0; color: #475569;"><b>NO₂</b></td><td style="text-align: right; font-weight: 700; color: #0f172a;">{row['no2_mean']:.1f}</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0; color: #475569;"><b>SO₂</b></td><td style="text-align: right; font-weight: 700; color: #0f172a;">{row['so2_mean']:.1f}</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0; color: #475569;"><b>Temp</b></td><td style="text-align: right; font-weight: 700; color: #0f172a;">{row['temperature_2m_mean']:.1f} °C</td></tr>
                <tr><td style="padding: 4px 0; color: #475569;"><b>Humidity</b></td><td style="text-align: right; font-weight: 700; color: #0f172a;">{row['relative_humidity_2m_mean']:.0f}%</td></tr>
            </table>
        </div>
        """)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=circle_radius,
            color=stroke_color,
            weight=stroke_weight,
            fill=True,
            fill_color=aqi_info["color"],
            fill_opacity=0.82,
            popup=folium.Popup(popup_content, max_width=270),
            tooltip=f"{row['city']} | AQI {aqi_val:.0f} ({aqi_info['category']})"
        ).add_to(m)

    st_folium(m, width=None, height=550, returned_objects=[], use_container_width=True)

    # Legend Bar
    st.markdown(textwrap.dedent("""
    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 20px; margin-top: 12px; margin-bottom: 22px; font-size: 12px; color: #1e293b; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.02);">
        <span style="font-weight: 800; color: #0f172a;">INDIAN AQI INDEX SCALE:</span>
        <span style="display: flex; align-items: center; gap: 4px;"><span style="color: #10B981; font-size: 14px;">●</span> <b>Good</b> (0–50)</span>
        <span style="display: flex; align-items: center; gap: 4px;"><span style="color: #84CC16; font-size: 14px;">●</span> <b>Satisfactory</b> (51–100)</span>
        <span style="display: flex; align-items: center; gap: 4px;"><span style="color: #F59E0B; font-size: 14px;">●</span> <b>Moderate</b> (101–200)</span>
        <span style="display: flex; align-items: center; gap: 4px;"><span style="color: #F97316; font-size: 14px;">●</span> <b>Poor</b> (201–300)</span>
        <span style="display: flex; align-items: center; gap: 4px;"><span style="color: #EF4444; font-size: 14px;">●</span> <b>Very Poor</b> (301–400)</span>
        <span style="display: flex; align-items: center; gap: 4px;"><span style="color: #881337; font-size: 14px;">●</span> <b>Severe</b> (401+)</span>
    </div>
    """), unsafe_allow_html=True)

    # --------------------------------------------------------
    # LOWER DETAILS (2 COLUMNS)
    # --------------------------------------------------------
    if selected_city_option != "All Cities":
        target_city_name = selected_city_option
    else:
        target_city_name = day_df.loc[day_df["india_aqi"].idxmax()]["city"]

    selected_city_row = day_df[day_df["city"] == target_city_name].iloc[0]
    c_aqi = float(selected_city_row["india_aqi"])
    c_info = get_aqi_info(c_aqi)

    l_col1, l_col2 = st.columns([1.2, 1.8])

    with l_col1:
        st.markdown(textwrap.dedent(f"""
        <div class="card-panel">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="card-header-label">SELECTED CITY FOCUS</div>
                    <div class="card-title-text">{selected_city_row['city']}</div>
                    <div style="font-size: 13px; color: #475569; font-weight: 500;">{selected_city_row['state']} • {selected_city_row['region']}</div>
                </div>
                <div style="background-color: {c_info['bg']}; color: {c_info['text']}; padding: 6px 14px; border-radius: 20px; font-weight: 800; font-size: 13px; border: 1px solid {c_info['color']}40;">
                    {c_info['category']}
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=c_aqi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "INDIA AQI METER", 'font': {'size': 12, 'color': '#475569', 'family': 'Inter'}},
            number={'font': {'size': 44, 'color': '#0f172a', 'family': 'Manrope'}},
            gauge={
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': c_info['color'], 'thickness': 0.24},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': '#ECFDF5'},
                    {'range': [50, 100], 'color': '#F7FEE7'},
                    {'range': [100, 200], 'color': '#FEF3C7'},
                    {'range': [200, 300], 'color': '#FFEDD5'},
                    {'range': [300, 400], 'color': '#FEF2F2'},
                    {'range': [400, 500], 'color': '#FFE4E6'}
                ]
            }
        ))
        
        gauge_fig.update_layout(
            height=210,
            margin=dict(l=15, r=15, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a")
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown(textwrap.dedent(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; font-size: 13px; color: #1e293b; line-height: 1.4;">
            💡 <b>Health Advisory:</b> {c_info['desc']}
        </div>
        """), unsafe_allow_html=True)

    with l_col2:
        city_history = df[df["city"] == target_city_name].sort_values("date")
        
        st.markdown(textwrap.dedent(f"""
        <div class="card-panel">
            <div class="card-header-label">HISTORICAL TREND</div>
            <div class="card-title-text" style="font-size: 20px;">Air Quality History ({target_city_name})</div>
        </div>
        """), unsafe_allow_html=True)

        trend_fig = px.line(
            city_history,
            x="date",
            y="india_aqi",
            labels={"india_aqi": "AQI", "date": "Date"}
        )
        
        trend_fig.update_traces(
            line_color="#0f172a",
            line_width=2.5,
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>AQI:</b> %{y:.0f}<extra></extra>"
        )
        
        trend_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="", tickfont=dict(color="#0f172a")),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="AQI", tickfont=dict(color="#0f172a"))
        )
        
        st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    # Pollutants Row
    st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 12px;'>Key Pollutant Concentrations</div>", unsafe_allow_html=True)

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    pollutant_cards = [
        ("PM2.5", f"{selected_city_row['pm2_5_mean']:.1f}", "µg/m³"),
        ("PM10", f"{selected_city_row['pm10_mean']:.1f}", "µg/m³"),
        ("NO₂", f"{selected_city_row['no2_mean']:.1f}", "µg/m³"),
        ("SO₂", f"{selected_city_row['so2_mean']:.1f}", "µg/m³"),
        ("O₃", f"{selected_city_row['o3_8h_max']:.1f}", "µg/m³"),
        ("CO", f"{selected_city_row['co_8h_max']:.1f}", "mg/m³")
    ]

    for col, (p_name, p_val, p_unit) in zip([p1, p2, p3, p4, p5, p6], pollutant_cards):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-tile">
                <div class="metric-tile-label">{p_name}</div>
                <div class="metric-tile-val">{p_val}</div>
                <div class="metric-tile-sub">{p_unit}</div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    # Weather Row
    st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 12px;'>Meteorology & Atmospheric Conditions</div>", unsafe_allow_html=True)

    w1, w2, w3, w4, w5 = st.columns(5)

    weather_cards = [
        ("Temperature", f"{selected_city_row['temperature_2m_mean']:.1f}", "°C"),
        ("Humidity", f"{selected_city_row['relative_humidity_2m_mean']:.0f}", "%"),
        ("Wind Speed", f"{selected_city_row['wind_speed_10m_max']:.1f}", "km/h"),
        ("Rainfall", f"{selected_city_row['precipitation_sum']:.1f}", "mm"),
        ("Elevation", f"{selected_city_row['elevation_m']:.0f}", "m")
    ]

    for col, (w_name, w_val, w_unit) in zip([w1, w2, w3, w4, w5], weather_cards):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-tile">
                <div class="metric-tile-label">{w_name}</div>
                <div class="metric-tile-val">{w_val} <span style="font-size: 12px; font-weight: 600; color: #64748b;">{w_unit}</span></div>
            </div>
            """), unsafe_allow_html=True)

# ============================================================
# 7. SCREEN 2: ANALYTICS
# ============================================================

elif st.session_state.page == "Analytics":
    
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 18px;">
        <div class="card-title-text" style="font-size: 26px;">Air Quality Analytics Engine</div>
        <div style="font-size: 13px; color: #475569; font-weight: 500;">Historical trends, city comparisons, and national severity breakdown</div>
    </div>
    """), unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1.5, 1])
    
    with col_a1:
        analyzed_city = st.selectbox("Select City", sorted(df["city"].unique().tolist()), index=0)
    
    with col_a2:
        aggregation_type = st.radio("Aggregation Level", ["Daily Raw Data", "Monthly Mean", "Yearly Mean"], horizontal=True)

    city_df = df[df["city"] == analyzed_city].sort_values("date").copy()

    if aggregation_type == "Monthly Mean":
        plot_df = city_df.set_index("date").resample("MS").mean(numeric_only=True).reset_index()
    elif aggregation_type == "Yearly Mean":
        plot_df = city_df.set_index("date").resample("YS").mean(numeric_only=True).reset_index()
    else:
        plot_df = city_df

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    avg_aqi = city_df["india_aqi"].mean()
    max_aqi = city_df["india_aqi"].max()
    min_aqi = city_df["india_aqi"].min()
    clean_days_pct = (city_df["india_aqi"] <= 100).mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(textwrap.dedent(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">AVERAGE AQI</div>
            <div class="metric-tile-val">{avg_aqi:.0f}</div>
            <div class="metric-tile-sub">{get_aqi_info(avg_aqi)['category']} Average</div>
        </div>
        """), unsafe_allow_html=True)
    with m2:
        st.markdown(textwrap.dedent(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">MAX POLLUTION</div>
            <div class="metric-tile-val" style="color: #ef4444;">{max_aqi:.0f}</div>
            <div class="metric-tile-sub">Peak AQI Recorded</div>
        </div>
        """), unsafe_allow_html=True)
    with m3:
        st.markdown(textwrap.dedent(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">CLEANEST AQI</div>
            <div class="metric-tile-val" style="color: #10b981;">{min_aqi:.0f}</div>
            <div class="metric-tile-sub">Best AQI Recorded</div>
        </div>
        """), unsafe_allow_html=True)
    with m4:
        st.markdown(textwrap.dedent(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">SATISFACTORY DAYS %</div>
            <div class="metric-tile-val">{clean_days_pct:.1f}%</div>
            <div class="metric-tile-sub">Days with AQI ≤ 100</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    # 1. AQI Timeline Plot
    st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>Historical AQI Timeline</div>", unsafe_allow_html=True)
    
    fig_line = px.line(plot_df, x="date", y="india_aqi", title="")
    fig_line.update_traces(line_color="#0f172a", line_width=2.2)
    fig_line.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="", tickfont=dict(color="#0f172a")),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="AQI", tickfont=dict(color="#0f172a"))
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    # 2. Multi-Pollutant Comparison & Distribution
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>PM2.5 vs PM10 Concentration</div>", unsafe_allow_html=True)
        
        fig_pm = px.line(plot_df, x="date", y=["pm2_5_mean", "pm10_mean"],
                        labels={"value": "µg/m³", "variable": "Pollutant"})
        fig_pm.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            legend=dict(orientation="h", y=1.1, font=dict(color="#0f172a")),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="", tickfont=dict(color="#0f172a")),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="µg/m³", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_pm, use_container_width=True)

    with col_chart2:
        st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>AQI Severity Category Breakdown</div>", unsafe_allow_html=True)
        
        cats = city_df["india_aqi"].apply(lambda x: get_aqi_info(x)["category"]).value_counts().reset_index()
        cats.columns = ["Category", "Count"]
        
        category_colors = {
            "Good": "#10B981",
            "Satisfactory": "#84CC16",
            "Moderate": "#F59E0B",
            "Poor": "#F97316",
            "Very Poor": "#EF4444",
            "Severe": "#881337"
        }
        
        fig_pie = px.pie(cats, names="Category", values="Count", color="Category", color_discrete_map=category_colors, hole=0.5)
        fig_pie.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#0f172a")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    # 3. National City Rankings
    st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>National City Air Quality Rankings</div>", unsafe_allow_html=True)

    latest_date_df = df[df["date"] == df["date"].max()].copy()
    city_rankings = latest_date_df.groupby("city")["india_aqi"].mean().reset_index().sort_values("india_aqi", ascending=False)

    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #ef4444; margin-bottom: 6px;'>Top 10 Most Polluted Cities</div>", unsafe_allow_html=True)
        top10_worst = city_rankings.head(10).sort_values("india_aqi", ascending=True)
        fig_worst = px.bar(top10_worst, x="india_aqi", y="city", orientation="h", color="india_aqi",
                           color_continuous_scale=["#f59e0b", "#f97316", "#ef4444", "#881337"])
        fig_worst.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            coloraxis_showscale=False,
            xaxis=dict(title="AQI", tickfont=dict(color="#0f172a")),
            yaxis=dict(title="", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_worst, use_container_width=True)

    with col_rank2:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #10b981; margin-bottom: 6px;'>Top 10 Cleanest Air Quality Cities</div>", unsafe_allow_html=True)
        top10_best = city_rankings.tail(10).sort_values("india_aqi", ascending=False)
        fig_best = px.bar(top10_best, x="india_aqi", y="city", orientation="h", color="india_aqi",
                          color_continuous_scale=["#10b981", "#84cc16", "#f59e0b"])
        fig_best.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            coloraxis_showscale=False,
            xaxis=dict(title="AQI", tickfont=dict(color="#0f172a")),
            yaxis=dict(title="", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_best, use_container_width=True)

# ============================================================
# 8. SCREEN 3: AI PREDICTION
# ============================================================

elif st.session_state.page == "AI Prediction":
    
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 18px;">
        <div class="card-title-text" style="font-size: 26px;">AI Air Quality Prediction Engine</div>
        <div style="font-size: 13px; color: #475569; font-weight: 500;">Estimate India AQI using trained Gradient Boosting Regressor model</div>
    </div>
    """), unsafe_allow_html=True)

    # City preset selector for quick filling
    preset_col1, preset_col2 = st.columns([2, 1])
    with preset_col1:
        preset_city = st.selectbox("Quick Prefill Parameters from Monitored City", ["Custom Input"] + sorted(df["city"].unique().tolist()))
    
    if preset_city != "Custom Input":
        preset_row = df[df["city"] == preset_city].iloc[0]
        def_pm25 = float(preset_row["pm2_5_mean"])
        def_pm25_max = float(preset_row["pm2_5_max"])
        def_pm10 = float(preset_row["pm10_mean"])
        def_pm10_max = float(preset_row["pm10_max"])
        def_no2 = float(preset_row["no2_mean"])
        def_so2 = float(preset_row["so2_mean"])
        def_o3 = float(preset_row["o3_8h_max"])
        def_co = float(preset_row["co_8h_max"])
        def_dust = float(preset_row["dust_mean"])
        def_aod = float(preset_row["aod_mean"])
        def_temp = float(preset_row["temperature_2m_mean"])
        def_hum = float(preset_row["relative_humidity_2m_mean"])
        def_wind = float(preset_row["wind_speed_10m_max"])
        def_precip = float(preset_row["precipitation_sum"])
        def_precip_h = float(preset_row["precipitation_hours"])
        def_wind_dir = float(preset_row["wind_direction_10m_dominant"])
        def_rad = float(preset_row["shortwave_radiation_sum"])
        def_elev = float(preset_row["elevation_m"])
    else:
        def_pm25, def_pm25_max = 52.4, 85.0
        def_pm10, def_pm10_max = 95.2, 140.0
        def_no2, def_so2 = 28.5, 12.4
        def_o3, def_co = 58.0, 650.0
        def_dust, def_aod = 18.2, 0.45
        def_temp, def_hum = 27.5, 62.0
        def_wind, def_precip = 11.2, 1.5
        def_precip_h, def_wind_dir = 0.5, 195.0
        def_rad, def_elev = 18.5, 120.0

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    with st.form("aqi_prediction_form"):
        st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 15px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>Particulate Matter</div>", unsafe_allow_html=True)
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            pm25 = st.number_input("PM2.5 Mean (µg/m³)", min_value=0.0, value=def_pm25, step=1.0, key="inp_pm25")
        with p_col2:
            pm25_max = st.number_input("PM2.5 Max (µg/m³)", min_value=0.0, value=def_pm25_max, step=1.0, key="inp_pm25_max")
        with p_col3:
            pm10 = st.number_input("PM10 Mean (µg/m³)", min_value=0.0, value=def_pm10, step=1.0, key="inp_pm10")
        with p_col4:
            pm10_max = st.number_input("PM10 Max (µg/m³)", min_value=0.0, value=def_pm10_max, step=1.0, key="inp_pm10_max")

        st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 15px; font-weight: 800; color: #0f172a; margin-top: 18px; margin-bottom: 10px;'>Gaseous Pollutants & Aerosols</div>", unsafe_allow_html=True)
        
        # Row 1 of Gaseous
        g1_col1, g1_col2, g1_col3, g1_col4 = st.columns(4)
        with g1_col1:
            no2 = st.number_input("NO₂ Mean (µg/m³)", min_value=0.0, value=def_no2, step=1.0, key="inp_no2")
        with g1_col2:
            so2 = st.number_input("SO₂ Mean (µg/m³)", min_value=0.0, value=def_so2, step=1.0, key="inp_so2")
        with g1_col3:
            o3 = st.number_input("O₃ 8h Max (µg/m³)", min_value=0.0, value=def_o3, step=1.0, key="inp_o3")
        with g1_col4:
            co = st.number_input("CO 8h Max (mg/m³)", min_value=0.0, value=def_co, step=10.0, key="inp_co")

        # Row 2 of Gaseous & Dust
        g2_col1, g2_col2, g2_col3, g2_col4 = st.columns(4)
        with g2_col1:
            dust = st.number_input("Dust Mean", min_value=0.0, value=def_dust, step=1.0, key="inp_dust")
        with g2_col2:
            aod = st.number_input("AOD Mean", min_value=0.0, value=def_aod, step=0.05, key="inp_aod")
        with g2_col3:
            st.empty()
        with g2_col4:
            st.empty()

        st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 15px; font-weight: 800; color: #0f172a; margin-top: 18px; margin-bottom: 10px;'>Weather & Geography</div>", unsafe_allow_html=True)
        
        # Row 1 of Weather
        w1_col1, w1_col2, w1_col3, w1_col4 = st.columns(4)
        with w1_col1:
            temp = st.number_input("Temperature (°C)", value=def_temp, step=0.5, key="inp_temp")
        with w1_col2:
            hum = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=def_hum, step=1.0, key="inp_hum")
        with w1_col3:
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=def_wind, step=0.5, key="inp_wind")
        with w1_col4:
            wind_dir = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=def_wind_dir, step=5.0, key="inp_wind_dir")

        # Row 2 of Weather
        w2_col1, w2_col2, w2_col3, w2_col4 = st.columns(4)
        with w2_col1:
            precip = st.number_input("Precipitation Sum (mm)", min_value=0.0, value=def_precip, step=0.5, key="inp_precip")
        with w2_col2:
            precip_h = st.number_input("Precipitation Hours", min_value=0.0, max_value=24.0, value=def_precip_h, step=0.5, key="inp_precip_h")
        with w2_col3:
            rad = st.number_input("Shortwave Radiation", min_value=0.0, value=def_rad, step=1.0, key="inp_rad")
        with w2_col4:
            elev = st.number_input("Elevation (m)", min_value=0.0, value=def_elev, step=10.0, key="inp_elev")

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Predict Air Quality Index", use_container_width=True, type="primary")

    if submit_btn:
        input_features = pd.DataFrame([[
            pm25, pm25_max, pm10, pm10_max, no2, so2, o3, co,
            dust, aod, temp, hum, wind_speed, precip, precip_h,
            wind_dir, rad, elev
        ]], columns=[
            "pm2_5_mean", "pm2_5_max", "pm10_mean", "pm10_max", "no2_mean", "so2_mean",
            "o3_8h_max", "co_8h_max", "dust_mean", "aod_mean", "temperature_2m_mean",
            "relative_humidity_2m_mean", "wind_speed_10m_max", "precipitation_sum",
            "precipitation_hours", "wind_direction_10m_dominant", "shortwave_radiation_sum",
            "elevation_m"
        ])

        prediction_val = float(model.predict(input_features)[0])
        pred_info = get_aqi_info(prediction_val)

        st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)
        
        st.markdown(textwrap.dedent(f"""
        <div class="card-panel" style="border-left: 5px solid {pred_info['color']};">
            <div class="card-header-label">GRADIENT BOOSTING PREDICTION RESULT</div>
            <div style="display: flex; align-items: baseline; gap: 16px; margin-top: 6px;">
                <div style="font-family: 'Manrope', sans-serif; font-size: 52px; font-weight: 800; color: #0f172a;">{prediction_val:.1f}</div>
                <div style="background-color: {pred_info['bg']}; color: {pred_info['text']}; padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 15px; border: 1px solid {pred_info['color']}40;">
                    {pred_info['dot']} {pred_info['category']}
                </div>
            </div>
            <div style="font-size: 13px; color: #334155; margin-top: 10px;">
                💡 <b>Health & Environmental Guidance:</b> {pred_info['desc']}
            </div>
        </div>
        """), unsafe_allow_html=True)

# ============================================================
# 9. SCREEN 4: MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "Model":
    
    st.markdown(textwrap.dedent("""
    <div style="margin-bottom: 18px;">
        <div class="card-title-text" style="font-size: 26px;">Model Evaluation & Benchmarking</div>
        <div style="font-size: 13px; color: #475569; font-weight: 500;">Evaluated on 15,250 unseen test samples (Gradient Boosting Regressor)</div>
    </div>
    """), unsafe_allow_html=True)

    mod1, mod2, mod3 = st.columns(3)

    with mod1:
        st.markdown(textwrap.dedent("""
        <div class="metric-tile">
            <div class="metric-tile-label">ROOT MEAN SQUARED ERROR (RMSE)</div>
            <div class="metric-tile-val" style="color: #10b981;">5.54</div>
            <div class="metric-tile-sub">Lowest error among tested models</div>
        </div>
        """), unsafe_allow_html=True)

    with mod2:
        st.markdown(textwrap.dedent("""
        <div class="metric-tile">
            <div class="metric-tile-label">MEAN ABSOLUTE ERROR (MAE)</div>
            <div class="metric-tile-val">1.25</div>
            <div class="metric-tile-sub">Average absolute prediction offset</div>
        </div>
        """), unsafe_allow_html=True)

    with mod3:
        st.markdown(textwrap.dedent("""
        <div class="metric-tile">
            <div class="metric-tile-label">R² SCORE (VARIANCE EXPLAINED)</div>
            <div class="metric-tile-val" style="color: #0f172a;">0.9941</div>
            <div class="metric-tile-sub">99.41% Variance Explained</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>Regression Algorithm Comparison</div>", unsafe_allow_html=True)

    comparison_data = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting (Selected)"],
        "MAE": [13.95, 0.78, 0.67, 1.25],
        "RMSE": [23.23, 8.52, 7.32, 5.54],
        "R² Score": [0.8959, 0.9860, 0.9897, 0.9941]
    })

    col_comp1, col_comp2 = st.columns([1.2, 1.8])

    with col_comp1:
        st.dataframe(
            comparison_data.style.highlight_min(subset=["RMSE", "MAE"], color="#dcfce7")
                                  .highlight_max(subset=["R² Score"], color="#dcfce7"),
            use_container_width=True,
            hide_index=True
        )

    with col_comp2:
        fig_comp = px.bar(comparison_data, x="Model", y="RMSE", text_auto=".2f",
                          color="Model", color_discrete_sequence=["#cbd5e1", "#cbd5e1", "#cbd5e1", "#0f172a"])
        fig_comp.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            showlegend=False,
            yaxis=dict(title="RMSE", tickfont=dict(color="#0f172a")),
            xaxis=dict(tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>Model Feature Importances</div>", unsafe_allow_html=True)

    if hasattr(model, "feature_importances_"):
        feature_names = [
            "PM2.5 Mean", "PM2.5 Max", "PM10 Mean", "PM10 Max", "NO₂ Mean", "SO₂ Mean",
            "O₃ 8h Max", "CO 8h Max", "Dust Mean", "AOD Mean", "Temperature",
            "Humidity", "Wind Speed", "Precipitation Sum", "Precipitation Hours",
            "Wind Direction", "Solar Radiation", "Elevation"
        ]
        
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance",
                        color_continuous_scale="Greys")
        fig_fi.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            coloraxis_showscale=False,
            xaxis=dict(title="Importance Weight", tickfont=dict(color="#0f172a")),
            yaxis=dict(title="", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    if df_test is not None:
        st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family: \"Manrope\", sans-serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>Actual vs Predicted AQI (Test Set Scatter Plot)</div>", unsafe_allow_html=True)

        sample_test = df_test.sample(min(2000, len(df_test)), random_state=42)
        fig_scatter = px.scatter(
            sample_test, x="india_aqi", y="predicted_aqi",
            labels={"india_aqi": "Actual AQI", "predicted_aqi": "Predicted AQI"},
            opacity=0.5
        )
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=500, y1=500,
            line=dict(color="#ef4444", width=2, dash="dash")
        )
        fig_scatter.update_traces(marker=dict(color="#0f172a", size=4))
        fig_scatter.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color="#0f172a")),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color="#0f172a"))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(textwrap.dedent("""
<div style="text-align: center; color: #64748b; font-size: 12px; font-weight: 600; padding-top: 30px; border-top: 1px solid #e2e8f0; margin-top: 30px;">
    Air Quality Intelligence • India Environmental Monitoring Platform
</div>
"""), unsafe_allow_html=True)