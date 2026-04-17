import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
import re
import os
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    ARIMA = None
    HAS_ARIMA = False

st.set_page_config(page_title="FMR Intelligence", layout="wide", page_icon="🏠")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp, .main,
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
    color: #1a2332 !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container { padding: 3rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] > div {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
.sidebar-brand {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #1a2332 !important;
    letter-spacing: -0.01em !important;
}
.sidebar-sub {
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
    margin-top: 2px !important;
}

/* ══ HERO ══ */
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1a2332 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 6px !important;
}
.hero-accent { color: #0ea5e9; }
.hero-sub {
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
    margin-bottom: 1.6rem !important;
}

/* ══ METRIC CARDS ══ */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: #0ea5e9 !important;
    box-shadow: 0 4px 16px rgba(14,165,233,0.1) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1a2332 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #10b981 !important; font-size: 0.8rem !important; }

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid #e2e8f0 !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 7px 16px !important;
    border: none !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #475569 !important; }
.stTabs [aria-selected="true"] {
    background: #f0f9ff !important;
    color: #0ea5e9 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ══ INPUTS ══ */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1a2332 !important;
}

/* ══ BUTTONS ══ */
.stButton > button, .stFormSubmitButton > button {
    background: #ffffff !important;
    color: #0ea5e9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stFormSubmitButton > button {
    background: #0ea5e9 !important;
    color: #ffffff !important;
    border-color: #0ea5e9 !important;
    font-weight: 600 !important;
}
.stFormSubmitButton > button:hover {
    background: #0284c7 !important;
    border-color: #0284c7 !important;
    box-shadow: 0 4px 16px rgba(14,165,233,0.25) !important;
}

/* ══ FORM ══ */
[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* ══ ALERTS ══ */
.stSuccess > div { background: #f0fdf4 !important; border-color: #86efac !important; color: #166534 !important; border-radius: 8px !important; }
.stInfo    > div { background: #f0f9ff !important; border-color: #7dd3fc !important; color: #075985 !important; border-radius: 8px !important; }
.stWarning > div { background: #fffbeb !important; border-color: #fcd34d !important; color: #92400e !important; border-radius: 8px !important; }
.stError   > div { background: #fef2f2 !important; border-color: #fca5a5 !important; color: #991b1b !important; border-radius: 8px !important; }

/* ══ EXPANDER ══ */
details { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important; }
summary { color: #475569 !important; font-weight: 500 !important; padding: 0.6rem 1rem !important; }

/* ══ HEADINGS ══ */
h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #1a2332 !important; letter-spacing: -0.01em !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; margin-bottom: 0.8rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

/* ══ CAPTION ══ */
.stCaption, small { color: #94a3b8 !important; font-size: 0.75rem !important; }

/* ══ DIVIDER ══ */
hr { border-color: #e2e8f0 !important; margin: 1rem 0 !important; }

/* ══ DATAFRAME ══ */
.stDataFrame, .stDataFrame > div { border-radius: 10px !important; overflow: hidden !important; border: 1px solid #e2e8f0 !important; }

/* ══ AI BOX ══ */
.ai-box {
    background: #f8fafc;
    border: 1px solid #0ea5e9;
    border-radius: 12px;
    padding: 1.6rem;
    color: #1a2332;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(14,165,233,0.08);
}
.ai-box h4 { color: #0ea5e9 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; margin: 0 0 1rem 0 !important; }

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f8fafc; }
::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    def repair(in_path, out_path):
        with zipfile.ZipFile(in_path, "r") as zin, zipfile.ZipFile(out_path, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "docProps/core.xml":
                    text = data.decode("utf-8", errors="ignore")
                    text = re.sub(
                        r"(\d{4})-\s*(\d{1,2})-(\d{1,2})T",
                        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T",
                        text,
                    )
                    data = text.encode("utf-8")
                zout.writestr(item, data)

    fy25_raw = "FY25_FMRs_revised.xlsx"
    fy25_fixed = "FY25_fixed.xlsx"
    if not os.path.exists(fy25_fixed):
        repair(fy25_raw, fy25_fixed)

    fy25 = pd.read_excel(fy25_fixed, sheet_name="FY25_FMRs_revised")
    fy26 = pd.read_excel("FY26_FMRs.xlsx", sheet_name="FY26_FMRs")

    keep = ["stusps", "countyname", "fips", "fmr_0", "fmr_1", "fmr_2", "fmr_3", "fmr_4"]
    agg = {f"fmr_{i}": "mean" for i in range(5)}
    agg["stusps"] = "first"
    agg["countyname"] = "first"
    fy25c = fy25[keep].groupby("fips", as_index=False).agg(agg).rename(columns={f"fmr_{i}": f"fmr_{i}_2025" for i in range(5)})
    fy26c = fy26[keep].groupby("fips", as_index=False).agg(agg).rename(columns={f"fmr_{i}": f"fmr_{i}_2026" for i in range(5)})

    df = fy25c.merge(fy26c, on="fips", how="inner")
    df = df.rename(columns={"stusps_x": "state", "countyname_x": "county"})
    df = df.drop(columns=["stusps_y", "countyname_y"])

    for i in range(5):
        df[f"change_{i}"] = df[f"fmr_{i}_2026"] - df[f"fmr_{i}_2025"]
        df[f"pct_{i}"] = (df[f"change_{i}"] / df[f"fmr_{i}_2025"]) * 100

    # HUD uses non-standard FIPS - convert to real 5-digit census FIPS
    
    df["fips_short"] = df["fips"].astype(str).str[:5].str.zfill(5)
    df = df[df["fips_short"].notna()]

    df = df.drop_duplicates(subset=["state", "county"])

    for i in range(5):
        rent_norm   = 1 - (df[f"fmr_{i}_2026"] - df[f"fmr_{i}_2026"].min()) / (df[f"fmr_{i}_2026"].max() - df[f"fmr_{i}_2026"].min())
        growth_norm = 1 - (df[f"pct_{i}"] - df[f"pct_{i}"].min()) / (df[f"pct_{i}"].max() - df[f"pct_{i}"].min())
        df[f"afford_score_{i}"] = ((rent_norm * 0.6 + growth_norm * 0.4) * 100).round(1)

    # ── FY27 Predictive Forecasting (ML: Linear Regression + ARIMA) ──
    def arima_one_step_forecast(series):
        if (not HAS_ARIMA) or len(series) < 4:
            return np.nan
        try:
            ts = pd.Series(series, dtype=float)
            model = ARIMA(ts, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            pred = float(fitted.forecast(steps=1).iloc[0])
            return max(pred, 0)
        except Exception:
            return np.nan

    def repair_excel(in_path, out_path):
        with zipfile.ZipFile(in_path, "r") as zin, zipfile.ZipFile(out_path, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "docProps/core.xml":
                    text = data.decode("utf-8", errors="ignore")
                    text = re.sub(
                        r"(\d{4})-\s*(\d{1,2})-(\d{1,2})T\s*(\d{1,2}):\s*(\d{1,2}):\s*(\d{1,2})Z",
                        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T{int(m.group(4)):02d}:{int(m.group(5)):02d}:{int(m.group(6)):02d}Z",
                        text,
                    )
                    text = re.sub(
                        r"(\d{4})-\s*(\d{1,2})-(\d{1,2})T",
                        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T",
                        text,
                    )
                    data = text.encode("utf-8")
                zout.writestr(item, data)

    def load_historical_year(file_name, sheet_name, year):
        fixed_name = f"{os.path.splitext(file_name)[0]}_ml_fixed.xlsx"
        if not os.path.exists(fixed_name):
            repair_excel(file_name, fixed_name)

        hist = pd.read_excel(fixed_name, sheet_name=sheet_name)

        if year == 2022:
            keep_cols = ["fips2010", "state_alpha", "countyname", "fmr_0", "fmr_1", "fmr_2", "fmr_3", "fmr_4"]
            hist = hist[keep_cols].rename(columns={"fips2010": "fips", "state_alpha": "state", "countyname": "county"})
        elif year == 2023:
            keep_cols = ["fips", "state_alpha", "countyname", "fmr_0", "fmr_1", "fmr_2", "fmr_3", "fmr_4"]
            hist = hist[keep_cols].rename(columns={"state_alpha": "state", "countyname": "county"})
        else:
            keep_cols = ["fips", "stusps", "countyname", "fmr_0", "fmr_1", "fmr_2", "fmr_3", "fmr_4"]
            hist = hist[keep_cols].rename(columns={"stusps": "state", "countyname": "county"})

        agg_hist = {f"fmr_{i}": "mean" for i in range(5)}
        agg_hist["state"] = "first"
        agg_hist["county"] = "first"

        hist = hist.groupby("fips", as_index=False).agg(agg_hist)
        hist = hist.rename(columns={f"fmr_{i}": f"fmr_{i}_{year}" for i in range(5)})
        return hist

    hist22 = load_historical_year("FY22_FMRs_revised.xlsx", "FY22_FMRs_revised", 2022)
    hist23 = load_historical_year("FY23_FMRs_revised.xlsx", "FY23_FMRs_revised", 2023)
    hist24 = load_historical_year("FMR2024_final_revised.xlsx", "FY24_FMRs_rev", 2024)

    history = hist22[["fips"] + [f"fmr_{i}_2022" for i in range(5)]].merge(
        hist23[["fips"] + [f"fmr_{i}_2023" for i in range(5)]], on="fips", how="outer"
    ).merge(
        hist24[["fips"] + [f"fmr_{i}_2024" for i in range(5)]], on="fips", how="outer"
    )

    df = df.merge(history, on="fips", how="left")

    years = np.array([2022, 2023, 2024, 2025, 2026], dtype=float)

    for i in range(5):
        final_preds = []
        lr_preds = []
        arima_preds = []
        r2_vals = []
        n_points = []
        model_labels = []

        for _, row in df.iterrows():
            y = np.array([
                row.get(f"fmr_{i}_2022", np.nan),
                row.get(f"fmr_{i}_2023", np.nan),
                row.get(f"fmr_{i}_2024", np.nan),
                row.get(f"fmr_{i}_2025", np.nan),
                row.get(f"fmr_{i}_2026", np.nan),
            ], dtype=float)

            mask = ~np.isnan(y)
            x_train = years[mask]
            y_train = y[mask]

            lr_pred = np.nan
            arima_pred = np.nan
            final_pred = np.nan
            model_label = "Fallback"

            if len(y_train) >= 3:
                model = LinearRegression()
                model.fit(x_train.reshape(-1, 1), y_train)
                lr_pred = float(model.predict(np.array([[2027.0]]))[0])
                lr_pred = max(lr_pred, 0)
                fitted = model.predict(x_train.reshape(-1, 1))
                r2 = float(r2_score(y_train, fitted)) if len(y_train) >= 2 else np.nan

                arima_pred = arima_one_step_forecast(y_train)
                if pd.notna(arima_pred):
                    final_pred = 0.6 * lr_pred + 0.4 * arima_pred
                    model_label = "LR + ARIMA Ensemble"
                else:
                    final_pred = lr_pred
                    model_label = "Linear Regression"
            else:
                growth_rate = (row[f"pct_{i}"] / 100) if pd.notna(row[f"pct_{i}"]) else 0
                growth_rate = float(np.clip(growth_rate, -0.25, 0.25))
                base_val = row[f"fmr_{i}_2026"] if pd.notna(row[f"fmr_{i}_2026"]) else np.nan
                final_pred = base_val * (1 + growth_rate) if pd.notna(base_val) else np.nan
                r2 = np.nan

            final_pred = round(max(final_pred, 0)) if pd.notna(final_pred) else np.nan
            lr_pred = round(max(lr_pred, 0)) if pd.notna(lr_pred) else np.nan
            arima_pred = round(max(arima_pred, 0)) if pd.notna(arima_pred) else np.nan

            final_preds.append(final_pred)
            lr_preds.append(lr_pred)
            arima_preds.append(arima_pred)
            r2_vals.append(r2)
            n_points.append(int(len(y_train)))
            model_labels.append(model_label)

        df[f"fmr_{i}_2027_lr"] = lr_preds
        df[f"fmr_{i}_2027_arima"] = arima_preds
        df[f"fmr_{i}_2027"] = final_preds
        df[f"forecast_model_{i}"] = model_labels
        df[f"change_27_{i}"] = (df[f"fmr_{i}_2027"] - df[f"fmr_{i}_2026"]).round(0)
        df[f"pct_27_{i}"] = ((df[f"change_27_{i}"] / df[f"fmr_{i}_2026"]) * 100).replace([np.inf, -np.inf], np.nan).round(2)
        df[f"r2_{i}"] = pd.Series(r2_vals).round(3)
        df[f"train_years_{i}"] = n_points

    return df

BEDROOM_LABELS = {0: "Studio", 1: "1-Bedroom", 2: "2-Bedroom", 3: "3-Bedroom", 4: "4-Bedroom"}
BEDROOM_EMOJI  = {0: "🛋️", 1: "🛏️", 2: "🛏️🛏️", 3: "🏠", 4: "🏡"}

df = load_data()
all_states = sorted(df["state"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-brand">🏠 FMR Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">HUD Fair Market Rents · FY25 → FY26</div>', unsafe_allow_html=True)
st.sidebar.divider()

bedroom = st.sidebar.selectbox(
    "Bedroom Size",
    options=[0, 1, 2, 3, 4],
    index=2,
    format_func=lambda x: f"{BEDROOM_EMOJI[x]} {BEDROOM_LABELS[x]}",
)
st.sidebar.divider()
st.sidebar.caption("Visual Analytics · Team 13 · Spring 2026")

pct_col   = f"pct_{bedroom}"
fy25_col  = f"fmr_{bedroom}_2025"
fy26_col  = f"fmr_{bedroom}_2026"
chg_col   = f"change_{bedroom}"
score_col = f"afford_score_{bedroom}"

filtered = df.copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">HUD Fair Market Rent Intelligence</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Analyzing {len(filtered):,} counties &nbsp;·&nbsp; {BEDROOM_EMOJI[bedroom]} {BEDROOM_LABELS[bedroom]} &nbsp;·&nbsp; FY2025 → FY2026 &nbsp;·&nbsp; Source: U.S. Department of Housing and Urban Development</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Market Overview",
    "🧮 Affordability Calculator",
    "🤖 AI County Recommender",
    "📋 Data Explorer",
    "🔮 FY27 Forecast",
])

# ═══════════════════════════════════════════════════════
# TAB 1 — Market Overview (Interactive)
# ═══════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg FY25 Rent",    f"${filtered[fy25_col].mean():,.0f}")
    c2.metric("Avg FY26 Rent",    f"${filtered[fy26_col].mean():,.0f}", delta=f"+${filtered[chg_col].mean():,.0f}")
    c3.metric("Avg % Change",     f"{filtered[pct_col].mean():.2f}%")
    c4.metric("Counties Tracked", f"{len(filtered):,}")

    st.divider()

    CHART_STYLE = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a2332", family="Plus Jakarta Sans"),
        title_font=dict(family="Plus Jakarta Sans", color="#1a2332", size=14),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    if "clicked_state" not in st.session_state:
        st.session_state["clicked_state"] = None

    state_df = filtered.groupby("state", as_index=False).agg(
        avg_pct=(pct_col, "mean"),
        avg_fy25=(fy25_col, "mean"),
        avg_fy26=(fy26_col, "mean"),
        county_count=("county", "count"),
    ).round(2)

    # ── National choropleth ──
    fig_national = px.choropleth(
        state_df, locations="state", locationmode="USA-states",
        color="avg_pct", scope="usa",
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        title=f"{BEDROOM_LABELS[bedroom]} Rent Change by State (FY25 → FY26)",
        labels={"avg_pct": "YoY Change (%)"},
        custom_data=["state", "avg_fy25", "avg_fy26", "avg_pct", "county_count"],
    )
    fig_national.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>"
                      "FY25 Avg: $%{customdata[1]:,.0f}<br>"
                      "FY26 Avg: $%{customdata[2]:,.0f}<br>"
                      "YoY Change: %{customdata[3]:.2f}%<br>"
                      "Counties: %{customdata[4]}<br>"
                      "<i>Click to drill down ↓</i><extra></extra>"
    )
    fig_national.update_layout(
        height=500,
        geo=dict(
            scope="usa", showlakes=False, showland=True,
            landcolor="rgba(240,244,248,1)",
            bgcolor="rgba(0,0,0,0)",
            subunitcolor="#ffffff", subunitwidth=1,
        ),
        **CHART_STYLE
    )
    clicked = st.plotly_chart(fig_national, use_container_width=True,
                               on_select="rerun", key="national_map")

    # Handle click
    if clicked and clicked.get("selection", {}).get("points"):
        pt = clicked["selection"]["points"][0]
        location = pt.get("location")
        if location and location in all_states:
            st.session_state["clicked_state"] = location

    # ── County drill-down ──
    if st.session_state["clicked_state"]:
        sel = st.session_state["clicked_state"]
        county_df = df[df["state"] == sel].copy()

        st.divider()
        col_title, col_reset = st.columns([5, 1])
        with col_title:
            st.success(f"📍 Drilling into **{sel}** — {len(county_df)} counties")
        with col_reset:
            if st.button("🔄 Reset", key="reset_drill"):
                st.session_state["clicked_state"] = None
                st.rerun()

        col_map, col_info = st.columns([2, 1])
        with col_map:
            fig_drill = px.choropleth(
                county_df,
                geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                locations="fips_short", color=pct_col,
                color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                scope="usa",
                title=f"{sel}: {BEDROOM_LABELS[bedroom]} Rent Change by County",
                custom_data=["county", fy25_col, fy26_col, pct_col, f"afford_score_{bedroom}"],
                labels={pct_col: "YoY %"},
            )
            fig_drill.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "FY25: $%{customdata[1]:,.0f}/mo<br>"
                              "FY26: $%{customdata[2]:,.0f}/mo<br>"
                              "Change: %{customdata[3]:.2f}%<br>"
                              "Afford. Score: %{customdata[4]:.0f}/100<extra></extra>"
            )
            fig_drill.update_geos(fitbounds="locations", visible=False)
            fig_drill.update_layout(height=420, **CHART_STYLE)
            st.plotly_chart(fig_drill, use_container_width=True)

        with col_info:
            sel_state = state_df[state_df["state"] == sel].iloc[0]
            st.markdown(f"### 📊 {sel} Summary")
            st.metric("Avg FY26 Rent",  f"${sel_state['avg_fy26']:,.0f}")
            st.metric("Avg YoY Change", f"{sel_state['avg_pct']:.2f}%")
            st.metric("Counties",       int(sel_state["county_count"]))
            st.divider()
            st.markdown("**🏆 Top 5 Cheapest Counties**")
            top5 = county_df.nsmallest(5, fy26_col)[["county", fy26_col, pct_col]]
            for _, row in top5.iterrows():
                arrow = "↑" if row[pct_col] >= 0 else "↓"
                st.markdown(f"**{row['county']}** — ${row[fy26_col]:,.0f}/mo ({arrow}{abs(row[pct_col]):.1f}%)")

        # County bar chart
        st.subheader(f"All Counties in {sel} — Ranked by Rent Change")
        county_sorted = county_df.sort_values(pct_col)
        county_sorted["Trend"] = county_sorted[pct_col].apply(lambda x: "Rising" if x >= 0 else "Stable/Declining")
        fig_cb = px.bar(
            county_sorted, x=pct_col, y="county", orientation="h",
            color="Trend",
            color_discrete_map={"Rising": "#ef4444", "Stable/Declining": "#22c55e"},
            custom_data=["county", fy25_col, fy26_col, pct_col],
            labels={pct_col: "YoY %", "county": ""},
            title=f"{sel}: All Counties Ranked by Rent Change",
            height=max(400, len(county_sorted) * 18),
        )
        fig_cb.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "FY25: $%{customdata[1]:,.0f} → FY26: $%{customdata[2]:,.0f}<br>"
                          "Change: %{customdata[3]:.2f}%<extra></extra>"
        )
        fig_cb.update_layout(showlegend=True, **CHART_STYLE)
        st.plotly_chart(fig_cb, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — Affordability Calculator
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("🧮 Personal Affordability Calculator")
    st.markdown("*Based on the **30% rule**: monthly rent ≤ 30% of monthly gross income.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        annual_salary = st.number_input("Annual Salary ($)", 10000, 500000, 60000, 5000)
    with col2:
        pref_states_calc = st.selectbox("Preferred State (blank = all)", ["All States"] + all_states, key="calc_states")
        pref_states_calc = [] if pref_states_calc == "All States" else [pref_states_calc]
    with col3:
        max_growth = st.slider("Max Rent Growth Tolerance (%)", -15, 30, 10)

    monthly_budget = (annual_salary / 12) * 0.30
    col_budget1, col_budget2, col_budget3 = st.columns(3)
    col_budget1.metric("Your Monthly Rent Budget", f"${monthly_budget:,.0f}", help="30% of your monthly gross income")
    col_budget2.metric("Monthly Gross Income", f"${annual_salary/12:,.0f}")
    col_budget3.metric("Annual Salary", f"${annual_salary:,.0f}")

    calc_df = df.copy()
    if pref_states_calc:
        calc_df = calc_df[calc_df["state"].isin(pref_states_calc)]

    affordable = calc_df[
        (calc_df[fy26_col] <= monthly_budget) &
        (calc_df[pct_col] <= max_growth)
    ].copy()
    affordable["Monthly Savings"] = (monthly_budget - affordable[fy26_col]).round(0)
    affordable["Budget Used %"]   = (affordable[fy26_col] / monthly_budget * 100).round(1)
    affordable = affordable.sort_values(score_col, ascending=False)

    st.divider()

    if len(affordable) == 0:
        st.warning("😕 No counties match your criteria. Try increasing salary or relaxing the growth filter.")
    else:
        st.success(f"✅ Found **{len(affordable):,}** affordable counties!")

        fig_scatter = px.scatter(
            affordable.head(300),
            x=pct_col, y=fy26_col,
            color=score_col, size="Monthly Savings",
            hover_data={"state": True, "county": True,
                        fy26_col: ":$,.0f", pct_col: ":.1f", score_col: True},
            color_continuous_scale="RdYlGn",
            labels={pct_col: "Rent Growth (%)", fy26_col: "FY26 Rent ($/mo)",
                    score_col: "Affordability Score"},
            title="Rent Level vs Growth Rate — Bubble Size = Monthly Savings",
        )
        fig_scatter.add_hline(y=monthly_budget, line_dash="dash", line_color="red",
                               annotation_text=f"Your Budget ${monthly_budget:,.0f}")
        fig_scatter.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)", font=dict(color="#1a2332", family="DM Sans"), title_font=dict(family="Syne", color="#1a2332"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # State map - show all states colored by how many affordable counties
        all_state_agg = df.groupby("state", as_index=False).agg(total=("county","count"))
        aff_state = affordable.groupby("state", as_index=False).agg(
            count=("county", "count"),
            avg_rent=(fy26_col, "mean"),
        )
        state_map_df = all_state_agg.merge(aff_state, on="state", how="left").fillna(0)
        state_map_df["pct_affordable"] = (state_map_df["count"] / state_map_df["total"] * 100).round(1)

        fig_aff_map = px.choropleth(
            state_map_df, locations="state", locationmode="USA-states",
            color="pct_affordable", scope="usa", color_continuous_scale="RdYlGn",
            title="% of Counties Affordable on Your Budget by State",
            hover_data={"count": True, "pct_affordable": ":.1f"},
            labels={"pct_affordable": "% Affordable", "count": "# Counties"},
        )
        fig_aff_map.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0)", font=dict(color="#1a2332", family="DM Sans"), title_font=dict(family="Syne", color="#1a2332"))
        st.plotly_chart(fig_aff_map, use_container_width=True)

        # County-level map
        st.subheader("🔍 County-Level Affordability Map")
        # Auto-select state based on preferred states filter using session state
        if pref_states_calc and st.session_state.get("aff_county_state") != pref_states_calc[0]:
            st.session_state["aff_county_state"] = pref_states_calc[0]

        # Default to NC or first continental state, skip territories
        continental = [s for s in all_states if s not in ["AS", "GU", "MP", "PR", "VI"]]
        default_state = pref_states_calc[0] if pref_states_calc else ("NC" if "NC" in continental else continental[0])
        if "aff_county_state" not in st.session_state:
            st.session_state["aff_county_state"] = default_state

        county_state_sel = st.selectbox("Select a State to explore county-level affordability",
                                         continental,
                                         index=continental.index(st.session_state["aff_county_state"]) if st.session_state["aff_county_state"] in continental else 0,
                                         key="aff_county_state")
        county_aff = affordable[affordable["state"] == county_state_sel].copy()
        all_county_state = df[df["state"] == county_state_sel].copy()
        all_county_state["affordable"] = all_county_state["county"].isin(county_aff["county"])
        all_county_state["label"] = all_county_state["affordable"].map({True: "✅ Affordable", False: "❌ Over Budget"})

        fig_county_aff = px.choropleth(
            all_county_state,
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            locations="fips_short", color="label",
            color_discrete_map={"✅ Affordable": "#22c55e", "❌ Over Budget": "#ef4444"},
            scope="usa",
            title=f"{county_state_sel}: Affordable Counties on ${monthly_budget:,.0f}/mo Budget",
            hover_data={"fips_short": False, "county": True, fy26_col: ":$,.0f", pct_col: ":.1f"},
            labels={"label": "Status"},
        )
        fig_county_aff.update_geos(fitbounds="locations", visible=False)
        fig_county_aff.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0)", font=dict(color="#1a2332", family="DM Sans"), title_font=dict(family="Syne", color="#1a2332"))
        st.plotly_chart(fig_county_aff, use_container_width=True)

        st.subheader("🏆 Top Affordable Counties")
        top_aff_display = affordable.copy()
        if pref_states_calc:
            top_aff_display = top_aff_display[top_aff_display["state"].isin(pref_states_calc)]
        top_aff = top_aff_display.head(20)[["state", "county", fy26_col, pct_col, score_col, "Monthly Savings", "Budget Used %"]].rename(columns={
            fy26_col: "FY26 Rent", pct_col: "Growth %", score_col: "Afford. Score"
        }).reset_index(drop=True)
        st.dataframe(
            top_aff.style
                .format({"FY26 Rent": "${:,.0f}", "Growth %": "{:.1f}%",
                         "Afford. Score": "{:.0f}", "Monthly Savings": "${:,.0f}", "Budget Used %": "{:.1f}%"})
                .background_gradient(subset=["Afford. Score"], cmap="RdYlGn")
                .background_gradient(subset=["Budget Used %"], cmap="RdYlGn_r"),
            use_container_width=True, height=420,
        )

# ═══════════════════════════════════════════════════════
# TAB 3 — AI County Recommender
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 AI-Powered County Recommender")
    st.markdown("*Tell the AI your situation — get personalized county picks backed by real HUD data.*")

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("⚠️ No Groq API key found. Set it in terminal: `export GROQ_API_KEY='your-key-here'` then restart the app.")

    with st.form("ai_form"):
        col1, col2 = st.columns(2)
        with col1:
            ai_salary  = st.number_input("Annual Salary ($)", 20000, 500000, 65000, 5000, key="ai_sal")
            ai_bedroom = st.selectbox("Bedroom Size Needed", options=[0,1,2,3,4],
                                       format_func=lambda x: f"{BEDROOM_EMOJI[x]} {BEDROOM_LABELS[x]}", index=2, key="ai_bed")
            ai_states  = st.multiselect("Preferred States (blank = open to all)", all_states, key="ai_st")
        with col2:
            ai_priorities = st.multiselect("Your Priorities",
                ["Lowest possible rent", "Rent stability (low growth)", "Maximize monthly savings",
                 "Avoid high cost-of-living areas", "College/university town", "Near a major city"],
                default=["Lowest possible rent", "Rent stability (low growth)"], key="ai_pri")
            ai_situation = st.text_area("Anything else?",
                placeholder="e.g. grad student, remote worker, moving with family, prefer rural areas...",
                height=110, key="ai_sit")
        submitted = st.form_submit_button("🚀 Get My AI Recommendations", use_container_width=True)

    if submitted:
        b2         = ai_bedroom
        b2_pct     = f"pct_{b2}"
        b2_fy26    = f"fmr_{b2}_2026"
        b2_score   = f"afford_score_{b2}"
        budget_ai  = (ai_salary / 12) * 0.30

        cand = df.copy()
        if ai_states:
            cand = cand[cand["state"].isin(ai_states)]
        cand = cand[cand[b2_fy26] <= budget_ai * 1.15].sort_values(b2_score, ascending=False).head(20)

        context_lines = "\n".join([
            f"- {r['county']}, {r['state']}: rent ${r[b2_fy26]:.0f}/mo, growth {r[b2_pct]:.1f}%, score {r[b2_score]:.0f}/100"
            for _, r in cand.iterrows()
        ]) or "No counties found within budget."

        prompt = f"""You are a housing market expert with access to HUD Fair Market Rent data.

User profile:
- Annual salary: ${ai_salary:,} → monthly rent budget (30% rule): ${budget_ai:,.0f}
- Bedroom size: {BEDROOM_LABELS[ai_bedroom]}
- Preferred states: {', '.join(ai_states) if ai_states else 'Any state'}
- Priorities: {', '.join(ai_priorities)}
- Notes: {ai_situation or 'None'}

Top candidate counties from HUD FY2026 data (sorted by affordability score):
{context_lines}

Provide:
1. **Top 5 County Recommendations** — for each, bold the county name, give actual rent figure, and 2-3 sentences on why it fits this user
2. **Market Insight** — one paragraph key takeaway from this data
3. **Watch Out** — one important caveat or consideration

Be specific, data-driven, and concise (under 500 words total). Do NOT use backticks or code formatting anywhere in your response. Write all numbers as plain text."""

        with st.spinner("🤖 AI is analyzing HUD data for you..."):
            try:
                api_key = os.environ.get("GROQ_API_KEY", "")
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "max_tokens": 1000,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                result = resp.json()
                if "error" in result:
                    st.error(f"API Error: {result['error']['message']}")
                else:
                    ai_text = result["choices"][0]["message"]["content"]
                    # Remove any backtick code formatting the AI might produce
                    import re as _re
                    ai_text = _re.sub(r'`([^`]*)`', r'\1', ai_text)

                    st.markdown('<div class="ai-box"><h4 style="color:#4ecca3;margin-top:0">🤖 AI Recommendations</h4>', unsafe_allow_html=True)
                    st.markdown(ai_text)
                    st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("📊 Data Claude analyzed"):
                        disp = cand[["state", "county", b2_fy26, b2_pct, b2_score]].rename(columns={
                            b2_fy26: "FY26 Rent", b2_pct: "Growth %", b2_score: "Afford. Score"
                        }).reset_index(drop=True)
                        st.dataframe(
                            disp.style.format({"FY26 Rent": "${:,.0f}", "Growth %": "{:.1f}%", "Afford. Score": "{:.0f}"})
                                .background_gradient(subset=["Afford. Score"], cmap="RdYlGn"),
                            use_container_width=True,
                        )

                    fig_ai = px.scatter(
                        cand, x=b2_pct, y=b2_fy26, text="county",
                        color=b2_score, color_continuous_scale="RdYlGn",
                        labels={b2_pct: "Rent Growth (%)", b2_fy26: "Monthly Rent ($)", b2_score: "Score"},
                        title="Candidate Counties — Growth vs Rent",
                    )
                    fig_ai.add_hline(y=budget_ai, line_dash="dash", line_color="red",
                                      annotation_text=f"Your Budget ${budget_ai:,.0f}")
                    fig_ai.update_traces(textposition="top center")
                    fig_ai.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)", font=dict(color="#1a2332", family="DM Sans"), title_font=dict(family="Syne", color="#1a2332"))
                    st.plotly_chart(fig_ai, use_container_width=True)

            except Exception as e:
                st.error(f"Error calling Claude API: {e}. Make sure ANTHROPIC_API_KEY is set in your terminal before running the app.")

# ═══════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ═══════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Full Data Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        sort_label = st.selectbox("Sort by", ["% Change", "FY26 Rent", "FY25 Rent", "$ Change", "Afford. Score"])
    with col2:
        sort_dir = st.radio("Direction", ["Descending", "Ascending"], horizontal=True)
    with col3:
        state_filter2 = st.multiselect("Filter States", all_states, key="exp_states")

    exp_df = df.copy()
    if state_filter2:
        exp_df = exp_df[exp_df["state"].isin(state_filter2)]

    sort_map = {"% Change": pct_col, "FY26 Rent": fy26_col, "FY25 Rent": fy25_col,
                "$ Change": chg_col, "Afford. Score": score_col}
    exp_df = exp_df.sort_values(sort_map[sort_label], ascending=(sort_dir == "Ascending"))

    show = exp_df[["state", "county", fy25_col, fy26_col, chg_col, pct_col, score_col]].rename(columns={
        fy25_col: "FY25 Rent", fy26_col: "FY26 Rent",
        chg_col: "$ Change", pct_col: "% Change", score_col: "Afford. Score",
    }).reset_index(drop=True)

    st.dataframe(
        show.style
            .format({"FY25 Rent": "${:,.0f}", "FY26 Rent": "${:,.0f}",
                     "$ Change": "${:,.0f}", "% Change": "{:.2f}%", "Afford. Score": "{:.0f}"})
            .background_gradient(subset=["% Change"], cmap="RdYlGn")
            .background_gradient(subset=["Afford. Score"], cmap="RdYlGn"),
        use_container_width=True, height=520,
    )
    st.caption(f"Showing {len(show):,} counties · Source: HUD FMR FY2025 & FY2026 · Visual Analytics Team 13")

# ═══════════════════════════════════════════════════════
# TAB 5 — FY27 Predictive Forecast
# ═══════════════════════════════════════════════════════
with tab5:
    fy22_col   = f"fmr_{bedroom}_2022"
    fy23_col   = f"fmr_{bedroom}_2023"
    fy24_col   = f"fmr_{bedroom}_2024"
    fy27_col   = f"fmr_{bedroom}_2027"
    pct27_col  = f"pct_27_{bedroom}"
    chg27_col  = f"change_27_{bedroom}"
    r2_col     = f"r2_{bedroom}"
    years_col  = f"train_years_{bedroom}"
    lr27_col    = f"fmr_{bedroom}_2027_lr"
    arima27_col = f"fmr_{bedroom}_2027_arima"
    model_col   = f"forecast_model_{bedroom}"

    CHART_STYLE_F = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a2332", family="Plus Jakarta Sans"),
        title_font=dict(family="Plus Jakarta Sans", color="#1a2332", size=14),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    st.subheader("🔮 FY27 Rent Forecast")
    st.markdown(
        "*Predicted FY2027 rents using a county-level Linear Regression model trained on HUD Fair Market Rent history "
        "(FY2022–FY2026). The model learns the trend for each county and bedroom size, then forecasts FY2027.*"
    )

    model_df = df.dropna(subset=[fy27_col]).copy()

    # ── KPI Cards ──
    avg_fy26 = model_df[fy26_col].mean()
    avg_fy27 = model_df[fy27_col].mean()
    avg_chg  = avg_fy27 - avg_fy26
    rising   = (model_df[pct27_col] > 0).sum()
    declining = (model_df[pct27_col] <= 0).sum()
    avg_r2 = model_df[r2_col].dropna().mean()
    model_coverage = (model_df[years_col] >= 3).sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg FY26 Rent",      f"${avg_fy26:,.0f}")
    k2.metric("Avg FY27 Forecast",  f"${avg_fy27:,.0f}", delta=f"+${avg_chg:,.0f}")
    k3.metric("Projected Change",   f"{(avg_chg/avg_fy26*100):.2f}%")
    k4.metric("Model Coverage",     f"{model_coverage:,}")
    k5.metric("Avg Model R²",       f"{avg_r2:.2f}" if pd.notna(avg_r2) else "N/A")
    st.caption("Model logic: LR + ARIMA ensemble where possible, otherwise Linear Regression fallback.")

    st.divider()

    # ── National Forecast Choropleth ──
    state_f = model_df.groupby("state", as_index=False).agg(
        avg_fy26=(fy26_col, "mean"),
        avg_fy27=(fy27_col, "mean"),
        avg_pct27=(pct27_col, "mean"),
        avg_r2=(r2_col, "mean"),
        county_count=("county", "count"),
    ).round(2)
    state_f["avg_chg27"] = (state_f["avg_fy27"] - state_f["avg_fy26"]).round(0)

    fig_f_map = px.choropleth(
        state_f, locations="state", locationmode="USA-states",
        color="avg_pct27", scope="usa",
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        title=f"Predicted {BEDROOM_LABELS[bedroom]} Rent Change FY26 → FY27 (Ensemble Forecast)",
        labels={"avg_pct27": "Predicted Change (%)"},
        custom_data=["state", "avg_fy26", "avg_fy27", "avg_pct27", "avg_chg27", "avg_r2"],
    )
    fig_f_map.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>"
                      "FY26 Avg: $%{customdata[1]:,.0f}<br>"
                      "FY27 Pred: $%{customdata[2]:,.0f}<br>"
                      "Change: %{customdata[3]:.2f}% ($%{customdata[4]:,.0f})<br>"
                      "Avg R²: %{customdata[5]:.2f}<extra></extra>"
    )
    fig_f_map.update_layout(
        height=480,
        geo=dict(scope="usa", showlakes=False, showland=True,
                 landcolor="rgba(240,244,248,1)", bgcolor="rgba(0,0,0,0)",
                 subunitcolor="#ffffff", subunitwidth=1),
        **CHART_STYLE_F
    )
    st.plotly_chart(fig_f_map, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

    st.divider()

    # ── Top Rising vs Most Stable side by side ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 📈 Top 10 States — Predicted Rent Increase")
        top_rising = state_f.nlargest(10, "avg_pct27").sort_values("avg_pct27")
        fig_rising = px.bar(
            top_rising, x="avg_pct27", y="state", orientation="h",
            color="avg_pct27", color_continuous_scale="Reds",
            labels={"avg_pct27": "Predicted Change (%)"},
            custom_data=["state", "avg_fy27", "avg_pct27", "avg_r2"],
        )
        fig_rising.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>FY27 Pred: $%{customdata[1]:,.0f}<br>Change: %{customdata[2]:.2f}%<br>Avg R²: %{customdata[3]:.2f}<extra></extra>"
        )
        fig_rising.update_layout(height=370, showlegend=False, coloraxis_showscale=False, **CHART_STYLE_F)
        st.plotly_chart(fig_rising, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

    with col_r:
        st.markdown("#### 💚 Top 10 States — Most Stable / Declining")
        top_stable = state_f.nsmallest(10, "avg_pct27").sort_values("avg_pct27", ascending=False)
        fig_stable = px.bar(
            top_stable, x="avg_pct27", y="state", orientation="h",
            color="avg_pct27", color_continuous_scale="Greens_r",
            labels={"avg_pct27": "Predicted Change (%)"},
            custom_data=["state", "avg_fy27", "avg_pct27", "avg_r2"],
        )
        fig_stable.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>FY27 Pred: $%{customdata[1]:,.0f}<br>Change: %{customdata[2]:.2f}%<br>Avg R²: %{customdata[3]:.2f}<extra></extra>"
        )
        fig_stable.update_layout(height=370, showlegend=False, coloraxis_showscale=False, **CHART_STYLE_F)
        st.plotly_chart(fig_stable, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

    st.divider()

    # ── 6-Year Trend line chart ──
    st.markdown("#### 📊 6-Year Rent Trend by State (FY22 → FY27 Prediction)")
    continental = [s for s in sorted(model_df["state"].unique()) if s not in ["AS","GU","MP","PR","VI"]]
    trend_state = st.selectbox(
        "Select a state",
        continental,
        index=continental.index("NC") if "NC" in continental else 0,
        key="trend_state"
    )
    trend_df = model_df[model_df["state"] == trend_state].copy()

    trend_rows = []
    for _, row in trend_df.iterrows():
        trend_rows.extend([
            {"county": row["county"], "year": "FY2022",        "rent": row[fy22_col]},
            {"county": row["county"], "year": "FY2023",        "rent": row[fy23_col]},
            {"county": row["county"], "year": "FY2024",        "rent": row[fy24_col]},
            {"county": row["county"], "year": "FY2025",        "rent": row[fy25_col]},
            {"county": row["county"], "year": "FY2026",        "rent": row[fy26_col]},
            {"county": row["county"], "year": "FY2027 (Pred.)","rent": row[fy27_col]},
        ])
    trend_long = pd.DataFrame(trend_rows).dropna(subset=["rent"])

    state_avg = trend_long.groupby("year", as_index=False)["rent"].mean().round(0)
    state_avg["order"] = state_avg["year"].map({
        "FY2022": 0, "FY2023": 1, "FY2024": 2,
        "FY2025": 3, "FY2026": 4, "FY2027 (Pred.)": 5
    })
    state_avg = state_avg.sort_values("order")

    col_trend, col_side = st.columns([2, 1])
    with col_trend:
        fig_trend = px.line(
            state_avg, x="year", y="rent", markers=True,
            title=f"{trend_state}: Avg {BEDROOM_LABELS[bedroom]} Rent — 6 Year Trend",
            labels={"year": "", "rent": "Avg Monthly Rent ($)"},
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_trend.update_traces(
            line=dict(width=3), marker=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Avg Rent: $%{y:,.0f}<extra></extra>",
        )
        fig_trend.add_vrect(
            x0="FY2026", x1="FY2027 (Pred.)",
            fillcolor="rgba(14,165,233,0.06)", layer="below", line_width=0,
            annotation_text="Prediction Zone", annotation_position="top left",
            annotation_font_color="#94a3b8", annotation_font_size=10,
        )
        fig_trend.update_layout(height=360, **CHART_STYLE_F)
        st.plotly_chart(fig_trend, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

    with col_side:
        fy24_avg = trend_df[fy24_col].mean()
        fy25_avg = trend_df[fy25_col].mean()
        fy26_avg = trend_df[fy26_col].mean()
        fy27_avg = trend_df[fy27_col].mean()
        state_r2 = trend_df[r2_col].dropna().mean()
        st.markdown(f"**{trend_state} Summary**")
        st.metric("FY24 Avg",  f"${fy24_avg:,.0f}")
        st.metric("FY25 Avg",  f"${fy25_avg:,.0f}", delta=f"+${fy25_avg-fy24_avg:,.0f}")
        st.metric("FY26 Avg",  f"${fy26_avg:,.0f}", delta=f"+${fy26_avg-fy25_avg:,.0f}")
        st.metric("FY27 Pred.", f"${fy27_avg:,.0f}", delta=f"+${fy27_avg-fy26_avg:,.0f}")
        st.metric("Avg State R²", f"{state_r2:.2f}" if pd.notna(state_r2) else "N/A")
        st.divider()
        st.markdown("**📉 Cheapest FY27 Counties**")
        for _, row in trend_df.nsmallest(5, fy27_col)[["county", fy27_col, pct27_col]].iterrows():
            arrow = "↑" if row[pct27_col] > 0 else "↓"
            st.markdown(f"**{row['county']}** — ${row[fy27_col]:,.0f}/mo ({arrow}{abs(row[pct27_col]):.1f}%)")

    st.divider()

    # ── Full county forecast table ──
    st.markdown("#### 🔍 County-Level FY27 Forecast Table")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fcast_state = st.selectbox("Filter by State", ["All States"] + continental, key="fcast_state")
    with col_f2:
        fcast_sort = st.selectbox(
            "Sort by",
            ["Highest Projected Rent", "Lowest Projected Rent", "Biggest Increase", "Biggest Decrease", "Highest R²"],
            key="fcast_sort"
        )

    fcast_df = model_df.copy()
    if fcast_state != "All States":
        fcast_df = fcast_df[fcast_df["state"] == fcast_state]

    sort_map_f = {
        "Highest Projected Rent": (fy27_col, False),
        "Lowest Projected Rent":  (fy27_col, True),
        "Biggest Increase":       (pct27_col, False),
        "Biggest Decrease":       (pct27_col, True),
        "Highest R²":             (r2_col, False),
    }
    scol, sasc = sort_map_f[fcast_sort]
    fcast_df = fcast_df.sort_values(scol, ascending=sasc, na_position="last")

    fcast_show = fcast_df[["state", "county", fy22_col, fy23_col, fy24_col, fy25_col, fy26_col, lr27_col, arima27_col, fy27_col, pct27_col, r2_col, years_col, model_col]].rename(columns={
        fy22_col:  "FY22 Rent",
        fy23_col:  "FY23 Rent",
        fy24_col:  "FY24 Rent",
        fy25_col:  "FY25 Rent",
        fy26_col:  "FY26 Rent",
        lr27_col:  "FY27 LR",
        arima27_col: "FY27 ARIMA",
        fy27_col:  "FY27 Final",
        pct27_col: "FY26→27 % (Pred.)",
        r2_col:    "Model R²",
        years_col: "Train Years",
        model_col: "Model Used",
    }).reset_index(drop=True)

    st.dataframe(
        fcast_show.style
            .format({
                "FY22 Rent": "${:,.0f}",
                "FY23 Rent": "${:,.0f}",
                "FY24 Rent": "${:,.0f}",
                "FY25 Rent": "${:,.0f}",
                "FY26 Rent": "${:,.0f}",
                "FY27 LR": "${:,.0f}",
                "FY27 ARIMA": "${:,.0f}",
                "FY27 Final": "${:,.0f}",
                "FY26→27 % (Pred.)": "{:.2f}%",
                "Model R²": "{:.2f}",
                "Train Years": "{:.0f}",
            })
            .background_gradient(subset=["FY26→27 % (Pred.)"], cmap="RdYlGn_r")
            .background_gradient(subset=["Model R²"], cmap="Blues"),
        use_container_width=True, height=480,
    )
    st.caption("⚠️ FY27 values are machine learning forecasts from county-level Linear Regression + ARIMA models trained on HUD FMR history (FY2022–FY2026). If ARIMA is unavailable or unstable for a county, the app falls back safely to Linear Regression or capped trend extrapolation. Visual Analytics Team 13")

