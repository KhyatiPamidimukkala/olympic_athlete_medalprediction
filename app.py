"""
Olympic Medal Data Analysis Using Machine Learning - Interactive Web Dashboard
Run locally with: python -m streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, silhouette_score
)

plt.style.use("dark_background")

mpl.rcParams["figure.facecolor"] = "#0F172A"
mpl.rcParams["axes.facecolor"] = "#0F172A"
mpl.rcParams["axes.edgecolor"] = "#475569"
mpl.rcParams["axes.labelcolor"] = "#F1F5F9"
mpl.rcParams["text.color"] = "#F1F5F9"
mpl.rcParams["xtick.color"] = "#CBD5E1"
mpl.rcParams["ytick.color"] = "#CBD5E1"
mpl.rcParams["grid.color"] = "#334155"
mpl.rcParams["legend.facecolor"] = "#1E293B"
mpl.rcParams["legend.edgecolor"] = "#475569"
mpl.rcParams["savefig.facecolor"] = "#0F172A"



def style_graph(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(
            color="#F1F5F9",
            family="Arial"
        ),
        title_font=dict(
            color="#FFFFFF",
            size=20
        ),
        legend=dict(
            font=dict(color="#F1F5F9")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#334155",
            zerolinecolor="#475569"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#334155",
            zerolinecolor="#475569"
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig



st.set_page_config(
    page_title="Olympic Medal Analysis - ML Dashboard",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #101525, #191E35, #222744);
    color: #F1F5F9;
}

.block-container {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3, h4 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

/* General text */
p, li, label, .stMarkdown {
    color: #E2E8F0;
}

/* Filter labels */
.stSelectbox label,
.stTextInput label,
.stMultiSelect label,
.stDateInput label {
    color: #A5B4FC !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* Dropdown fields */
div[data-baseweb="select"] > div {
    background-color: #252B45 !important;
    border: 1px solid #6875B8 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

/* Text input and search field */
.stTextInput input,
.stDateInput input,
.stNumberInput input {
    background-color: #252B45 !important;
    color: #FFFFFF !important;
    border: 1px solid #6875B8 !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder {
    color: #C0C9E0 !important;
    opacity: 1 !important;
}

/* Dropdown options */
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #252B45 !important;
}

div[data-baseweb="menu"] *,
div[data-baseweb="popover"] * {
    color: #FFFFFF !important;
}

/* Dataset table */
div[data-testid="stDataFrame"] {
    background-color: #171D31 !important;
    border: 1px solid #414B70 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(42, 50, 79, 0.85);
    border: 1px solid #414B70;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #C7D2FE !important;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #4F46E5 !important;
    border: 1px solid #C4B5FD !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111629 !important;
    border-right: 1px solid #343D5C;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* Sidebar radio and navigation text */
section[data-testid="stSidebar"] label {
    color: #E0E7FF !important;
}

/* Dividers */
hr {
    border-color: #343D5C !important;
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #B8C4DC !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #C7D2FE !important;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
}

/* Remove the white top header bar */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0rem !important;
}

/* Remove extra top spacing */
.block-container {
    padding-top: 1rem !important;
}

/* Keep the toolbar area transparent */
[data-testid="stToolbar"] {
    background: transparent !important;
}

/* Hide the white decoration at the top */
div[data-testid="stDecoration"] {
    display: none !important;
}

/* Advanced sidebar appearance */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1730 0%, #101B32 100%) !important;
    border-right: 1px solid #293A5D !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1rem 0.8rem !important;
}

/* Navigation labels */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #AFC4E8 !important;
}

/* Radio navigation items */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 7px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #15233D !important;
    border: 1px solid #283957 !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    transition: all 0.2s ease-in-out;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #E5EDFC !important;
    font-size: 12px !important;
    font-weight: 550 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #223555 !important;
    border-color: #7189C5 !important;
}

/* Selected navigation item */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(100deg, #344D78, #293D64) !important;
    border: 1px solid #E0B84E !important;
    box-shadow: 0 0 12px #D4AF371A;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 750 !important;
}

/* Hide radio circles for a cleaner menu */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Consistent styling for Sections 5, 6, 7 and 8 */

.stApp {
    background: linear-gradient(135deg, #0B0E17, #101525, #1A1F36) !important;
    color: #F1F5F9 !important;
}

h1, h2, h3, h4, p, label {
    color: #F1F5F9;
}

/* Graph background and borders */
div[data-testid="stPlotlyChart"] {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid #334155 !important;
    padding: 16px !important;
    border-radius: 12px !important;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

/* Tables */
div[data-testid="stDataFrame"] {
    background-color: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox,
.stMultiSelect {
    color: #F1F5F9 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1120, #0F172A) !important;
    border-right: 1px solid #1E293B !important;
}

/* Common theme for all dashboard sections */

.stApp {
    background: linear-gradient(135deg, #0B0E17, #101525, #1A1F36) !important;
    color: #F1F5F9 !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0B0E17, #101525, #1A1F36) !important;
}

h1, h2, h3, h4, p, label {
    color: #F1F5F9;
}

/* Cards and metrics */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid #334155 !important;
    padding: 16px !important;
    border-radius: 12px !important;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

/* Graph containers */
div[data-testid="stPlotlyChart"],
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"],
div[data-testid="stPyplot"] {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

/* Data tables */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

/* Text inputs and number inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
}

/* Dropdowns */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #1E293B !important;
    color: #F1F5F9 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1120, #0F172A) !important;
    border-right: 1px solid #1E293B !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #CBD5E1 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIGURES_DIR = os.path.join(_BASE_DIR, 'outputs', 'figures')
DATA_PATH = os.path.join(_BASE_DIR, 'data', 'olympics.csv')

def display_fig(filename, caption=None):
    filepath = os.path.join(FIGURES_DIR, filename)

    if os.path.exists(filepath):
        try:
            st.image(filepath, caption=caption, width="stretch")
        except (TypeError, ValueError):
            st.image(filepath, caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: {filename}")


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-07-27/olympics.csv'
        df = pd.read_csv(url)
    return df

@st.cache_resource
def get_trained_models(_df_data):
    df_mod = _df_data[_df_data['year'] >= 1960].copy()
    
    # Impute missing biometrics (height, weight, age) with median to prevent NaN errors
    imputer = SimpleImputer(strategy='median')
    df_mod[['height', 'weight', 'age']] = imputer.fit_transform(df_mod[['height', 'weight', 'age']])
    
    df_mod['medal_won'] = df_mod['medal'].notnull().astype(int)
    df_mod['sex_male'] = (df_mod['sex'] == 'M').astype(int)
    df_mod['season_summer'] = (df_mod['season'] == 'Summer').astype(int)
    
    reg_features = ['height', 'age', 'sex_male', 'season_summer', 'year']
    
    # Fit regression model
    reg_model = LinearRegression().fit(df_mod[reg_features], df_mod['weight'])
    
    # Fit classification model
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(df_mod[reg_features])
    clf_model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42).fit(X_sc, df_mod['medal_won'])
    
    return reg_model, clf_model, scaler

df_raw = load_data()

# App Header

# Advanced Dashboard Header
st.markdown("""
<div class="hero">
    <div class="hero-tag">OLYMPIC ANALYTICS · MACHINE LEARNING</div>
    <h1>Olympic Intelligence Dashboard</h1>
    <p>
        Explore athlete performance, discover statistical patterns,
        evaluate machine-learning models, and experiment with live
        predictions through one interactive research platform.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation

# ADVANCED SIDEBAR NAVIGATION
with st.sidebar:

    st.markdown("""
    <div style="
        text-align:center;
        padding:22px 14px 18px 14px;
        background:linear-gradient(145deg,#142443,#0B1428);
        border:1px solid #293A5D;
        border-radius:18px;
        margin-bottom:24px;
    ">
        <div style="font-size:52px;">🏅</div>
        <h2 style="
            color:#FFFFFF;
            margin:8px 0 4px 0;
            font-size:23px;
            font-weight:800;
        ">OLYMPIC ML</h2>
        <p style="
            color:#AFC4E8;
            font-size:12px;
            margin:0;
        ">Analytics & Prediction Platform</p>
        <div style="
            margin:14px auto 14px auto;
            height:3px;
            width:65px;
            border-radius:5px;
            background:linear-gradient(90deg,#E4B847,#8B9DFF);
        "></div>
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 6px 14px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid #2A3C63;
            border-radius: 30px;
            margin-top: 4px;
        ">
            <img src="https://github.com/Aditipaul17.png" 
                 style="width: 28px; height: 28px; border-radius: 50%; border: 1.5px solid #E4B847; object-fit: cover;" 
                 alt="Aditi Paul">
            <div style="text-align: left; line-height: 1.2;">
                <div style="color: #94A3B8; font-size: 9px; text-transform: uppercase;">Creator</div>
                <a href="https://github.com/Aditipaul17" target="_blank" style="color: #F8FAFC; font-size: 12px; font-weight: 700; text-decoration: none;">Aditi Paul</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        color:#D8E4FF;
        font-size:17px;
        font-weight:750;
        padding:4px 4px 12px 4px;
    ">🧭  NAVIGATION MENU</div>
    """, unsafe_allow_html=True)

    menu_options = [
        "🔴 Dashboard Overview",
        "🧹 Missing Value Imputation & Data Preprocessing",
        "📈 Simple & Multiple Linear Regression",
        "🤖 Supervised Classification: KNN, Naive Bayes & Logistic Regression",
        "🧩 Morphological Clustering: K-Means & GMM",
        "🌲 Ensemble Learning: Bagging & Random Forest",
        "🏆 Best Model",
        "🎯 Cross-Validation & Hyperparameter Tuning",
        "🔮 Interactive Athlete Predictor (Live AI)"
    ]

    # Map display labels to the original section names
    menu_mapping = {
        "🔴 Dashboard Overview": "Project Overview & Data Explorer",
        "🧹 Missing Value Imputation & Data Preprocessing":
            "Missing Value Imputation & Data Preprocessing",
        "📈 Simple & Multiple Linear Regression":
            "Simple & Multiple Linear Regression",
        "🤖 Supervised Classification: KNN, Naive Bayes & Logistic Regression":
            "Supervised Classification: KNN, Naive Bayes & Logistic Regression",
        "🧩 Morphological Clustering: K-Means & GMM":
            "Morphological Clustering: K-Means & Gaussian Mixture Models",
        "🌲 Ensemble Learning: Bagging & Random Forest":
            "Ensemble Learning: Bagging & Random Forest",
        "🏆 Best Model": "Best Model",
        "🎯 Cross-Validation & Hyperparameter Tuning":
            "Cross-Validation & Hyperparameter Tuning",
        "🔮 Interactive Athlete Predictor (Live AI)":
            "Interactive ML Predictor (Live Inference)"
    }

    # Keep the existing section routing variable named menu
    selected_menu = st.radio(
        "Choose a section",
        menu_options,
        label_visibility="collapsed",
        key="advanced_navigation"
    )

    menu = menu_mapping[selected_menu]

    st.markdown("""
    <div style="
        margin-top:24px;
        padding:14px;
        text-align:center;
        background:#131F36;
        border:1px solid #293A5D;
        border-radius:12px;
        color:#9EB2D5;
        font-size:11px;
    ">
        🥇 OLYMPIC DATA INTELLIGENCE<br>
        <span style="color:#657FA8;">Developed by</span>
        <a href="https://github.com/Aditipaul17" target="_blank" style="color:#E4B847; text-decoration:none; font-weight:700;"> Aditi Paul ↗</a>
    </div>
    """, unsafe_allow_html=True)



# SECTION 1: OVERVIEW & DATA EXPLORER
if menu == "Project Overview & Data Explorer":

    st.header("📊 Olympic Data Explorer")
    st.caption("Explore participation, medal statistics and Olympic athlete records.")

    # Dashboard metrics
    total_records = len(df_raw)
    total_sports = df_raw["sport"].nunique()
    total_medals = df_raw["medal"].notna().sum()
    total_years = df_raw["year"].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{total_records:,}")
    m2.metric("Sports", f"{total_sports:,}")
    m3.metric("Medal Records", f"{total_medals:,}")
    m4.metric("Olympic Years", f"{total_years:,}")

    st.markdown("---")

    # Charts side by side ABOVE the dataset
    
    # Advanced charts side by side
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
        gap: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    chart1, chart2 = st.columns(2, gap="large")

    chart_bg = "#171D31"
    text_color = "#E2E8F0"
    grid_color = "#343D5C"

    with chart1:
        st.subheader("🏅 Medal Distribution")
        st.caption("Breakdown of athletes by medal outcome")

        medal_counts = (
            df_raw["medal"]
            .fillna("No Medal")
            .value_counts()
            .reindex(["Gold", "Silver", "Bronze", "No Medal"])
            .dropna()
        )

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=chart_bg)
        ax.set_facecolor(chart_bg)

        medal_colors = ["#F6C453", "#B8C5D6", "#CD8965", "#64748B"]
        bars = ax.bar(
            medal_counts.index,
            medal_counts.values,
            color=medal_colors,
            width=0.62,
            edgecolor="#FFFFFF22",
            linewidth=1.2,
            zorder=3
        )

        ax.set_title(
            "Medal Outcome Overview",
            color="#FFFFFF",
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        ax.set_ylabel("Number of Records", color=text_color, fontsize=10)
        ax.tick_params(axis="x", colors=text_color, labelsize=10)
        ax.tick_params(axis="y", colors="#AAB5CF", labelsize=9)

        ax.grid(axis="y", color=grid_color, linestyle="--",
                linewidth=0.7, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,.0f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color="#FFFFFF",
                fontsize=9,
                fontweight="bold"
            )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_ylim(0, max(medal_counts.values) * 1.16)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with chart2:
        st.subheader("🏃 Top 10 Sports")
        st.caption("Sports ranked by total participation records")

        top_sports = df_raw["sport"].value_counts().head(10).sort_values()

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=chart_bg)
        ax.set_facecolor(chart_bg)

        values = top_sports.values
        max_value = max(values) if len(values) else 1

        # Color intensity increases with participation
        bar_colors = [
            plt.cm.viridis(0.35 + 0.60 * (v / max_value))
            for v in values
        ]

        bars = ax.barh(
            top_sports.index,
            values,
            color=bar_colors,
            height=0.68,
            edgecolor="#FFFFFF22",
            linewidth=0.8,
            zorder=3
        )

        ax.set_title(
            "Most Popular Sports",
            color="#FFFFFF",
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        ax.set_xlabel("Number of Records", color=text_color, fontsize=10)
        ax.tick_params(axis="x", colors="#AAB5CF", labelsize=9)
        ax.tick_params(axis="y", colors=text_color, labelsize=9)

        ax.grid(axis="x", color=grid_color, linestyle="--",
                linewidth=0.7, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.bar_label(
            bars,
            labels=[f"{v:,.0f}" for v in values],
            padding=6,
            color="#FFFFFF",
            fontsize=8,
            fontweight="bold"
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlim(0, max_value * 1.20)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


    st.markdown("---")
    st.subheader("🔎 Filter and Explore Dataset")
    st.write("Choose one or more filters. The table and summary update automatically.")

    # Dropdown filters
    filter1, filter2, filter3, filter4 = st.columns(4)

    with filter1:
        years = sorted(
            pd.to_numeric(df_raw["year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        selected_year = st.selectbox(
            "Olympic Year",
            ["All Years"] + years,
            key="explorer_year"
        )

    with filter2:
        sports = sorted(df_raw["sport"].dropna().astype(str).unique().tolist())
        selected_sport = st.selectbox(
            "Sport",
            ["All Sports"] + sports,
            key="explorer_sport"
        )

    with filter3:
        medals = ["Gold", "Silver", "Bronze", "No Medal"]
        selected_medal = st.selectbox(
            "Medal Type",
            ["All Medals"] + medals,
            key="explorer_medal"
        )

    with filter4:
        if "season" in df_raw.columns:
            seasons = sorted(df_raw["season"].dropna().astype(str).unique().tolist())
            selected_season = st.selectbox(
                "Olympic Season",
                ["All Seasons"] + seasons,
                key="explorer_season"
            )
        else:
            selected_season = "All Seasons"
            st.selectbox(
                "Olympic Season",
                ["All Seasons"],
                key="explorer_season"
            )

    # Search and sorting controls
    search_col, sort_col, order_col = st.columns([2, 1.5, 1])

    with search_col:
        search_text = st.text_input(
            "🔍 Search records",
            placeholder="Search athlete, team, sport, city...",
            key="explorer_search"
        )

    with sort_col:
        sort_options = [
            c for c in
            ["year", "name", "sport", "medal", "season", "team", "city"]
            if c in df_raw.columns
        ]
        sort_by = st.selectbox(
            "Sort by",
            ["Default"] + sort_options,
            key="explorer_sort"
        )

    with order_col:
        sort_order = st.selectbox(
            "Order",
            ["Ascending", "Descending"],
            key="explorer_order"
        )

    # Apply filters without modifying the original dataset
    filtered_df = df_raw.copy()

    if selected_year != "All Years":
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df["year"], errors="coerce")
            == selected_year
        ]

    if selected_sport != "All Sports":
        filtered_df = filtered_df[
            filtered_df["sport"].astype(str) == selected_sport
        ]

    if selected_medal != "All Medals":
        if selected_medal == "No Medal":
            filtered_df = filtered_df[filtered_df["medal"].isna()]
        else:
            filtered_df = filtered_df[
                filtered_df["medal"].astype(str) == selected_medal
            ]

    if selected_season != "All Seasons" and "season" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["season"].astype(str) == selected_season
        ]

    if search_text.strip():
        term = search_text.strip().lower()
        search_cols = [c for c in ['name', 'team', 'sport', 'city', 'event', 'noc'] if c in filtered_df.columns]
        if search_cols:
            masks = [filtered_df[col].astype(str).str.lower().str.contains(term, regex=False) for col in search_cols]
            filtered_df = filtered_df[np.logical_or.reduce(masks)]

    if sort_by != "Default":
        filtered_df = filtered_df.sort_values(
            by=sort_by,
            ascending=(sort_order == "Ascending"),
            na_position="last"
        )

    # Results summary
    st.markdown("### 📋 Dataset Results")

    result1, result2, result3 = st.columns(3)
    result1.metric("Matching Records", f"{len(filtered_df):,}")
    result2.metric("Columns", f"{filtered_df.shape[1]:,}")
    result3.metric(
        "Matching Medal Records",
        f"{filtered_df['medal'].notna().sum():,}"
    )

    # Download the filtered dataset
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    download_col, reset_col = st.columns([1, 5])

    with download_col:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="filtered_olympic_dataset.csv",
            mime="text/csv",
            width="stretch"
        )

    with reset_col:
        if st.button("🔄 Reset All Filters"):
            for key in [
                "explorer_year",
                "explorer_sport",
                "explorer_medal",
                "explorer_season",
                "explorer_search",
                "explorer_sort",
                "explorer_order"
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    # Large, full-width data table
    
    st.dataframe(
        filtered_df.head(5000) if len(filtered_df) > 5000 else filtered_df,
        width="stretch",
        height=650,
        hide_index=True,
        column_config={
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "name": st.column_config.TextColumn("Athlete"),
            "sport": st.column_config.TextColumn("Sport"),
            "medal": st.column_config.TextColumn("Medal"),
            "season": st.column_config.TextColumn("Season")
        }
    )

    if len(filtered_df) > 5000:
        st.caption(
            f"Showing top 5,000 preview records (out of {len(filtered_df):,} matching, "
            f"{len(df_raw):,} total) for optimal browser performance. "
            "Use the 'Download CSV' button above for the complete matching dataset."
        )
    else:
        st.caption(
            f"Showing {len(filtered_df):,} matching records out of "
            f"{len(df_raw):,} total records. Use the filters above to narrow "
            "the results. You can also sort columns directly in the table."
        )




# SECTION 2: IMPUTATION & PREPROCESSING
elif menu == "Missing Value Imputation & Data Preprocessing":

    st.header("🧹 Missing Value Intelligence Center")
    st.caption(
        "Explore data quality, identify missing values, compare imputation "
        "strategies, and review preprocessing."
    )

    chart_bg = "#171D31"
    text_color = "#E8EDF7"
    grid_color = "#343D5C"
    accent_colors = ["#D4AF37", "#818CF8", "#38BDF8", "#F97316"]

    # ---------------- FILTERS ----------------
    st.markdown("### 🔎 Data Quality Filters")

    filter1, filter2, filter3 = st.columns(3)
    df_view = df_raw.copy()

    with filter1:
        if "year" in df_view.columns:
            year_values = pd.to_numeric(
                df_view["year"], errors="coerce"
            ).dropna()

            if not year_values.empty:
                min_year = int(year_values.min())
                max_year = int(year_values.max())

                year_range = st.slider(
                    "Olympic year range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    key="missing_year_range"
                )

                numeric_year = pd.to_numeric(
                    df_view["year"], errors="coerce"
                )
                df_view = df_view[
                    numeric_year.between(year_range[0], year_range[1])
                ]

    with filter2:
        if "sport" in df_view.columns:
            sport_options = ["All Sports"] + sorted(
                df_view["sport"].dropna().astype(str).unique().tolist()
            )
            selected_sport = st.selectbox(
                "Sport",
                sport_options,
                key="missing_sport_filter"
            )

            if selected_sport != "All Sports":
                df_view = df_view[
                    df_view["sport"].astype(str) == selected_sport
                ]

    with filter3:
        if "season" in df_view.columns:
            season_options = ["All Seasons"] + sorted(
                df_view["season"].dropna().astype(str).unique().tolist()
            )
            selected_season = st.selectbox(
                "Olympic season",
                season_options,
                key="missing_season_filter"
            )

            if selected_season != "All Seasons":
                df_view = df_view[
                    df_view["season"].astype(str) == selected_season
                ]

    if df_view.empty:
        st.warning("No records match these filters. Please change your selection.")
    else:
        # ---------------- DATA QUALITY METRICS ----------------
        missing_total = int(df_view.isna().sum().sum())
        total_cells = int(df_view.shape[0] * df_view.shape[1])
        completeness = (
            100 * (1 - missing_total / total_cells)
            if total_cells else 100
        )
        affected_columns = int((df_view.isna().sum() > 0).sum())

        st.markdown("### 📊 Data Quality Overview")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records Analysed", f"{len(df_view):,}")
        k2.metric("Missing Cells", f"{missing_total:,}")
        k3.metric("Data Completeness", f"{completeness:.2f}%")
        k4.metric("Affected Features", f"{affected_columns:,}")

        st.markdown("---")

        # ---------------- MISSING VALUE ANALYSIS ----------------
        st.markdown("### 📉 Missing Value Analysis")

        missing_summary = pd.DataFrame({
            "Feature": df_view.columns,
            "Missing Count": df_view.isna().sum().values,
            "Missing Percentage": (
                df_view.isna().mean().values * 100
            )
        })

        missing_summary = (
            missing_summary[missing_summary["Missing Count"] > 0]
            .sort_values("Missing Percentage", ascending=True)
        )

        available_features = df_view.columns.tolist()
        selected_features = st.multiselect(
            "Select features to analyse",
            options=available_features,
            default=(
                missing_summary["Feature"].tolist()
                if not missing_summary.empty
                else available_features[:min(8, len(available_features))]
            ),
            key="missing_feature_filter"
        )

        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            st.markdown("#### Missingness by Feature")

            plot_missing = missing_summary[
                missing_summary["Feature"].isin(selected_features)
            ].copy()

            if plot_missing.empty:
                st.success("No missing values in the selected features.")
            else:
                fig, ax = plt.subplots(
                    figsize=(8, max(4, len(plot_missing) * 0.38)),
                    facecolor=chart_bg
                )
                ax.set_facecolor(chart_bg)

                bars = ax.barh(
                    plot_missing["Feature"],
                    plot_missing["Missing Percentage"],
                    color=[
                        plt.cm.magma(
                            0.35 + 0.6 * value / 100
                        )
                        for value in plot_missing["Missing Percentage"]
                    ],
                    height=0.68
                )

                ax.bar_label(
                    bars,
                    labels=[
                        f"{v:.1f}%"
                        for v in plot_missing["Missing Percentage"]
                    ],
                    padding=5,
                    color=text_color,
                    fontsize=9
                )

                ax.set_xlabel("Missing Values (%)", color=text_color)
                ax.tick_params(colors=text_color, labelsize=9)
                ax.grid(
                    axis="x",
                    color=grid_color,
                    linestyle="--",
                    alpha=0.7
                )
                ax.set_axisbelow(True)
                ax.set_xlim(
                    0,
                    min(
                        110,
                        max(
                            10,
                            plot_missing["Missing Percentage"].max() * 1.2
                        )
                    )
                )

                for spine in ax.spines.values():
                    spine.set_visible(False)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

        with chart2:
            st.markdown("#### Data Completeness by Feature")

            if selected_features:
                completeness_data = pd.DataFrame({
                    "Feature": selected_features,
                    "Complete": [
                        100 * df_view[col].notna().mean()
                        for col in selected_features
                    ]
                }).sort_values("Complete", ascending=True)

                fig, ax = plt.subplots(
                    figsize=(8, max(4, len(completeness_data) * 0.38)),
                    facecolor=chart_bg
                )
                ax.set_facecolor(chart_bg)

                bars = ax.barh(
                    completeness_data["Feature"],
                    completeness_data["Complete"],
                    color="#38BDF8",
                    height=0.68
                )

                ax.bar_label(
                    bars,
                    labels=[
                        f"{v:.1f}%"
                        for v in completeness_data["Complete"]
                    ],
                    padding=5,
                    color=text_color,
                    fontsize=9
                )

                ax.set_xlabel("Completeness (%)", color=text_color)
                ax.tick_params(colors=text_color, labelsize=9)
                ax.set_xlim(0, 115)
                ax.grid(
                    axis="x",
                    color=grid_color,
                    linestyle="--",
                    alpha=0.7
                )
                ax.set_axisbelow(True)

                for spine in ax.spines.values():
                    spine.set_visible(False)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            else:
                st.info("Select at least one feature to display completeness.")

        # ---------------- MISSINGNESS TABLE ----------------
        st.markdown("---")
        st.markdown("### 📋 Detailed Missing Value Report")

        if selected_features:
            report = pd.DataFrame({
                "Feature": selected_features,
                "Data Type": [
                    str(df_view[col].dtype) for col in selected_features
                ],
                "Total Records": [
                    len(df_view) for col in selected_features
                ],
                "Missing Count": [
                    int(df_view[col].isna().sum())
                    for col in selected_features
                ],
                "Missing (%)": [
                    round(df_view[col].isna().mean() * 100, 2)
                    for col in selected_features
                ],
                "Available Values": [
                    int(df_view[col].notna().sum())
                    for col in selected_features
                ]
            }).sort_values("Missing (%)", ascending=False)

            st.dataframe(
                report,
                width="stretch",
                height=300,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Missing Value Report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="olympic_missing_value_report.csv",
                mime="text/csv"
            )

        # ---------------- IMPUTATION COMPARISON ----------------
        st.markdown("---")
        st.markdown("### 🧪 Imputation Strategy Lab")
        st.write(
            "Compare how mean, median, and mode fill missing values "
            "in a selected numeric feature. This preview does not modify "
            "your original dataset or retrain your models."
        )

        numeric_features = df_view.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_features:
            selected_numeric = st.selectbox(
                "Choose a numeric feature",
                numeric_features,
                key="imputation_numeric_feature"
            )

            original = pd.to_numeric(
                df_view[selected_numeric], errors="coerce"
            )
            observed = original.dropna()

            if original.isna().sum() == 0:
                st.success(
                    f"{selected_numeric} has no missing values in this filtered dataset."
                )
            elif observed.empty:
                st.warning(
                    "This feature contains no observed numeric values to use for imputation."
                )
            else:
                mean_value = observed.mean()
                median_value = observed.median()
                mode_values = observed.mode()
                mode_value = (
                    mode_values.iloc[0]
                    if not mode_values.empty
                    else median_value
                )

                missing_count = int(original.isna().sum())

                a, b, c = st.columns(3)
                a.metric("Mean Fill Value", f"{mean_value:.3f}")
                b.metric("Median Fill Value", f"{median_value:.3f}")
                c.metric("Mode Fill Value", f"{mode_value:.3f}")

                strategy = st.radio(
                    "Choose an imputation strategy",
                    ["Mean", "Median", "Mode"],
                    horizontal=True,
                    key="imputation_strategy"
                )

                fill_values = {
                    "Mean": mean_value,
                    "Median": median_value,
                    "Mode": mode_value
                }

                imputed = original.fillna(fill_values[strategy])

                left, right = st.columns(2, gap="large")

                with left:
                    st.markdown("#### Before Imputation")
                    st.write(
                        f"Observed: {len(observed):,} | "
                        f"Missing: {missing_count:,}"
                    )

                    fig, ax = plt.subplots(
                        figsize=(7, 4),
                        facecolor=chart_bg
                    )
                    ax.set_facecolor(chart_bg)
                    ax.hist(
                        observed,
                        bins=30,
                        color="#818CF8",
                        edgecolor=chart_bg,
                        alpha=0.9
                    )
                    ax.set_xlabel(selected_numeric, color=text_color)
                    ax.set_ylabel("Frequency", color=text_color)
                    ax.tick_params(colors=text_color)
                    ax.grid(axis="y", color=grid_color, alpha=0.5)

                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                with right:
                    st.markdown(f"#### After {strategy} Imputation")
                    st.write(
                        f"Records after filling: {len(imputed):,} | "
                        f"Remaining missing: {int(imputed.isna().sum()):,}"
                    )

                    fig, ax = plt.subplots(
                        figsize=(7, 4),
                        facecolor=chart_bg
                    )
                    ax.set_facecolor(chart_bg)
                    ax.hist(
                        imputed,
                        bins=30,
                        color="#D4AF37",
                        edgecolor=chart_bg,
                        alpha=0.9
                    )
                    ax.axvline(
                        fill_values[strategy],
                        color="#F97316",
                        linestyle="--",
                        linewidth=2,
                        label=f"Fill value: {fill_values[strategy]:.2f}"
                    )
                    ax.set_xlabel(selected_numeric, color=text_color)
                    ax.set_ylabel("Frequency", color=text_color)
                    ax.tick_params(colors=text_color)
                    ax.grid(axis="y", color=grid_color, alpha=0.5)
                    ax.legend(facecolor=chart_bg, labelcolor=text_color)

                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                with st.expander("View imputation summary"):
                    st.write({
                        "Feature": selected_numeric,
                        "Strategy": strategy,
                        "Fill value": float(fill_values[strategy]),
                        "Missing values before": missing_count,
                        "Missing values after": int(imputed.isna().sum()),
                        "Original observed mean": float(observed.mean()),
                        "Mean after imputation": float(imputed.mean())
                    })

        else:
            st.info("No numeric features are available for imputation.")

        # ---------------- EXISTING SCALING SECTION ----------------
        st.markdown("---")
        st.markdown("### ⚙️ Data Preprocessing & Feature Scaling")

        st.markdown("""
        <div style="
            background:linear-gradient(135deg,#17233B,#202944);
            border:1px solid #343D5C;
            border-radius:14px;
            padding:18px;
            color:#E8EDF7;
        ">
            <b style="color:#D4AF37;">Anti-Leakage Protocol</b><br>
            Scalers should be fitted only on the training data, then applied
            to validation or test data. This prevents information from the
            evaluation set leaking into model training.
            <br><br>
            <b>StandardScaler:</b> centers values around zero and scales
            using standard deviation.<br>
            <b>MinMaxScaler:</b> scales values to a defined range, commonly
            0 to 1.
        </div>
        """, unsafe_allow_html=True)

        display_fig(
            "p2_scaling_comparison_boxplots.png",
            "Raw vs. StandardScaler vs. MinMaxScaler Distributions"
        )


# SECTION 3: REGRESSION
elif menu == "Simple & Multiple Linear Regression":

    st.header("📈 Regression Analytics Studio")
    st.caption("Interactive height-weight analysis and regression model evaluation.")

    # Theme
    bg = "#171D31"
    gold = "#D4AF37"
    blue = "#818CF8"
    text_color = "#E8EDF7"
    grid_color = "#343D5C"

    # Prepare data
    required = ["height", "weight", "year"]
    if not all(col in df_raw.columns for col in required):
        st.error("The dataset must contain height, weight, and year columns.")
    else:
        data = df_raw.copy()

        for col in ["height", "weight", "age", "year"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        st.markdown("### 🔎 Analysis Filters")

        f1, f2, f3 = st.columns(3)

        with f1:
            year_data = data["year"].dropna()
            if not year_data.empty:
                min_y = int(year_data.min())
                max_y = int(year_data.max())
                year_range = st.slider(
                    "Olympic year range",
                    min_value=min_y,
                    max_value=max_y,
                    value=(min_y, max_y),
                    key="reg_year_range"
                )
                data = data[data["year"].between(*year_range)]

        with f2:
            if "sport" in data.columns:
                sport_options = ["All Sports"] + sorted(
                    data["sport"].dropna().astype(str).unique().tolist()
                )
                sport_filter = st.selectbox(
                    "Sport",
                    sport_options,
                    key="reg_sport_filter"
                )
                if sport_filter != "All Sports":
                    data = data[data["sport"].astype(str) == sport_filter]

        with f3:
            if "sex" in data.columns:
                sex_options = ["All"] + sorted(
                    data["sex"].dropna().astype(str).unique().tolist()
                )
                sex_filter = st.selectbox(
                    "Athlete sex",
                    sex_options,
                    key="reg_sex_filter"
                )
                if sex_filter != "All":
                    data = data[data["sex"].astype(str) == sex_filter]

        if "season" in data.columns:
            seasons = ["All Seasons"] + sorted(
                data["season"].dropna().astype(str).unique().tolist()
            )
            season_filter = st.selectbox(
                "Olympic season",
                seasons,
                key="reg_season_filter"
            )
            if season_filter != "All Seasons":
                data = data[data["season"].astype(str) == season_filter]

        # Clean data for regression
        simple_data = data.dropna(subset=["height", "weight"]).copy()

        if len(simple_data) < 10:
            st.warning(
                "Not enough complete height and weight records for regression. "
                "Adjust the filters."
            )
        else:
            # Simple regression: Height -> Weight
            X_simple = simple_data[["height"]]
            y_simple = simple_data["weight"]

            simple_model = LinearRegression()
            simple_model.fit(X_simple, y_simple)

            simple_pred = simple_model.predict(X_simple)
            simple_r2 = simple_model.score(X_simple, y_simple)
            simple_rmse = float(np.sqrt(np.mean((y_simple - simple_pred) ** 2)))
            simple_mae = float(np.mean(np.abs(y_simple - simple_pred)))

            st.markdown("### 📊 Simple Regression Performance")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Records Analysed", f"{len(simple_data):,}")
            m2.metric("R² Score", f"{simple_r2:.3f}")
            m3.metric("RMSE (kg)", f"{simple_rmse:.2f}")
            m4.metric("MAE (kg)", f"{simple_mae:.2f}")

            chart1, chart2 = st.columns(2, gap="large")

            with chart1:
                st.markdown("#### Height vs Weight")

                fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg)
                ax.set_facecolor(bg)

                # Sample for readable rendering on large datasets
                plot_data = simple_data
                if len(plot_data) > 5000:
                    plot_data = plot_data.sample(5000, random_state=42)

                ax.scatter(
                    plot_data["height"],
                    plot_data["weight"],
                    color=blue,
                    alpha=0.25,
                    s=13,
                    edgecolors="none",
                    label="Athlete records"
                )

                x_line = np.linspace(
                    simple_data["height"].min(),
                    simple_data["height"].max(),
                    150
                )
                y_line = simple_model.predict(
                    pd.DataFrame({"height": x_line})
                )

                ax.plot(
                    x_line,
                    y_line,
                    color=gold,
                    linewidth=3,
                    label="Regression line"
                )

                ax.set_xlabel("Height (cm)", color=text_color)
                ax.set_ylabel("Weight (kg)", color=text_color)
                ax.tick_params(colors=text_color)
                ax.grid(color=grid_color, linestyle="--", alpha=0.5)
                ax.legend(
                    facecolor=bg,
                    edgecolor=grid_color,
                    labelcolor=text_color
                )

                for spine in ax.spines.values():
                    spine.set_color(grid_color)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with chart2:
                st.markdown("#### Residual Distribution")

                residuals = y_simple - simple_pred

                fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg)
                ax.set_facecolor(bg)

                ax.hist(
                    residuals,
                    bins=35,
                    color=blue,
                    alpha=0.9,
                    edgecolor=bg
                )
                ax.axvline(
                    0,
                    color=gold,
                    linewidth=2,
                    linestyle="--",
                    label="Zero error"
                )

                ax.set_xlabel("Residual (actual − predicted), kg", color=text_color)
                ax.set_ylabel("Frequency", color=text_color)
                ax.tick_params(colors=text_color)
                ax.grid(axis="y", color=grid_color, linestyle="--", alpha=0.5)
                ax.legend(
                    facecolor=bg,
                    edgecolor=grid_color,
                    labelcolor=text_color
                )

                for spine in ax.spines.values():
                    spine.set_color(grid_color)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with st.expander("View simple regression equation"):
                st.code(
                    f"Predicted weight = {simple_model.intercept_:.3f} "
                    f"+ ({simple_model.coef_[0]:.3f} × height)"
                )

            st.download_button(
                "⬇️ Download Simple Regression Predictions",
                data=pd.DataFrame({
                    "Actual Weight": y_simple,
                    "Predicted Weight": simple_pred,
                    "Residual": residuals
                }).to_csv(index=False).encode("utf-8"),
                file_name="simple_regression_predictions.csv",
                mime="text/csv"
            )

            # Multiple regression
            st.markdown("---")
            st.markdown("### 🧠 Multiple Regression Lab")
            st.caption("Select predictors and compare model performance.")

            candidate_features = [
                col for col in
                ["height", "age", "year", "sex", "season", "sport"]
                if col in data.columns
            ]

            default_features = [
                col for col in ["height", "age", "year", "sex", "season", "sport"]
                if col in candidate_features
            ]

            selected_features = st.multiselect(
                "Select predictor features",
                options=candidate_features,
                default=default_features,
                key="reg_multiple_features"
            )

            if not selected_features:
                st.info("Select at least one predictor feature.")
            else:
                model_data = data[
                    selected_features + ["weight"]
                ].dropna().copy()

                if len(model_data) < 10:
                    st.warning(
                        "Not enough complete records for multiple regression. "
                        "Select different features or broaden the filters."
                    )
                else:
                    X_multi = pd.get_dummies(
                        model_data[selected_features],
                        drop_first=True,
                        dtype=float
                    )
                    y_multi = model_data["weight"].astype(float)

                    if X_multi.shape[1] == 0:
                        st.warning("The selected features cannot be used for regression.")
                    else:
                        multi_model = LinearRegression()
                        multi_model.fit(X_multi, y_multi)
                        multi_pred = multi_model.predict(X_multi)

                        multi_r2 = multi_model.score(X_multi, y_multi)
                        multi_rmse = float(
                            np.sqrt(np.mean((y_multi - multi_pred) ** 2))
                        )
                        multi_mae = float(
                            np.mean(np.abs(y_multi - multi_pred))
                        )

                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("Complete Records", f"{len(model_data):,}")
                        p2.metric("Predictor Columns", f"{X_multi.shape[1]:,}")
                        p3.metric("R² Score", f"{multi_r2:.3f}")
                        p4.metric("RMSE (kg)", f"{multi_rmse:.2f}")

                        left, right = st.columns(2, gap="large")

                        with left:
                            st.markdown("#### Actual vs Predicted Weight")

                            fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg)
                            ax.set_facecolor(bg)

                            plot_idx = np.random.RandomState(42).choice(len(y_multi), size=min(5000, len(y_multi)), replace=False)
                            ax.scatter(
                                y_multi.iloc[plot_idx],
                                multi_pred[plot_idx],
                                color=blue,
                                alpha=0.3,
                                s=14,
                                edgecolors="none"
                            )

                            low = min(y_multi.min(), multi_pred.min())
                            high = max(y_multi.max(), multi_pred.max())

                            ax.plot(
                                [low, high],
                                [low, high],
                                color=gold,
                                linestyle="--",
                                linewidth=2,
                                label="Ideal prediction"
                            )

                            ax.set_xlabel("Actual Weight (kg)", color=text_color)
                            ax.set_ylabel("Predicted Weight (kg)", color=text_color)
                            ax.tick_params(colors=text_color)
                            ax.grid(color=grid_color, linestyle="--", alpha=0.5)
                            ax.legend(
                                facecolor=bg,
                                edgecolor=grid_color,
                                labelcolor=text_color
                            )

                            for spine in ax.spines.values():
                                spine.set_color(grid_color)

                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                        with right:
                            st.markdown("#### Predictor Coefficients")

                            coef_data = pd.DataFrame({
                                "Feature": X_multi.columns,
                                "Coefficient": multi_model.coef_
                            }).sort_values("Coefficient")

                            coef_plot = coef_data.tail(15)

                            fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg)
                            ax.set_facecolor(bg)

                            colors = [
                                gold if v >= 0 else blue
                                for v in coef_plot["Coefficient"]
                            ]

                            ax.barh(
                                coef_plot["Feature"],
                                coef_plot["Coefficient"],
                                color=colors
                            )
                            ax.axvline(0, color=text_color, linewidth=0.8)
                            ax.set_xlabel("Coefficient", color=text_color)
                            ax.tick_params(colors=text_color, labelsize=8)
                            ax.grid(axis="x", color=grid_color, linestyle="--", alpha=0.5)

                            for spine in ax.spines.values():
                                spine.set_color(grid_color)

                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                        st.download_button(
                            "⬇️ Download Multiple Regression Predictions",
                            data=pd.DataFrame({
                                "Actual Weight": y_multi,
                                "Predicted Weight": multi_pred,
                                "Residual": y_multi - multi_pred
                            }).to_csv(index=False).encode("utf-8"),
                            file_name="multiple_regression_predictions.csv",
                            mime="text/csv"
                        )

                        with st.expander("View model coefficients"):
                            st.dataframe(
                                coef_data,
                                width="stretch",
                                hide_index=True
                            )



# SECTION 4: CLASSIFICATION
elif menu == "Supervised Classification: KNN, Naive Bayes & Logistic Regression":

    st.header("🤖 Olympic Classification Studio")
    st.caption("Filter athlete records, train classifiers, and compare medal prediction performance.")

    bg = "#171D31"
    gold = "#D4AF37"
    blue = "#818CF8"
    text_color = "#E8EDF7"
    grid_color = "#343D5C"

    data = df_raw.copy()

    if "medal" not in data.columns:
        st.error("The dataset does not contain a medal column.")
    else:
        # ---------------- FILTERS ----------------
        st.markdown("### 🔎 Dataset Filters")
        f1, f2, f3 = st.columns(3)

        with f1:
            if "year" in data.columns:
                year_values = pd.to_numeric(
                    data["year"], errors="coerce"
                ).dropna()

                if not year_values.empty:
                    year_range = st.slider(
                        "Olympic year range",
                        int(year_values.min()),
                        int(year_values.max()),
                        (int(year_values.min()), int(year_values.max())),
                        key="cls_year_range"
                    )
                    data["year"] = pd.to_numeric(
                        data["year"], errors="coerce"
                    )
                    data = data[data["year"].between(*year_range)]

        with f2:
            if "sport" in data.columns:
                sport_options = ["All Sports"] + sorted(
                    data["sport"].dropna().astype(str).unique().tolist()
                )
                sport_filter = st.selectbox(
                    "Sport",
                    sport_options,
                    key="cls_sport_filter"
                )
                if sport_filter != "All Sports":
                    data = data[data["sport"].astype(str) == sport_filter]

        with f3:
            if "season" in data.columns:
                season_options = ["All Seasons"] + sorted(
                    data["season"].dropna().astype(str).unique().tolist()
                )
                season_filter = st.selectbox(
                    "Season",
                    season_options,
                    key="cls_season_filter"
                )
                if season_filter != "All Seasons":
                    data = data[data["season"].astype(str) == season_filter]

        # ---------------- TARGET AND FEATURES ----------------
        data["medal_won"] = data["medal"].notna().astype(int)

        available_features = [
            c for c in ["age", "height", "weight", "sex", "season", "year", "sport"]
            if c in data.columns
        ]

        default_features = [
            c for c in ["age", "height", "weight", "sex", "season", "year", "sport"]
            if c in available_features
        ]

        selected_features = st.multiselect(
            "Select prediction features",
            options=available_features,
            default=default_features,
            key="cls_features"
        )

        model_name = st.selectbox(
            "Choose classification model",
            [
                "Compare All Models",
                "K-Nearest Neighbors (KNN)",
                "Gaussian Naive Bayes",
                "Logistic Regression"
            ],
            key="cls_model_choice"
        )

        test_size = st.slider(
            "Test data percentage",
            min_value=10,
            max_value=40,
            value=25,
            step=5,
            key="cls_test_size"
        )

        threshold = st.slider(
            "Medal prediction probability threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key="cls_threshold"
        )

        if not selected_features:
            st.warning("Select at least one prediction feature.")
        elif len(data) < 30 or data["medal_won"].nunique() < 2:
            st.warning(
                "The selected filters do not provide enough records or both target classes. "
                "Broaden the filters and try again."
            )
        else:
            # Convert numeric columns and identify feature types
            X = data[selected_features].copy()
            y = data["medal_won"].copy()

            for col in X.columns:
                if col in ["age", "height", "weight", "year"]:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

            # Remove rows with missing target (target is already constructed from medal)
            valid_rows = y.notna()
            X = X.loc[valid_rows]
            y = y.loc[valid_rows].astype(int)

            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = [
                c for c in X.columns if c not in numeric_features
            ]

            if len(y) < 30 or y.value_counts().min() < 2:
                st.warning(
                    "Not enough examples of both medal outcomes for a reliable split. "
                    "Change the filters."
                )
            else:
                numeric_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                categorical_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ])

                transformers = []
                if numeric_features:
                    transformers.append(
                        ("numeric", numeric_transformer, numeric_features)
                    )
                if categorical_features:
                    transformers.append(
                        ("categorical", categorical_transformer, categorical_features)
                    )

                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder="drop"
                )

                if len(X) > 25000:
                    X_sample, _, y_sample, _ = train_test_split(
                        X, y, train_size=25000, random_state=42, stratify=y
                    )
                else:
                    X_sample, y_sample = X, y

                X_train, X_test, y_train, y_test = train_test_split(
                    X_sample,
                    y_sample,
                    test_size=test_size / 100,
                    random_state=42,
                    stratify=y_sample
                )

                models = {
                    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
                        n_neighbors=min(5, len(X_train))
                    ),
                    "Gaussian Naive Bayes": GaussianNB(),
                    "Logistic Regression": LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )
                }

                requested_models = (
                    list(models.keys())
                    if model_name == "Compare All Models"
                    else [model_name]
                )

                results = {}
                predictions = {}

                with st.spinner("Training and evaluating classifiers..."):
                    for name in requested_models:
                        pipeline = Pipeline([
                            ("preprocessing", preprocessor),
                            ("classifier", models[name])
                        ])

                        pipeline.fit(X_train, y_train)

                        probabilities = pipeline.predict_proba(X_test)[:, 1]
                        predicted = (probabilities >= threshold).astype(int)

                        results[name] = {
                            "Accuracy": accuracy_score(y_test, predicted),
                            "Precision": precision_score(
                                y_test, predicted, zero_division=0
                            ),
                            "Recall": recall_score(
                                y_test, predicted, zero_division=0
                            ),
                            "F1 Score": f1_score(
                                y_test, predicted, zero_division=0
                            ),
                            "ROC-AUC": roc_auc_score(y_test, probabilities)
                        }

                        predictions[name] = {
                            "model": pipeline,
                            "predicted": predicted,
                            "probabilities": probabilities
                        }

                # ---------------- METRICS ----------------
                st.markdown("---")
                st.markdown("### 📊 Model Performance")

                chosen_name = (
                    model_name
                    if model_name != "Compare All Models"
                    else max(results, key=lambda n: results[n]["F1 Score"])
                )

                best = results[chosen_name]

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Test Records", f"{len(y_test):,}")
                k2.metric("Accuracy", f"{best['Accuracy']:.3f}")
                k3.metric("Precision", f"{best['Precision']:.3f}")
                k4.metric("Recall", f"{best['Recall']:.3f}")
                k5.metric("F1 Score", f"{best['F1 Score']:.3f}")

                st.caption(
                    f"Selected model: {chosen_name} · "
                    f"ROC-AUC: {best['ROC-AUC']:.3f} · "
                    f"Threshold: {threshold:.2f}"
                )
                
                # ---------------- INDIVIDUAL MODEL GRAPHS ----------------
                st.markdown("---")
                st.markdown("### 🧠 Individual Classifier Analytics")

                model_tabs = st.tabs([
                    "📍 KNN",
                    "🔔 Naive Bayes",
                    "📈 Logistic Regression"
                ])

                model_tab_names = [
                    "K-Nearest Neighbors (KNN)",
                    "Gaussian Naive Bayes",
                    "Logistic Regression"
                ]

                for tab, classifier_name in zip(model_tabs, model_tab_names):
                    with tab:
                        if classifier_name not in predictions:
                            st.info(
                                "Select 'Compare All Models' above to display "
                                "graphs for all three classifiers."
                            )
                            continue

                        model_result = predictions[classifier_name]
                        actual = y_test.to_numpy()
                        predicted = model_result["predicted"]
                        probabilities = model_result["probabilities"]

                        graph1, graph2 = st.columns(2, gap="large")

                        # Confusion matrix
                        with graph1:
                            st.markdown("#### Confusion Matrix")

                            cm_individual = confusion_matrix(
                                actual, predicted, labels=[0, 1]
                            )

                            fig, ax = plt.subplots(
                                figsize=(6, 4),
                                facecolor=bg
                            )
                            ax.set_facecolor(bg)

                            sns.heatmap(
                                cm_individual,
                                annot=True,
                                fmt=",",
                                cmap="mako",
                                linewidths=1,
                                linecolor=bg,
                                xticklabels=["No Medal", "Medalist"],
                                yticklabels=["No Medal", "Medalist"],
                                annot_kws={"size": 12},
                                ax=ax
                            )

                            ax.set_xlabel("Predicted", color=text_color)
                            ax.set_ylabel("Actual", color=text_color)
                            ax.tick_params(colors=text_color)

                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                        # Probability distribution
                        with graph2:
                            st.markdown("#### Predicted Probability Distribution")

                            fig, ax = plt.subplots(
                                figsize=(6, 4),
                                facecolor=bg
                            )
                            ax.set_facecolor(bg)

                            ax.hist(
                                probabilities[actual == 0],
                                bins=20,
                                alpha=0.75,
                                color=blue,
                                label="No Medal",
                                edgecolor=bg
                            )

                            ax.hist(
                                probabilities[actual == 1],
                                bins=20,
                                alpha=0.75,
                                color=gold,
                                label="Medalist",
                                edgecolor=bg
                            )

                            ax.axvline(
                                threshold,
                                color="#F97316",
                                linestyle="--",
                                linewidth=2,
                                label=f"Threshold = {threshold:.2f}"
                            )

                            ax.set_xlabel("Predicted Medal Probability", color=text_color)
                            ax.set_ylabel("Number of Athletes", color=text_color)
                            ax.tick_params(colors=text_color)
                            ax.grid(
                                axis="y",
                                color=grid_color,
                                linestyle="--",
                                alpha=0.5
                            )
                            ax.legend(
                                facecolor=bg,
                                edgecolor=grid_color,
                                labelcolor=text_color,
                                fontsize=8
                            )

                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                        # Individual model metrics
                        metric_values = results[classifier_name]

                        st.markdown("#### Performance Metrics")

                        q1, q2, q3, q4, q5 = st.columns(5)
                        q1.metric("Accuracy", f"{metric_values['Accuracy']:.3f}")
                        q2.metric("Precision", f"{metric_values['Precision']:.3f}")
                        q3.metric("Recall", f"{metric_values['Recall']:.3f}")
                        q4.metric("F1 Score", f"{metric_values['F1 Score']:.3f}")
                        q5.metric("ROC-AUC", f"{metric_values['ROC-AUC']:.3f}")


                # ---------------- VISUALS ----------------
                left, right = st.columns(2, gap="large")

                with left:
                    st.markdown("#### Confusion Matrix")

                    pred = predictions[chosen_name]["predicted"]
                    cm = confusion_matrix(y_test, pred, labels=[0, 1])

                    fig, ax = plt.subplots(
                        figsize=(7, 5),
                        facecolor=bg
                    )
                    ax.set_facecolor(bg)

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt=",",
                        cmap="mako",
                        linewidths=1,
                        linecolor=bg,
                        xticklabels=["No Medal", "Medalist"],
                        yticklabels=["No Medal", "Medalist"],
                        annot_kws={"size": 13},
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Class", color=text_color)
                    ax.set_ylabel("Actual Class", color=text_color)
                    ax.tick_params(colors=text_color)

                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                with right:
                    st.markdown("#### Model Comparison")

                    comparison = pd.DataFrame(results).T
                    comparison = comparison[
                        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
                    ]

                    fig, ax = plt.subplots(
                        figsize=(8, 5),
                        facecolor=bg
                    )
                    ax.set_facecolor(bg)

                    comparison.plot(
                        kind="bar",
                        ax=ax,
                        color=[gold, blue, "#38BDF8", "#34D399", "#F97316"],
                        width=0.78
                    )

                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Score", color=text_color)
                    ax.set_xlabel("")
                    ax.tick_params(colors=text_color, axis="x", rotation=15)
                    ax.tick_params(colors=text_color, axis="y")
                    ax.grid(axis="y", color=grid_color, linestyle="--", alpha=0.5)
                    ax.set_axisbelow(True)
                    ax.legend(
                        facecolor=bg,
                        edgecolor=grid_color,
                        labelcolor=text_color,
                        fontsize=8
                    )

                    for spine in ax.spines.values():
                        spine.set_color(grid_color)

                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                # ---------------- ROC CURVE ----------------
                st.markdown("#### ROC Curve")

                fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg)
                ax.set_facecolor(bg)

                for name, result in predictions.items():
                    fpr, tpr, _ = roc_curve(
                        y_test,
                        result["probabilities"]
                    )
                    auc_value = results[name]["ROC-AUC"]
                    ax.plot(
                        fpr,
                        tpr,
                        linewidth=2.5,
                        label=f"{name} (AUC={auc_value:.3f})"
                    )

                ax.plot(
                    [0, 1], [0, 1],
                    linestyle="--",
                    color="#94A3B8",
                    label="Random baseline"
                )

                ax.set_xlabel("False Positive Rate", color=text_color)
                ax.set_ylabel("True Positive Rate", color=text_color)
                ax.tick_params(colors=text_color)
                ax.grid(color=grid_color, linestyle="--", alpha=0.5)
                ax.legend(
                    facecolor=bg,
                    edgecolor=grid_color,
                    labelcolor=text_color
                )

                for spine in ax.spines.values():
                    spine.set_color(grid_color)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                # ---------------- DOWNLOADS ----------------
                st.markdown("---")
                st.markdown("### 📥 Export Results")

                st.dataframe(
                    comparison.style.format("{:.3f}"),
                    width="stretch"
                )

                st.download_button(
                    "⬇️ Download Model Comparison",
                    data=comparison.to_csv().encode("utf-8"),
                    file_name="olympic_classifier_comparison.csv",
                    mime="text/csv"
                )

                selected_prediction = predictions[chosen_name]

                prediction_report = pd.DataFrame({
                    "Actual Medal Outcome": y_test.to_numpy(),
                    "Predicted Medal Outcome": selected_prediction["predicted"],
                    "Medal Probability": selected_prediction["probabilities"]
                })

                st.download_button(
                    "⬇️ Download Test Predictions",
                    data=prediction_report.to_csv(index=False).encode("utf-8"),
                    file_name="olympic_classification_predictions.csv",
                    mime="text/csv"
                )



# SECTION 5: ADVANCED CLUSTERING
elif menu == "Morphological Clustering: K-Means & Gaussian Mixture Models":
    st.header("🧬 Athlete Clustering & Morphological Discovery")
    st.caption("Explore athlete profiles and discover groups with similar physical characteristics.")

    data = df_raw.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")

    f1, f2, f3 = st.columns(3)

    with f1:
        year_min = int(data["year"].dropna().min())
        year_max = int(data["year"].dropna().max())
        year_range = st.slider(
            "Olympic year range",
            year_min,
            year_max,
            (max(year_min, 1960), year_max),
            key="cluster_year"
        )

    with f2:
        sport_options = ["All"] + sorted(data["sport"].dropna().astype(str).unique().tolist())
        selected_sport = st.selectbox("Sport", sport_options, key="cluster_sport")

    with f3:
        season_options = ["All"] + sorted(data["season"].dropna().astype(str).unique().tolist())
        selected_season = st.selectbox("Season", season_options, key="cluster_season")

    numeric_features = ["age", "height", "weight", "year"]
    available_features = [c for c in numeric_features if c in data.columns]

    selected_features = st.multiselect(
        "Features used for clustering",
        available_features,
        default=[c for c in ["height", "weight", "age"] if c in available_features],
        key="cluster_features"
    )

    k1, k2 = st.columns(2)

    with k1:
        cluster_count = st.slider("Number of clusters", 2, 8, 3, key="cluster_k")

    with k2:
        max_points = st.select_slider(
            "Maximum records",
            options=[1000, 3000, 5000, 10000],
            value=3000,
            key="cluster_sample"
        )

    filtered = data[data["year"].between(year_range[0], year_range[1])].copy()

    if selected_sport != "All":
        filtered = filtered[filtered["sport"].astype(str) == selected_sport]

    if selected_season != "All":
        filtered = filtered[filtered["season"].astype(str) == selected_season]

    if len(selected_features) < 2:
        st.warning("Select at least two numeric features to perform clustering.")
    else:
        cluster_data = filtered[selected_features].apply(pd.to_numeric, errors="coerce")
        cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(cluster_data) > max_points:
            cluster_data = cluster_data.sample(max_points, random_state=42)

        if len(cluster_data) < max(20, cluster_count * 3):
            st.warning("Not enough complete records. Select a wider year range or another sport.")
        else:
            scaler_cluster = StandardScaler()
            X_scaled = scaler_cluster.fit_transform(cluster_data)

            kmeans_model = KMeans(
                n_clusters=cluster_count,
                n_init=10,
                random_state=42
            )
            kmeans_labels = kmeans_model.fit_predict(X_scaled)

            gmm_model = GaussianMixture(
                n_components=cluster_count,
                covariance_type="full",
                random_state=42
            )
            gmm_labels = gmm_model.fit_predict(X_scaled)

            kmeans_score = (
                silhouette_score(X_scaled, kmeans_labels)
                if len(set(kmeans_labels)) > 1 else np.nan
            )
            gmm_score = (
                silhouette_score(X_scaled, gmm_labels)
                if len(set(gmm_labels)) > 1 else np.nan
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Athlete Records", f"{len(cluster_data):,}")
            m2.metric("Number of Clusters", cluster_count)
            m3.metric(
                "K-Means Silhouette",
                f"{kmeans_score:.3f}" if not np.isnan(kmeans_score) else "N/A"
            )
            m4.metric(
                "GMM Silhouette",
                f"{gmm_score:.3f}" if not np.isnan(gmm_score) else "N/A"
            )

            kmeans_data = cluster_data.copy()
            kmeans_data["Cluster"] = kmeans_labels.astype(str)

            gmm_data = cluster_data.copy()
            gmm_data["Cluster"] = gmm_labels.astype(str)

            st.subheader("Clustering Algorithm Comparison")

            x_feature = st.selectbox(
                "X-axis feature", selected_features, index=0, key="advanced_cluster_x"
            )
            y_options = [feature for feature in selected_features if feature != x_feature]
            y_feature = st.selectbox(
                "Y-axis feature", y_options, index=0, key="advanced_cluster_y"
            )

            graph1, graph2 = st.columns(2, gap="large")

            with graph1:
                st.markdown("### K-Means")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    data=kmeans_data,
                    x=x_feature,
                    y=y_feature,
                    hue="Cluster",
                    palette="viridis",
                    alpha=0.7,
                    s=40,
                    ax=ax
                )
                ax.set_title("K-Means Athlete Groups")
                ax.grid(alpha=0.2)
                ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(f"Inertia: {kmeans_model.inertia_:,.2f}")

            with graph2:
                st.markdown("### Gaussian Mixture Model")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    data=gmm_data,
                    x=x_feature,
                    y=y_feature,
                    hue="Cluster",
                    palette="plasma",
                    alpha=0.7,
                    s=40,
                    ax=ax
                )
                ax.set_title("Gaussian Mixture Athlete Groups")
                ax.grid(alpha=0.2)
                ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"AIC: {gmm_model.aic(X_scaled):,.1f} | "
                    f"BIC: {gmm_model.bic(X_scaled):,.1f}"
                )

            st.subheader("Cluster Quality Comparison")
            comparison = pd.DataFrame({
                "Algorithm": ["K-Means", "Gaussian Mixture Model"],
                "Silhouette Score": [kmeans_score, gmm_score]
            }).dropna()

            if not comparison.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    data=comparison,
                    x="Algorithm",
                    y="Silhouette Score",
                    hue="Algorithm",
                    legend=False,
                    palette="Set2",
                    ax=ax
                )
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylim(-1, 1)
                ax.set_title("Silhouette Score (higher is generally better)")
                ax.grid(axis="y", alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Average Athlete Profile by Cluster")
            profile1, profile2 = st.columns(2, gap="large")

            with profile1:
                st.markdown("**K-Means Feature Heatmap**")
                profile = kmeans_data.groupby("Cluster")[selected_features].mean()
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(profile, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                ax.set_title("Average Features per K-Means Cluster")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(profile.round(2), width="stretch")

            with profile2:
                st.markdown("**GMM Feature Heatmap**")
                profile_gmm = gmm_data.groupby("Cluster")[selected_features].mean()
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(profile_gmm, annot=True, fmt=".1f", cmap="magma", ax=ax)
                ax.set_title("Average Features per GMM Cluster")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(profile_gmm.round(2), width="stretch")

            st.subheader("Athlete Distribution Across Clusters")
            distribution = pd.concat([
                kmeans_data["Cluster"].value_counts().rename_axis("Cluster")
                .reset_index(name="Athletes").assign(Algorithm="K-Means"),
                gmm_data["Cluster"].value_counts().rename_axis("Cluster")
                .reset_index(name="Athletes").assign(Algorithm="GMM")
            ], ignore_index=True)

            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(
                data=distribution,
                x="Cluster",
                y="Athletes",
                hue="Algorithm",
                palette="Set2",
                ax=ax
            )
            ax.set_title("Cluster Size Comparison")
            ax.set_ylabel("Number of athletes")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            clustered = cluster_data.copy()
            clustered["KMeans_Cluster"] = kmeans_labels
            clustered["GMM_Cluster"] = gmm_labels
            clustered["Sport"] = filtered.loc[cluster_data.index, "sport"].astype(str)
            clustered["Year"] = filtered.loc[cluster_data.index, "year"]

            with st.expander("View clustered records"):
                st.dataframe(clustered, width="stretch")

            st.download_button(
                "Download Both Clustering Results",
                clustered.to_csv(index=True).encode("utf-8"),
                "olympic_clustering_comparison.csv",
                "text/csv"
            )


# SECTION 6: ADVANCED ENSEMBLE LEARNING
elif menu == "Ensemble Learning: Bagging & Random Forest":
    st.header("🌲 Ensemble Learning: Bagging & Random Forest")
    st.caption("Compare individual trees and ensemble models for medal prediction.")

    data = df_raw.copy()
    required = ["medal", "year", "age", "height", "weight", "sex", "season", "sport"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        st.error(f"Required columns missing from dataset: {', '.join(missing)}")
    else:
        f1, f2, f3 = st.columns(3)
        data["year"] = pd.to_numeric(data["year"], errors="coerce")
        year_min = int(data["year"].dropna().min())
        year_max = int(data["year"].dropna().max())

        with f1:
            ens_years = st.slider(
                "Year range", year_min, year_max,
                (max(year_min, 1960), year_max), key="ens_year"
            )
        with f2:
            ens_sport = st.selectbox(
                "Sport", ["All"] + sorted(data["sport"].dropna().astype(str).unique()),
                key="ens_sport"
            )
        with f3:
            ens_season = st.selectbox(
                "Season", ["All"] + sorted(data["season"].dropna().astype(str).unique()),
                key="ens_season"
            )

        e1, e2, e3 = st.columns(3)
        with e1:
            test_size = st.slider("Test data (%)", 15, 40, 25, key="ens_test") / 100
        with e2:
            n_estimators = st.slider(
                "Number of trees / estimators", 50, 300, 150, step=50,
                key="ens_estimators"
            )
        with e3:
            max_depth = st.select_slider(
                "Maximum tree depth", options=[3, 5, 8, 12, 20, None],
                value=8, key="ens_depth"
            )

        data = data[data["year"].between(ens_years[0], ens_years[1])].copy()
        if ens_sport != "All":
            data = data[data["sport"].astype(str) == ens_sport]
        if ens_season != "All":
            data = data[data["season"].astype(str) == ens_season]

        data["medal_won"] = data["medal"].notna().astype(int)
        features = ["age", "height", "weight", "year", "sex", "season", "sport"]
        model_data = data[features + ["medal_won"]].copy()

        for col in ["age", "height", "weight", "year"]:
            model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
            model_data[col] = model_data[col].fillna(model_data[col].median())

        for col in ["sex", "season", "sport"]:
            model_data[col] = model_data[col].fillna("Unknown").astype(str)

        X = pd.get_dummies(model_data[features], drop_first=False)
        y = model_data["medal_won"]

        if y.nunique() < 2 or y.value_counts().min() < 2:
            st.warning("The selected filters leave too few examples in one medal class. Widen the filters.")
        else:
            if len(X) > 20000:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, train_size=20000, random_state=42, stratify=y
                )
            else:
                X_sample, y_sample = X, y

            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=test_size, random_state=42, stratify=y_sample
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=max_depth, class_weight="balanced", random_state=42
                ),
                "Bagging": BaggingClassifier(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                )
            }

            rows = []
            fitted = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
                fitted[name] = (model, pred, prob)
                rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1 Score": f1_score(y_test, pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_test, prob)
                })

            results_df = pd.DataFrame(rows).set_index("Model")
            best_name = results_df["ROC-AUC"].idxmax()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best Model", best_name)
            m2.metric("Best ROC-AUC", f"{results_df.loc[best_name, 'ROC-AUC']:.3f}")
            m3.metric("Training Records", f"{len(X_train):,}")
            m4.metric("Testing Records", f"{len(X_test):,}")

            st.subheader("Model Performance Dashboard")
            metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

            fig, ax = plt.subplots(figsize=(11, 5))
            results_df[metric_columns].plot(
                kind="bar", ax=ax, width=0.78, colormap="Set2"
            )
            ax.set_title("Performance Comparison of Classification Models")
            ax.set_ylabel("Score")
            ax.set_xlabel("Model")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=0)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Detailed Metric Heatmap")
            fig, ax = plt.subplots(figsize=(9, 3.5))
            sns.heatmap(
                results_df[metric_columns], annot=True, fmt=".3f",
                cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5, ax=ax
            )
            ax.set_title("Model Metrics at a Glance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(results_df.round(3), width="stretch")

            chosen_model = st.selectbox(
                "Inspect model", list(fitted.keys()), key="ens_inspect"
            )
            model, pred, prob = fitted[chosen_model]
            cm = confusion_matrix(y_test, pred, labels=[0, 1])

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Medal", "Medal"],
                    yticklabels=["No Medal", "Medal"], ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{chosen_model} Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with c2:
                if hasattr(model, "feature_importances_"):
                    importance = pd.Series(
                        model.feature_importances_, index=X.columns
                    ).sort_values(ascending=False).head(15).sort_values()

                    st.subheader("Top Feature Importances")
                    fig, ax = plt.subplots(figsize=(7, 5))
                    importance.plot(kind="barh", ax=ax, color="#3984b6")
                    ax.set_xlabel("Feature Importance")
                    ax.set_title(f"Top Features — {chosen_model}")
                    ax.grid(axis="x", alpha=0.2)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Feature importance is not available for this model.")

            st.download_button(
                "Download Model Comparison",
                results_df.to_csv().encode("utf-8"),
                "ensemble_model_comparison.csv",
                "text/csv"
            )


# SECTION: BEST MODEL
elif menu == "Best Model":

    import numpy as np
    import pandas as pd
    import streamlit as st

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    st.header("🏆 Best Model — Athlete Medal Predictor")
    st.caption("Compare five regression models and automatically select the best one.")

    data = df_raw.copy()

    required = ["medal", "year", "age", "height", "weight",
                "sex", "season", "sport"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
    else:
        data["year"] = pd.to_numeric(data["year"], errors="coerce")
        data["medal_won"] = data["medal"].notna().astype(int)

        data = data.dropna(subset=["year"])

        if data.empty:
            st.error("No valid Olympic year data is available.")
            st.stop()

        st.subheader("1. Filter Data")

        c1, c2, c3 = st.columns(3)

        min_year = int(data["year"].min())
        max_year = int(data["year"].max())

        with c1:
            year_range = st.slider(
                "Olympic Year",
                min_year,
                max_year,
                (min_year, max_year),
                key="best_year"
            )

        with c2:
            sports = ["All"] + sorted(data["sport"].dropna().astype(str).unique())
            sport = st.selectbox("Sport", sports, key="best_sport")

        with c3:
            seasons = ["All"] + sorted(data["season"].dropna().astype(str).unique())
            season = st.selectbox("Season", seasons, key="best_season")

        sexes = ["All"] + sorted(data["sex"].dropna().astype(str).unique())
        sex = st.selectbox("Athlete Sex", sexes, key="best_sex")

        data = data[data["year"].between(*year_range)]

        if sport != "All":
            data = data[data["sport"].astype(str) == sport]

        if season != "All":
            data = data[data["season"].astype(str) == season]

        if sex != "All":
            data = data[data["sex"].astype(str) == sex]

        numeric_options = ["year", "age", "height", "weight"]

        features = st.multiselect(
            "Select numerical prediction features",
            numeric_options,
            default=["year", "age", "height", "weight"],
            key="best_features"
        )

        if not features:
            st.warning("Select at least one numerical feature.")
            st.stop()

        if len(data) < 30 or data["medal_won"].nunique() < 2:
            st.warning("Not enough data after filtering. Please widen your filters.")
            st.stop()

        # Use a smaller sample to speed up model training
        model_data = data[features + ["medal_won"]].copy()

        for col in features:
            model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

        if len(model_data) > 10000:
            model_data = model_data.sample(n=10000, random_state=42)

        st.metric("Training records", f"{len(model_data):,}")

        @st.cache_resource
        def train_best_models(model_data, features):
            X = model_data[list(features)]
            y = model_data["medal_won"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "Simple Linear Regression": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LinearRegression())
                ]),
                "Multiple Linear Regression": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LinearRegression())
                ]),
                "Decision Tree": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DecisionTreeRegressor(
                        max_depth=10, random_state=42
                    ))
                ]),
                "Random Forest": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(
                        n_estimators=30,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1
                    ))
                ]),
                "Bagging": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", BaggingRegressor(
                        n_estimators=10,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
            }

            # Simple regression uses only one numerical feature
            simple_feature = features[0]

            simple_X_train = X_train[[simple_feature]]
            simple_X_test = X_test[[simple_feature]]

            results = []
            fitted = {}

            for name, model in models.items():
                if name == "Simple Linear Regression":
                    model.fit(simple_X_train, y_train)
                    predictions = model.predict(simple_X_test)
                else:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                mae = float(mean_absolute_error(y_test, predictions))

                results.append({
                    "Model": name,
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4)
                })

                fitted[name] = model

            results_df = pd.DataFrame(results).sort_values(
                "RMSE"
            ).reset_index(drop=True)

            return fitted, results_df, simple_feature

        with st.spinner("Training and comparing five models..."):
            fitted_models, results_df, simple_feature = train_best_models(
                model_data, tuple(features)
            )

        best_name = results_df.iloc[0]["Model"]
        best_model = fitted_models[best_name]

        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", best_name)
        m2.metric("RMSE", f"{results_df.iloc[0]['RMSE']:.4f}")
        m3.metric("MAE", f"{results_df.iloc[0]['MAE']:.4f}")

        st.subheader("2. Model Comparison")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.bar_chart(results_df.set_index("Model")[["RMSE"]])

        st.caption("Lower RMSE indicates better performance on the test set.")

# SECTION 7: CROSS-VALIDATION & HYPERPARAMETER TUNING
elif menu == "Cross-Validation & Hyperparameter Tuning":
    st.header("🧪 Cross-Validation & Hyperparameter Tuning")
    st.caption("Estimate model reliability across folds and compare Random Forest settings.")

    data = df_raw.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")

    f1, f2, f3 = st.columns(3)
    ymin = int(data["year"].dropna().min())
    ymax = int(data["year"].dropna().max())

    with f1:
        cv_years = st.slider(
            "Year range", ymin, ymax, (max(ymin, 1960), ymax), key="cv_year"
        )
    with f2:
        cv_sport = st.selectbox(
            "Sport", ["All"] + sorted(data["sport"].dropna().astype(str).unique()),
            key="cv_sport"
        )
    with f3:
        cv_season = st.selectbox(
            "Season", ["All"] + sorted(data["season"].dropna().astype(str).unique()),
            key="cv_season"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        fold_count = st.slider("Number of folds", 3, 10, 5, key="cv_folds")
    with c2:
        cv_strategy = st.selectbox(
            "Validation strategy", ["Stratified K-Fold", "K-Fold"], key="cv_strategy"
        )
    with c3:
        search_method = st.selectbox(
            "Tuning method", ["Grid Search", "Randomized Search"], key="cv_search"
        )

    data = data[data["year"].between(cv_years[0], cv_years[1])].copy()
    if cv_sport != "All":
        data = data[data["sport"].astype(str) == cv_sport]
    if cv_season != "All":
        data = data[data["season"].astype(str) == cv_season]

    data["medal_won"] = data["medal"].notna().astype(int)
    features = ["age", "height", "weight", "year", "sex", "season", "sport"]
    model_data = data[features + ["medal_won"]].copy()

    for col in ["age", "height", "weight", "year"]:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
        model_data[col] = model_data[col].fillna(model_data[col].median())

    for col in ["sex", "season", "sport"]:
        model_data[col] = model_data[col].fillna("Unknown").astype(str)

    X = pd.get_dummies(model_data[features], drop_first=False)
    y = model_data["medal_won"]

    if y.nunique() < 2 or y.value_counts().min() < fold_count:
        st.warning("There are not enough examples in each class for this fold count. Reduce folds or widen the filters.")
    else:
        if cv_strategy == "Stratified K-Fold":
            cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=fold_count, shuffle=True, random_state=42)

        st.subheader("Cross-Validation Performance Analysis")
        cv_model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

        if len(X) > 15000:
            X_cv, _, y_cv, _ = train_test_split(
                X, y, train_size=15000, random_state=42, stratify=y
            )
        else:
            X_cv, y_cv = X, y

        with st.spinner("Evaluating cross-validation folds..."):
            scores = cross_val_score(
                cv_model, X_cv, y_cv, cv=cv, scoring="roc_auc", n_jobs=-1
            )

        cv_results = pd.DataFrame({
            "Fold": [f"Fold {i + 1}" for i in range(len(scores))],
            "ROC-AUC": scores
        })

        a, b, c, d = st.columns(4)
        a.metric("Mean ROC-AUC", f"{scores.mean():.3f}")
        b.metric("Score Std. Dev.", f"{scores.std():.3f}")
        c.metric("Best Fold", f"{scores.max():.3f}")
        d.metric("Worst Fold", f"{scores.min():.3f}")

        left, right = st.columns([1.5, 1], gap="large")

        with left:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(
                data=cv_results, x="Fold", y="ROC-AUC",
                color="#3984b6", ax=ax
            )
            ax.axhline(
                scores.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {scores.mean():.3f}"
            )
            ax.fill_between(
                [-0.5, len(scores) - 0.5],
                max(0, scores.mean() - scores.std()),
                min(1, scores.mean() + scores.std()),
                color="orange", alpha=0.15, label="±1 standard deviation"
            )
            ax.set_ylim(0, 1)
            ax.set_title("ROC-AUC Across Validation Folds")
            ax.set_ylabel("ROC-AUC")
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("**Fold Results**")
            st.dataframe(cv_results.round(3), width="stretch", hide_index=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(y=scores, color="#83c5be", width=0.35, ax=ax)
            sns.stripplot(y=scores, color="#1d3557", size=7, ax=ax)
            ax.set_title("Score Distribution")
            ax.set_ylabel("ROC-AUC")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Random Forest Hyperparameter Search")

        if search_method == "Grid Search":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }
            search = GridSearchCV(
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                ),
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1
            )
        else:
            param_dist = {
                "n_estimators": [50, 100, 150, 200, 250],
                "max_depth": [5, 8, 10, 15, None],
                "min_samples_split": [2, 4, 5, 8],
                "min_samples_leaf": [1, 2, 3, 5]
            }
            search = RandomizedSearchCV(
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                ),
                param_distributions=param_dist,
                n_iter=8,
                cv=cv,
                scoring="roc_auc",
                random_state=42,
                n_jobs=-1
            )

        if st.button("Run Hyperparameter Search", key="run_tuning"):
            with st.spinner("Searching for the best parameters..."):
                search.fit(X, y)

            st.success("Hyperparameter search completed.")
            st.metric("Best Cross-Validation ROC-AUC", f"{search.best_score_:.3f}")
            st.write("**Best parameters:**")
            st.json(search.best_params_)

            tuning_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
            display_cols = [
                c for c in ["rank_test_score", "mean_test_score", "std_test_score", "params"]
                if c in tuning_results.columns
            ]
            st.dataframe(tuning_results[display_cols].head(15), width="stretch")
            st.download_button(
                "Download Tuning Results",
                tuning_results[display_cols].to_csv(index=False).encode("utf-8"),
                "hyperparameter_tuning_results.csv",
                "text/csv"
            )


# SECTION 8: INTERACTIVE ML PREDICTOR
elif menu == "Interactive ML Predictor (Live Inference)":
    st.header("🎯 Interactive Athlete Predictor")
    st.caption("Enter an athlete profile to estimate body weight and the model's medal probability.")

    p1, p2, p3 = st.columns(3)

    with p1:
        height_in = st.slider("Height (cm)", 140, 220, 180, key="pred_height")
        age_in = st.slider("Age (years)", 14, 60, 25, key="pred_age")

    with p2:
        sex_in = st.selectbox("Sex", ["Male", "Female"], key="pred_sex")
        season_in = st.selectbox("Olympic season", ["Summer", "Winter"], key="pred_season")

    with p3:
        available_years = sorted(
            pd.to_numeric(df_raw["year"], errors="coerce").dropna().astype(int).unique().tolist()
        )
        year_in = st.select_slider(
            "Olympic year",
            options=available_years,
            value=available_years[-1],
            key="pred_year"
        )
        sport_options = sorted(df_raw["sport"].dropna().astype(str).unique().tolist())
        sport_in = st.selectbox("Sport", sport_options, key="pred_sport")

    threshold = st.slider(
        "Medal probability threshold (%)", 10, 90, 50, step=5,
        key="pred_threshold"
    )

    is_male = 1 if sex_in == "Male" else 0
    is_summer = 1 if season_in == "Summer" else 0
    reg_features = ["height", "age", "sex_male", "season_summer", "year"]

    reg_model, clf_model, scaler = get_trained_models(df_raw)

    input_df = pd.DataFrame(
        [[height_in, age_in, is_male, is_summer, year_in]],
        columns=reg_features
    )

    pred_weight = float(reg_model.predict(input_df)[0])
    input_sc = scaler.transform(input_df)
    pred_prob = float(clf_model.predict_proba(input_sc)[0][1] * 100)
    bmi = pred_weight / ((height_in / 100) ** 2)

    st.markdown("---")
    st.subheader("Prediction Summary")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Estimated Weight", f"{pred_weight:.1f} kg")
    r2.metric("Estimated BMI", f"{bmi:.1f}")
    r3.metric("Medal Probability", f"{pred_prob:.1f}%")
    r4.metric("Selected Threshold", f"{threshold}%")

    if pred_prob >= threshold:
        st.success("The model's estimated probability is at or above your selected threshold.")
    else:
        st.info("The model's estimated probability is below your selected threshold.")

    st.warning(
        "This is an experimental model estimate, not a real-world guarantee. "
        "The current trained models use height, age, sex, season, and year; "
        "the selected sport is displayed as profile context but is not an input to these models."
    )

    st.subheader("Medal Probability Analysis")
    gauge_col, profile_col = st.columns([1.2, 1], gap="large")

    with gauge_col:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(
            ["Estimated probability"], [100],
            color="#e9ecef", height=0.4, label="Full scale"
        )
        ax.barh(
            ["Estimated probability"], [pred_prob],
            color="#3984b6", height=0.4, label="Predicted probability"
        )
        ax.axvline(
            threshold, color="#e63946", linestyle="--", linewidth=2,
            label=f"Threshold: {threshold}%"
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Model-Estimated Medal Probability")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with profile_col:
        st.markdown("**Athlete Measurements**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(
            ["Height (cm)", "Age (years)", "Weight (kg)"],
            [height_in, age_in, pred_weight],
            color=["#3984b6", "#83c5be", "#e9c46a"]
        )
        ax.set_title("Selected and Estimated Values")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    profile = pd.DataFrame({
        "Parameter": ["Height", "Age", "Sex", "Season", "Year", "Sport"],
        "Selected Value": [
            f"{height_in} cm",
            f"{age_in} years",
            str(sex_in),
            str(season_in),
            str(year_in),
            str(sport_in)
        ]
    })

    st.subheader("Athlete Profile")
    st.dataframe(profile, hide_index=True, width="stretch")

    prediction_download = pd.DataFrame([{
        "height_cm": height_in,
        "age": age_in,
        "sex": sex_in,
        "season": season_in,
        "year": year_in,
        "sport": sport_in,
        "predicted_weight_kg": round(pred_weight, 2),
        "estimated_bmi": round(bmi, 2),
        "medal_probability_percent": round(pred_prob, 2),
        "probability_threshold_percent": threshold
    }])

    st.download_button(
        "Download Prediction",
        prediction_download.to_csv(index=False).encode("utf-8"),
        "athlete_prediction.csv",
        "text/csv"
    )
