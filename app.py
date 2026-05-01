import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle, os, io, time, warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor Pro",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">',
    unsafe_allow_html=True,
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #F4F6FB; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0D0D1A 0%, #1A1A2E 45%, #0F3460 100%);
    padding: 2.8rem 2.4rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.6rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(83,52,131,0.45) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p { opacity: 0.7; margin: 0; font-size: 1rem; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: 0.9rem;
    margin-right: 0.4rem;
    letter-spacing: 0.5px;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border-left: 4px solid #0F3460;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .label { font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 1.2px; }
.metric-card .value { font-size: 1.85rem; font-weight: 700; color: #1A1A2E; line-height: 1.2; }
.metric-card .delta { font-size: 0.78rem; margin-top: 0.2rem; }
.delta-up { color: #4CAF50; } .delta-down { color: #E53935; }

/* ── Prediction box ── */
.result-box {
    background: linear-gradient(135deg, #0F3460 0%, #533483 100%);
    color: white;
    padding: 2.2rem;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(15,52,96,0.3);
}
.result-box .price {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    line-height: 1.1;
}
.result-box .label {
    opacity: 0.75;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Section titles ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #5E5E7F;
    margin: 1.6rem 0 0.9rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid #E8EBF0;
}

/* ── Scenario card ── */
.scenario-card {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 0.6rem;
    border-left: 4px solid #533483;
}
.scenario-card h4 { margin: 0 0 0.3rem; font-size: 0.9rem; color: #555; }
.scenario-card .sc-price { font-size: 1.4rem; font-weight: 700; color: #0F3460; }

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #0F3460;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
}
.best-badge {
    background: #FFF3CD;
    color: #856404;
}

/* ── Info box ── */
.info-box {
    background: #EEF6FF;
    border-left: 4px solid #64B5F6;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #1A3A5C;
    margin: 0.8rem 0;
}
.warn-box {
    background: #FFF8E1;
    border-left: 4px solid #FFB300;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #5D4037;
    margin: 0.8rem 0;
}

/* ── Buttons ── */
div[data-testid="stSlider"] > div { padding-top: 0.2rem; }
.stButton > button {
    background: linear-gradient(135deg, #0F3460, #533483);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.72rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0D1A 0%, #1A1A2E 100%);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: rgba(255,255,255,0.6) !important; font-size: 0.8rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { font-size: 0.88rem; font-weight: 500; }

/* ── Dataframe ── */
.dataframe { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## <i class='bi bi-sliders'></i> Settings", unsafe_allow_html=True)
    st.markdown("---")

    test_size = st.slider("Test Split Size", 0.10, 0.40, 0.20, 0.05,
                          help="Fraction of data held out for evaluation")
    random_seed = st.selectbox("Random Seed", [42, 7, 99, 2024, 2026], index=0)
    n_estimators = st.slider("RF: n_estimators", 50, 300, 50, 50)
    max_depth = st.slider("RF: max_depth", 4, 20, 8, 2)

    st.markdown("---")
    st.markdown("### <i class='bi bi-palette'></i> Chart Theme", unsafe_allow_html=True)
    chart_theme = st.selectbox("Palette", ["Navy & Purple", "Teal & Coral", "Forest & Gold"])

    PALETTES = {
        "Navy & Purple": {"primary": "#0F3460", "secondary": "#533483",
                          "accent": "#64B5F6", "cmap": "Blues"},
        "Teal & Coral":  {"primary": "#00897B", "secondary": "#E64A19",
                          "accent": "#80CBC4", "cmap": "summer"},
        "Forest & Gold": {"primary": "#2E7D32", "secondary": "#F9A825",
                          "accent": "#A5D6A7", "cmap": "YlGn"},
    }
    PAL = PALETTES[chart_theme]

    st.markdown("---")
    show_raw = st.checkbox("Show raw dataset tab", value=False)
    st.markdown("---")
    st.markdown('<span style="font-size:0.75rem;opacity:0.5">California Housing · Phase 2</span>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════
MODEL_PATH = f"house_model_{n_estimators}_{max_depth}_{random_seed}.pkl"

@st.cache_resource(show_spinner="Training models — this takes ~20 seconds the first time…")
def load_everything(test_sz, seed, n_est, mx_depth):
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.columns = [
        "MedIncome", "HouseAge", "AvgRooms", "AvgBedrooms",
        "Population", "AvgOccupants", "Latitude", "Longitude", "Price"
    ]
    df["Price"] = df["Price"] * 100_000

    # Feature engineering
    df["RoomsPerPerson"]   = df["AvgRooms"] / df["AvgOccupants"].clip(0.5)
    df["BedroomRatio"]     = df["AvgBedrooms"] / df["AvgRooms"].clip(1)
    df["IncomePerPerson"]  = df["MedIncome"] / df["AvgOccupants"].clip(0.5)
    df["PopDensity"]       = df["Population"] / df["AvgOccupants"].clip(0.5)

    FEATURES = ["MedIncome", "HouseAge", "AvgRooms", "AvgBedrooms",
                "Population", "AvgOccupants", "Latitude", "Longitude",
                "RoomsPerPerson", "BedroomRatio", "IncomePerPerson", "PopDensity"]

    X = df[FEATURES]
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=seed)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train all models ──
    models_def = {
        "Random Forest":        RandomForestRegressor(n_estimators=n_est, max_depth=mx_depth,
                                                      random_state=seed, n_jobs=-1),
        "Extra Trees":          ExtraTreesRegressor(n_estimators=50, max_depth=mx_depth,
                                                    random_state=seed, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=50, max_depth=5,
                                                          random_state=seed, subsample=0.8),
        "Decision Tree":        DecisionTreeRegressor(max_depth=mx_depth, random_state=seed),
        "Ridge Regression":     Ridge(alpha=1.0),
        "Lasso Regression":     Lasso(alpha=50.0, max_iter=5000),
    }

    results = {}
    trained = {}
    for name, mdl in models_def.items():
        if name in ("Ridge Regression", "Lasso Regression"):
            mdl.fit(X_train_sc, y_train)
            preds = mdl.predict(X_test_sc)
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
        results[name] = {
            "mae":  mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2":   r2_score(y_test, preds),
            "preds": preds,
        }
        trained[name] = mdl

    # Best model = highest R²
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = trained[best_name]
    best_preds = results[best_name]["preds"]

    return (df, X, y, X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, scaler,
            trained, results, best_name, best_model, best_preds, FEATURES)

(df, X, y, X_train, X_test, y_train, y_test,
 X_train_sc, X_test_sc, scaler,
 trained_models, model_results, best_name, best_model, best_preds, FEATURES) = load_everything(
    test_size, random_seed, n_estimators, max_depth)

mae  = model_results[best_name]["mae"]
rmse = model_results[best_name]["rmse"]
r2   = model_results[best_name]["r2"]

# ══════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1><i class="bi bi-houses-fill" style="font-size:2rem;vertical-align:middle;margin-right:0.5rem;opacity:0.85"></i>House Price Predictor <em style="font-size:1.6rem;opacity:0.6">Pro</em></h1>
  <p>6 regression models · 12 engineered features · California Housing Dataset</p>
  <span class="badge"><i class="bi bi-trophy-fill"></i> Best: {best_name}</span>
  <span class="badge"><i class="bi bi-graph-up"></i> R² {r2:.3f}</span>
  <span class="badge">April 2026</span>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("Best Model",        best_name,         ""),
    ("R² Score",          f"{r2:.3f}",        '<span class="delta-up">▲ Strong fit</span>'),
    ("Mean Abs. Error",   f"${mae:,.0f}",     ""),
    ("RMSE",              f"${rmse:,.0f}",    ""),
    ("Training Samples",  f"{len(X_train):,}", ""),
]
for col, (lbl, val, delta) in zip([c1,c2,c3,c4,c5], metrics):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value" style="font-size:{'1.1rem' if len(val)>9 else '1.85rem'};padding-top:4px">{val}</div>
            <div class="delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CACHED EXPENSIVE COMPUTATIONS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_learning_curve(_model, _X_train, _y_train, model_name, is_linear,
                       _X_train_sc, test_sz, seed, n_est, mx_depth):
    """Cache learning curve results per model + hyperparams."""
    Xuse = _X_train_sc if is_linear else _X_train
    sizes = np.linspace(0.1, 1.0, 8)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        _model, Xuse, _y_train,
        train_sizes=sizes, cv=4,
        scoring="r2", n_jobs=-1,
        shuffle=True, random_state=42
    )
    return train_sizes_abs, train_scores, val_scores


@st.cache_data(show_spinner=False)
def get_cv_scores(_trained_models, _X_train, _y_train, _X_train_sc,
                  test_sz, seed, n_est, mx_depth):
    """Cache 5-fold CV scores for all models per hyperparams."""
    cv_results = {}
    for name, mdl in _trained_models.items():
        Xc = _X_train_sc if name in ("Ridge Regression", "Lasso Regression") else _X_train
        scores = cross_val_score(mdl, Xc, _y_train, cv=5, scoring="r2", n_jobs=-1)
        cv_results[name] = scores
    return cv_results


@st.cache_data(show_spinner=False)
def get_permutation_importance(_model, _X_test, _y_test, model_name,
                                test_sz, seed, n_est, mx_depth):
    """Cache permutation importance per model + hyperparams."""
    perm = permutation_importance(_model, _X_test, _y_test,
                                  n_repeats=8, random_state=42, n_jobs=-1)
    return perm.importances_mean, perm.importances_std


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_labels = ["Predict", "Map Explorer", "EDA", "Model Arena",
              "Insights", "Learning Curves", "Scenario Compare", "Export"]
if show_raw:
    tab_labels.append("Raw Data")

tabs = st.tabs(tab_labels)

# ═══════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Configure Your Property</div>', unsafe_allow_html=True)

    col_sel, col_gap = st.columns([1, 3])
    with col_sel:
        chosen_model_name = st.selectbox("Model to use for prediction",
                                         list(trained_models.keys()),
                                         index=list(trained_models.keys()).index(best_name))
    chosen_model = trained_models[chosen_model_name]
    is_linear = chosen_model_name in ("Ridge Regression", "Lasso Regression")

    col_in, col_out = st.columns([1.15, 0.85], gap="large")
    with col_in:
        st.markdown("**Location**")
        loc1, loc2 = st.columns(2)
        with loc1:
            latitude  = st.slider("Latitude",  32.5, 42.0, 37.5, 0.05)
        with loc2:
            longitude = st.slider("Longitude", -124.5, -114.0, -122.0, 0.05)

        st.markdown("**Property Details**")
        p1, p2 = st.columns(2)
        with p1:
            house_age  = st.slider("House Age (yrs)", 1, 52, 20)
            avg_rooms  = st.slider("Avg Rooms / HH",  1.0, 15.0, 5.5, 0.5)
        with p2:
            avg_beds   = st.slider("Avg Bedrooms / HH", 1.0, 5.0, 2.0, 0.5)
            avg_occ    = st.slider("Avg Occupants / HH", 1.0, 6.0, 2.8, 0.1)

        st.markdown("**Neighbourhood**")
        n1, n2 = st.columns(2)
        with n1:
            med_income = st.slider("Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
        with n2:
            population = st.slider("Block Population", 100, 5000, 1200)

        predict_btn = st.button("Run Prediction", key="main_predict")

    with col_out:
        if predict_btn:
            rooms_per_person  = avg_rooms / max(avg_occ, 0.5)
            bedroom_ratio     = avg_beds  / max(avg_rooms, 1)
            income_per_person = med_income / max(avg_occ, 0.5)
            pop_density       = population / max(avg_occ, 0.5)

            raw_input = [med_income, house_age, avg_rooms, avg_beds, population, avg_occ,
                         latitude, longitude, rooms_per_person, bedroom_ratio,
                         income_per_person, pop_density]
            input_df = pd.DataFrame([raw_input], columns=FEATURES)

            if is_linear:
                input_sc  = scaler.transform(input_df)
                prediction = chosen_model.predict(input_sc)[0]
            else:
                prediction = chosen_model.predict(input_df)[0]

            cur_mae  = model_results[chosen_model_name]["mae"]
            cur_r2   = model_results[chosen_model_name]["r2"]

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Estimated House Price</div>
                <div class="price">${prediction:,.0f}</div>
                <div style="opacity:0.6;margin-top:0.4rem;font-size:0.82rem">
                    ± ${cur_mae:,.0f} margin · {chosen_model_name} · R² {cur_r2:.3f}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Percentile gauge
            pct = np.percentile(df["Price"], [25, 50, 75])
            label_tier = ("Below Average", "Average", "Above Average", "Top Tier")
            tier_idx   = int(sum(prediction > p for p in pct))
            tier_colors = ["#64B5F6","#4CAF50","#FF9800","#E53935"]

            fig, ax = plt.subplots(figsize=(5, 1.4))
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            bounds = [0] + list(pct) + [df["Price"].max()]
            for i in range(4):
                ax.barh(0, bounds[i+1]-bounds[i], left=bounds[i],
                        height=0.55, color=tier_colors[i], alpha=0.85,
                        edgecolor="white", linewidth=2)
            ax.axvline(prediction, color="#1A1A2E", linewidth=2.5, linestyle="--", zorder=5)
            ax.set_xlim(0, df["Price"].max())
            ax.set_yticks([])
            ax.set_xlabel("Price (USD)", fontsize=8, color="#555")
            ax.set_title(f"Market tier: {label_tier[tier_idx]}", fontsize=9,
                         fontweight="bold", color="#1A1A2E")
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.tick_params(colors="#777", labelsize=7)
            fig.tight_layout()
            st.pyplot(fig, width="stretch"); plt.close(fig)

            # Feature influence mini-bar (only for tree-based)
            if not is_linear and hasattr(chosen_model, "feature_importances_"):
                fi = pd.Series(chosen_model.feature_importances_, index=FEATURES)
                top5 = fi.nlargest(5)
                fig2, ax2 = plt.subplots(figsize=(5, 2.2))
                fig2.patch.set_alpha(0)
                ax2.patch.set_alpha(0)
                bars = ax2.barh(top5.index[::-1], top5.values[::-1],
                                color=PAL["primary"], alpha=0.85, edgecolor="white")
                ax2.set_xlabel("Importance", fontsize=8, color="#555")
                ax2.set_title("Top 5 features driving this prediction",
                              fontsize=8.5, fontweight="bold", color="#1A1A2E")
                for spine in ax2.spines.values(): spine.set_visible(False)
                ax2.tick_params(colors="#777", labelsize=7.5)
                fig2.tight_layout()
                st.pyplot(fig2, width="stretch"); plt.close(fig2)
        else:
            st.markdown("""<div class="info-box">
            <i class="bi bi-arrow-left-circle"></i> &nbsp;Adjust the sliders and click <strong>Run Prediction</strong> to see the estimate,
            market position gauge, and the top 5 features driving the result.
            </div>""", unsafe_allow_html=True)

            # Show model comparison summary while waiting
            st.markdown("**Model Performance Summary**")
            perf_df = pd.DataFrame({
                "Model":  list(model_results.keys()),
                "R²":     [f"{v['r2']:.3f}"   for v in model_results.values()],
                "MAE ($)": [f"{v['mae']:,.0f}" for v in model_results.values()],
                "RMSE ($)":[f"{v['rmse']:,.0f}"for v in model_results.values()],
            }).sort_values("R²", ascending=False).reset_index(drop=True)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════
# TAB 2 — MAP EXPLORER
# ═══════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Geographic Price Distribution — California</div>',
                unsafe_allow_html=True)

    m1, m2 = st.columns([0.7, 0.3])
    with m2:
        map_metric  = st.selectbox("Colour by", ["Price", "MedIncome", "HouseAge",
                                                  "AvgRooms", "Population"])
        map_sample  = st.slider("Sample size (for speed)", 500, len(df), 3000, 500)
        map_size_by = st.selectbox("Dot size by", ["Uniform", "MedIncome", "Population"])

    sample = df.sample(map_sample, random_state=42)
    dot_sizes = 8
    if map_size_by == "MedIncome":
        dot_sizes = (sample["MedIncome"] / sample["MedIncome"].max() * 40 + 4).values
    elif map_size_by == "Population":
        dot_sizes = (sample["Population"] / sample["Population"].max() * 40 + 4).values

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(sample["Longitude"], sample["Latitude"],
                    c=sample[map_metric], cmap="plasma",
                    s=dot_sizes, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label=map_metric, shrink=0.7)
    ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(f"California Housing — coloured by {map_metric}", fontsize=11, fontweight="bold")
    ax.set_facecolor("#0D1117")
    fig.patch.set_color("#0D1117")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#aaa"); ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_color("#333")

    # Annotate major cities (approximate)
    cities = {"San Francisco": (-122.41, 37.77),
              "Los Angeles":   (-118.24, 34.05),
              "San Diego":     (-117.16, 32.72),
              "Sacramento":    (-121.49, 38.58)}
    for city, (lon, lat) in cities.items():
        ax.annotate(city, (lon, lat), color="white", fontsize=7.5,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0F3460", ec="none", alpha=0.7))
        ax.plot(lon, lat, "w^", ms=5, zorder=5)

    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # Price heatmap by region
    st.markdown("**Regional Price Heatmap (binned grid)**")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    h = ax2.hist2d(df["Longitude"], df["Latitude"], weights=df["Price"]/1000,
                   bins=[60, 40], cmap="hot", density=False)
    plt.colorbar(h[3], ax=ax2, label="Avg Price ($000s)", shrink=0.7)
    ax2.set_xlabel("Longitude", fontsize=9); ax2.set_ylabel("Latitude", fontsize=9)
    ax2.set_title("Price Intensity Grid", fontsize=11, fontweight="bold")
    ax2.set_facecolor("#0D1117"); fig2.patch.set_color("#0D1117")
    ax2.tick_params(colors="#aaa"); ax2.xaxis.label.set_color("#aaa"); ax2.yaxis.label.set_color("#aaa")
    ax2.title.set_color("white")
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch"); plt.close(fig2)

# ═══════════════════════════════════════════════════
# TAB 3 — EDA
# ═══════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # ── Row 1: distributions
    r1c1, r1c2, r1c3 = st.columns(3)
    for col, feat, title in zip(
        [r1c1, r1c2, r1c3],
        ["Price", "MedIncome", "HouseAge"],
        ["House Price ($000s)", "Median Income (×$10k)", "House Age (years)"]
    ):
        with col:
            fig, axes = plt.subplots(2, 1, figsize=(4.5, 4.5),
                                     gridspec_kw={"height_ratios": [3, 1]})
            data_col = df[feat] / (1000 if feat == "Price" else 1)
            axes[0].hist(data_col, bins=45, color=PAL["primary"],
                         edgecolor="white", linewidth=0.4, alpha=0.88)
            axes[0].set_title(f"Distribution: {title}", fontsize=9, fontweight="bold")
            axes[0].set_ylabel("Count", fontsize=8)
            for sp in axes[0].spines.values(): sp.set_visible(False)
            axes[0].tick_params(labelsize=7)

            axes[1].boxplot(data_col, vert=False, patch_artist=True,
                            boxprops=dict(facecolor=PAL["accent"], color=PAL["primary"]),
                            medianprops=dict(color=PAL["secondary"], linewidth=2),
                            whiskerprops=dict(color=PAL["primary"]),
                            capprops=dict(color=PAL["primary"]),
                            flierprops=dict(marker=".", color=PAL["primary"], alpha=0.4, ms=3))
            axes[1].set_yticks([])
            for sp in axes[1].spines.values(): sp.set_visible(False)
            axes[1].tick_params(labelsize=7)
            fig.tight_layout()
            st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── Scatter matrix of key features
    st.markdown("**Pairwise Scatter: Income, Age, Rooms vs Price**")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feat in zip(axes, ["MedIncome", "HouseAge", "AvgRooms"]):
        sc = ax.scatter(df[feat], df["Price"]/1000, alpha=0.1, s=4,
                        c=df["Latitude"], cmap="coolwarm")
        m, b = np.polyfit(df[feat], df["Price"]/1000, 1)
        ax.plot(df[feat].sort_values(),
                m * df[feat].sort_values() + b, color=PAL["secondary"],
                linewidth=1.8, linestyle="--", label="Trend")
        ax.set_xlabel(feat, fontsize=9); ax.set_ylabel("Price ($000s)", fontsize=9)
        ax.set_title(f"{feat} vs Price", fontsize=10, fontweight="bold")
        for sp in ax.spines.values(): sp.set_alpha(0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    plt.colorbar(sc, ax=axes[-1], label="Latitude", shrink=0.7)
    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── Correlation heatmap
    st.markdown("**Full Correlation Matrix**")
    fig, ax = plt.subplots(figsize=(11, 5))
    numeric_df = df[FEATURES + ["Price"]].copy()
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.6, annot_kws={"size": 7.5},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix — all features", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── Outlier analysis
    st.markdown("**Outlier Detection — IQR Method**")
    outlier_stats = []
    for feat in FEATURES:
        Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[feat] < Q1 - 1.5*IQR) | (df[feat] > Q3 + 1.5*IQR)).sum()
        outlier_stats.append({"Feature": feat, "Q1": f"{Q1:.2f}", "Q3": f"{Q3:.2f}",
                               "IQR": f"{IQR:.2f}", "Outliers (n)": n_out,
                               "Outlier %": f"{n_out/len(df)*100:.1f}%"})
    out_df = pd.DataFrame(outlier_stats)
    st.dataframe(out_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════
# TAB 4 — MODEL ARENA
# ═══════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Model Arena — Head-to-Head Comparison</div>',
                unsafe_allow_html=True)

    # Summary table
    arena_rows = []
    for name, res in model_results.items():
        is_best = name == best_name
        arena_rows.append({
            "Model": ("★ " if is_best else "  ") + name,
            "R²":    round(res["r2"], 4),
            "MAE":   int(res["mae"]),
            "RMSE":  int(res["rmse"]),
            "Best?": "Yes" if is_best else "",
        })
    arena_df = pd.DataFrame(arena_rows).sort_values("R²", ascending=False).reset_index(drop=True)
    st.dataframe(arena_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar charts: R², MAE, RMSE
    a1, a2, a3 = st.columns(3)
    model_names = list(model_results.keys())
    colors_bar  = [PAL["secondary"] if n == best_name else PAL["accent"] for n in model_names]

    for col, metric, label, fmt in zip(
        [a1, a2, a3],
        ["r2", "mae", "rmse"],
        ["R² Score (higher = better)", "MAE — lower = better", "RMSE — lower = better"],
        [".3f", ",.0f", ",.0f"]
    ):
        with col:
            vals = [model_results[n][metric] for n in model_names]
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            bars = ax.barh(model_names, vals, color=colors_bar, edgecolor="white", linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                        f"${v:{fmt}}" if metric != "r2" else f"{v:.3f}",
                        va="center", fontsize=7.5, color="#333")
            ax.set_title(label, fontsize=8.5, fontweight="bold")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(labelsize=7.5)
            ax.set_xlim(0, max(vals)*1.15 if metric != "r2" else 1.05)
            fig.tight_layout()
            st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── Predicted vs Actual for each model
    st.markdown("**Predicted vs Actual — all models**")
    n_models = len(model_results)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for i, (name, res) in enumerate(model_results.items()):
        ax = axes[i]
        ax.scatter(y_test/1000, res["preds"]/1000, alpha=0.18, s=5,
                   color=PAL["primary"] if name != best_name else PAL["secondary"])
        lims = [min(y_test.min(), res["preds"].min())/1000,
                max(y_test.max(), res["preds"].max())/1000]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
        ax.set_title(f"{name}\nR²={res['r2']:.3f}  MAE=${res['mae']:,.0f}",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Actual ($000s)", fontsize=7.5)
        ax.set_ylabel("Predicted ($000s)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values(): sp.set_alpha(0.3)
        if name == best_name:
            ax.set_facecolor("#FFF8E1")
    fig.suptitle("Predicted vs Actual Price — all models", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── Residual comparison
    st.markdown("**Residual Distributions — side-by-side**")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for name, res in model_results.items():
        residuals = (res["preds"] - y_test.values) / 1000
        ax.hist(residuals, bins=60, alpha=0.45, label=name, edgecolor="none")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Prediction Error ($000s)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Residual distributions — all models overlaid", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    for sp in ax.spines.values(): sp.set_alpha(0.3)
    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ═══════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Model Insights & Explainability</div>',
                unsafe_allow_html=True)

    ins_model_name = st.selectbox("Select model for insights",
                                  [n for n in trained_models if n not in ("Ridge Regression","Lasso Regression")],
                                  index=0, key="ins_sel")
    ins_model = trained_models[ins_model_name]

    i1, i2 = st.columns(2)
    with i1:
        # Feature importance
        st.markdown("**Built-in Feature Importance**")
        fi = pd.Series(ins_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        bar_colors = [PAL["secondary"] if v == fi.max() else PAL["primary"] for v in fi]
        bars = ax.barh(fi.index, fi.values, color=bar_colors, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, fi.values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_xlabel("Gini Importance", fontsize=9)
        ax.set_title(f"Feature Importances — {ins_model_name}", fontsize=10, fontweight="bold")
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, fi.max() * 1.18)
        fig.tight_layout()
        st.pyplot(fig, width="stretch"); plt.close(fig)

    with i2:
        # Permutation importance (more robust)
        st.markdown("**Permutation Importance (test set)**")
        with st.spinner("Computing permutation importance (cached after first run)…"):
            perm_mean, perm_std = get_permutation_importance(
                ins_model, X_test, y_test, ins_model_name,
                test_size, random_seed, n_estimators, max_depth
            )
        perm_means = pd.Series(perm_mean, index=FEATURES).sort_values()
        perm_stds  = pd.Series(perm_std,  index=FEATURES).reindex(perm_means.index)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.barh(perm_means.index, perm_means.values,
                xerr=perm_stds.values, color=PAL["accent"],
                edgecolor="white", linewidth=0.6,
                error_kw=dict(ecolor=PAL["primary"], linewidth=1.2, capsize=3))
        ax.set_xlabel("Mean Decrease in R²", fontsize=9)
        ax.set_title("Permutation Importance ± std", fontsize=10, fontweight="bold")
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(labelsize=8)
        ax.axvline(0, color="#ccc", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig, width="stretch"); plt.close(fig)

    # Partial dependence — manual implementation
    st.markdown("**Partial Dependence — how each feature affects predicted price**")
    pd_feat = st.selectbox("Feature for PD plot", FEATURES, index=0, key="pd_feat")
    grid_vals = np.linspace(X[pd_feat].quantile(0.02), X[pd_feat].quantile(0.98), 60)
    X_pd = X_test.copy().iloc[:200]  # use subset for speed
    pd_preds = []
    for val in grid_vals:
        X_temp = X_pd.copy()
        X_temp[pd_feat] = val
        pd_preds.append(ins_model.predict(X_temp).mean())

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(grid_vals, np.array(pd_preds)/1000, color=PAL["primary"], linewidth=2.5)
    ax.fill_between(grid_vals, np.array(pd_preds)/1000,
                    alpha=0.15, color=PAL["primary"])
    ax.set_xlabel(pd_feat, fontsize=9); ax.set_ylabel("Avg Predicted Price ($000s)", fontsize=9)
    ax.set_title(f"Partial Dependence: {pd_feat} → Price", fontsize=11, fontweight="bold")
    for sp in ax.spines.values(): sp.set_alpha(0.3)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # Error analysis
    st.markdown("**Residual Deep Dive**")
    residuals = (best_preds - y_test.values) / 1000

    ea1, ea2 = st.columns(2)
    with ea1:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.scatter(y_test/1000, residuals, alpha=0.2, s=6, color=PAL["primary"])
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axhline(residuals.mean(), color=PAL["secondary"], linestyle=":", linewidth=1.5,
                   label=f"Mean error: ${residuals.mean():.1f}k")
        ax.set_xlabel("Actual Price ($000s)", fontsize=9)
        ax.set_ylabel("Residual ($000s)", fontsize=9)
        ax.set_title("Residuals vs Actual (heteroscedasticity check)", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        for sp in ax.spines.values(): sp.set_alpha(0.3)
        ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        st.pyplot(fig, width="stretch"); plt.close(fig)

    with ea2:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.hist(residuals, bins=70, color=PAL["secondary"],
                edgecolor="white", linewidth=0.3, alpha=0.85, density=True)
        mu, sigma = residuals.mean(), residuals.std()
        x_norm = np.linspace(residuals.min(), residuals.max(), 300)
        ax.plot(x_norm, (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_norm-mu)/sigma)**2),
                color=PAL["primary"], linewidth=2, label="Normal fit")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual ($000s)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Residual Distribution  μ={mu:.1f}k  σ={sigma:.1f}k",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        for sp in ax.spines.values(): sp.set_alpha(0.3)
        ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        st.pyplot(fig, width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════
# TAB 6 — LEARNING CURVES
# ═══════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Learning Curves & Validation</div>',
                unsafe_allow_html=True)

    lc_model_name = st.selectbox("Model", list(trained_models.keys()), key="lc_sel")
    lc_model_obj  = trained_models[lc_model_name]
    is_lc_linear  = lc_model_name in ("Ridge Regression", "Lasso Regression")

    st.markdown("""<div class="info-box">
    Learning curves show how model performance changes as more training data is added.
    A large <strong>gap between train and validation</strong> signals <strong>overfitting</strong>;
    both curves being high signals <strong>underfitting</strong>.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Computing learning curves (cached after first run)…"):
        train_sizes_abs, train_scores, val_scores = get_learning_curve(
            lc_model_obj, X_train, y_train, lc_model_name, is_lc_linear,
            X_train_sc, test_size, random_seed, n_estimators, max_depth
        )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # R² learning curve
    ax = axes[0]
    ax.plot(train_sizes_abs, train_mean, "o-", color=PAL["primary"],
            label="Training R²", linewidth=2, ms=6)
    ax.fill_between(train_sizes_abs, train_mean-train_std, train_mean+train_std,
                    alpha=0.15, color=PAL["primary"])
    ax.plot(train_sizes_abs, val_mean, "s--", color=PAL["secondary"],
            label="CV Validation R²", linewidth=2, ms=6)
    ax.fill_between(train_sizes_abs, val_mean-val_std, val_mean+val_std,
                    alpha=0.15, color=PAL["secondary"])
    ax.set_xlabel("Training Samples", fontsize=9); ax.set_ylabel("R²", fontsize=9)
    ax.set_title(f"Learning Curve — R² ({lc_model_name})", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5); ax.set_ylim(-0.05, 1.05)
    for sp in ax.spines.values(): sp.set_alpha(0.3)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.15)

    # Bias-variance gap
    gap = train_mean - val_mean
    ax2 = axes[1]
    ax2.bar(train_sizes_abs, gap, width=train_sizes_abs[1]*0.4,
            color=[PAL["secondary"] if g > 0.05 else PAL["accent"] for g in gap],
            edgecolor="white", linewidth=0.8, alpha=0.85)
    ax2.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="Overfit threshold")
    ax2.set_xlabel("Training Samples", fontsize=9)
    ax2.set_ylabel("Train R² − Val R² (gap)", fontsize=9)
    ax2.set_title("Bias–Variance Gap", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5)
    for sp in ax2.spines.values(): sp.set_alpha(0.3)
    ax2.tick_params(labelsize=8); ax2.grid(alpha=0.15)

    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # ── Cross-validation score distribution
    st.markdown("**5-Fold Cross-Validation Scores — all models**")
    with st.spinner("Running cross-validation (cached after first run)…"):
        cv_results = get_cv_scores(trained_models, X_train, y_train, X_train_sc,
                                   test_size, random_seed, n_estimators, max_depth)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    positions = list(range(len(cv_results)))
    bps = ax.boxplot([s for s in cv_results.values()],
                     positions=positions, patch_artist=True, vert=True,
                     boxprops=dict(facecolor=PAL["accent"], color=PAL["primary"]),
                     medianprops=dict(color=PAL["secondary"], linewidth=2.2),
                     whiskerprops=dict(color=PAL["primary"]),
                     capprops=dict(color=PAL["primary"]),
                     flierprops=dict(marker=".", color=PAL["primary"], alpha=0.5))
    ax.set_xticks(positions)
    ax.set_xticklabels(list(cv_results.keys()), rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel("R² Score", fontsize=9)
    ax.set_title("5-Fold CV Distribution — all models", fontsize=10, fontweight="bold")
    for sp in ax.spines.values(): sp.set_alpha(0.3)
    ax.tick_params(labelsize=8); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch"); plt.close(fig)

    # CV summary table
    cv_summary = pd.DataFrame({
        "Model":   list(cv_results.keys()),
        "CV Mean R²": [f"{s.mean():.4f}" for s in cv_results.values()],
        "CV Std":  [f"±{s.std():.4f}"   for s in cv_results.values()],
        "CV Min":  [f"{s.min():.4f}"    for s in cv_results.values()],
        "CV Max":  [f"{s.max():.4f}"    for s in cv_results.values()],
    })
    st.dataframe(cv_summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════
# TAB 7 — SCENARIO COMPARE
# ═══════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">What-If Scenario Comparison</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Build up to <strong>4 scenarios</strong> and compare their predicted prices side-by-side.
    Great for exploring how location, income, or house size changes affect price.
    </div>""", unsafe_allow_html=True)

    sc_model_name = st.selectbox("Model for scenarios", list(trained_models.keys()),
                                 index=list(trained_models.keys()).index(best_name), key="sc_model")
    sc_model_obj  = trained_models[sc_model_name]
    sc_is_linear  = sc_model_name in ("Ridge Regression", "Lasso Regression")

    n_scenarios = st.slider("Number of scenarios", 2, 4, 3)

    scenario_inputs  = []
    scenario_labels  = []
    sc_cols = st.columns(n_scenarios)

    for i, col in enumerate(sc_cols[:n_scenarios]):
        with col:
            st.markdown(f"**Scenario {i+1}**")
            label = st.text_input("Label", value=f"Scenario {i+1}", key=f"sc_label_{i}")
            inc  = st.slider("Med. Income (×$10k)",  0.5, 15.0, [2.5, 5.0, 8.0, 12.0][i], 0.1, key=f"inc_{i}")
            age  = st.slider("House Age (yrs)",       1, 52,     [40,  20,  10,  5][i],    key=f"age_{i}")
            rm   = st.slider("Avg Rooms",             1.0, 15.0, [4.0, 5.5, 7.0, 9.0][i], 0.5, key=f"rm_{i}")
            bd   = st.slider("Avg Bedrooms",          1.0, 5.0,  [1.5, 2.0, 2.5, 3.0][i], 0.5, key=f"bd_{i}")
            pop  = st.slider("Population",            100, 5000, [2000,1200,800, 400][i],  key=f"pop_{i}")
            occ  = st.slider("Avg Occupants",         1.0, 6.0,  [3.5, 2.8, 2.2, 1.8][i], 0.1, key=f"occ_{i}")
            lat  = st.slider("Latitude",              32.5, 42.0,[33.9,37.5,34.0,37.7][i], 0.1, key=f"lat_{i}")
            lon  = st.slider("Longitude",             -124.5,-114.0,[-118.2,-122.0,-117.2,-122.4][i], 0.1, key=f"lon_{i}")

            rpp = rm / max(occ, 0.5)
            bratio = bd / max(rm, 1)
            ipp = inc / max(occ, 0.5)
            pdense = pop / max(occ, 0.5)

            raw = [inc, age, rm, bd, pop, occ, lat, lon, rpp, bratio, ipp, pdense]
            scenario_inputs.append(raw)
            scenario_labels.append(label)

    if st.button("Compare All Scenarios", key="compare_btn"):
        sc_predictions = []
        for raw in scenario_inputs:
            inp_df = pd.DataFrame([raw], columns=FEATURES)
            if sc_is_linear:
                inp_sc = scaler.transform(inp_df)
                p = sc_model_obj.predict(inp_sc)[0]
            else:
                p = sc_model_obj.predict(inp_df)[0]
            sc_predictions.append(p)

        # Result cards
        res_cols = st.columns(n_scenarios)
        for col, label, pred in zip(res_cols, scenario_labels, sc_predictions):
            with col:
                pct_rank = (df["Price"] < pred).mean() * 100
                st.markdown(f"""<div class="scenario-card">
                    <h4>{label}</h4>
                    <div class="sc-price">${pred:,.0f}</div>
                    <div style="font-size:0.78rem;color:#888;margin-top:0.2rem">
                        Percentile: {pct_rank:.0f}th
                    </div>
                </div>""", unsafe_allow_html=True)

        # Comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        bar_colors = [PAL["primary"], PAL["secondary"], PAL["accent"], "#E53935"][:n_scenarios]

        axes[0].bar(scenario_labels, [p/1000 for p in sc_predictions],
                    color=bar_colors, edgecolor="white", linewidth=0.8, width=0.55)
        for i, p in enumerate(sc_predictions):
            axes[0].text(i, p/1000 + 5, f"${p/1000:,.0f}k",
                         ha="center", fontsize=8.5, fontweight="600")
        axes[0].set_ylabel("Price ($000s)", fontsize=9)
        axes[0].set_title("Price Comparison", fontsize=10, fontweight="bold")
        axes[0].axhline(df["Price"].median()/1000, color="gray",
                        linestyle="--", linewidth=1.2, label=f"Dataset median")
        axes[0].legend(fontsize=8)
        for sp in axes[0].spines.values(): sp.set_alpha(0.3)
        axes[0].tick_params(labelsize=8.5)

        # Radar chart — feature comparison
        cats = ["MedIncome", "HouseAge", "AvgRooms", "AvgBedrooms", "Population", "AvgOccupants"]
        cat_idx = [FEATURES.index(c) for c in cats]
        N = len(cats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        ax_r = axes[1]
        ax_r.remove()
        ax_r = fig.add_subplot(1, 2, 2, polar=True)

        for i, (raw, label) in enumerate(zip(scenario_inputs, scenario_labels)):
            vals = [raw[j] for j in cat_idx]
            mins = [df[c].quantile(0.05) for c in cats]
            maxs = [df[c].quantile(0.95) for c in cats]
            norm = [(v - mn) / max(mx - mn, 1e-6) for v, mn, mx in zip(vals, mins, maxs)]
            norm += norm[:1]
            ax_r.plot(angles, norm, "o-", linewidth=2, label=label, color=bar_colors[i])
            ax_r.fill(angles, norm, alpha=0.08, color=bar_colors[i])

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(cats, fontsize=8)
        ax_r.set_title("Feature Radar (normalised)", fontsize=9, fontweight="bold", pad=20)
        ax_r.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.35, 1.15))
        ax_r.set_yticklabels([])

        fig.tight_layout()
        st.pyplot(fig, width="stretch"); plt.close(fig)

        # Difference table
        baseline = sc_predictions[0]
        diff_data = {
            "Scenario": scenario_labels,
            "Predicted Price": [f"${p:,.0f}" for p in sc_predictions],
            "vs Scenario 1":   [f"{'+' if p-baseline>=0 else ''}{(p-baseline)/1000:,.1f}k"
                                  for p in sc_predictions],
            "% Change":        [f"{'+' if p-baseline>=0 else ''}{(p-baseline)/baseline*100:.1f}%"
                                  for p in sc_predictions],
            "Percentile":      [f"{(df['Price'] < p).mean()*100:.0f}th" for p in sc_predictions],
        }
        st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════
# TAB 8 — EXPORT
# ═══════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("**<i class='bi bi-download'></i> Test Set Predictions**", unsafe_allow_html=True)
        export_df = X_test.copy()
        export_df["Actual_Price"]    = y_test.values
        export_df["Predicted_Price"] = best_preds
        export_df["Residual"]        = best_preds - y_test.values
        export_df["Abs_Error"]       = np.abs(best_preds - y_test.values)
        export_df["Model"]           = best_name

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download predictions CSV ({len(export_df):,} rows)",
            data=csv_bytes,
            file_name="house_price_predictions.csv",
            mime="text/csv",
        )

        st.markdown("**<i class='bi bi-download'></i> Full Dataset with Engineered Features**", unsafe_allow_html=True)
        full_csv = df[FEATURES + ["Price"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download full dataset CSV ({len(df):,} rows)",
            data=full_csv,
            file_name="california_housing_engineered.csv",
            mime="text/csv",
        )

    with e2:
        st.markdown("**<i class='bi bi-file-earmark-text'></i> Model Performance Report**", unsafe_allow_html=True)
        report_lines = ["# House Price Predictor — Model Report\n",
                        f"**Generated:** Phase 2 Final\n\n",
                        f"## Settings\n- Test size: {test_size}\n- Random seed: {random_seed}\n",
                        f"- RF n_estimators: {n_estimators}\n- RF max_depth: {max_depth}\n\n",
                        "## Model Performance\n",
                        "| Model | R² | MAE | RMSE |\n|---|---|---|---|\n"]
        for name, res in sorted(model_results.items(), key=lambda x: -x[1]["r2"]):
            star = " [best]" if name == best_name else ""
            report_lines.append(f"| {name}{star} | {res['r2']:.4f} | ${res['mae']:,.0f} | ${res['rmse']:,.0f} |\n")
        report_lines.append(f"\n## Best Model: {best_name}\n")
        report_lines.append(f"- R²: {r2:.4f}\n- MAE: ${mae:,.0f}\n- RMSE: ${rmse:,.0f}\n")

        report_md = "".join(report_lines)
        st.download_button(
            label="Download report (.md)",
            data=report_md.encode("utf-8"),
            file_name="model_report.md",
            mime="text/markdown",
        )

        st.markdown("**Preview:**")
        st.code(report_md[:600] + "\n…", language="markdown")

    # Dataset summary
    st.markdown("**Dataset Summary Statistics**")
    st.dataframe(df[FEATURES + ["Price"]].describe().round(3), use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 9 — RAW DATA (optional)
# ═══════════════════════════════════════════════════
if show_raw:
    with tabs[8]:
        st.markdown('<div class="section-title">Raw Dataset Explorer</div>', unsafe_allow_html=True)
        rd1, rd2, rd3 = st.columns(3)
        with rd1:
            price_range = st.slider("Filter by Price ($)",
                                    int(df["Price"].min()), int(df["Price"].max()),
                                    (int(df["Price"].quantile(0.1)), int(df["Price"].quantile(0.9))),
                                    step=5000)
        with rd2:
            income_range = st.slider("Filter by MedIncome",
                                     float(df["MedIncome"].min()), float(df["MedIncome"].max()),
                                     (float(df["MedIncome"].quantile(0.1)),
                                      float(df["MedIncome"].quantile(0.9))), step=0.1)
        with rd3:
            n_show = st.slider("Rows to display", 50, 2000, 200, 50)

        filtered = df[
            (df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1]) &
            (df["MedIncome"] >= income_range[0]) & (df["MedIncome"] <= income_range[1])
        ].head(n_show)

        st.markdown(f"Showing **{len(filtered):,}** rows matching filters")
        st.dataframe(filtered[FEATURES + ["Price"]].round(3),
                     use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:0.78rem;color:#aaa;">'
    'House Price Predictor Pro &nbsp;·&nbsp; California Housing Dataset &nbsp;·&nbsp; April 2026</p>',
    unsafe_allow_html=True
)