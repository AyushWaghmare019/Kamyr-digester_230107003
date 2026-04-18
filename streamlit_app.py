"""
================================================================================
 Kamyr Digester Kappa Soft-Sensor — Streamlit Dashboard
================================================================================
 Run locally:
     pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib plotly
     streamlit run streamlit_app.py

 Expected layout:
     ./streamlit_app.py
     ./artifacts/           <-- produced by kamyr_pipeline.py
         scaler.joblib
         model_*.joblib
         pca.joblib
         metadata.joblib
         df_clean.csv
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import subprocess
subprocess.run(["pip", "install", "statsmodels"])
# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Kamyr Digester Kappa Soft Sensor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------
ARTDIR = "artifacts"

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{ARTDIR}/scaler.joblib")
    pca    = joblib.load(f"{ARTDIR}/pca.joblib")
    meta   = joblib.load(f"{ARTDIR}/metadata.joblib")
    models = {}
    for f in os.listdir(ARTDIR):
        if f.startswith("model_") and f.endswith(".joblib"):
            name = f.replace("model_", "").replace(".joblib", "")
            models[name] = joblib.load(f"{ARTDIR}/{f}")
    return scaler, pca, meta, models

@st.cache_data
def load_dataframe():
    return pd.read_csv(f"{ARTDIR}/df_clean.csv")

# Physical meanings of each process tag -- used for tooltips
VAR_INFO = {
    "ChipRate"       : "Wood chip feed rate to the digester (throughput)",
    "BF-CMratio"     : "Blow-flow to chip-mass ratio (dilution / consistency)",
    "BlowFlow"       : "Pulp flow leaving the digester blow line",
    "ChipLevel4"     : "Chip column height in the digester",
    "T-upperExt-2"   : "Upper extraction zone temperature (lag 2 h)",
    "T-lowerExt-2"   : "Lower extraction (wash) zone temperature (lag 2 h)",
    "UCZAA"          : "Upper cook zone active alkali concentration",
    "WhiteFlow-4"    : "White liquor flow, cook zone (lag 4 h)",
    "AA-Wood-4"      : "Active-alkali-to-wood charge ratio (lag 4 h)",
    "ChipMoisture-4" : "Chip moisture at feed (lag 4 h)",
    "SteamFlow-4"    : "Cooking steam flow (lag 4 h)",
    "Lower-HeatT-3"  : "Lower heater outlet temperature (lag 3 h)",
    "Upper-HeatT-3"  : "Upper heater outlet temperature (lag 3 h)",
    "ChipMass-4"     : "Chip mass throughput (lag 4 h)",
    "WeakLiquorF"    : "Weak liquor (black liquor) flow",
    "BlackFlow-2"    : "Black liquor flow at extraction (lag 2 h)",
    "WeakWashF"      : "Weak wash flow to wash zone",
    "SteamHeatF-3"   : "Steam flow to heaters (lag 3 h)",
    "T-Top-Chips-4"  : "Top-of-digester chip temperature (lag 4 h)",
    "Hfactor_proxy"  : "H-factor proxy: exp((T_upperExt − 100) / 15)",
    "AA_charge_rate" : "Active alkali mass rate = AA-Wood × ChipRate",
}


# ===================================================================
# HEADER
# ===================================================================
st.title("🧪 Kamyr Digester — Kappa Number Soft Sensor")
st.markdown(
    """
    **Course:** CL653 — Applications of AI/ML for Chemical Engineering &nbsp;|&nbsp;
    **Dataset:** OpenMV.net Kamyr digester (Dayal *et al.*, 1994) &nbsp;|&nbsp;
    **Target:** residual lignin (Kappa number) at the blow line.

    The Kappa number quantifies un-reacted lignin in kraft pulp. It is normally
    measured off-line every 1–4 h in a laboratory; this soft sensor infers Kappa
    continuously from easily-measured process variables so operators can react
    to disturbances in **minutes instead of hours**.
    """
)

# Load everything once
scaler, pca, meta, models = load_artifacts()
df_full = load_dataframe()

feature_names = meta["feature_names"]
target        = meta["target"]
best_name     = meta["best_model"]

# ===================================================================
# SIDEBAR — process-variable inputs
# ===================================================================
st.sidebar.header("🎛️ Process Inputs")
st.sidebar.caption("Adjust to match current plant conditions.")

model_choice = st.sidebar.selectbox(
    "Predictive model",
    options=list(models.keys()),
    index=list(models.keys()).index(best_name) if best_name in models else 0,
    help=f"Default is the best performer on the held-out test set: **{best_name}**",
)

if st.sidebar.button("🔄 Reset to median operating point"):
    st.session_state.clear()

stats = meta["feature_stats"]

# Group the variables for a cleaner sidebar
GROUPS = {
    "Feed & throughput": ["ChipRate", "ChipMass-4", "ChipMoisture-4",
                          "ChipLevel4", "T-Top-Chips-4"],
    "Cooking chemicals": ["UCZAA", "WhiteFlow-4", "AA-Wood-4",
                          "AA_charge_rate"],
    "Temperature / heat": ["T-upperExt-2", "T-lowerExt-2",
                           "Upper-HeatT-3", "Lower-HeatT-3",
                           "SteamFlow-4", "SteamHeatF-3", "Hfactor_proxy"],
    "Liquor flows": ["BlowFlow", "BF-CMratio", "WeakLiquorF",
                     "BlackFlow-2", "WeakWashF"],
}

user_inputs = {}
for group, cols in GROUPS.items():
    with st.sidebar.expander(group, expanded=(group == "Cooking chemicals")):
        for col in cols:
            if col not in feature_names:
                continue
            lo   = float(stats[col]["min"])
            hi   = float(stats[col]["max"])
            med  = float(stats[col]["median"])
            span = hi - lo if hi > lo else 1.0
            step = max(span / 200.0, 0.001)
            user_inputs[col] = st.slider(
                col,
                min_value=float(round(lo - 0.05 * span, 3)),
                max_value=float(round(hi + 0.05 * span, 3)),
                value=float(round(med, 3)),
                step=float(round(step, 4)),
                help=VAR_INFO.get(col, ""),
            )

# Build input vector in the exact training order (as a DataFrame so scaler
# keeps its feature names and doesn't emit a warning)
x_vec = pd.DataFrame([[user_inputs[c] for c in feature_names]],
                     columns=feature_names)
x_scaled = scaler.transform(x_vec)


# ===================================================================
# MAIN — tabbed layout
# ===================================================================
tab_pred, tab_eda, tab_models, tab_feat, tab_pca, tab_about = st.tabs([
    "🎯 Live Prediction",
    "📊 EDA",
    "🏆 Model Comparison",
    "🔍 Feature Importance",
    "🧭 PCA & t-SNE",
    "ℹ️ About",
])


# -------------------------------------------------------------------
# TAB 1 — Live Prediction
# -------------------------------------------------------------------
with tab_pred:
    st.subheader("Real-time Kappa prediction")

    chosen_model = models[model_choice]
    y_pred = float(chosen_model.predict(x_scaled).ravel()[0])

    # Kraft-pulp rule-of-thumb Kappa bands for softwood bleachable-grade pulp
    if y_pred < 18:
        band, colour, advice = "Over-cooked", "#d62728", (
            "Kappa below soft-wood bleachable target (≈ 25–32). Pulp yield will "
            "drop and fibre strength is compromised. Reduce cooking temperature "
            "or white-liquor flow."
        )
    elif y_pred <= 32:
        band, colour, advice = "On-target", "#2ca02c", (
            "Kappa is within the typical bleachable-grade softwood window. "
            "Maintain current operation."
        )
    else:
        band, colour, advice = "Under-cooked", "#ff7f0e", (
            "Kappa high — insufficient delignification. Increase effective alkali, "
            "raise upper-extraction temperature, or extend residence time."
        )

    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    c1.metric("Predicted Kappa", f"{y_pred:.2f}")
    c2.metric("Status", band)
    c3.markdown(
        f"<div style='background-color:{colour}22;border-left:4px solid {colour};"
        f"padding:10px;border-radius:4px'><b>Operator advice:</b> {advice}</div>",
        unsafe_allow_html=True,
    )

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=y_pred,
        title={"text": "Kappa number"},
        gauge={
            "axis": {"range": [10, 40]},
            "bar":  {"color": colour},
            "steps": [
                {"range": [10, 18], "color": "#f7b6b6"},
                {"range": [18, 32], "color": "#b6e2b6"},
                {"range": [32, 40], "color": "#fdd7a1"},
            ],
        },
    ))
    gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(gauge, use_container_width=True)

    # Show all model predictions at the same operating point
    st.markdown("#### Cross-model comparison at this operating point")
    cross = {name: float(m.predict(x_scaled).ravel()[0]) for name, m in models.items()}
    cross_df = pd.DataFrame(
        {"Model": list(cross.keys()), "Predicted Kappa": list(cross.values())}
    ).sort_values("Predicted Kappa")
    st.dataframe(cross_df.style.format({"Predicted Kappa": "{:.2f}"}),
                 use_container_width=True, hide_index=True)

    # What-if: sweep the single most important variable
    st.markdown("#### What-if analysis")
    top_vars = sorted(meta["perm_imp"].items(), key=lambda kv: kv[1], reverse=True)[:8]
    sweep_var = st.selectbox(
        "Variable to sweep (others held at sidebar values)",
        [v for v, _ in top_vars],
    )
    lo = float(stats[sweep_var]["min"])
    hi = float(stats[sweep_var]["max"])
    sweep = np.linspace(lo, hi, 60)
    rows = pd.DataFrame(np.tile(x_vec.values, (len(sweep), 1)),
                        columns=feature_names)
    rows[sweep_var] = sweep
    sweep_preds = chosen_model.predict(scaler.transform(rows)).ravel()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep, y=sweep_preds, mode="lines",
                             line=dict(width=3, color="#2E75B6"),
                             name=f"{model_choice} prediction"))
    fig.add_trace(go.Scatter(x=[user_inputs[sweep_var]], y=[y_pred],
                             mode="markers",
                             marker=dict(size=12, color="crimson", symbol="x"),
                             name="current operating point"))
    fig.add_hline(y=32, line_dash="dash", line_color="orange",
                  annotation_text="Under-cooked limit", annotation_position="right")
    fig.add_hline(y=18, line_dash="dash", line_color="red",
                  annotation_text="Over-cooked limit", annotation_position="right")
    fig.update_layout(
        xaxis_title=sweep_var, yaxis_title="Predicted Kappa",
        height=430, margin=dict(l=20, r=20, t=40, b=20),
        title=f"Sensitivity of Kappa to {sweep_var}",
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# TAB 2 — EDA
# -------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory data analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df_full)}")
    c2.metric("Process variables", f"{len(feature_names)}")
    c3.metric("Kappa mean", f"{df_full[target].mean():.2f}")
    c4.metric("Kappa std",  f"{df_full[target].std():.2f}")

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Kappa distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.histplot(df_full[target], kde=True, ax=ax, color="teal")
        ax.set_xlabel("Kappa number")
        st.pyplot(fig, clear_figure=True)

    with colR:
        st.markdown("**Kappa time series (hour-by-hour)**")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(df_full.index, df_full[target], color="darkred", lw=1)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Kappa")
        st.pyplot(fig, clear_figure=True)

    st.markdown("**Correlation heatmap (all variables vs Kappa)**")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df_full.corr(), cmap="RdBu_r", center=0, square=True,
                annot=False, cbar_kws={"shrink": 0.6}, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.markdown("**Top 10 |corr| with Kappa**")
    corr_ser = (pd.Series(meta["corr_with_target"])
                  .sort_values(ascending=False).head(10))
    fig = px.bar(corr_ser.reset_index(),
                 x=0, y="index", orientation="h",
                 labels={"index": "Variable", "0": "|r| with Kappa"})
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Scatter: pick any variable vs Kappa**")
    choice = st.selectbox("Variable", feature_names, index=feature_names.index("WhiteFlow-4")
                          if "WhiteFlow-4" in feature_names else 0)
try:
    fig = px.scatter(
        df_full,
        x=choice,
        y=target,
        trendline="ols",
        opacity=0.65,
        height=430
    )
except ModuleNotFoundError:
    fig = px.scatter(
        df_full,
        x=choice,
        y=target,
        opacity=0.65,
        height=430
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# TAB 3 — Model Comparison
# -------------------------------------------------------------------
with tab_models:
    st.subheader("Model comparison (held-out test set)")

    res_df = pd.DataFrame(meta["results_table"])
    st.dataframe(
        res_df.style
              .format({"R2_train": "{:.3f}", "R2_test": "{:.3f}",
                       "RMSE_test": "{:.3f}", "MAE_test": "{:.3f}"})
              .background_gradient(subset=["R2_test"], cmap="Greens")
              .background_gradient(subset=["RMSE_test"], cmap="Reds_r"),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(res_df, x="Model", y="R2_test",
                     title="Test-set R²", color="R2_test",
                     color_continuous_scale="Viridis")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(res_df, x="Model", y="RMSE_test",
                     title="Test-set RMSE", color="RMSE_test",
                     color_continuous_scale="Reds")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Temporal robustness — TimeSeriesSplit CV of best model")
    st.caption(
        "A chronological split stresses the model against future operating regimes. "
        "Negative R² in some folds reflects genuine process drift over the 301-hour "
        "record — a normal feature of industrial data and a motivator for periodic "
        "model re-training."
    )
    ts = meta["ts_cv_scores"]
    ts_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(ts))],
                          "R²": ts})
    fig = px.bar(ts_df, x="Fold", y="R²",
                 title=f"{best_name}: TimeSeriesSplit CV (mean = {meta['ts_cv_mean']:.3f})",
                 color="R²", color_continuous_scale="RdYlGn")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# TAB 4 — Feature importance
# -------------------------------------------------------------------
with tab_feat:
    st.subheader("What drives Kappa?")
    st.caption(
        "PLS VIP, tree-based impurity importances, and permutation importance are "
        "computed on different model families; convergent rankings across methods "
        "give a more trustworthy causal story."
    )

    imp_sources = {
        "PLS VIP":                 meta["vip_scores"],
        "Random Forest":           meta["rf_imp"],
        "Gradient Boosting":       meta["gb_imp"],
        f"Permutation ({best_name})": meta["perm_imp"],
    }

    cols = st.columns(2)
    for i, (label, d) in enumerate(imp_sources.items()):
        with cols[i % 2]:
            s = pd.Series(d).sort_values(ascending=False).head(10)
            fig = px.bar(s.reset_index(), x=0, y="index", orientation="h",
                         labels={"index": "Variable", "0": "Importance"},
                         title=label,
                         color=0, color_continuous_scale="Viridis")
            fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"},
                              showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Consensus top drivers")
    ranks = pd.DataFrame(imp_sources).rank(ascending=False)
    consensus = ranks.mean(axis=1).sort_values().head(10)
    cdf = consensus.reset_index()
    cdf.columns = ["Variable", "Average rank (lower = more important)"]
    st.dataframe(cdf, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# TAB 5 — PCA & t-SNE
# -------------------------------------------------------------------
with tab_pca:
    st.subheader("Dimensionality reduction")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PCA scree plot**")
        evr  = meta["pca_evr"]
        cevr = meta["pca_cum_evr"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(evr))], y=evr,
                             name="individual", marker_color="steelblue"))
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cevr))], y=cevr,
                                 name="cumulative", mode="lines+markers",
                                 line=dict(color="orange")))
        fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                      annotation_text="90%")
        fig.update_layout(height=400, yaxis_title="Explained variance ratio")
        st.plotly_chart(fig, use_container_width=True)

        n90 = next((i + 1 for i, v in enumerate(cevr) if v >= 0.9), len(cevr))
        st.info(f"**{n90} PCs** capture ≥ 90 % of process variance "
                f"(original dimensionality = {len(feature_names)}).")

    with c2:
        st.markdown("**PC1–PC2 score plot (colour = Kappa)**")
        X_all = df_full[feature_names].values
        scores = pca.transform(scaler.transform(X_all))[:, :2]
        fig = px.scatter(x=scores[:, 0], y=scores[:, 1],
                         color=df_full[target],
                         color_continuous_scale="Viridis",
                         labels={"x": "PC1", "y": "PC2", "color": "Kappa"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### t-SNE embedding (2-D, nonlinear)")
    if os.path.exists("outputs/06b_tsne.png"):
        st.image("outputs/06b_tsne.png", use_column_width=True)
    else:
        st.warning("t-SNE image not found — re-run kamyr_pipeline.py to regenerate.")


# -------------------------------------------------------------------
# TAB 6 — About
# -------------------------------------------------------------------
with tab_about:
    st.subheader("About this soft sensor")
    st.markdown(
        f"""
**Pipeline stages**

1. Data ingested from `kamyr-digester.csv` — 301 hourly observations, 22 raw variables.
2. Columns with > 45 % missing values (`AAWhiteSt-4`, `SulphidityL-4`) were dropped;
   remainder imputed with the median.
3. Univariate robust cleaning via the **Hampel filter** (rolling-median MAD rule).
4. Multivariate outlier screening via **Isolation Forest** (contamination = 0.05).
5. Two physics-motivated engineered features:
   * **H-factor proxy** — Arrhenius-like exponential temperature integral.
   * **AA charge rate** — alkali-to-wood ratio × chip feed rate.
6. `StandardScaler` → PCA (scree + score plots) and t-SNE for visualisation.
7. Five models trained and tuned with `GridSearchCV` on 5-fold CV:
   Linear Regression, PLS, SVR (RBF), Random Forest, Gradient Boosting.
8. Evaluation on a held-out 20 % test set; time-series robustness quantified with
   `TimeSeriesSplit`.

**Best model:** `{best_name}` — test R² = {pd.DataFrame(meta['results_table']).set_index('Model').loc[best_name, 'R2_test']:.3f}

**Industrial framing**

* Kappa is measured off-line every 1–4 h; a continuous inferential estimate
  closes the loop between control action and response.
* The top drivers (consistent across methods) — BF-CMratio, WhiteFlow-4,
  SteamFlow-4, SteamHeatF-3, BlackFlow-2 — are exactly the variables kraft
  cooking theory predicts should dominate delignification rate.
* Future work: online retraining on a sliding window, hybrid first-principles /
  ML residual modelling, and digital-twin integration.
        """
    )
    st.caption("© 2026 —  Chemical Engineering Project (CL653).")
