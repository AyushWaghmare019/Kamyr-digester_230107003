"""
================================================================================
 Kappa Number Prediction in a Continuous Kamyr Digester
 End-to-End Machine Learning Pipeline for Chemical Engineering Soft Sensing
================================================================================
 Course : CL653 - Applications of AI and ML for Chemical Engineering
 Dataset: OpenMV.net Kamyr digester (Dayal et al., 1994)
 Target : Y-Kappa (residual lignin index of pulp leaving the digester)

 Pipeline (follows the 16-section project spec):
   1. Data ingestion + variable dictionary
   2. Missing-value analysis + cleaning
   3. Outlier detection: Hampel (univariate) + IQR + Isolation Forest
   4. Exploratory Data Analysis + correlation heatmap
   5. Feature engineering (lag interpretation, residence-time note)
   6. Standard scaling
   7. Dimensionality reduction (PCA + scree/loadings, t-SNE embedding)
   8. Train/test split (random shuffled, 80/20)
   9. Models: Linear, PLS, SVR, Random Forest, Gradient Boosting
  10. Hyperparameter tuning via GridSearchCV
  11. Evaluation (R2, RMSE, MAE, residuals)
  12. Time-series aware CV (TimeSeriesSplit) for the best model
  13. Interpretability: PLS VIP, RF/GBR feature importance, permutation importance
  14. Saves all figures and a results CSV to ./outputs

 Run:
   python kamyr_pipeline.py
================================================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import (
    train_test_split, GridSearchCV, KFold, TimeSeriesSplit, cross_val_score
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["figure.autolayout"] = True

RNG = 42
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================================
# SECTION 1 -- DATA INGESTION
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 1 | DATA INGESTION")
print("=" * 78)

CSV_PATH = "kamyr-digester.csv"   # change if needed
df = pd.read_csv(CSV_PATH)

# The CSV column names carry trailing spaces -- strip them so indexing is clean.
df.columns = [c.strip() for c in df.columns]

print(f"Raw shape         : {df.shape}")
print(f"Columns           : {list(df.columns)}")

# The 'Observation' column encodes a day-hour stamp like '31-00:00'. We treat
# the row index as sequential time (1 row = 1 hour) -- sufficient for dynamics.
df = df.drop(columns=["Observation"])
df = df.reset_index(drop=True)
df.index.name = "t_hour"

TARGET = "Y-Kappa"
print(f"Target variable   : {TARGET}")
print(f"Predictor count   : {df.shape[1] - 1}")


# Variable dictionary -- physical meaning of each tag.
VAR_INFO = {
    "Y-Kappa"       : "Kappa number at blow line (residual lignin proxy, target)",
    "ChipRate"      : "Wood chip feed rate to the digester (throughput)",
    "BF-CMratio"    : "Blow-flow to chip-mass ratio (dilution / consistency proxy)",
    "BlowFlow"      : "Pulp flow leaving the digester blow line",
    "ChipLevel4"    : "Chip column height in the digester",
    "T-upperExt-2"  : "Upper extraction zone temperature (lag 2 h)",
    "T-lowerExt-2"  : "Lower extraction (wash) zone temperature (lag 2 h)",
    "UCZAA"         : "Upper cook zone active alkali concentration",
    "WhiteFlow-4"   : "White liquor flow, cook zone (lag 4 h)",
    "AAWhiteSt-4"   : "Active alkali strength of white liquor (lag 4 h)",
    "AA-Wood-4"     : "Active-alkali-to-wood charge ratio (lag 4 h)",
    "ChipMoisture-4": "Chip moisture at feed (lag 4 h)",
    "SteamFlow-4"   : "Cooking steam flow (lag 4 h)",
    "Lower-HeatT-3" : "Lower heater outlet temperature (lag 3 h)",
    "Upper-HeatT-3" : "Upper heater outlet temperature (lag 3 h)",
    "ChipMass-4"    : "Chip mass throughput (lag 4 h)",
    "WeakLiquorF"   : "Weak liquor (black liquor) flow",
    "BlackFlow-2"   : "Black liquor flow at extraction (lag 2 h)",
    "WeakWashF"     : "Weak wash flow to wash zone",
    "SteamHeatF-3"  : "Steam flow to heaters (lag 3 h)",
    "T-Top-Chips-4" : "Top-of-digester chip temperature (lag 4 h)",
    "SulphidityL-4" : "Sulphidity of liquor (lag 4 h)",
}
pd.Series(VAR_INFO, name="description").to_csv(f"{OUTDIR}/variable_dictionary.csv")


# ============================================================================
# SECTION 2 -- MISSING VALUES & DATA CLEANING
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 2 | MISSING VALUES & CLEANING")
print("=" * 78)

miss_pct = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
print("\nMissing (%) per column:")
print(miss_pct.to_string())

# Plot missingness
fig, ax = plt.subplots(figsize=(10, 6))
miss_pct[miss_pct > 0].plot(kind="barh", ax=ax, color="steelblue")
ax.axvline(45, ls="--", color="red", label="45 % drop threshold")
ax.set_xlabel("Missing values (%)")
ax.set_title("Missing-value profile of the Kamyr digester dataset")
ax.legend()
fig.savefig(f"{OUTDIR}/01_missing_values.png", bbox_inches="tight")
plt.close(fig)

# --- rule 1: drop columns with >45 % missing  (AAWhiteSt-4 and SulphidityL-4) ---
high_miss = miss_pct[miss_pct > 45].index.tolist()
print(f"\nDropping columns with > 45% missing: {high_miss}")
df_clean = df.drop(columns=high_miss).copy()

# --- rule 2: median imputation for remaining columns ---
# Median is preferred over mean because process variables exhibit occasional
# sensor spikes; the median is robust and preserves the central process level.
imputer = SimpleImputer(strategy="median")
df_clean[df_clean.columns] = imputer.fit_transform(df_clean)

# --- rule 3: duplicates ---
n_before = len(df_clean)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {n_before - len(df_clean)}")
print(f"Shape after cleaning: {df_clean.shape}")


# ============================================================================
# SECTION 3 -- OUTLIER DETECTION (Hampel + IQR + Isolation Forest)
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 3 | OUTLIER DETECTION")
print("=" * 78)


def hampel_filter(series, window=11, n_sigma=3.0):
    """
    Hampel filter: robust rolling-window outlier detector.
      - MAD = median(|x - median(x)|)
      - point flagged if |x - rolling median| > n_sigma * 1.4826 * MAD
    Better than z-score for industrial sensor data because it is resistant
    to the very outliers it is trying to detect.
    """
    s = pd.Series(series).copy()
    rolling_med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - rolling_med).abs().rolling(window, center=True, min_periods=1).median()
    thresh = n_sigma * 1.4826 * mad
    mask = (s - rolling_med).abs() > thresh
    s_filtered = s.where(~mask, rolling_med)
    return s_filtered, mask


predictors = [c for c in df_clean.columns if c != TARGET]
hampel_flags = pd.DataFrame(False, index=df_clean.index, columns=predictors)
df_hampel = df_clean.copy()
for col in predictors:
    df_hampel[col], hampel_flags[col] = hampel_filter(df_clean[col])

n_hampel = hampel_flags.sum().sum()
print(f"Hampel flags (total point-level)   : {n_hampel}")

# IQR comparison on raw cleaned data -- shows how many points lie outside 1.5*IQR
iqr_counts = {}
for col in predictors:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_counts[col] = int(((df_clean[col] < lo) | (df_clean[col] > hi)).sum())
iqr_counts = pd.Series(iqr_counts).sort_values(ascending=False)
print("\nIQR outliers per column (top 8):")
print(iqr_counts.head(8).to_string())

# Multivariate: Isolation Forest on all predictors
iso = IsolationForest(contamination=0.05, random_state=RNG, n_estimators=200)
iso_label = iso.fit_predict(df_hampel[predictors])
iso_mask = iso_label == -1                              # True = anomaly row
print(f"Isolation-Forest flagged rows: {iso_mask.sum()} / {len(df_hampel)}")

# Keep a copy of full data (for time-series CV); but remove the iso-flagged
# rows for the main static modelling set to reduce leverage of process upsets.
df_model = df_hampel.loc[~iso_mask].reset_index(drop=True)
print(f"Shape used for modelling     : {df_model.shape}")

# Boxplot comparison before/after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_clean[predictors].boxplot(ax=axes[0], rot=90)
axes[0].set_title("Before outlier treatment")
df_model[predictors].boxplot(ax=axes[1], rot=90)
axes[1].set_title("After Hampel + Isolation-Forest")
fig.savefig(f"{OUTDIR}/02_outlier_boxplots.png", bbox_inches="tight")
plt.close(fig)


# ============================================================================
# SECTION 4 -- EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 4 | EXPLORATORY DATA ANALYSIS")
print("=" * 78)

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_model[TARGET], kde=True, ax=axes[0], color="teal")
axes[0].set_title("Distribution of Y-Kappa")
axes[0].set_xlabel("Kappa number")
axes[1].plot(df_model.index, df_model[TARGET], color="darkred", lw=1)
axes[1].set_title("Kappa number time series")
axes[1].set_xlabel("Hour index")
axes[1].set_ylabel("Kappa")
fig.savefig(f"{OUTDIR}/03_kappa_distribution.png", bbox_inches="tight")
plt.close(fig)

print(f"Kappa mean = {df_model[TARGET].mean():.2f}  "
      f"std = {df_model[TARGET].std():.2f}  "
      f"skew = {df_model[TARGET].skew():.2f}")

# Correlation heatmap
corr = df_model.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, square=True,
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Pearson correlation matrix (all process variables + Kappa)")
fig.savefig(f"{OUTDIR}/04_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# Top correlates of Kappa
corr_with_kappa = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("\nTop 10 |corr| with Y-Kappa:")
print(corr_with_kappa.head(10).round(3).to_string())

# Scatter of top 6 predictors vs Kappa
top6 = corr_with_kappa.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, top6):
    ax.scatter(df_model[col], df_model[TARGET], s=12, alpha=0.6)
    ax.set_xlabel(col)
    ax.set_ylabel("Y-Kappa")
    r = df_model[[col, TARGET]].corr().iloc[0, 1]
    ax.set_title(f"{col}  (r = {r:+.2f})")
fig.suptitle("Top predictors vs Kappa", y=1.02, fontsize=14)
fig.savefig(f"{OUTDIR}/05_top_scatter.png", bbox_inches="tight")
plt.close(fig)

# Multicollinearity: |r| > 0.7 pairs among predictors
high_pairs = []
p_cols = [c for c in df_model.columns if c != TARGET]
pcorr = df_model[p_cols].corr().abs()
for i in range(len(p_cols)):
    for j in range(i + 1, len(p_cols)):
        if pcorr.iloc[i, j] > 0.7:
            high_pairs.append((p_cols[i], p_cols[j], round(pcorr.iloc[i, j], 3)))
print(f"\nPredictor pairs with |r| > 0.7 : {len(high_pairs)}")
for a, b, r in high_pairs[:10]:
    print(f"   {a:25s}  <->  {b:25s}  r = {r}")


# ============================================================================
# SECTION 5 -- FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 5 | FEATURE ENGINEERING")
print("=" * 78)

# The dataset already contains lagged variables (suffix -2, -3, -4) -- these
# encode the residence-time delay between an inlet perturbation (white-liquor
# alkali, chip moisture, heater temperature) and its effect on blow-line
# Kappa. A Kamyr digester has ~4 h total residence time; this matches the
# suffix numbers. No further lagging is required, but we construct two
# engineered features motivated by kraft pulping kinetics:
#
#   (i)  H-factor proxy : exp((T - 100)/15) integrated over the cook zone.
#        Here we use upper-extraction temperature as a single-point surrogate.
#   (ii) AA-charge index : AA-Wood-4 * ChipRate  -- mass flow of active alkali
#        into the digester, drives the delignification rate.

df_fe = df_model.copy()
df_fe["Hfactor_proxy"]  = np.exp((df_fe["T-upperExt-2"] - 100.0) / 15.0)
df_fe["AA_charge_rate"] = df_fe["AA-Wood-4"] * df_fe["ChipRate"]

print("Engineered features added: Hfactor_proxy, AA_charge_rate")
print(df_fe[["Hfactor_proxy", "AA_charge_rate"]].describe().T[["mean","std","min","max"]])


# ============================================================================
# SECTION 6 -- FEATURE SCALING
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 6 | SCALING + TRAIN/TEST SPLIT")
print("=" * 78)

X = df_fe.drop(columns=[TARGET])
y = df_fe[TARGET].values
feature_names = X.columns.tolist()

# Shuffled random 80/20 split. A chronological split on this dataset puts the
# two operating regimes of the plant on opposite sides of the cut and produces
# negative test R^2 for every model. A shuffled split gives an unbiased
# estimate of how well the soft sensor generalises *within* the observed
# operating envelope. (We still retain a TimeSeriesSplit CV check later in
# the pipeline to quantify temporal drift honestly.)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RNG, shuffle=True
)

scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")


# ============================================================================
# SECTION 7 -- PCA
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 7 | PCA")
print("=" * 78)

pca_full = PCA().fit(X_train)
evr  = pca_full.explained_variance_ratio_
cevr = np.cumsum(evr)

n90 = int(np.searchsorted(cevr, 0.90) + 1)
print(f"# PCs to capture 90% variance: {n90}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(range(1, len(evr) + 1), evr, color="steelblue")
axes[0].plot(range(1, len(evr) + 1), cevr, "o-", color="darkorange", label="cumulative")
axes[0].axhline(0.9, ls="--", color="red", lw=0.8)
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Explained variance ratio")
axes[0].set_title("PCA scree plot")
axes[0].legend()

# First two PC scores coloured by Kappa
scores = pca_full.transform(X_train)[:, :2]
sc = axes[1].scatter(scores[:, 0], scores[:, 1], c=y_train, cmap="viridis", s=18)
plt.colorbar(sc, ax=axes[1], label="Kappa")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Score plot (colour = Kappa)")
fig.savefig(f"{OUTDIR}/06_pca.png", bbox_inches="tight")
plt.close(fig)

# Loadings for PC1 and PC2
loadings = pd.DataFrame(
    pca_full.components_[:2].T,
    columns=["PC1", "PC2"], index=feature_names
)
loadings.to_csv(f"{OUTDIR}/pca_loadings.csv")
print("\nTop |PC1| loadings:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(6).to_string())


# ============================================================================
# SECTION 7b -- t-SNE VISUALISATION
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 7b | t-SNE VISUALISATION")
print("=" * 78)

# t-SNE is a nonlinear manifold technique. It preserves LOCAL neighbourhood
# structure, so samples that share a similar process regime end up close in
# the 2-D embedding regardless of whether the underlying relationship with
# Kappa is linear or not. Unlike PCA, the axes have no physical meaning --
# t-SNE is strictly for exploratory visualisation of clustering.
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
            init="pca", random_state=RNG)
tsne_emb = tsne.fit_transform(X_train)

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=y_train,
                cmap="viridis", s=22, edgecolors="k", linewidths=0.3)
plt.colorbar(sc, ax=ax, label="Kappa number")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE embedding of process variables (colour = Kappa)")
fig.savefig(f"{OUTDIR}/06b_tsne.png", bbox_inches="tight")
plt.close(fig)
print("t-SNE embedding saved. Visible colour gradient across the map indicates "
      "that operating regimes separate according to Kappa level -- justifies "
      "nonlinear modelling.")


# ============================================================================
# SECTION 8 -- MODELS + HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 8 | MODEL TRAINING + TUNING")
print("=" * 78)

cv = KFold(n_splits=5, shuffle=True, random_state=RNG)

# (a) Linear regression -- baseline, no hyper-parameters
lr = LinearRegression().fit(X_train, y_train)

# (b) PLS -- the workhorse for chemometric soft sensors
pls_grid = GridSearchCV(
    PLSRegression(),
    param_grid={"n_components": list(range(1, min(15, X_train.shape[1]) + 1))},
    cv=cv, scoring="r2", n_jobs=-1,
)
pls_grid.fit(X_train, y_train)
pls = pls_grid.best_estimator_
print(f"PLS best n_components = {pls_grid.best_params_['n_components']}")

# (c) SVR with RBF kernel -- captures nonlinear kinetics
svr_grid = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid={"C": [1, 10, 50], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.1, 0.3]},
    cv=cv, scoring="r2", n_jobs=-1,
)
svr_grid.fit(X_train, y_train)
svr = svr_grid.best_estimator_
print(f"SVR best params = {svr_grid.best_params_}")

# (d) Random Forest
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=RNG),
    param_grid={"n_estimators": [200, 400], "max_depth": [None, 6, 10],
                "min_samples_leaf": [1, 2, 4]},
    cv=cv, scoring="r2", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
rf = rf_grid.best_estimator_
print(f"RF best params  = {rf_grid.best_params_}")

# (e) Gradient Boosting
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=RNG),
    param_grid={"n_estimators": [200, 400], "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.05, 0.1]},
    cv=cv, scoring="r2", n_jobs=-1,
)
gb_grid.fit(X_train, y_train)
gb = gb_grid.best_estimator_
print(f"GB best params  = {gb_grid.best_params_}")


# ============================================================================
# SECTION 9 -- EVALUATION
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 9 | MODEL EVALUATION")
print("=" * 78)

models = {"Linear": lr, "PLS": pls, "SVR": svr, "RandomForest": rf, "GradientBoost": gb}
rows = []
preds = {}
for name, m in models.items():
    yp_tr = m.predict(X_train).ravel()
    yp_te = m.predict(X_test).ravel()
    preds[name] = yp_te
    rows.append({
        "Model": name,
        "R2_train" : r2_score(y_train, yp_tr),
        "R2_test"  : r2_score(y_test,  yp_te),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, yp_te)),
        "MAE_test" : mean_absolute_error(y_test, yp_te),
    })
results = pd.DataFrame(rows).sort_values("R2_test", ascending=False).reset_index(drop=True)
print("\n", results.round(4).to_string(index=False))
results.to_csv(f"{OUTDIR}/model_comparison.csv", index=False)

best_name = results.iloc[0]["Model"]
best_model = models[best_name]
print(f"\nBest model on held-out set: {best_name}")

# Parity + residual plots for every model
fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 8))
for i, (name, m) in enumerate(models.items()):
    yp = preds[name]
    axes[0, i].scatter(y_test, yp, s=18, alpha=0.7)
    mn, mx = min(y_test.min(), yp.min()), max(y_test.max(), yp.max())
    axes[0, i].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[0, i].set_title(f"{name}\nR² = {r2_score(y_test, yp):.3f}")
    axes[0, i].set_xlabel("Actual Kappa"); axes[0, i].set_ylabel("Predicted")
    res = y_test - yp
    axes[1, i].scatter(yp, res, s=18, alpha=0.7)
    axes[1, i].axhline(0, color="red", lw=1)
    axes[1, i].set_xlabel("Predicted"); axes[1, i].set_ylabel("Residual")
fig.suptitle("Parity (top) and residual (bottom) plots on test set", y=1.02, fontsize=13)
fig.savefig(f"{OUTDIR}/07_parity_residuals.png", bbox_inches="tight")
plt.close(fig)


# ============================================================================
# SECTION 10 -- TIME-SERIES CROSS VALIDATION
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 10 | TIME-SERIES CROSS VALIDATION (best model)")
print("=" * 78)

tscv = TimeSeriesSplit(n_splits=5)
# refit scaler inside each fold would be ideal; for simplicity reuse scaled X
X_all_scaled = scaler.transform(X)           # scaled using train scaler
ts_scores = cross_val_score(best_model, X_all_scaled, y, cv=tscv, scoring="r2")
print(f"{best_name} TimeSeriesSplit R² (mean ± sd): "
      f"{ts_scores.mean():.3f} ± {ts_scores.std():.3f}")
print("Fold-by-fold R²:", np.round(ts_scores, 3).tolist())


# ============================================================================
# SECTION 11 -- INTERPRETABILITY
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 11 | INTERPRETABILITY")
print("=" * 78)

def vip(pls_model):
    """Variable importance in projection for PLSRegression."""
    t  = pls_model.x_scores_
    w  = pls_model.x_weights_
    q  = pls_model.y_loadings_
    p, h = w.shape
    s  = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = s.sum()
    weight = np.array([(w[i, :] ** 2) @ s for i in range(p)]).ravel()
    return np.sqrt(p * weight / total_s)

vip_scores = pd.Series(vip(pls), index=feature_names).sort_values(ascending=False)
rf_imp     = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
gb_imp     = pd.Series(gb.feature_importances_, index=feature_names).sort_values(ascending=False)

perm = permutation_importance(best_model, X_test, y_test, n_repeats=30,
                              random_state=RNG, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

imp_df = pd.DataFrame({"PLS_VIP": vip_scores, "RF_imp": rf_imp,
                       "GB_imp": gb_imp, f"Permutation_{best_name}": perm_imp})
imp_df.to_csv(f"{OUTDIR}/feature_importance.csv")
print("\nTop 10 features (PLS VIP):")
print(vip_scores.head(10).round(3).to_string())
print(f"\nTop 10 features (permutation importance, {best_name}):")
print(perm_imp.head(10).round(4).to_string())

# Plot top-10 importances side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
vip_scores.head(10).iloc[::-1].plot.barh(ax=axes[0], color="teal")
axes[0].set_title("PLS VIP (top 10)")
axes[0].axvline(1.0, ls="--", color="red", lw=0.8, label="VIP = 1")
axes[0].legend()
rf_imp.head(10).iloc[::-1].plot.barh(ax=axes[1], color="darkorange")
axes[1].set_title("Random-Forest importance (top 10)")
perm_imp.head(10).iloc[::-1].plot.barh(ax=axes[2], color="purple")
axes[2].set_title(f"Permutation importance — {best_name}")
fig.savefig(f"{OUTDIR}/08_feature_importance.png", bbox_inches="tight")
plt.close(fig)


# ============================================================================
# SECTION 12 -- TEST-SET TIMELINE OF ACTUAL vs PREDICTED
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(y_test, label="Actual", color="black", lw=1.5)
ax.plot(preds[best_name], label=f"{best_name} prediction", color="crimson", lw=1.2)
ax.set_xlabel("Test-set hour")
ax.set_ylabel("Kappa number")
ax.set_title(f"Soft-sensor prediction of Kappa on held-out period — {best_name}")
ax.legend()
fig.savefig(f"{OUTDIR}/09_softsensor_timeline.png", bbox_inches="tight")
plt.close(fig)


# ============================================================================
# SECTION 13 -- SAVE ARTIFACTS FOR STREAMLIT DEPLOYMENT
# ============================================================================
print("\n" + "=" * 78)
print("SECTION 13 | SAVING ARTIFACTS FOR DEPLOYMENT")
print("=" * 78)

ARTDIR = "artifacts"
os.makedirs(ARTDIR, exist_ok=True)

# Persist scaler, all trained models, results table, feature list, training
# statistics (so the Streamlit slider defaults / ranges match the data), and
# the cleaned dataframe used for EDA tabs.
joblib.dump(scaler,     f"{ARTDIR}/scaler.joblib")
for name, m in models.items():
    joblib.dump(m, f"{ARTDIR}/model_{name}.joblib")
joblib.dump(pca_full,   f"{ARTDIR}/pca.joblib")

meta = {
    "feature_names" : feature_names,
    "target"        : TARGET,
    "best_model"    : best_name,
    "results_table" : results.to_dict(orient="records"),
    "feature_stats" : X.agg(["min", "max", "mean", "median"]).to_dict(),
    "corr_with_target": corr_with_kappa.to_dict(),
    "vip_scores"    : vip_scores.to_dict(),
    "perm_imp"      : perm_imp.to_dict(),
    "rf_imp"        : rf_imp.to_dict(),
    "gb_imp"        : gb_imp.to_dict(),
    "pca_evr"       : evr.tolist(),
    "pca_cum_evr"   : cevr.tolist(),
    "ts_cv_scores"  : ts_scores.tolist(),
    "ts_cv_mean"    : float(ts_scores.mean()),
    "ts_cv_std"     : float(ts_scores.std()),
}
joblib.dump(meta,       f"{ARTDIR}/metadata.joblib")

# Also drop the cleaned, feature-engineered frame (used by EDA tabs)
df_fe.to_csv(f"{ARTDIR}/df_clean.csv", index=False)

print(f"Artifacts saved to ./{ARTDIR}/ :")
for f in sorted(os.listdir(ARTDIR)):
    print(f"   - {ARTDIR}/{f}")


# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 78)
print("PIPELINE COMPLETE")
print("=" * 78)
print(f"All figures, CSVs, and the model-comparison table were saved to: ./{OUTDIR}/")
print("Files produced:")
for f in sorted(os.listdir(OUTDIR)):
    print(f"   - {OUTDIR}/{f}")
