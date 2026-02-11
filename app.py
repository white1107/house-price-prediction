"""Streamlit Web App - House Price Prediction."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from house_prices.features.advanced_engineering import build_features


# ==================================================
# Model Loading
# ==================================================
@st.cache_resource
def load_model():
    models_dir = PROJECT_ROOT / "models"
    model = joblib.load(models_dir / "best_model.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")
    return model, scaler, feature_names


@st.cache_resource
def load_shap_explainer(_model, _scaler, feature_names):
    """Create SHAP explainer (cached to avoid recomputation)."""
    # Use a background dataset for KernelExplainer or LinearExplainer
    model_name = type(_model).__name__
    if model_name in ("Lasso", "Ridge", "ElasticNet", "LinearRegression"):
        explainer = shap.LinearExplainer(_model, np.zeros((1, len(feature_names))))
    else:
        # For tree models, use TreeExplainer
        try:
            explainer = shap.TreeExplainer(_model)
        except Exception:
            explainer = None
    return explainer


def prepare_input(features_dict, feature_names, scaler):
    """Prepare input DataFrame for prediction."""
    for key, val in features_dict.items():
        if val is None:
            features_dict[key] = np.nan

    df = pd.DataFrame([features_dict, features_dict])
    df = build_features(df, encode_nominal=True, fix_skew=True)

    missing_cols = set(feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[feature_names]

    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    return df_scaled.iloc[[0]]


def predict_price(features_dict, model, scaler, feature_names):
    """Run prediction pipeline on a single house."""
    df_scaled = prepare_input(features_dict, feature_names, scaler)
    pred = model.predict(df_scaled)[0]
    return float(np.expm1(pred))


# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Predictor")
st.caption("Kaggle House Prices Competition - Interactive Prediction Tool")

model, scaler, feature_names = load_model()
st.sidebar.success(f"Model: **{type(model).__name__}** ({len(feature_names)} features)")

# ==================================================
# Sample Presets
# ==================================================
PRESETS = {
    "Custom": {},
    "Starter Home": {
        "overall_qual": 5, "overall_cond": 6, "year_built": 1965, "year_remod": 1965,
        "gr_liv_area": 900, "lot_area": 6000, "total_bsmt_sf": 900, "first_flr_sf": 900,
        "second_flr_sf": 0, "garage_cars": 1, "garage_area": 280, "garage_type": "Detchd",
        "garage_finish": "Unf", "mas_vnr_area": 0, "mas_vnr_type": "None",
        "full_bath": 1, "half_bath": 0, "bsmt_full_bath": 0, "bsmt_half_bath": 0,
        "bedrooms": 2, "kitchen_qual": "TA", "tot_rms": 5, "neighborhood": "Edwards",
        "ms_zoning": "RL", "bldg_type": "1Fam", "house_style": "1Story",
        "exter_qual": "TA", "bsmt_qual": "TA", "fireplace_qu": None, "fireplaces": 0,
        "central_air": "Y", "heating_qc": "TA", "pool_area": 0, "wood_deck_sf": 0,
        "open_porch_sf": 0, "enclosed_porch": 0, "screen_porch": 0, "fence": None,
    },
    "Average Family Home": {
        "overall_qual": 6, "overall_cond": 5, "year_built": 1995, "year_remod": 1995,
        "gr_liv_area": 1500, "lot_area": 9500, "total_bsmt_sf": 800, "first_flr_sf": 1000,
        "second_flr_sf": 500, "garage_cars": 2, "garage_area": 480, "garage_type": "Attchd",
        "garage_finish": "RFn", "mas_vnr_area": 100, "mas_vnr_type": "BrkFace",
        "full_bath": 2, "half_bath": 1, "bsmt_full_bath": 0, "bsmt_half_bath": 0,
        "bedrooms": 3, "kitchen_qual": "TA", "tot_rms": 7, "neighborhood": "NAmes",
        "ms_zoning": "RL", "bldg_type": "1Fam", "house_style": "2Story",
        "exter_qual": "TA", "bsmt_qual": "Gd", "fireplace_qu": "TA", "fireplaces": 1,
        "central_air": "Y", "heating_qc": "Ex", "pool_area": 0, "wood_deck_sf": 100,
        "open_porch_sf": 40, "enclosed_porch": 0, "screen_porch": 0, "fence": None,
    },
    "Modern Suburban": {
        "overall_qual": 7, "overall_cond": 5, "year_built": 2005, "year_remod": 2005,
        "gr_liv_area": 1800, "lot_area": 10000, "total_bsmt_sf": 1000, "first_flr_sf": 1000,
        "second_flr_sf": 800, "garage_cars": 2, "garage_area": 550, "garage_type": "Attchd",
        "garage_finish": "Fin", "mas_vnr_area": 200, "mas_vnr_type": "BrkFace",
        "full_bath": 2, "half_bath": 1, "bsmt_full_bath": 1, "bsmt_half_bath": 0,
        "bedrooms": 3, "kitchen_qual": "Gd", "tot_rms": 8, "neighborhood": "CollgCr",
        "ms_zoning": "RL", "bldg_type": "1Fam", "house_style": "2Story",
        "exter_qual": "Gd", "bsmt_qual": "Gd", "fireplace_qu": "Gd", "fireplaces": 1,
        "central_air": "Y", "heating_qc": "Ex", "pool_area": 0, "wood_deck_sf": 150,
        "open_porch_sf": 60, "enclosed_porch": 0, "screen_porch": 0, "fence": None,
    },
    "Luxury Estate": {
        "overall_qual": 10, "overall_cond": 5, "year_built": 2008, "year_remod": 2008,
        "gr_liv_area": 3500, "lot_area": 15000, "total_bsmt_sf": 2000, "first_flr_sf": 2000,
        "second_flr_sf": 1500, "garage_cars": 3, "garage_area": 900, "garage_type": "Attchd",
        "garage_finish": "Fin", "mas_vnr_area": 600, "mas_vnr_type": "Stone",
        "full_bath": 3, "half_bath": 1, "bsmt_full_bath": 1, "bsmt_half_bath": 0,
        "bedrooms": 4, "kitchen_qual": "Ex", "tot_rms": 12, "neighborhood": "NridgHt",
        "ms_zoning": "RL", "bldg_type": "1Fam", "house_style": "2Story",
        "exter_qual": "Ex", "bsmt_qual": "Ex", "fireplace_qu": "Ex", "fireplaces": 2,
        "central_air": "Y", "heating_qc": "Ex", "pool_area": 500, "wood_deck_sf": 300,
        "open_porch_sf": 200, "enclosed_porch": 0, "screen_porch": 200, "fence": "GdPrv",
    },
    "Vintage Fixer-Upper": {
        "overall_qual": 4, "overall_cond": 4, "year_built": 1920, "year_remod": 1950,
        "gr_liv_area": 1200, "lot_area": 7500, "total_bsmt_sf": 600, "first_flr_sf": 700,
        "second_flr_sf": 500, "garage_cars": 1, "garage_area": 250, "garage_type": "Detchd",
        "garage_finish": "Unf", "mas_vnr_area": 0, "mas_vnr_type": "None",
        "full_bath": 1, "half_bath": 0, "bsmt_full_bath": 0, "bsmt_half_bath": 0,
        "bedrooms": 3, "kitchen_qual": "Fa", "tot_rms": 6, "neighborhood": "OldTown",
        "ms_zoning": "RM", "bldg_type": "1Fam", "house_style": "1.5Fin",
        "exter_qual": "Fa", "bsmt_qual": "TA", "fireplace_qu": "Gd", "fireplaces": 1,
        "central_air": "N", "heating_qc": "TA", "pool_area": 0, "wood_deck_sf": 0,
        "open_porch_sf": 30, "enclosed_porch": 50, "screen_porch": 0, "fence": "MnWw",
    },
    "New Construction": {
        "overall_qual": 8, "overall_cond": 5, "year_built": 2009, "year_remod": 2009,
        "gr_liv_area": 2200, "lot_area": 11000, "total_bsmt_sf": 1200, "first_flr_sf": 1200,
        "second_flr_sf": 1000, "garage_cars": 3, "garage_area": 700, "garage_type": "Attchd",
        "garage_finish": "Fin", "mas_vnr_area": 300, "mas_vnr_type": "Stone",
        "full_bath": 2, "half_bath": 1, "bsmt_full_bath": 1, "bsmt_half_bath": 0,
        "bedrooms": 4, "kitchen_qual": "Ex", "tot_rms": 9, "neighborhood": "Somerst",
        "ms_zoning": "RL", "bldg_type": "1Fam", "house_style": "2Story",
        "exter_qual": "Gd", "bsmt_qual": "Ex", "fireplace_qu": "Gd", "fireplaces": 1,
        "central_air": "Y", "heating_qc": "Ex", "pool_area": 0, "wood_deck_sf": 200,
        "open_porch_sf": 80, "enclosed_porch": 0, "screen_porch": 0, "fence": None,
    },
}

st.sidebar.header("📋 Sample Presets")
preset_name = st.sidebar.selectbox(
    "Select a preset to load",
    list(PRESETS.keys()),
    help="Choose a sample house type to auto-fill all fields",
)
preset = PRESETS[preset_name]

def get_val(key, default):
    """Get value from preset or use default."""
    return preset.get(key, default)

# ==================================================
# Sidebar - Key Parameters
# ==================================================
st.sidebar.header("🔧 Key Parameters")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, get_val("overall_qual", 7), help="Overall material and finish quality (1-10)")
overall_cond = st.sidebar.slider("Overall Condition", 1, 10, get_val("overall_cond", 5), help="Overall condition rating (1-10)")
year_built = st.sidebar.slider("Year Built", 1870, 2025, get_val("year_built", 2003))
year_remod = st.sidebar.slider("Year Remodeled", 1870, 2025, get_val("year_remod", 2003))
yr_sold = st.sidebar.selectbox("Year Sold", [2006, 2007, 2008, 2009, 2010], index=2)

# ==================================================
# Main - Area Features
# ==================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📐 Area")
    gr_liv_area = st.number_input("Living Area (sqft)", 300, 6000, get_val("gr_liv_area", 1710), step=50)
    lot_area = st.number_input("Lot Area (sqft)", 1000, 100000, get_val("lot_area", 8450), step=500)
    total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 5000, get_val("total_bsmt_sf", 856), step=50)
    first_flr_sf = st.number_input("1st Floor (sqft)", 300, 5000, get_val("first_flr_sf", 856), step=50)
    second_flr_sf = st.number_input("2nd Floor (sqft)", 0, 3000, get_val("second_flr_sf", 854), step=50)

with col2:
    st.subheader("🚗 Garage & Exterior")
    _garage_cars_opts = [0, 1, 2, 3, 4]
    garage_cars = st.selectbox("Garage Cars", _garage_cars_opts, index=_garage_cars_opts.index(get_val("garage_cars", 2)))
    garage_area = st.number_input("Garage Area (sqft)", 0, 1500, get_val("garage_area", 548), step=50)
    _garage_type_opts = ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types", None]
    garage_type = st.selectbox("Garage Type", _garage_type_opts, index=_garage_type_opts.index(get_val("garage_type", "Attchd")))
    _garage_fin_opts = ["Fin", "RFn", "Unf", None]
    garage_finish = st.selectbox("Garage Finish", _garage_fin_opts, index=_garage_fin_opts.index(get_val("garage_finish", "RFn")))
    mas_vnr_area = st.number_input("Masonry Veneer Area", 0, 1500, get_val("mas_vnr_area", 196), step=25)
    _mvt_opts = ["BrkFace", "Stone", "BrkCmn", "None"]
    mas_vnr_type = st.selectbox("Masonry Veneer Type", _mvt_opts, index=_mvt_opts.index(get_val("mas_vnr_type", "BrkFace")))

with col3:
    st.subheader("🛁 Rooms & Bath")
    _fb_opts = [0, 1, 2, 3, 4]
    full_bath = st.selectbox("Full Bathrooms", _fb_opts, index=_fb_opts.index(get_val("full_bath", 2)))
    _hb_opts = [0, 1, 2]
    half_bath = st.selectbox("Half Bathrooms", _hb_opts, index=_hb_opts.index(get_val("half_bath", 1)))
    _bfb_opts = [0, 1, 2, 3]
    bsmt_full_bath = st.selectbox("Basement Full Bath", _bfb_opts, index=_bfb_opts.index(get_val("bsmt_full_bath", 1)))
    _bhb_opts = [0, 1, 2]
    bsmt_half_bath = st.selectbox("Basement Half Bath", _bhb_opts, index=_bhb_opts.index(get_val("bsmt_half_bath", 0)))
    _bed_opts = [0, 1, 2, 3, 4, 5, 6]
    bedrooms = st.selectbox("Bedrooms", _bed_opts, index=_bed_opts.index(get_val("bedrooms", 3)))
    _kq_opts = ["Ex", "Gd", "TA", "Fa", "Po"]
    kitchen_qual = st.selectbox("Kitchen Quality", _kq_opts, index=_kq_opts.index(get_val("kitchen_qual", "Gd")))
    tot_rms = st.slider("Total Rooms Above Grade", 2, 15, get_val("tot_rms", 8))

# ==================================================
# Additional Features (Expandable)
# ==================================================
with st.expander("🏗️ Additional Features", expanded=False):
    acol1, acol2, acol3 = st.columns(3)

    with acol1:
        _neigh_opts = [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt",
            "Gilbert", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor",
            "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
            "SWISU", "Blmngtn", "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste",
        ]
        neighborhood = st.selectbox("Neighborhood", _neigh_opts, index=_neigh_opts.index(get_val("neighborhood", "NAmes")))
        _zone_opts = ["RL", "RM", "FV", "RH", "C (all)"]
        ms_zoning = st.selectbox("Zoning", _zone_opts, index=_zone_opts.index(get_val("ms_zoning", "RL")))
        _btype_opts = ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"]
        bldg_type = st.selectbox("Building Type", _btype_opts, index=_btype_opts.index(get_val("bldg_type", "1Fam")))
        _hstyle_opts = ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf", "2.5Unf", "2.5Fin"]
        house_style = st.selectbox("House Style", _hstyle_opts, index=_hstyle_opts.index(get_val("house_style", "2Story")))

    with acol2:
        _eq_opts = ["Ex", "Gd", "TA", "Fa", "Po"]
        exter_qual = st.selectbox("Exterior Quality", _eq_opts, index=_eq_opts.index(get_val("exter_qual", "Gd")))
        _bq_opts = ["Ex", "Gd", "TA", "Fa", "Po", None]
        bsmt_qual = st.selectbox("Basement Quality", _bq_opts, index=_bq_opts.index(get_val("bsmt_qual", "Gd")))
        _fq_opts = ["Ex", "Gd", "TA", "Fa", "Po", None]
        fireplace_qu = st.selectbox("Fireplace Quality", _fq_opts, index=_fq_opts.index(get_val("fireplace_qu", None)))
        _fp_opts = [0, 1, 2, 3]
        fireplaces = st.selectbox("Fireplaces", _fp_opts, index=_fp_opts.index(get_val("fireplaces", 0)))
        _ca_opts = ["Y", "N"]
        central_air = st.selectbox("Central Air", _ca_opts, index=_ca_opts.index(get_val("central_air", "Y")))
        _hqc_opts = ["Ex", "Gd", "TA", "Fa", "Po"]
        heating_qc = st.selectbox("Heating Quality", _hqc_opts, index=_hqc_opts.index(get_val("heating_qc", "Ex")))

    with acol3:
        pool_area = st.number_input("Pool Area", 0, 800, get_val("pool_area", 0), step=50)
        wood_deck_sf = st.number_input("Wood Deck (sqft)", 0, 1000, get_val("wood_deck_sf", 0), step=25)
        open_porch_sf = st.number_input("Open Porch (sqft)", 0, 600, get_val("open_porch_sf", 61), step=25)
        enclosed_porch = st.number_input("Enclosed Porch (sqft)", 0, 600, get_val("enclosed_porch", 0), step=25)
        screen_porch = st.number_input("Screen Porch (sqft)", 0, 600, get_val("screen_porch", 0), step=25)
        _fence_opts = [None, "GdPrv", "MnPrv", "GdWo", "MnWw"]
        fence = st.selectbox("Fence", _fence_opts, index=_fence_opts.index(get_val("fence", None)))

# ==================================================
# Build features dict
# ==================================================
features = {
    "MSSubClass": 60,
    "MSZoning": ms_zoning,
    "LotFrontage": 65.0,
    "LotArea": lot_area,
    "Street": "Pave",
    "Alley": None,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": neighborhood,
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": bldg_type,
    "HouseStyle": house_style,
    "OverallQual": overall_qual,
    "OverallCond": overall_cond,
    "YearBuilt": year_built,
    "YearRemodAdd": year_remod,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": mas_vnr_type if mas_vnr_type != "None" else None,
    "MasVnrArea": float(mas_vnr_area),
    "ExterQual": exter_qual,
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": bsmt_qual,
    "BsmtCond": "TA" if total_bsmt_sf > 0 else None,
    "BsmtExposure": "No" if total_bsmt_sf > 0 else None,
    "BsmtFinType1": "GLQ" if total_bsmt_sf > 0 else None,
    "BsmtFinSF1": float(total_bsmt_sf * 0.6) if total_bsmt_sf > 0 else 0.0,
    "BsmtFinType2": "Unf" if total_bsmt_sf > 0 else None,
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": float(total_bsmt_sf * 0.4) if total_bsmt_sf > 0 else 0.0,
    "TotalBsmtSF": float(total_bsmt_sf),
    "Heating": "GasA",
    "HeatingQC": heating_qc,
    "CentralAir": central_air,
    "Electrical": "SBrkr",
    "1stFlrSF": first_flr_sf,
    "2ndFlrSF": second_flr_sf,
    "LowQualFinSF": 0,
    "GrLivArea": gr_liv_area,
    "BsmtFullBath": bsmt_full_bath,
    "BsmtHalfBath": bsmt_half_bath,
    "FullBath": full_bath,
    "HalfBath": half_bath,
    "BedroomAbvGr": bedrooms,
    "KitchenAbvGr": 1,
    "KitchenQual": kitchen_qual,
    "TotRmsAbvGrd": tot_rms,
    "Functional": "Typ",
    "Fireplaces": fireplaces,
    "FireplaceQu": fireplace_qu,
    "GarageType": garage_type,
    "GarageYrBlt": float(year_built) if garage_cars > 0 else np.nan,
    "GarageFinish": garage_finish,
    "GarageCars": garage_cars,
    "GarageArea": float(garage_area),
    "GarageQual": "TA" if garage_cars > 0 else None,
    "GarageCond": "TA" if garage_cars > 0 else None,
    "PavedDrive": "Y",
    "WoodDeckSF": wood_deck_sf,
    "OpenPorchSF": open_porch_sf,
    "EnclosedPorch": enclosed_porch,
    "3SsnPorch": 0,
    "ScreenPorch": screen_porch,
    "PoolArea": pool_area,
    "PoolQC": "Gd" if pool_area > 0 else None,
    "Fence": fence,
    "MiscFeature": None,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": yr_sold,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}

# ==================================================
# Prediction
# ==================================================
st.divider()

predicted_price = predict_price(features, model, scaler, feature_names)

# Display
pcol1, pcol2, pcol3 = st.columns([2, 1, 1])

with pcol1:
    st.metric(
        label="💰 Predicted Sale Price",
        value=f"${predicted_price:,.0f}",
    )

with pcol2:
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    if total_sf > 0:
        price_per_sqft = predicted_price / total_sf
        st.metric("Price per sqft", f"${price_per_sqft:,.0f}")

with pcol3:
    st.metric("Total SF", f"{total_sf:,}")

# Price gauge
st.progress(min(predicted_price / 600000, 1.0))
st.caption(f"Range: $0 - $600,000+")

# ==================================================
# SHAP Explanation
# ==================================================
st.subheader("🔍 SHAP - Why this price?")

explainer = load_shap_explainer(model, scaler, feature_names)

if explainer is not None:
    df_input = prepare_input(dict(features), feature_names, scaler)
    shap_values = explainer.shap_values(df_input)

    if isinstance(shap_values, list):
        shap_vals = shap_values[0].flatten()
    else:
        shap_vals = shap_values.flatten()

    # Top contributing features
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals,
        "Abs SHAP": np.abs(shap_vals),
    }).sort_values("Abs SHAP", ascending=False)

    top_n = 15
    top_features = shap_df.head(top_n).sort_values("SHAP Value")

    # Waterfall-style horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#ff4b4b" if v < 0 else "#0068c9" for v in top_features["SHAP Value"]]
    ax.barh(range(len(top_features)), top_features["SHAP Value"], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"], fontsize=10)
    ax.set_xlabel("SHAP Value (impact on log price)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Contributions", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")

    # Add value labels
    for i, (val, feat) in enumerate(zip(top_features["SHAP Value"], top_features["Feature"])):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            i,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Legend
    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown("🔵 **Blue** = Increases price")
    with scol2:
        st.markdown("🔴 **Red** = Decreases price")

    # Detailed SHAP table
    with st.expander("📋 Full SHAP Values (all features)"):
        display_df = shap_df[["Feature", "SHAP Value"]].copy()
        display_df["SHAP Value"] = display_df["SHAP Value"].apply(lambda x: f"{x:+.6f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("SHAP is not available for this model type. Train with a tree-based or linear model to enable SHAP explanations.")

# ==================================================
# Feature Summary
# ==================================================
with st.expander("📊 Feature Summary"):
    summary = pd.DataFrame({
        "Category": ["Quality", "Quality", "Area", "Area", "Area", "Rooms", "Age", "Garage"],
        "Feature": ["Overall Qual", "Overall Cond", "Living Area", "Total SF", "Lot Area",
                     "Bathrooms", "House Age", "Garage Cars"],
        "Value": [
            f"{overall_qual}/10",
            f"{overall_cond}/10",
            f"{gr_liv_area:,} sqft",
            f"{total_sf:,} sqft",
            f"{lot_area:,} sqft",
            f"{full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath:.1f}",
            f"{yr_sold - year_built} years",
            f"{garage_cars}",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
