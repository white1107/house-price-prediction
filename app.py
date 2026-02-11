"""Streamlit Web App - House Price Prediction."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


def predict_price(features_dict, model, scaler, feature_names):
    """Run prediction pipeline on a single house."""
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
    pred = model.predict(df_scaled.iloc[[0]])[0]
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
# Sidebar - Key Parameters
# ==================================================
st.sidebar.header("🔧 Key Parameters")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 7, help="Overall material and finish quality (1-10)")
overall_cond = st.sidebar.slider("Overall Condition", 1, 10, 5, help="Overall condition rating (1-10)")
year_built = st.sidebar.slider("Year Built", 1870, 2025, 2003)
year_remod = st.sidebar.slider("Year Remodeled", 1870, 2025, 2003)
yr_sold = st.sidebar.selectbox("Year Sold", [2006, 2007, 2008, 2009, 2010], index=2)

# ==================================================
# Main - Area Features
# ==================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📐 Area")
    gr_liv_area = st.number_input("Living Area (sqft)", 300, 6000, 1710, step=50)
    lot_area = st.number_input("Lot Area (sqft)", 1000, 100000, 8450, step=500)
    total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 5000, 856, step=50)
    first_flr_sf = st.number_input("1st Floor (sqft)", 300, 5000, 856, step=50)
    second_flr_sf = st.number_input("2nd Floor (sqft)", 0, 3000, 854, step=50)

with col2:
    st.subheader("🚗 Garage & Exterior")
    garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4], index=2)
    garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 548, step=50)
    garage_type = st.selectbox("Garage Type", ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types", None], index=0)
    garage_finish = st.selectbox("Garage Finish", ["Fin", "RFn", "Unf", None], index=1)
    mas_vnr_area = st.number_input("Masonry Veneer Area", 0, 1500, 196, step=25)
    mas_vnr_type = st.selectbox("Masonry Veneer Type", ["BrkFace", "Stone", "BrkCmn", "None"], index=0)

with col3:
    st.subheader("🛁 Rooms & Bath")
    full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
    half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=1)
    bsmt_full_bath = st.selectbox("Basement Full Bath", [0, 1, 2, 3], index=1)
    bsmt_half_bath = st.selectbox("Basement Half Bath", [0, 1, 2], index=0)
    bedrooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5, 6], index=3)
    kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
    tot_rms = st.slider("Total Rooms Above Grade", 2, 15, 8)

# ==================================================
# Additional Features (Expandable)
# ==================================================
with st.expander("🏗️ Additional Features", expanded=False):
    acol1, acol2, acol3 = st.columns(3)

    with acol1:
        neighborhood = st.selectbox("Neighborhood", [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt",
            "Gilbert", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor",
            "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
            "SWISU", "Blmngtn", "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste",
        ], index=0)
        ms_zoning = st.selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"], index=0)
        bldg_type = st.selectbox("Building Type", ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"], index=0)
        house_style = st.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf", "2.5Unf", "2.5Fin"], index=1)

    with acol2:
        exter_qual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
        bsmt_qual = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po", None], index=1)
        fireplace_qu = st.selectbox("Fireplace Quality", ["Ex", "Gd", "TA", "Fa", "Po", None], index=5)
        fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=0)
        central_air = st.selectbox("Central Air", ["Y", "N"], index=0)
        heating_qc = st.selectbox("Heating Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=0)

    with acol3:
        pool_area = st.number_input("Pool Area", 0, 800, 0, step=50)
        wood_deck_sf = st.number_input("Wood Deck (sqft)", 0, 1000, 0, step=25)
        open_porch_sf = st.number_input("Open Porch (sqft)", 0, 600, 61, step=25)
        enclosed_porch = st.number_input("Enclosed Porch (sqft)", 0, 600, 0, step=25)
        screen_porch = st.number_input("Screen Porch (sqft)", 0, 600, 0, step=25)
        fence = st.selectbox("Fence", [None, "GdPrv", "MnPrv", "GdWo", "MnWw"], index=0)

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

# Feature summary
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
