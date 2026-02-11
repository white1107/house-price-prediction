"""Request/Response schemas for House Price Prediction API."""

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Input features for house price prediction."""

    MSSubClass: int = Field(60, description="Building class")
    MSZoning: str = Field("RL", description="General zoning classification")
    LotFrontage: float = Field(65.0, description="Linear feet of street connected to property")
    LotArea: int = Field(8450, description="Lot size in square feet")
    Street: str = Field("Pave", description="Type of road access")
    Alley: str | None = Field(None, description="Type of alley access")
    LotShape: str = Field("Reg", description="General shape of property")
    LandContour: str = Field("Lvl", description="Flatness of the property")
    Utilities: str = Field("AllPub", description="Type of utilities available")
    LotConfig: str = Field("Inside", description="Lot configuration")
    LandSlope: str = Field("Gtl", description="Slope of property")
    Neighborhood: str = Field("NAmes", description="Physical location")
    Condition1: str = Field("Norm", description="Proximity to main road or railroad")
    Condition2: str = Field("Norm", description="Proximity to main road or railroad (2nd)")
    BldgType: str = Field("1Fam", description="Type of dwelling")
    HouseStyle: str = Field("2Story", description="Style of dwelling")
    OverallQual: int = Field(7, ge=1, le=10, description="Overall material and finish quality")
    OverallCond: int = Field(5, ge=1, le=10, description="Overall condition rating")
    YearBuilt: int = Field(2003, description="Original construction date")
    YearRemodAdd: int = Field(2003, description="Remodel date")
    RoofStyle: str = Field("Gable", description="Type of roof")
    RoofMatl: str = Field("CompShg", description="Roof material")
    Exterior1st: str = Field("VinylSd", description="Exterior covering on house")
    Exterior2nd: str = Field("VinylSd", description="Exterior covering on house (2nd)")
    MasVnrType: str | None = Field("BrkFace", description="Masonry veneer type")
    MasVnrArea: float = Field(196.0, description="Masonry veneer area in square feet")
    ExterQual: str = Field("Gd", description="Exterior material quality")
    ExterCond: str = Field("TA", description="Present condition of the material on the exterior")
    Foundation: str = Field("PConc", description="Type of foundation")
    BsmtQual: str | None = Field("Gd", description="Height of the basement")
    BsmtCond: str | None = Field("TA", description="General condition of the basement")
    BsmtExposure: str | None = Field("No", description="Walkout or garden level basement walls")
    BsmtFinType1: str | None = Field("GLQ", description="Quality of basement finished area")
    BsmtFinSF1: float = Field(706.0, description="Type 1 finished square feet")
    BsmtFinType2: str | None = Field("Unf", description="Quality of second finished area")
    BsmtFinSF2: float = Field(0.0, description="Type 2 finished square feet")
    BsmtUnfSF: float = Field(150.0, description="Unfinished square feet of basement area")
    TotalBsmtSF: float = Field(856.0, description="Total square feet of basement area")
    Heating: str = Field("GasA", description="Type of heating")
    HeatingQC: str = Field("Ex", description="Heating quality and condition")
    CentralAir: str = Field("Y", description="Central air conditioning")
    Electrical: str = Field("SBrkr", description="Electrical system")
    FirstFlrSF: int = Field(856, description="First floor square feet", alias="1stFlrSF")
    SecondFlrSF: int = Field(854, description="Second floor square feet", alias="2ndFlrSF")
    LowQualFinSF: int = Field(0, description="Low quality finished square feet")
    GrLivArea: int = Field(1710, description="Above grade living area square feet")
    BsmtFullBath: int = Field(1, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(0, description="Basement half bathrooms")
    FullBath: int = Field(2, description="Full bathrooms above grade")
    HalfBath: int = Field(1, description="Half baths above grade")
    BedroomAbvGr: int = Field(3, description="Number of bedrooms above basement level")
    KitchenAbvGr: int = Field(1, description="Number of kitchens")
    KitchenQual: str = Field("Gd", description="Kitchen quality")
    TotRmsAbvGrd: int = Field(8, description="Total rooms above grade")
    Functional: str = Field("Typ", description="Home functionality rating")
    Fireplaces: int = Field(0, description="Number of fireplaces")
    FireplaceQu: str | None = Field(None, description="Fireplace quality")
    GarageType: str | None = Field("Attchd", description="Garage location")
    GarageYrBlt: float | None = Field(2003.0, description="Year garage was built")
    GarageFinish: str | None = Field("RFn", description="Interior finish of the garage")
    GarageCars: int = Field(2, description="Size of garage in car capacity")
    GarageArea: float = Field(548.0, description="Size of garage in square feet")
    GarageQual: str | None = Field("TA", description="Garage quality")
    GarageCond: str | None = Field("TA", description="Garage condition")
    PavedDrive: str = Field("Y", description="Paved driveway")
    WoodDeckSF: int = Field(0, description="Wood deck area in square feet")
    OpenPorchSF: int = Field(61, description="Open porch area in square feet")
    EnclosedPorch: int = Field(0, description="Enclosed porch area in square feet")
    ThreeSsnPorch: int = Field(0, description="Three season porch area", alias="3SsnPorch")
    ScreenPorch: int = Field(0, description="Screen porch area in square feet")
    PoolArea: int = Field(0, description="Pool area in square feet")
    PoolQC: str | None = Field(None, description="Pool quality")
    Fence: str | None = Field(None, description="Fence quality")
    MiscFeature: str | None = Field(None, description="Miscellaneous feature")
    MiscVal: int = Field(0, description="Value of miscellaneous feature")
    MoSold: int = Field(2, description="Month sold")
    YrSold: int = Field(2008, description="Year sold")
    SaleType: str = Field("WD", description="Type of sale")
    SaleCondition: str = Field("Normal", description="Condition of sale")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Prediction response."""

    predicted_price: float = Field(description="Predicted sale price in USD")
    model_name: str = Field(description="Model used for prediction")
    model_version: str = Field(description="Model version")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    houses: list[HouseFeatures] = Field(description="List of houses to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse] = Field(description="List of predictions")
    count: int = Field(description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str | None
    features_count: int | None
