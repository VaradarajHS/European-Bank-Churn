# ============================================================
# EUROPEAN BANK CUSTOMER CHURN
# PREDICTIVE MODELING & RISK SCORING
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="European Bank Churn Risk",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #666666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">Bank Customer Churn Prediction & Risk Scoring</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">European Bank — Predictive Modeling, Customer Risk Analysis & What-If Simulation</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

from pathlib import Path


@st.cache_data
def load_data():
    file_path = Path("/Users/vardaraj/Downloads/European_Bank.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find: {file_path}"
        )

    df = pd.read_csv(file_path)
    df_model = df.copy()

    return df, df_model
# ============================================================
# LOAD DATA WITH ERROR HANDLING
# ============================================================

try:

    raw_df, df_model = load_data()

except FileNotFoundError:

    st.error(
        "❌ Dataset not found.\n\n"
        "Please make sure the following file is in the same folder "
        "as European-Bank.py:\n\n"
        "**European_Bank.csv**"
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading dataset: {e}"
    )

    st.stop()


# ============================================================
# DATASET VALIDATION
# ============================================================

required_columns = [
    "Exited"
]

missing_columns = [
    col
    for col in required_columns
    if col not in raw_df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing:\n\n"
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# DATA COPY
# ============================================================

df = raw_df.copy()


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = [
    "CustomerId",
    "Surname",
    "Year",
    "AgeGroup"
]

existing_columns_to_drop = [
    col
    for col in columns_to_drop
    if col in df_model.columns
]

df_model = df_model.drop(
    columns=existing_columns_to_drop
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # Balance / Salary Ratio
    # --------------------------------------------------------

    if (
        "Balance" in data.columns
        and "EstimatedSalary" in data.columns
    ):

        data["BalanceSalaryRatio"] = (
            data["Balance"]
            / (data["EstimatedSalary"] + 1)
        )


    # --------------------------------------------------------
    # Product Density
    # --------------------------------------------------------

    if (
        "NumOfProducts" in data.columns
        and "Tenure" in data.columns
    ):

        data["ProductDensity"] = (
            data["NumOfProducts"]
            / (data["Tenure"] + 1)
        )


    # --------------------------------------------------------
    # Engagement × Product
    # --------------------------------------------------------

    if (
        "IsActiveMember" in data.columns
        and "NumOfProducts" in data.columns
    ):

        data["EngagementProduct"] = (
            data["IsActiveMember"]
            * data["NumOfProducts"]
        )


    # --------------------------------------------------------
    # Age × Tenure
    # --------------------------------------------------------

    if (
        "Age" in data.columns
        and "Tenure" in data.columns
    ):

        data["AgeTenure"] = (
            data["Age"]
            * data["Tenure"]
        )


    return data


df_model = create_features(df_model)


# ============================================================
# TARGET & FEATURES
# ============================================================

X = df_model.drop(
    "Exited",
    axis=1
)

y = df_model["Exited"]


# ============================================================
# IDENTIFY NUMERICAL & CATEGORICAL FEATURES
# ============================================================

categorical_features = (
    X.select_dtypes(
        include=["object"]
    )
    .columns
    .tolist()
)

numerical_features = (
    X.select_dtypes(
        exclude=["object"]
    )
    .columns
    .tolist()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            StandardScaler(),

            numerical_features
        ),

        (
            "cat",

            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ]
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

random_forest = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",

            RandomForestClassifier(

                n_estimators=200,

                random_state=42,

                class_weight="balanced",

                n_jobs=-1
            )
        )

    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X_train, y_train):

    model = Pipeline(

        steps=[

            (
                "preprocessor",

                ColumnTransformer(

                    transformers=[

                        (
                            "num",

                            StandardScaler(),

                            numerical_features
                        ),

                        (
                            "cat",

                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore"
                            ),

                            categorical_features
                        )

                    ]
                )
            ),

            (
                "classifier",

                RandomForestClassifier(

                    n_estimators=200,

                    random_state=42,

                    class_weight="balanced",

                    n_jobs=-1
                )
            )

        ]
    )

    model.fit(
        X_train,
        y_train
    )

    return model


random_forest = train_model(
    X_train,
    y_train
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Customer Risk Calculator")

st.sidebar.success(
    "European_Bank.csv loaded successfully"
)

st.sidebar.divider()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.sidebar.subheader("Customer Features")


# Credit Score
credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650,
    step=1
)


# Geography
if "Geography" in X.columns:

    geography_options = sorted(
        df["Geography"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    geography = st.sidebar.selectbox(
        "Geography",
        geography_options
    )


# Gender
if "Gender" in X.columns:

    gender_options = sorted(
        df["Gender"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    gender = st.sidebar.selectbox(
        "Gender",
        gender_options
    )


# Age
age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40,
    step=1
)


# Tenure
tenure = st.sidebar.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)


# Balance
balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)


# Number of Products
num_products = st.sidebar.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2,
    step=1
)


# Has Credit Card
has_credit_card = st.sidebar.selectbox(
    "Has Credit Card?",
    [0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# Active Member
is_active_member = st.sidebar.selectbox(
    "Active Member?",
    [0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# Estimated Salary
estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)


# ============================================================
# CREATE CUSTOMER INPUT DATAFRAME
# ============================================================

customer_data = {}


if "CreditScore" in X.columns:

    customer_data["CreditScore"] = credit_score


if "Geography" in X.columns:

    customer_data["Geography"] = geography


if "Gender" in X.columns:

    customer_data["Gender"] = gender


if "Age" in X.columns:

    customer_data["Age"] = age


if "Tenure" in X.columns:

    customer_data["Tenure"] = tenure


if "Balance" in X.columns:

    customer_data["Balance"] = balance


if "NumOfProducts" in X.columns:

    customer_data["NumOfProducts"] = num_products


if "HasCrCard" in X.columns:

    customer_data["HasCrCard"] = has_credit_card


if "IsActiveMember" in X.columns:

    customer_data["IsActiveMember"] = is_active_member


if "EstimatedSalary" in X.columns:

    customer_data["EstimatedSalary"] = estimated_salary


customer = pd.DataFrame(
    [customer_data]
)


# ============================================================
# CUSTOMER FEATURE ENGINEERING
# ============================================================

customer = create_features(
    customer
)


# Ensure same columns and order
customer = customer.reindex(
    columns=X.columns
)


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

churn_probability = (

    random_forest
    .predict_proba(customer)[0][1]

)


# Default threshold
default_threshold = 0.50


churn_prediction = int(
    churn_probability >= default_threshold
)


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(probability):

    if probability < 0.30:

        return "Low Risk"

    elif probability < 0.60:

        return "Medium Risk"

    else:

        return "High Risk"


risk = risk_category(
    churn_probability
)


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.markdown(
    '<div class="section-title">Customer Churn Risk Assessment</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )


with col2:

    st.metric(
        "Risk Category",
        risk
    )


with col3:

    st.metric(
        "Predicted Churn",
        "Yes"
        if churn_prediction == 1
        else "No"
    )


with col4:

    st.metric(
        "Decision Threshold",
        "50%"
    )


# ============================================================
# RISK MESSAGE
# ============================================================

if risk == "High Risk":

    st.error(
        "⚠️ High-risk customer. "
        "Targeted retention action is recommended."
    )

elif risk == "Medium Risk":

    st.warning(
        "⚠️ Medium-risk customer. "
        "Additional customer engagement may be useful."
    )

else:

    st.success(
        "✅ Low-risk customer."
    )


# ============================================================
# PROBABILITY BAR
# ============================================================

st.subheader("Churn Probability Score")

st.progress(
    float(churn_probability)
)


# ============================================================
# CUSTOMER DETAILS
# ============================================================

with st.expander(
    "View Customer Input Details"
):

    st.dataframe(
        customer,
        use_container_width=True
    )


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Model Overview</div>',
    unsafe_allow_html=True
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Training Customers",
        f"{len(X_train):,}"
    )


with m2:

    st.metric(
        "Testing Customers",
        f"{len(X_test):,}"
    )


with m3:

    st.metric(
        "Features",
        f"{X.shape[1]}"
    )


with m4:

    st.metric(
        "Model",
        "Random Forest"
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Churn Probability Distribution</div>',
    unsafe_allow_html=True
)


test_probabilities = (

    random_forest
    .predict_proba(X_test)[:, 1]

)


probability_df = pd.DataFrame({

    "Churn Probability":
        test_probabilities

})


fig_probability = px.histogram(

    probability_df,

    x="Churn Probability",

    nbins=20,

    title="Distribution of Predicted Churn Probabilities"

)


fig_probability.add_vline(

    x=0.30,

    line_dash="dash",

    annotation_text="Low → Medium"

)


fig_probability.add_vline(

    x=0.60,

    line_dash="dash",

    annotation_text="Medium → High"

)


fig_probability.add_vline(

    x=churn_probability,

    line_dash="dot",

    annotation_text="Selected Customer"

)


st.plotly_chart(

    fig_probability,

    use_container_width=True

)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Feature Importance Dashboard</div>',
    unsafe_allow_html=True
)


feature_names = (

    random_forest
    .named_steps["preprocessor"]
    .get_feature_names_out()

)


importance_values = (

    random_forest
    .named_steps["classifier"]
    .feature_importances_

)


importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})


importance_df = (

    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)

)


fig_importance = px.bar(

    importance_df.sort_values(
        "Importance"
    ),

    x="Importance",

    y="Feature",

    orientation="h",

    title="Top 15 Features Influencing Customer Churn"

)


st.plotly_chart(

    fig_importance,

    use_container_width=True

)


st.dataframe(

    importance_df,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# WHAT-IF SCENARIO SIMULATOR
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🔮 What-If Scenario Simulator</div>',
    unsafe_allow_html=True
)


st.write(
    "Change customer engagement and number of products "
    "to observe how the predicted churn probability changes."
)


# ------------------------------------------------------------
# Scenario inputs
# ------------------------------------------------------------

scenario_active = st.selectbox(

    "Scenario — Active Member",

    [0, 1],

    index=is_active_member,

    format_func=lambda x:
        "Active" if x == 1 else "Inactive"

)


scenario_products = st.slider(

    "Scenario — Number of Products",

    min_value=1,

    max_value=4,

    value=num_products

)


# ============================================================
# CREATE SCENARIO
# ============================================================

scenario = customer.copy()


scenario["IsActiveMember"] = (
    scenario_active
)


scenario["NumOfProducts"] = (
    scenario_products
)


# Recalculate engineered features
scenario = create_features(
    scenario
)


scenario = scenario.reindex(
    columns=X.columns
)


# ============================================================
# SCENARIO PREDICTION
# ============================================================

scenario_probability = (

    random_forest
    .predict_proba(scenario)[0][1]

)


probability_change = (

    scenario_probability
    - churn_probability

)


# ============================================================
# SCENARIO RESULTS
# ============================================================

s1, s2, s3 = st.columns(3)


with s1:

    st.metric(
        "Original Churn Risk",
        f"{churn_probability:.1%}"
    )


with s2:

    st.metric(
        "Scenario Churn Risk",
        f"{scenario_probability:.1%}"
    )


with s3:

    st.metric(
        "Probability Change",
        f"{probability_change:+.1%}"
    )


# ============================================================
# SCENARIO INTERPRETATION
# ============================================================

if scenario_probability < churn_probability:

    st.success(
        "✅ The selected scenario decreases the predicted "
        "churn probability."
    )

elif scenario_probability > churn_probability:

    st.error(
        "⚠️ The selected scenario increases the predicted "
        "churn probability."
    )

else:

    st.info(
        "The selected scenario does not change the "
        "predicted churn probability."
    )


# ============================================================
# SCENARIO COMPARISON TABLE
# ============================================================

st.subheader(
    "Scenario Comparison"
)


scenario_comparison = pd.DataFrame({

    "Scenario": [
        "Original Customer",
        "What-If Scenario"
    ],

    "Active Member": [

        "Yes"
        if is_active_member == 1
        else "No",

        "Yes"
        if scenario_active == 1
        else "No"

    ],

    "Number of Products": [

        num_products,

        scenario_products

    ],

    "Churn Probability": [

        f"{churn_probability:.2%}",

        f"{scenario_probability:.2%}"

    ]

})


st.dataframe(

    scenario_comparison,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# RISK DISTRIBUTION BY CATEGORY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Customer Risk Categories</div>',
    unsafe_allow_html=True
)


risk_distribution = pd.DataFrame({

    "Churn Probability":
        test_probabilities

})


risk_distribution["Risk Category"] = (

    risk_distribution["Churn Probability"]
    .apply(risk_category)

)


risk_counts = (

    risk_distribution
    .groupby("Risk Category")
    .size()
    .reset_index(name="Customers")

)


fig_risk = px.bar(

    risk_counts,

    x="Risk Category",

    y="Customers",

    title="Customers by Churn Risk Category"

)


st.plotly_chart(

    fig_risk,

    use_container_width=True

)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Dataset Information"
):

    d1, d2, d3, d4 = st.columns(4)


    with d1:

        st.metric(
            "Total Customers",
            f"{len(df):,}"
        )


    with d2:

        st.metric(
            "Total Columns",
            f"{len(raw_df.columns):,}"
        )


    with d3:

        st.metric(
            "Churned Customers",
            f"{int(df['Exited'].sum()):,}"
        )


    with d4:

        churn_rate = (
            df["Exited"].mean() * 100
        )

        st.metric(
            "Overall Churn Rate",
            f"{churn_rate:.1f}%"
        )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "View Dataset"
):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "European Bank Customer Churn | "
    "Predictive Modeling and Risk Scoring"
)