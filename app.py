import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "logistic_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "diabetes.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_preprocessing_statistics():
    df = pd.read_csv(DATA_PATH)

    # Replace zero values with missing values
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    # Fill missing values using median
    medians = df[ZERO_AS_MISSING].median()

    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].fillna(medians)

    # Calculate IQR bounds
    bounds = {}

    for col in FEATURES + ["Outcome"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        bounds[col] = (lower_bound, upper_bound)

    return medians.to_dict(), bounds


def preprocess_input(input_df, medians, bounds):

    X = input_df.copy()

    # Same zero-value handling used during training
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    # Fill missing values
    for col in ZERO_AS_MISSING:
        X[col] = X[col].fillna(medians[col])

    # Apply IQR capping
    for col in FEATURES:
        lower_bound, upper_bound = bounds[col]

        X[col] = X[col].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return X[FEATURES]


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction")
st.subheader("Logistic Regression Model")

st.write(
    "Enter the patient information below to obtain "
    "a diabetes outcome prediction."
)

st.info(
    "This application is created for educational and "
    "machine-learning purposes. It is not a medical diagnosis."
)


# ---------------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------------

try:

    model, scaler = load_artifacts()

    medians, bounds = load_preprocessing_statistics()

except Exception as exc:

    st.error(
        "The model, scaler, or dataset could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ---------------------------------------------------------
# User Input Form
# ---------------------------------------------------------

with st.form("prediction_form"):

    st.markdown("### Patient Information")

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=30.0,
        step=0.1,
        format="%.1f"
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.2f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    input_df = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=FEATURES
    )

    # Preprocess input
    processed_data = preprocess_input(
        input_df,
        medians,
        bounds
    )

    # Scale input
    scaled_data = scaler.transform(processed_data)

    # Prediction
    prediction = int(
        model.predict(scaled_data)[0]
    )

    probability = float(
        model.predict_proba(scaled_data)[0, 1]
    )

    st.markdown("### Prediction Result")

    if prediction == 1:

        st.error(
            "Model Prediction: Diabetes Outcome = 1"
        )

    else:

        st.success(
            "Model Prediction: Diabetes Outcome = 0"
        )

    st.metric(
        "Probability of Outcome = 1",
        f"{probability:.2%}"
    )

    with st.expander(
        "View Processed Input"
    ):

        st.dataframe(
            processed_data,
            use_container_width=True
        )


st.caption(
    "Developed for academic and learning purposes."
)