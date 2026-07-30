import streamlit as st
import joblib
import numpy as np

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Multi Model ML App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Machine Learning Multi-Model App")
st.write("Select Problem Type and Enter Features for Prediction")

# =====================================
# Select Problem Type
# =====================================
problem = st.selectbox(
    "Choose Problem Type",
    ["Classification", "Regression"]
)

# =====================================
# CLASSIFICATION
# =====================================
if problem == "Classification":

    st.header("🌸 Iris Flower Classification")

    # Load Model
    try:
        model = joblib.load("best_classification_model.pkl")
        scaler = joblib.load("classification_scaler.pkl")
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
        st.stop()

    # Input Fields
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        format="%.2f"
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        format="%.2f"
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        format="%.2f"
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        format="%.2f"
    )

    # Prediction
    if st.button("Predict Class"):

        data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        data = scaler.transform(data)
        prediction = model.predict(data)[0]

        class_names = {
            0: "🌸 Setosa",
            1: "🌼 Versicolor",
            2: "🌺 Virginica"
        }

        st.success(
            f"Predicted Flower: {class_names.get(prediction, prediction)}"
        )

# =====================================
# REGRESSION
# =====================================
else:

    st.header("🩺 Diabetes Disease Progression Prediction")

    # Load Model
    try:
        model = joblib.load("best_regression_model.pkl")
        scaler = joblib.load("regression_scaler.pkl")
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
        st.stop()

    st.subheader("Enter Patient Details")

    # Input Fields
    age = st.number_input(
        "Age",
        value=0.0,
        format="%.4f"
    )

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    # Encoding
    # Female = 0
    # Male = 1
    sex = 1.0 if sex == "Male" else 0.0

    bmi = st.number_input(
        "BMI (Body Mass Index)",
        value=0.0,
        format="%.4f"
    )

    bp = st.number_input(
        "Average Blood Pressure",
        value=0.0,
        format="%.4f"
    )

    s1 = st.number_input(
        "S1 (Total Cholesterol)",
        value=0.0,
        format="%.4f"
    )

    s2 = st.number_input(
        "S2 (LDL)",
        value=0.0,
        format="%.4f"
    )

    s3 = st.number_input(
        "S3 (HDL)",
        value=0.0,
        format="%.4f"
    )

    s4 = st.number_input(
        "S4 (Cholesterol / HDL Ratio)",
        value=0.0,
        format="%.4f"
    )

    s5 = st.number_input(
        "S5 (Triglycerides)",
        value=0.0,
        format="%.4f"
    )

    s6 = st.number_input(
        "S6 (Blood Sugar)",
        value=0.0,
        format="%.4f"
    )

    # Prediction Button
    if st.button("Predict Disease Progression"):

        features = np.array([[
            age,
            sex,
            bmi,
            bp,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        st.success(
            f"Predicted Disease Progression Score: {prediction:.2f}"
        )