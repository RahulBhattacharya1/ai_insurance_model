import os
import joblib
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Insurance Charge Predictor", page_icon=":bar_chart:", layout="centered")

st.title("Insurance Charge Predictor")
st.write("Enter details and get an estimated insurance **charges** prediction.")

# ---------- Load model ----------
MODEL_PATH = os.path.join("models", "insurance_model.joblib")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- Input form ----------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1, format="%.1f")
        children = st.number_input("Children", min_value=0, max_value=10, value=1, step=1)
    with col2:
        sex = st.selectbox("Sex", options=["male","female"], index=0)
        smoker = st.selectbox("Smoker", options=["yes","no"], index=1)
        region = st.selectbox("Region", options=["northeast","northwest","southeast","southwest"], index=0)

    submitted = st.form_submit_button("Predict")

# ---------- Predict ----------
if submitted:
    # Build a single-row DataFrame with the exact training columns
    data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    try:
        pred = float(model.predict(data)[0])
        st.success(f"Estimated Charges: ${pred:,.2f}")
        st.caption("This is a model estimate based on your inputs.")
    except Exception as e:
        st.error("Prediction failed. Make sure the model file matches these inputs.")
        st.exception(e)

# ---------- About section ----------
with st.expander("How this works"):
    st.write(
        """
        The model is a scikit-learn Pipeline that one-hot encodes categorical columns
        (sex, smoker, region) and uses a RandomForestRegressor trained on your Kaggle
        insurance.csv data to predict charges.
        """
    )
