import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')
st.title("🧠 Alzheimer's Disease Prediction")
st.caption(f"Model loaded: `{type(model)}`")

# Input form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ['M', 'F'])
    age = st.number_input("Age", min_value=50, max_value=100, value=70)
    educ = st.number_input("Years of Education (Educ)", min_value=0, max_value=25, value=12)
    ses = st.number_input("Socioeconomic Status (SES)", min_value=1, max_value=5, value=2)
    mmse = st.number_input("MMSE Score", min_value=0, max_value=30, value=28)
    etiv = st.number_input("Estimated Total Intracranial Volume (eTIV)", min_value=1000, max_value=2000, value=1500)
    nwbv = st.number_input("Normalized Whole Brain Volume (nWBV)", min_value=0.6, max_value=0.9, value=0.7, step=0.01)
    asf = st.number_input("Atlas Scaling Factor (ASF)", min_value=1.0, max_value=2.0, value=1.5, step=0.01)
    
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Convert gender
    gender_numeric = 1 if gender == 'M' else 0

    # Prepare input data
    input_data = pd.DataFrame({
        'M/F': [gender_numeric],
        'Age': [age],
        'Educ': [educ],
        'SES': [ses],
        'MMSE': [mmse],
        'eTIV': [etiv],
        'nWBV': [nwbv],
        'ASF': [asf]
    })

    # Prediction
    demented_prob = float(model.predict_proba(input_data)[0][1])
    non_demented_prob = 1 - demented_prob

    # Show probabilities
    st.subheader("Prediction Results")
    st.write(f"🟩 **Non-Demented** Probability: `{non_demented_prob * 100:.2f}%`")
    st.write(f"🟥 **Demented** Probability: `{demented_prob * 100:.2f}%`")

    # Interpretation
    if demented_prob > 0.3:
        st.error("⚠️ High Risk: Likely **Demented**")
    else:
        st.success("✅ Low Risk: Likely **Non-Demented**")

    # Visual Progress
    st.progress(min(demented_prob, 1.0))

