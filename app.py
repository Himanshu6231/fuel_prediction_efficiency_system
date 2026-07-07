import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_mpg_model.pkl")

st.title("🚗 Fuel Efficiency Predictor (MPG)")
st.markdown("Enter the car features to predict fuel efficiency (MPG).")

# User Inputs
with st.form("input_form"):
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)
    cylinders = st.number_input("Cylinders", min_value=1, value=4)
    horsepower = st.number_input("Horsepower", min_value=10.0, value=150.0)
    torque = st.number_input("Torque (lb-ft)", min_value=10.0, value=200.0)
    weight = st.number_input("Weight (lbs)", min_value=500.0, value=3000.0)
    drag_coefficient = st.number_input("Drag Coefficient", min_value=0.2, value=0.3)
    acceleration_time = st.number_input("0-60 mph Time (s)", min_value=1.0, value=8.0)

    transmission = st.selectbox("Transmission", ['Automatic', 'Manual'])
    drivetrain = st.selectbox("Drivetrain", ['FWD', 'RWD', 'AWD'])
    tire_type = st.selectbox("Tire Type", ['All-Season', 'Summer', 'Winter'])
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
    fuel_injection = st.selectbox("Fuel Injection", ['Direct', 'Port'])
    turbocharged = st.selectbox("Turbocharged", ['Yes', 'No'])
    hybrid_system = st.selectbox("Hybrid System", ['Yes', 'No'])

    submitted = st.form_submit_button("Predict MPG")

# Predict
if submitted:
    df = pd.DataFrame([{
        "engine_size": engine_size,
        "cylinders": cylinders,
        "horsepower": horsepower,
        "torque": torque,
        "weight": weight,
        "drag_coefficient": drag_coefficient,
        "acceleration_time": acceleration_time,
        "power_to_weight": horsepower / weight,
        "torque_to_weight": torque / weight,
        "engine_efficiency": horsepower / (engine_size if engine_size > 0 else 0.1),
        "transmission": transmission,
        "drivetrain": drivetrain,
        "tire_type": tire_type,
        "fuel_type": fuel_type,
        "fuel_injection": fuel_injection,
        "turbocharged": turbocharged,
        "hybrid_system": hybrid_system
    }])

    prediction = model.predict(df)[0]
    st.success(f"🎯 Estimated MPG: **{prediction:.2f}**")

# Charts Section
st.header("📈 Model Visualizations")

with st.expander("Correlation Heatmap"):
    st.image("correlation_heatmap.png", use_column_width=True)

with st.expander("Top 10 Feature Importances"):
    st.image("feature_importance.png", use_column_width=True)

with st.expander("Actual vs Predicted MPG"):
    st.image("actual_vs_predicted.png", use_column_width=True)

st.markdown("---")
st.caption("Powered by XGBoost + Streamlit")
