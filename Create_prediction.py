
import os
import json
import joblib
import streamlit as st
import pandas as pd

from ML_model import FEATURES, TARGET


MODEL_PATH = "model_artifacts/best_model.pkl"
METRICS_PATH = "model_artifacts/model_metrics.json"


def render_prediction_page(df):

    st.title("🚗 CO2 Emissions Prediction")

    st.markdown(
        "Predict vehicle CO2 emissions using machine learning."
    )

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "Prediction model not found. "
            "Please run: python train_model.py"
        )
        return

    model = joblib.load(MODEL_PATH)

    st.subheader("Vehicle Specifications")

    col1, col2 = st.columns(2)

    with col1:

        make = st.selectbox(
            "Vehicle Manufacturer",
            sorted(df["Make"].dropna().unique())
        )

        vehicle_class = st.selectbox(
            "Vehicle Class",
            sorted(df["Vehicle Class"].dropna().unique())
        )

        engine_size = st.number_input(
            "Engine Size (L)",
            min_value=0.1,
            max_value=20.0,
            value=float(df["Engine Size(L)"].median()),
            step=0.1
        )

        cylinders = st.selectbox(
            "Cylinders",
            sorted(df["Cylinders"].dropna().unique())
        )

        transmission = st.selectbox(
            "Transmission",
            sorted(df["Transmission"].dropna().unique())
        )

    with col2:

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(df["Fuel Type"].dropna().unique())
        )

        city = st.number_input(
            "City Fuel Consumption (L/100 km)",
            min_value=0.1,
            max_value=100.0,
            value=float(
                df["Fuel Consumption City (L/100 km)"].median()
            ),
            step=0.1
        )

        highway = st.number_input(
            "Highway Fuel Consumption (L/100 km)",
            min_value=0.1,
            max_value=100.0,
            value=float(
                df["Fuel Consumption Hwy (L/100 km)"].median()
            ),
            step=0.1
        )

        combined = st.number_input(
            "Combined Fuel Consumption (L/100 km)",
            min_value=0.1,
            max_value=100.0,
            value=float(
                df["Fuel Consumption Comb (L/100 km)"].median()
            ),
            step=0.1
        )

    if st.button("🔮 Predict CO2 Emissions", type="primary"):

        input_data = pd.DataFrame([{
            "Make": make,
            "Vehicle Class": vehicle_class,
            "Engine Size(L)": engine_size,
            "Cylinders": cylinders,
            "Transmission": transmission,
            "Fuel Type": fuel_type,
            "Fuel Consumption City (L/100 km)": city,
            "Fuel Consumption Hwy (L/100 km)": highway,
            "Fuel Consumption Comb (L/100 km)": combined
        }])

        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted CO2 Emissions",
            value=f"{prediction:.2f} g/km"
        )

        st.success(
            f"Estimated vehicle CO2 emissions: "
            f"{prediction:.2f} g/km"
        )

    st.divider()

    st.subheader("📊 Model Performance Comparison")

    if os.path.exists(METRICS_PATH):

        metrics = pd.read_json(METRICS_PATH)

        st.dataframe(
            metrics.style.format({
                "R2 Score": "{:.4f}",
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}"
            }),
            use_container_width=True
        )

        import plotly.express as px

        fig = px.bar(
            metrics,
            x="Model",
            y="RMSE",
            title="Model Comparison - RMSE",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)
