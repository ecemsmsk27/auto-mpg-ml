import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Load the trained model
model = joblib.load("model.pkl")

# Title of the web app
st.title("Auto MPG Prediction")

# Sidebar for navigation
st.sidebar.header("Options")

# Sayfa durumu için session_state kullanalım
if "page" not in st.session_state:
    st.session_state["page"] = "Prediction"  # Varsayılan sayfa

# CSS ile butonu sidebar genişliğinde yapalım
st.sidebar.markdown(
    """
    <style>
        div.stButton > button {
            width: 100%;
            border: 2px solid black;
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Butonlara tıklanınca sayfa değiştir
if st.sidebar.button("Prediction"):
    st.session_state["page"] = "Prediction"

if st.sidebar.button("Data Visualization"):
    st.session_state["page"] = "Data Visualization"

# Seçilen sayfayı göster
page = st.session_state["page"]
st.write(f"Current Page: {page}")

if page == "Prediction":
    st.header("Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cylinders = st.number_input("Cylinders", min_value=1, max_value=8, value=4)
        displacement = st.number_input("Displacement", min_value=50, max_value=500, value=150)
        horsepower = st.number_input("Horsepower", min_value=50, max_value=300, value=100)
        weight = st.number_input("Weight", min_value=1500, max_value=5000, value=3000)
    
    with col2:
        acceleration = st.number_input("Acceleration", min_value=0.0, max_value=30.0, value=15.0)
        model_year = st.number_input("Model Year", min_value=70, max_value=82, value=75)
        origin = st.selectbox("Origin", options=["USA", "Europe", "Japan"])

    # Convert origin to numeric
    origin_dict = {"USA": 1, "Europe": 2, "Japan": 3}
    origin_numeric = origin_dict[origin]

    # Feature engineering
    horsepower_per_weight = horsepower / weight
    displacement_per_cylinder = displacement / cylinders
    horsepower_squared = horsepower ** 2
    weight_cubed = weight ** 3
    displacement_squared = displacement ** 2
    weight_horsepower_interaction = weight * horsepower
    displacement_weight_interaction = displacement * weight

    # Button to make prediction
    if st.button("Predict MPG"):
        input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin_numeric,
                                horsepower_per_weight, displacement_per_cylinder, horsepower_squared, weight_cubed,
                                displacement_squared, weight_horsepower_interaction, displacement_weight_interaction]])
        prediction = model.predict(input_data)
        st.success(f"Predicted MPG: {prediction[0]:.2f}")

elif page == "Data Visualization":
    st.header("Data Visualization")
    chart_data = pd.read_csv('auto-mpg.csv')

    # MPG Distribution
    st.subheader("MPG Distribution")
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('mpg', bin=True),
        y='count()',
        tooltip=['mpg', 'count()']
    ).properties(
        title='MPG Distribution'
    )
    st.altair_chart(chart, use_container_width=True)

    # Weight vs MPG
    st.subheader("Weight vs MPG")
    chart = alt.Chart(chart_data).mark_circle(size=60).encode(
        x='weight',
        y='mpg',
        color='origin',
        tooltip=['weight', 'mpg', 'origin']
    ).properties(
        title='Weight vs MPG'
    )
    st.altair_chart(chart, use_container_width=True)

    # Horsepower vs MPG
    st.subheader("Horsepower vs MPG")
    chart = alt.Chart(chart_data).mark_circle(size=60).encode(
        x='horsepower',
        y='mpg',
        color='origin',
        tooltip=['horsepower', 'mpg', 'origin']
    ).properties(
        title='Horsepower vs MPG'
    )
    st.altair_chart(chart, use_container_width=True)

    # Origin vs MPG Box Plot
    st.subheader("MPG by Origin")
    chart = alt.Chart(chart_data).mark_boxplot().encode(
        x='origin',
        y='mpg',
        color='origin',
        tooltip=['origin', 'mpg']
    ).properties(
        title='MPG by Origin'
    )
    st.altair_chart(chart, use_container_width=True)
