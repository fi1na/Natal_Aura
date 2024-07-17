import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import warnings
from io import StringIO
import requests

# Load models
maternal_model = pickle.load(open("Models/finalized_maternal_model.sav", 'rb'))
fetal_model = pickle.load(open("Models/fetal_health_classifier.sav", 'rb'))


class MaternalHealthDashboard:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.maternal_health_data = self.fetch_data()

    def fetch_data(self):
        try:
            response = requests.get(self.api_endpoint)
            if response.status_code == 200:
                data = pd.read_csv(StringIO(response.text))
                return data
            else:
                st.error(f"Failed to fetch data. Status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error during API request: {e}")
            return None

    def drop_all_india(self, df):
        return df[df["State/UT"] != "All India"]

    def create_bubble_chart(self):
        df = self.drop_all_india(self.maternal_health_data)
        st.subheader("Bubble Chart provides a visual representation of how well different regions have performed in achieving institutional deliveries compared to their assessed needs")
        fig = px.scatter(
            df,
            x="Need Assessed (2019-20) - (A)",
            y="Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)",
            size="% Achvt of need assessed (2019-20) - (E=(B/A)*100)",
            color="State/UT",
            hover_name="State/UT",
            labels={
                "Need Assessed (2019-20) - (A)": "Need Assessed",
                "Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)": "Achievement",
                "% Achvt of need assessed (2019-20) - (E=(B/A)*100)": "% Achievement",
            },
        )
        st.plotly_chart(fig)

    def create_pie_chart(self):
        st.subheader("Visualize the proportion of institutional deliveries across different states/union territories (UTs) during the specified period (April to June 2019-20)")
        df = self.drop_all_india(self.maternal_health_data)

        fig = px.pie(
            df,
            names="State/UT",
            values="Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)",
            labels={
                "Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)": "Institutional Deliveries"}
        )
        st.plotly_chart(fig)

    def get_bubble_chart_data(self):
        content = """
Bubble Chart provides a visual representation of how well different regions have performed in achieving institutional deliveries compared to their assessed needs. 

The Bubble Chart presented in the example is visualizing maternal health data, particularly focusing on the achievement of institutional deliveries in different states or union territories during the period of April to June for the year 2019-20. Let's break down what the chart is showing:

1: X-axis (horizontal axis): Need Assessed (2019-20) - (A)

This axis represents the assessed needs for maternal health in different states or union territories. Each point on the X-axis corresponds to a specific region, and the position along the axis indicates the magnitude of the assessed needs.

2: Y-axis (vertical axis): Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)

The Y-axis represents the actual achievement in terms of the number of institutional deliveries during the specified period (April to June) in the year 2019-20. Each point on the Y-axis corresponds to a specific region, and the position along the axis indicates the magnitude of the achieved institutional deliveries.

3: Bubble Size: % Achvt of need assessed (2019-20) - (E=(B/A)100)

The size of each bubble is determined by the percentage achievement of the assessed needs, calculated as % Achvt = (B/A) * 100. Larger bubbles indicate a higher percentage of achievement compared to the assessed needs, suggesting a better performance in delivering institutional healthcare.

4: Color: State/UT

Each bubble is color-coded based on the respective state or union territory it represents. Different colors distinguish between regions, making it easy to identify and compare data points for different states or union territories.

5: Hover Name: State/UT

Hovering over a bubble reveals additional information, such as the name of the state or union territory it represents. This interactive feature allows users to explore specific data points on the chart.
"""
        return content

    def get_pie_graph_data(self):
        content = """
Visualize the proportion of institutional deliveries across different states/union territories (UTs) during the specified period (April to June 2019-20). Let's break down the components of the graph and its interpretation:

Key Components:
Slices of the Pie:

Each slice of the pie represents a specific state or UT.
Size of Slices:

The size of each slice corresponds to the proportion of institutional deliveries achieved during April to June 2019-20 for the respective state or UT.
Hover Information:

Hovering over a slice provides additional information, such as the name of the state/UT and the exact proportion of institutional deliveries.
"""
        return content


# Sidebar navigation
with st.sidebar:
    st.title("NatalAura")
    st.write("Welcome to NatalAura")
    selected = st.radio('Navigation',
                        ['About us',
                         'Pregnancy Risk Prediction',
                         'Fetal Health Prediction',
                         'Dashboard'],
                        index=1)

if selected == 'About us':
    st.title("Welcome to NatalAura")
    st.write("At NatalAura, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. "
             "Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate "
             "predictions and proactive risk management.")

    # Main image centered
    st.image("images/mat-fet.png", caption="NatalAura", use_column_width=True)

    # Section 1: Pregnancy Risk Prediction
    st.header("1. Pregnancy Risk Prediction")
    st.write("Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, "
             "body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of "
             "potential risks during pregnancy.")

    # Section 2: Fetal Health Prediction
    st.header("2. Fetal Health Prediction")
    st.write("Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the "
             "health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, "
             "and genetic factors, we deliver insights into the well-being of the unborn child.")

    # Section 3: Dashboard
    st.header("3. Dashboard")
    st.write("Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic "
             "view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard "
             "is designed for ease of use and accessibility.")

    # Closing note
    st.write("Thank you for choosing NatalAura. We are committed to advancing healthcare through technology and predictive analytics. "
             "Feel free to explore our features and take advantage of the insights we provide.")

# Main content based on selection
if selected == 'Pregnancy Risk Prediction':
    st.title('Pregnancy Risk Prediction')
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{
                content}</b></div></br>", unsafe_allow_html=True)

    # Input fields for user data
    age = st.text_input('Age of the Person', key="age")
    diastolicBP = st.text_input(
        'Diastolic Blood Pressure (mmHg)', key="diastolicBP")
    BS = st.text_input('Blood Glucose (mmol/L)', key="BS")
    bodyTemp = st.text_input('Body Temperature (Celsius)', key="bodyTemp")
    heartRate = st.text_input('Heart Rate (beats per minute)', key="heartRate")

    # Prediction and result display
    if st.button('Predict Pregnancy Risk'):
        if age and diastolicBP and BS and bodyTemp and heartRate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Convert inputs to float and predict risk
                try:
                    predicted_risk = maternal_model.predict(
                        [[float(age), float(diastolicBP), float(BS), float(bodyTemp), float(heartRate)]])
                    st.subheader("Risk Level:")
                    if predicted_risk[0] == 0:
                        st.markdown(
                            '<p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p>', unsafe_allow_html=True)
                    elif predicted_risk[0] == 1:
                        st.markdown(
                            '<p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
        else:
            st.error("Please fill in all fields.")

    # Clear button to reset inputs
    if st.button("Clear"):
        st.experimental_rerun()

elif selected == 'Fetal Health Prediction':
    st.title('Fetal Health Prediction')
    content = "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    # Input fields for fetal health prediction
    BaselineValue = st.text_input('Baseline Value')
    Accelerations = st.text_input('Accelerations')
    fetal_movement = st.text_input('Fetal Movement')
    uterine_contractions = st.text_input('Uterine Contractions')
    light_decelerations = st.text_input('Light Decelerations')
    severe_decelerations = st.text_input('Severe Decelerations')
    prolongued_decelerations = st.text_input('Prolongued Decelerations')
    abnormal_short_term_variability = st.text_input(
        'Abnormal Short Term Variability')
    mean_value_of_short_term_variability = st.text_input(
        'Mean Value Of Short Term Variability')
    percentage_of_time_with_abnormal_long_term_variability = st.text_input(
        'Percentage Of Time With ALTV')
    mean_value_of_long_term_variability = st.text_input(
        'Mean Value Long Term Variability')
    histogram_width = st.text_input('Histogram Width')
    histogram_min = st.text_input('Histogram Min')
    histogram_max = st.text_input('Histogram Max')
    histogram_number_of_peaks = st.text_input('Histogram Number Of Peaks')
    histogram_number_of_zeroes = st.text_input('Histogram Number Of Zeroes')
    histogram_mode = st.text_input('Histogram Mode')
    histogram_mean = st.text_input('Histogram Mean')
    histogram_median = st.text_input('Histogram Median')
    histogram_variance = st.text_input('Histogram Variance')
    histogram_tendency = st.text_input('Histogram Tendency')

    # Prediction button
    if st.button('Predict Fetal Health'):
        if all([BaselineValue, Accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations, prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability, histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median, histogram_variance, histogram_tendency]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Convert inputs to float
                    inputs = list(map(float, [BaselineValue, Accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations, prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability,
                                  mean_value_of_long_term_variability, histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median, histogram_variance, histogram_tendency]))
                    predicted_health = fetal_model.predict([inputs])
                    # Display prediction result
                    st.subheader("Fetal Health Prediction:")
                    if predicted_health[0] == 0:
                        st.markdown(
                            '<p style="font-weight: bold; font-size: 20px; color: green;">Normal</p>', unsafe_allow_html=True)
                    elif predicted_health[0] == 1:
                        st.markdown(
                            '<p style="font-weight: bold; font-size: 20px; color: orange;">Suspect</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<p style="font-weight: bold; font-size: 20px; color: red;">Pathological</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
        else:
            st.error("Please fill in all fields.")

    # Clear button for fetal health inputs
    if st.button("Clear"):
        st.experimental_rerun()

elif selected == "Dashboard":
    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={
        api_key}&format=csv"
    st.header("Dashboard")
    content = "Our interactive dashboard offers a comprehensive visual representation of maternal health achievements across diverse regions. The featured chart provides insights into the performance of each region concerning institutional deliveries compared to their assessed needs. It serves as a dynamic tool for assessing healthcare effectiveness, allowing users to quickly gauge the success of maternal health initiatives."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{
                content}</b></div></br>", unsafe_allow_html=True)

    dashboard = MaternalHealthDashboard(api_endpoint)
    dashboard.create_bubble_chart()
    with st.expander("Show More"):
        content = dashboard.get_bubble_chart_data()
        st.markdown(
            f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)

    dashboard.create_pie_chart()
    with st.expander("Show More"):
        content = dashboard.get_pie_graph_data()
        st.markdown(
            f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)
