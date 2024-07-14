import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Streamlit app title
st.title("Machine Predictive Maintenance Classification")

page = st.sidebar.selectbox('Page Navigation', ["Problem statement", "Predictor",])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Shaloy Lewis](https://www.linkedin.com/in/shaloy-lewis/)")

if page == "Problem statement":
    st.write("""
    Predicting machine failures and classifying their probable types offer significant business advantages, including minimized downtime, cost savings, increased equipment lifespan, enhanced safety, and improved productivity. By enabling **proactive maintenance** and **scheduled repairs**, businesses can reduce repair costs, manage inventory efficiently, and prevent catastrophic failures. This leads to **consistent performance**, **operational efficiency**, and **data-driven decision-making**.
    
Additionally, it boosts **customer satisfaction** through reliable delivery and quality control, supports **sustainability** through resource efficiency, and provides a **competitive edge** and improved brand reputation. Overall, **predictive maintenance** ensures smoother operations and a stronger bottom line.
    
This neural network-driven model specifically addresses these needs by:
1. **Predicting Failure Probability**: It calculates the likelihood of tool failure based on operational conditions, allowing for timely intervention.
2. **Classifying Failure Types**: It identifies the most likely type of failure under given conditions, aiding in accurate and efficient troubleshooting and repairs.
    """)

else:
    st.subheader("Enter tool environment parameters")
    col1 = st.columns(2)
    air_temperature_k = col1[0].number_input('Air temperature [K]', value=298.9, min_value=295.0, max_value=304.0)
    process_temperature_k = col1[1].number_input('Process temperature [K]', value=309.1, min_value=305.0, max_value=314.0)
    
    col2 = st.columns(2)
    rotational_speed_rpm = col2[0].number_input('Rotational speed [rpm]', value=2861.0, min_value=1168.0, max_value=2886.0)
    torque_nm = col2[1].number_input('Torque [Nm]', value=4.6, min_value=3.8, max_value=76.6)
    
    col3 = st.columns(2)
    tool_wear_min = col3[0].number_input('Tool wear [min]', value=143, min_value=0, max_value=253)
    Type = col3[1].selectbox('Tool quality', ['H', 'L', 'M'])

    if st.button('Predict'):
        # Create a CustomData instance
        data = CustomData(
            air_temperature_k=air_temperature_k,
            process_temperature_k=process_temperature_k,
            rotational_speed_rpm=rotational_speed_rpm,
            torque_nm=torque_nm,
            tool_wear_min=tool_wear_min,
            Type=Type
        )

        # Convert the data to a DataFrame
        input_df = data.get_data_as_dataframe()
        
        failure_prob, failure_type = predict_pipeline.predict(input_df)

        # Construct the output in the desired format
        failure_type_probs = {
            "Heat Dissipation Failure": round(failure_type[0].tolist()[0], 6),
            "Overstrain Failure": round(failure_type[0].tolist()[2], 6),
            "Power Failure": round(failure_type[0].tolist()[3], 6),
            "Random Failures": round(failure_type[0].tolist()[4], 6),
            "Tool Wear Failure": round(failure_type[0].tolist()[5], 6)
        }

        # Sort the failure types by probability in descending order
        sorted_failure_type_probs = dict(sorted(failure_type_probs.items(), key=lambda item: item[1], reverse=True))

        output = {
            "probability_of_failure": round(failure_prob[0].tolist()[0], 6),
            "failure_type": sorted_failure_type_probs
        }
        
        st.subheader(f"Probability of tool failure: {output['probability_of_failure']}")
        st.subheader("Tool failure type probabilities:")
        for failure_type, prob in output['failure_type'].items():
            st.write(f"**{failure_type}:** {prob}")