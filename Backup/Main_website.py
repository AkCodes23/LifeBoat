import pandas as pd
import joblib

# Load the saved model and scaler
model_file = 'models/random_forest_risk_model.pkl'
scaler_file = 'models/scaler.pkl'

loaded_model = joblib.load(model_file)
loaded_scaler = joblib.load(scaler_file)

# Function to calculate derived values
def calculate_derived_values(weight, height, systolic_bp, diastolic_bp):
    pulse_pressure = systolic_bp - diastolic_bp
    bmi = weight / (height ** 2)
    map_value = (systolic_bp + 2 * diastolic_bp) / 3
    return pulse_pressure, bmi, map_value

# Function to test with new values
def predict_risk(weight, height, gender, age, systolic_bp, diastolic_bp, heart_rate, body_temp, spo2):
    # Calculate derived values
    pulse_pressure, bmi, map_value = calculate_derived_values(weight, height, systolic_bp, diastolic_bp)
    
    # Prepare input data
    input_values = [
        heart_rate, body_temp, spo2, systolic_bp, diastolic_bp,
        age, gender, weight, height, pulse_pressure, bmi, map_value
    ]
    
    # Convert input values to a DataFrame
    columns = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 
               'Systolic Blood Pressure', 'Diastolic Blood Pressure', 
               'Age', 'Gender', 'Weight (kg)', 'Height (m)', 
               'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP']
    
    input_data = pd.DataFrame([input_values], columns=columns)
    
    # Scale the input data
    input_data_scaled = loaded_scaler.transform(input_data)
    
    # Predict
    prediction = loaded_model.predict(input_data_scaled)
    
    risk_category = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    return risk_category

# Example usage with user inputs and sensor readings
weight = 85  # kg
height = 1.75  # m
gender = 1  # Male
age = 45  # years
systolic_bp = 120  # mmHg
diastolic_bp = 80  # mmHg
heart_rate = 100  # bpm
body_temp = 38.5  # °C
spo2 = 92  # %

risk_prediction = predict_risk(weight, height, gender, age, systolic_bp, diastolic_bp, heart_rate, body_temp, spo2)
print(f"Predicted Risk Category: {risk_prediction}")
