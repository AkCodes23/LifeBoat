import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Load the dataset
try:
    data = pd.read_csv(r"D:\Projects\LifeBoat\Backup\Main.csv")
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Inspect the first few rows to verify loading
print(data.head())

# Encode target variable (Risk Category)
data['Risk Category'] = data['Risk Category'].apply(lambda x: 1 if x == 'High Risk' else 0)

# Drop rows with missing values
data.dropna(inplace=True)

# Check for any missing columns
print("Columns after preprocessing:", data.columns)

# Encode categorical feature (Gender)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features and target variable
X = data[['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Systolic Blood Pressure', 
          'Diastolic Blood Pressure', 'Age', 'Gender', 'Weight (kg)', 'Height (m)', 
          'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP']]

y = data['Risk Category']

# Train-test split (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a Random Forest model with reduced complexity
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, 
                               min_samples_leaf=5, class_weight='balanced', random_state=42)

# Cross-validation (5-fold)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")

# Train the model
print("Training model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create directory for saving models
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
model_file = os.path.join(model_dir, 'random_forest_risk_model.pkl')
joblib.dump(model, model_file)
print(f"Random Forest model saved to {model_file}")

# Save the scaler
scaler_file = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}")

# Load the saved model and scaler for testing
loaded_model = joblib.load(model_file)
loaded_scaler = joblib.load(scaler_file)

# Function to test with new values
def test_model(input_values):
    # Convert input values to a DataFrame
    columns = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Systolic Blood Pressure', 
               'Diastolic Blood Pressure', 'Age', 'Gender', 'Weight (kg)', 'Height (m)', 
               'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP']
    
    input_data = pd.DataFrame([input_values], columns=columns)
    print("Input Data:")
    print(input_data)
    
    # Scale the input data
    input_data_scaled = loaded_scaler.transform(input_data)
    print("Scaled Data:")
    print(input_data_scaled)
    
    # Predict
    prediction = loaded_model.predict(input_data_scaled)
    
    risk_category = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    print(f"Predicted Risk Category: {risk_category}")

# Example: Test the model with new input values
test_model([85, 36.5, 98, 120, 80, 45, 1, 70, 1.75, 40, 22.9, 95])
