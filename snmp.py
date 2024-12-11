import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import streamlit as st

file_path = "dataset_router1.csv"

ERROR_THRESHOLD = 500

def label_failure(row):
    """Label rows based on error thresholds."""
    return int(row['Inbound Errors'] > ERROR_THRESHOLD or row['Outbound Errors'] > ERROR_THRESHOLD)

# Streamlit UI
st.title("Device Failure Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Create a target column based on error thresholds
    df['Device Failure'] = df.apply(label_failure, axis=1)

    # Encode categorical columns
    categorical_cols = ['Interface Type', 'Admin Status', 'Oper Status']
    encoder = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(encoder.fit_transform)

    # Define feature columns and target column
    FEATURES = ['Inbound Octets', 'Outbound Octets', 'Inbound Errors', 'Outbound Errors',
                'Interface Type', 'Admin Status', 'Oper Status']
    TARGET = 'Device Failure'

    X = df[FEATURES]
    y = df[TARGET]

    # Scale the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Display evaluation metrics
    st.subheader("Model Performance")
    st.text("Best Parameters:")
    st.json(grid_search.best_params_)

    st.text("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred))

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Accuracy Score:")
    st.text(accuracy_score(y_test, y_pred))


    # Save the trained model
    model_file = "device_failure_model.pkl"
    joblib.dump(best_model, model_file)
    st.success(f"Model saved to {model_file}")
