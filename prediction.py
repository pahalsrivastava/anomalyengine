# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('scada_data.csv')

np.random.seed(42)
data['failure'] = np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])

print("Dataset Preview with Failure Column:")
print(data.head())

data.drop(columns=['timestamp'], inplace=True, errors='ignore')

X = data.drop(columns=['failure'])
y = data['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

def predict_failure(new_data):
    """
    Predict if a device will fail based on sensor readings.
    :param new_data: List or numpy array of new sensor readings
    :return: "Failure" or "No Failure"
    """
    prediction = model.predict([new_data])
    return "Failure" if prediction[0] == 1 else "No Failure"

new_sensor_readings = [80.0, 1.5, 27.0, 220.0, 10.5, 40]
result = predict_failure(new_sensor_readings)
print("\nPrediction for New Sensor Readings:", result)
