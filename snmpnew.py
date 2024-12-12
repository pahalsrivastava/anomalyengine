import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

file_path = 'MOCK_DATA.csv'
data = pd.read_csv(file_path)

imputer = SimpleImputer(strategy='mean')
data[['Inbound Octets', 'Outbound Octets', 'Inbound Errors', 'Outbound Errors']] = imputer.fit_transform(
    data[['Inbound Octets', 'Outbound Octets', 'Inbound Errors', 'Outbound Errors']]
)

le_admin_status = LabelEncoder()
le_oper_status = LabelEncoder()

data['Admin Status'] = le_admin_status.fit_transform(data['Admin Status'])
data['Oper Status'] = le_oper_status.fit_transform(data['Oper Status'])

data['Inbound Error Rate'] = data['Inbound Errors'] / (data['Inbound Octets'] + 1)
data['Outbound Error Rate'] = data['Outbound Errors'] / (data['Outbound Octets'] + 1)
data['Traffic Rate'] = (data['Inbound Octets'] + data['Outbound Octets']) / 2

data['Failure'] = data['Oper Status'].apply(lambda x: 1 if x == le_oper_status.transform(['Down'])[0] else 0)

features = [
    'Inbound Octets', 'Outbound Octets', 'Inbound Errors', 
    'Outbound Errors', 'Inbound Error Rate', 
    'Outbound Error Rate', 'Traffic Rate', 'Admin Status'
]
X = data[features]
y = data['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

data['Failure Probability'] = model.predict_proba(X)[:, 1] 

# Display predictions with probabilities
print("Predictions with probabilities on all devices:")
print(data[['Metric Name', 'Failure Probability']].head())
