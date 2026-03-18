import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("data/loan_data.csv")

# -----------------------------
# Data Cleaning (same as before)
# -----------------------------
categorical_cols = [
    "Gender", "Married", "Dependents",
    "Self_Employed", "Credit_History",
    "Loan_Amount_Term"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

text_cols = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed",
    "Property_Area", "Loan_Status"
]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# -----------------------------
# Encoding categorical data
# -----------------------------
label_encoders = {}

for col in text_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

# -----------------------------
# Train test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# -----------------------------
# Save model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model and encoders saved successfully!")