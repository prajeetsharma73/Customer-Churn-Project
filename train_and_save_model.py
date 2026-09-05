

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# --- Cleaning (identical to notebook) ---
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
total_charges_median = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)

# --- Encoding (identical to notebook) ---
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
categorical_cols = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]
model_columns = X.columns.tolist()
print(f"Total features after one-hot encoding: {len(model_columns)}")

# --- Split + scale (identical to notebook) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# --- SMOTE + train (identical to notebook) ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_res, y_train_res)

# --- Evaluate ---
y_pred = rf_model.predict(X_test)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save artifacts for the Streamlit app ---
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model_columns, "model_columns.pkl")
joblib.dump(total_charges_median, "total_charges_median.pkl")

print("\nSaved: rf_model.pkl, scaler.pkl, model_columns.pkl, total_charges_median.pkl")
print("You can now run: streamlit run app.py")
