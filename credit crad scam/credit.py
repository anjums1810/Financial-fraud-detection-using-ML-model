import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from imblearn.over_sampling import SMOTE

# Step 1: Load the data
data = pd.read_csv(r"C:\Users\anju.ms\Downloads\archive (1)\creditcard.csv")  # Replace with your file path

# Step 2: Split into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column (target)

# Step 3: Split into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Train a model (e.g., Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Step 6: Evaluate the model
y_test_pred = model.predict(X_test)
y_test_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for Class 1

# Metrics
print("Testing Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print(f"Binary Cross-Entropy Loss: {log_loss(y_test, y_test_probs)}")

# Step 7: Validate the model
y_val_pred = model.predict(X_val)
y_val_probs = model.predict_proba(X_val)[:, 1]  # Probabilities for Class 1

print("\nValidation Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))
print(f"Binary Cross-Entropy Loss: {log_loss(y_val, y_val_probs)}")