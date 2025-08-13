import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import os

# Create a directory to save plots (if it doesn't exist)
output_dir = "model_evaluation_plots"
os.makedirs(output_dir, exist_ok=True)

# Load data
data = pd.read_csv(r"C:\Users\anju.ms\Downloads\archive (1)\creditcard.csv")  # Replace with your file path
X = data.iloc[:, :-1]  # All columns except last
y = data.iloc[:, -1]   # Last column (target: 0=non-fraud, 1=fraud)

# Split into train (60%), test (20%), validation (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Random Forest with class weights
model = RandomForestClassifier(
    class_weight='balanced',  # Penalize misclassifying fraud more
    random_state=42
)
model.fit(X_train_res, y_train_res)

# --- Evaluation with Plot Saving ---
def evaluate_model(X, y, model, threshold=0.5, dataset_name="dataset"):
    y_probs = model.predict_proba(X)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y, y_probs)
    avg_precision = average_precision_score(y, y_probs)
    
    print(f"\n=== {dataset_name.upper()} Evaluation (Threshold={threshold:.2f}) ===")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Plot ROC and Precision-Recall curves
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y, y_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_roc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y, y_probs)
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_filename = os.path.join(output_dir, f"{dataset_name}_curves.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save Confusion Matrix separately
    plt.figure()
    cm = confusion_matrix(y, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
    plt.yticks([0, 1], ['Non-Fraud', 'Fraud'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    
    cm_filename = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved plots to folder: '{output_dir}'")
    print(f"- {dataset_name}_curves.png (ROC & Precision-Recall)")
    print(f"- {dataset_name}_confusion_matrix.png")
    
    return y_probs

# Evaluate on test set and save plots
print("\n" + "="*50)
test_probs = evaluate_model(X_test, y_test, model, dataset_name="test_set")

# Evaluate on validation set and save plots
print("\n" + "="*50)
val_probs = evaluate_model(X_val, y_val, model, dataset_name="val_set")

# --- Threshold Tuning ---
def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

optimal_threshold = find_optimal_threshold(y_test, test_probs)
print(f"\nOptimal Threshold (Max F1): {optimal_threshold:.3f}")

# Re-evaluate with optimal threshold
print("\n" + "="*50)
print("=== Test Set with Optimal Threshold ===")
evaluate_model(X_test, y_test, model, threshold=optimal_threshold, dataset_name="test_set_optimal")

# --- Feature Importance ---
importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1]

print("\nTop 10 Important Features:")
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Save Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[sorted_idx[:10]], align='center')
plt.yticks(range(10), [feature_names[i] for i in sorted_idx[:10]])
plt.xlabel('Feature Importance')
plt.title('Top 10 Important Features')
plt.gca().invert_yaxis()  # Highest importance at top
plt.tight_layout()

feat_imp_filename = os.path.join(output_dir, "feature_importance.png")
plt.savefig(feat_imp_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n- feature_importance.png (Top 10 features)")