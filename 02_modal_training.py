import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import seaborn as sns

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

print("--- Starting Phase 3: Model Training ---")

# 1. Load the pre-cleaned data (Skipping Phase 1 & 2!)
print("Loading pre-cleaned dataset...")
df = pd.read_csv("cleaned_banking_data.csv")

# 2. Define Features (X) and Target (y)
y = df['TARGET']
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# 3. Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize and Train LightGBM
print("Initializing and Training LightGBM Model (Please wait 10-30 seconds)...")
clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    class_weight='balanced', # Handles our 92/8 class imbalance
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 5. Make Predictions
print("Evaluating the model...")
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# 6. Print Results
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n--- Model Performance ---")
print(f"ROC-AUC Score: {auc_score:.4f} (1.0 is perfect, 0.5 is random guessing)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualizing ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'LightGBM (AUC = {auc_score:.4f})', color='#3498db', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC Curve - Credit Risk Model')
plt.xlabel('False Positive Rate (Incorrectly flagged)')
plt.ylabel('True Positive Rate (Correctly identified defaulters)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()