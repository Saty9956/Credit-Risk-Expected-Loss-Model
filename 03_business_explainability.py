import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') # Hides unnecessary SHAP warnings

print("--- Starting Phase 4: Business Impact & Explainable AI ---")

# 1. Load Data & Quick Retrain (Fast configuration to load the model into memory)
print("Loading data and initializing model...")
df = pd.read_csv("cleaned_banking_data.csv")
y = df['TARGET']
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# Grab a small sample of the data to make the SHAP calculations run quickly on your laptop
X_sample = X.sample(2000, random_state=42)
y_sample = y.loc[X_sample.index]

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

# Retrain the model rapidly
clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
clf.fit(X_train, y_train)

# ==========================================
# 2. BUSINESS STRATEGY: EXPECTED LOSS (EL)
# ==========================================
print("\n--- Simulating Bank Loan Desk ---")
# Let's pull 5 random customers from our test set
customers = X_test.sample(5, random_state=99)
pd_scores = clf.predict_proba(customers)[:, 1] # Probability of Default (PD)

# Bank Math: Expected Loss (EL) = PD * LGD * EAD
lgd = 0.45 # Loss Given Default (Industry standard assumption: bank loses 45% of the money)
ead = customers['AMT_CREDIT'].values # Exposure at Default (The total loan amount requested)

expected_loss = pd_scores * lgd * ead

for i in range(5):
    print(f"\nApplicant {i+1}:")
    print(f"  Loan Requested (EAD): ${ead[i]:,.2f}")
    print(f"  Risk of Default (PD): {pd_scores[i]:.2%}")
    print(f"  Expected Financial Loss (EL): ${expected_loss[i]:,.2f}")
    
    if pd_scores[i] > 0.50:
        print("  Decision: ❌ REJECT (Risk Exceeds Tolerance)")
    elif pd_scores[i] > 0.30:
        print("  Decision: ⚠️ APPROVE WITH PREMIUM (+3% Interest Rate)")
    else:
        print("  Decision: ✅ APPROVE (Prime Rate)")

# ==========================================
# 3. COMPLIANCE: EXPLAINABLE AI (SHAP)
# ==========================================
print("\n--- Generating Regulatory Compliance Report (SHAP) ---")
print("Calculating SHAP values to explain the AI's logic (this takes a few seconds)...")

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# LightGBM returns a list of arrays for binary classification. Index 1 is the 'Default' class.
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

print("Opening SHAP Feature Importance Plot...")
print("(Close the graph window when you are done to end the script)")

# Generate the visual report
plt.figure(figsize=(10, 6))
plt.title("Global Risk Drivers (What makes an applicant high risk?)")
shap.summary_plot(shap_values_to_plot, X_test, show=False)
plt.tight_layout()
plt.show()