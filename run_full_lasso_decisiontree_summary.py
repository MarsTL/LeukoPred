import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from regression_tree import build_tree, predict

# Load dataset
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Define gene and drug columns
gene_cols = df.columns[1:944]
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"

# Scale gene expression
gene_expr = df[gene_cols]
gene_expr_scaled = (gene_expr - gene_expr.mean()) / gene_expr.std()

# --- LASSO REGULARIZATION ---
lasso_mask = df[single_drugs].notnull().all(axis=1)
X_lasso_df = gene_expr_scaled[lasso_mask]
y_lasso = df.loc[lasso_mask, single_drugs[0]]  # Use Doxorubicin AUC

# Save column names before imputation
feature_names = X_lasso_df.columns

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_lasso = imputer.fit_transform(X_lasso_df)

# Apply Lasso
lasso = Lasso(alpha=0.05, max_iter=10000)
lasso.fit(X_lasso, y_lasso)
selected_features = feature_names[(lasso.coef_ != 0)]

print(f" Lasso selected {len(selected_features)} features from {len(gene_cols)}\n")

# Reduce dataset
gene_expr_scaled = gene_expr_scaled[selected_features]

# --- TRAIN DATA ---
train_frames = []
for drug in single_drugs:
    mask = df[drug].notnull()
    temp = gene_expr_scaled[mask].copy()
    for d in single_drugs:
        temp[f"Drug_{d}"] = 1 if d == drug else 0
    temp["AUC"] = df.loc[mask, drug]
    train_frames.append(temp)

train_df = pd.concat(train_frames).reset_index(drop=True)

# Balance by AUC
df_high = train_df[train_df["AUC"] <= 0.33]
df_moderate = train_df[(train_df["AUC"] > 0.33) & (train_df["AUC"] <= 0.66)]
df_low = train_df[train_df["AUC"] > 0.66]

max_count = max(len(df_high), len(df_moderate), len(df_low))
df_high_bal = resample(df_high, replace=True, n_samples=max_count, random_state=42)
df_mod_bal = resample(df_moderate, replace=True, n_samples=max_count, random_state=42)
df_low_bal = resample(df_low, replace=True, n_samples=max_count, random_state=42)

train_df_balanced = pd.concat([df_high_bal, df_mod_bal, df_low_bal])
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# --- TEST DATA ---
combo_mask = df[combo_drug].notnull()
test_df = gene_expr_scaled[combo_mask].copy()
for d in single_drugs:
    test_df[f"Drug_{d}"] = 1
y_test = df.loc[combo_mask, combo_drug].values

# --- TRAIN TREE ---
X_train = train_df_balanced.drop(columns=["AUC"]).values
y_train = train_df_balanced["AUC"].values
X_test = test_df.values

tree_model = build_tree(X_train, y_train, max_depth=5, min_samples_leaf=5, min_std=0.15)
y_pred = predict(tree_model, X_test)

# --- EVALUATE ---
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Test RMSE:", rmse)
print(" Test R²:  ", r2)

# --- SUMMARY REPORT ---
results = {
    "Lasso": {
        "Selected_Features": len(selected_features),
        "Original_Features": len(gene_cols),
        "Drugs_Used_For_Training": single_drugs,
        "Combo_Drug_For_Test": combo_drug
    },
    "Decision_Tree": {
        "Max_Depth": 5,
        "Min_Samples_Leaf": 5,
        "Min_Std_Deviation": 0.15,
        "Stopping_Criteria": [
            "Max depth = 5",
            "Minimum samples per leaf = 5",
            "Standard deviation of AUC ≤ 0.15"
        ]
    },
    "Evaluation": {
        "Test_RMSE": rmse,
        "Test_R2": r2
    }
}

report_df = pd.DataFrame([
    ["Lasso Selected Features", results["Lasso"]["Selected_Features"]],
    ["Original Feature Count", results["Lasso"]["Original_Features"]],
    ["Drugs Used (Train)", ", ".join(results["Lasso"]["Drugs_Used_For_Training"])],
    ["Drug Used (Test)", results["Lasso"]["Combo_Drug_For_Test"]],
    ["Max Tree Depth", results["Decision_Tree"]["Max_Depth"]],
    ["Min Samples Per Leaf", results["Decision_Tree"]["Min_Samples_Leaf"]],
    ["AUC Std Threshold", results["Decision_Tree"]["Min_Std_Deviation"]],
    ["Stopping Conditions", "; ".join(results["Decision_Tree"]["Stopping_Criteria"])],
    ["Test RMSE", results["Evaluation"]["Test_RMSE"]],
    ["Test R² Score", results["Evaluation"]["Test_R2"]],
], columns=["Metric", "Value"])

print("\n Summary Report:")
print(report_df.to_string(index=False))
