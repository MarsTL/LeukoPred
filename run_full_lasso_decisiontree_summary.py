#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from regression_tree import build_tree, predict, train_model, calculate
from regression_tree import print_tree, plot_custom_tree

# Load dataset
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")
feat_csv = pd.read_csv("lasso_selected_features_combo.csv")

# Define gene and drug columns
# changing from 944 to 943 so that we are not accidentally pulling the erlotinib drug column
gene_cols = df.columns[1:943]
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"

# Scale gene expression
gene_expr = df[gene_cols]
gene_expr_scaled = (gene_expr - gene_expr.mean()) / gene_expr.std()

# now this is performed in a separate script
# --- LASSO REGULARIZATION ---
#lasso_mask = df[single_drugs].notnull().all(axis=1)
#X_lasso_df = gene_expr_scaled[lasso_mask]
#y_lasso = df.loc[lasso_mask, single_drugs[0]]  # Use Doxorubicin AUC

# Save column names before imputation
#feature_names = X_lasso_df.columns

# Impute missing values
# there are no missing values if things are being filtered to only include relevant drugs/responses
#imputer = SimpleImputer(strategy='mean')
#X_lasso = imputer.fit_transform(X_lasso_df)

# Apply Lasso
#lasso = Lasso(alpha=0.01, max_iter=10000)
#lasso.fit(X_lasso_df, y_lasso)
selected_features = feat_csv["0"]

#print(f" Lasso selected {len(selected_features)} features from {len(gene_cols)}\n")

# Reduce dataset
#gene_expr_scaled = gene_expr_scaled[selected_features]

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
train_df = train_df.rename(columns = {"Drug_Doxorubicin" : "Doxorubicin_Treated",
                                      "Drug_Navitoclax" : "Navitoclax_Treated"})
target_name = pd.Series(["AUC"])
training_features = pd.concat([selected_features, target_name])
train_df = train_df[training_features]

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

test_df = test_df.rename(columns = {"Drug_Doxorubicin" : "Doxorubicin_Treated",
                                      "Drug_Navitoclax" : "Navitoclax_Treated"})
test_df = test_df[selected_features]

# --- TRAIN TREE ---
X_train = train_df_balanced.drop(columns=["AUC"]).values
y_train = train_df_balanced["AUC"].values
X_test = test_df.values

# had to change min_samples_leaf to min_leaf_size, otherwise script fails
#tree_model = build_tree(X_train, y_train, max_depth=5, min_leaf_size=5, min_std=0.15)
   # don't actually need to call build_tree by itself, as train model already calls build_tree
tree_model = train_model(X_train, y_train, max_depth=5, min_leaf_size=5, min_std=0.01, alpha=0.01)

y_pred = predict(tree_model, X_test)
calculate(y_test, y_pred)

# Tree structure 
print("\nCombo Drug Tree Structure **************")
print_tree(tree_model, selected_features.tolist())

print("\nDisplaying Combo Drug Tree of {combo_drug}  *************")
plot_custom_tree(tree_model, selected_features.tolist())


# --- EVALUATE ---
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Test RMSE:", rmse)
print(" Test R²:  ", r2)

# --- SUMMARY REPORT ---
results = {
    "Lasso": {
        "Selected_Features": selected_features,
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
