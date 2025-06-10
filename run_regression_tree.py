    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
from regression_tree import build_tree, predict

# Load dataset
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Gene columns and drugs of interest
gene_cols = df.columns[1:944]
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"

# Scale gene expression
gene_expr = df[gene_cols]
gene_expr_scaled = (gene_expr - gene_expr.mean()) / gene_expr.std()

# Build training data
train_frames = []
for drug in single_drugs:
    mask = df[drug].notnull()
    temp = gene_expr_scaled[mask].copy()
    for d in single_drugs:
        temp[f"Drug_{d}"] = 1 if d == drug else 0
    temp["AUC"] = df.loc[mask, drug]  # raw AUC as regression target
    train_frames.append(temp)

train_df = pd.concat(train_frames).reset_index(drop=True)

# Upsample to balance by AUC range
df_high = train_df[train_df["AUC"] <= 0.33]
df_moderate = train_df[(train_df["AUC"] > 0.33) & (train_df["AUC"] <= 0.66)]
df_low = train_df[train_df["AUC"] > 0.66]

max_count = max(len(df_high), len(df_moderate), len(df_low))

df_high_balanced = resample(df_high, replace=True, n_samples=max_count, random_state=42)
df_moderate_balanced = resample(df_moderate, replace=True, n_samples=max_count, random_state=42)
df_low_balanced = resample(df_low, replace=True, n_samples=max_count, random_state=42)

train_df_balanced = pd.concat([df_high_balanced, df_moderate_balanced, df_low_balanced])
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced training class distribution (by AUC bins):")
print(f"High: {len(df_high_balanced)}, Moderate: {len(df_moderate_balanced)}, Low: {len(df_low_balanced)}\n")

# Prepare test data
combo_mask = df[combo_drug].notnull()
test_df = gene_expr_scaled[combo_mask].copy()
for d in single_drugs:
    test_df[f"Drug_{d}"] = 1

# Train/Test arrays
X_train = train_df_balanced.drop(columns=["AUC"]).values
y_train = train_df_balanced["AUC"].values

X_test = test_df.values
y_test = df.loc[combo_mask, combo_drug].values

# Train the regression tree
tree_model = build_tree(X_train, y_train, max_depth=5, min_samples_leaf=5, min_std=0.15)

# Predict and evaluate
y_pred = predict(tree_model, X_test)

print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("Test R²:  ", r2_score(y_test, y_pred))