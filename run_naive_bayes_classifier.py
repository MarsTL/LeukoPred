#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from naive_bayes_classifier import run_classifier
from sklearn.utils import resample

# Load the preprocessed data (or run your discretization steps here)
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Define columns
gene_cols = df.columns[1:944]
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"

# Discretize gene expression
gene_expr = df[gene_cols]
gene_expr_discrete = gene_expr.apply(lambda col: pd.qcut(col, 3, labels=[0, 1, 2]))

# Discretize AUC
def discretize_auc(auc_values):
    return pd.cut(auc_values, bins=[-float("inf"), 0.33, 0.66, float("inf")], labels=["high", "moderate", "low"])

# Prepare training data
train_frames = []
for drug in single_drugs:
    mask = df[drug].notnull()
    temp = gene_expr_discrete[mask].copy()
    for d in single_drugs:
        temp[f"Drug_{d}"] = 1 if d == drug else 0
    temp["AUC"] = discretize_auc(df.loc[mask, drug])
    train_frames.append(temp)

train_df = pd.concat(train_frames).reset_index(drop=True)
# === Upsample training data to balance classes ===
df_high = train_df[train_df["AUC"] == "high"]
df_moderate = train_df[train_df["AUC"] == "moderate"]
df_low = train_df[train_df["AUC"] == "low"]

# Determine the class with the largest size
max_count = max(len(df_high), len(df_moderate), len(df_low))

# Upsample each class to match the largest class size
df_high_balanced = resample(df_high, replace=True, n_samples=max_count, random_state=42)
df_moderate_balanced = resample(df_moderate, replace=True, n_samples=max_count, random_state=42)
df_low_balanced = resample(df_low, replace=True, n_samples=max_count, random_state=42)

# Combine into a new balanced dataframe
train_df_balanced = pd.concat([df_high_balanced, df_moderate_balanced, df_low_balanced])
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced training class counts (upsampled):\n", train_df_balanced["AUC"].value_counts()) 

# Prepare test data
combo_mask = df[combo_drug].notnull()
test_df = gene_expr_discrete[combo_mask].copy()
for d in single_drugs:
    test_df[f"Drug_{d}"] = 1
test_df["AUC"] = discretize_auc(df.loc[combo_mask, combo_drug])

# Combine features and target for training data
training_full = train_df.copy()
test_full = test_df.copy()
'''
# Rebalance training data by downsampling to the smallest class 
balanced_train_df = train_df.copy()

df_high = balanced_train_df[balanced_train_df["AUC"] == "high"]
df_moderate = balanced_train_df[balanced_train_df["AUC"] == "moderate"]
df_low = balanced_train_df[balanced_train_df["AUC"] == "low"]

min_count = min(len(df_high), len(df_moderate), len(df_low))

df_high_balanced = resample(df_high, replace=True, n_samples=min_count, random_state=42)
df_moderate_balanced = resample(df_moderate, replace=True, n_samples=min_count, random_state=42)
df_low_balanced = resample(df_low, replace=True, n_samples=min_count, random_state=42)

train_df_balanced = pd.concat([df_high_balanced, df_moderate_balanced, df_low_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced training class counts:\n", train_df_balanced["AUC"].value_counts())
'''
run_classifier(
    training_data=train_df_balanced.drop(columns=["AUC"]),
    training_target=train_df_balanced["AUC"],
    test_data=test_df.drop(columns=["AUC"]),
    test_target=test_df["AUC"]
)