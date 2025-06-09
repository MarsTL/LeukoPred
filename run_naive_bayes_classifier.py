import pandas as pd
from naive_bayes_classifier import run_classifier

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

# Prepare test data
combo_mask = df[combo_drug].notnull()
test_df = gene_expr_discrete[combo_mask].copy()
for d in single_drugs:
    test_df[f"Drug_{d}"] = 1
test_df["AUC"] = discretize_auc(df.loc[combo_mask, combo_drug])

# Combine features and target for training data
training_full = train_df.copy()
test_full = test_df.copy()

run_classifier(
    training_data=training_full.drop(columns=["AUC"]),
    training_target=training_full["AUC"],
    test_data=test_full.drop(columns=["AUC"]),
    test_target=test_full["AUC"]
)