#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''separate drugs'''

import pandas as pd

df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Define the gene columns (assuming they are from column index 1 to 943)
gene_cols = df.columns[1:944]  # adjust as needed based on actual file

# Define the drug columns to filter by
drug_cols = ["Doxorubicin", "Navitoclax", "Doxorubicin:navitoclax"]

# Filter rows where any of the selected drug columns are not null
filtered_df = df[df[drug_cols].notnull().any(axis=1)]

# Keep gene expression columns + selected drug columns
selected_cols = list(gene_cols) + drug_cols
filtered_df[selected_cols].to_csv("unique_meds_genes_and_drugs.csv", index=False)

print("Saved as unique_meds.csv")


