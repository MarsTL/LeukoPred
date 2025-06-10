#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''separate drugs'''

import pandas as pd

df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Select only the Sample column and the three drug columns
selected_cols = ["Sample", "Doxorubicin", "Navitoclax", "Doxorubicin:navitoclax"]
subset_df = df[selected_cols]

# Save to a new CSV
subset_df.to_csv("unique_meds.csv", index=False)

print("Saved drug columns to unique_meds.csv")

