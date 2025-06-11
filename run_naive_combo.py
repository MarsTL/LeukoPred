#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train on single and test on combo

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from naive_bayes_classifier import run_classifier

# Load data
features_df = pd.read_csv("leukemia_features_combo_discretize.csv")
targets_df = pd.read_csv("leukemia_targets_discrete.csv")

# Define drug columns
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"
