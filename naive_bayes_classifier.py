#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains functions for the naive bayes classifier.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def prior_probability(data, target_col):
    # for each of the classes in the target column, calculate the prior
    # probabilities
    prior_probabilities = data[target_col].value_counts().to_dict()
    return prior_probabilities


# calculate the counts of feature and target combinations from the training data
def calc_counts(data, target_col):

    counts = {}
    target_classes = data[target_col].unique()
    features = data.columns.drop(target_col)
    
    # for each of the given features
    for i in features:
        counts[i] = {}
        feature_values = data[i].unique()
        
        for j in feature_values:
            counts[i][j] = {}
            
            # filter for each value of a given feature
            data_filtered_by_feature = data[data[i] == j]
            
            target_counts_for_feature_val = data_filtered_by_feature[target_col].value_counts().to_dict()
            
            for k in target_classes:
                # count of the target value for the current feature value
                count_feature_val_and_target_val = target_counts_for_feature_val.get(k, 0)
                counts[i][j][k] = count_feature_val_and_target_val
                
    return counts


def naive_bayes_classifier(test_data, prior_probabilities, counts):
    # makes predictions of new instances based on the prior probabilities and counts
    # of the training data (incorporates laplace smoothing)
    predictions = []
    features = test_data.columns
    
    target_vals = list(prior_probabilities.keys())
    
    for i, obs in test_data.iterrows():
        posteriors = {}
        
        for j in target_vals:
            posterior = prior_probabilities[j]
            
            for k in features:
                #maybe we  want the feature valaue for  this test observation
                feature_val = obs[k]
                
                # Get the count of the feature value and target value from the counts table
                count_feature_val_and_target_val = counts.get(k, {}).get(feature_val, {}).get(j, 0)
                
                # Get the total count of the target value from the counts table (summing across all feature values for that feature)
                total_count_target_val = sum(counts.get(k, {}).get(fv, {}).get(j, 0) for fv in counts.get(k, {}).keys())
                
                # Determine the number of possible values for the feature for Laplace smoothing (V)
                # We get this from the keys present in the counts_table for this feature
                num_possible_feature_values = len(counts.get(k, {}).keys())

                # Calculate the laplace smoothing: (n_c^x=a + 1) / (n_c + K)
                laplace_res = (count_feature_val_and_target_val + 1) / (total_count_target_val + num_possible_feature_values)

                # Multiply the prior by the result of laplace smoothing to get prior
                posterior *= laplace_res
                
            posteriors[j] = posterior
            
        # Predict the class with the highest posterior probability
        predicted_class = max(posteriors, key=posteriors.get)
        predictions.append(predicted_class)
        
    return pd.Series(predictions)


def pred_results(y_true, y_pred):
    # calculates error for the naive bayes classifier
    total_samp = len(y_true)
    
    num_correct = np.sum(y_true == y_pred)
    
    success_rate = num_correct/total_samp
    
    conf_mat = confusion_matrix(y_true, y_pred)
    
    print(f"The prediction accuracy is: {success_rate:.2%}")
    print("\nThe results of prediction are:\n", conf_mat)


def run_classifier(training_data, training_target, test_data, test_target):
    priors = prior_probability( 
        data = training_data.assign(**{training_target.name: training_target}),
        target_col = training_target.name)

    counts = calc_counts(
        data = training_data.assign(**{training_target.name: training_target}),
        target_col = training_target.name
    )
  
    preds = naive_bayes_classifier(test_data= test_data,
                                   prior_probabilities = priors,
                                   counts = counts)
    
    pred_results(y_true=test_target.reset_index(drop=True), y_pred=preds.reset_index(drop=True))




