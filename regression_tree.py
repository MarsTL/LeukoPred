#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains functions for the regression tree.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Decision tree
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  
        self.threshold = threshold  # value of the feature to split on
        self.left = left  
        self.right = right  
        self.value = value  # predicted value at leaf

# mse used to evaluate quality of a split
def mse(target_values):
    if len(target_values) == 0:
        return 0
    return np.mean((target_values - np.mean(target_values)) ** 2)

# Splits the dataset from a feature and a threshold value
def split_data(features, target_values, feature_index, threshold):
    left_mask = features[:, feature_index] <= threshold
    right_mask = features[:, feature_index] > threshold
    return features[left_mask], target_values[left_mask], features[right_mask], target_values[right_mask]

# Best feature and threshold that minimizes MSE
def find_best_split(features, target_values, min_leaf_size, num_thresholds=10):
    best_feature, best_threshold = None, None
    best_score = float('inf')
    n_samples, n_features = features.shape

    for feature_index in range(n_features):
        values = features[:, feature_index]
        thresholds = np.unique(values)
        if len(thresholds) > num_thresholds:
            thresholds = np.linspace(values.min(), values.max(), num_thresholds)

        for threshold in thresholds:
            f_left, t_left, f_right, t_right = split_data(features, target_values, feature_index, threshold)
            if len(t_left) < min_leaf_size or len(t_right) < min_leaf_size:
                continue
            score = (len(t_left) * mse(t_left) + len(t_right) * mse(t_right)) / n_samples
            if score < best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

# Decision tree recursive
# Stops when reaching max depth, min samples, or low std deviation
def build_tree(features, target_values, depth=0, max_depth=5, min_leaf_size=5, min_std=0.15):
    # All sample in partition within similar AUC range (+-0.15)
    # Stops the  tree from splitting further if AUC value in current node are already very close together (low standard deviation)
    # min_leaf_size = 5, the tree will not split a node if it would create a child with fewer than 5 sample.  Avoids splitting when too few samples remain 
    # max_depth limits the height of the tree, no matter the data
    if len(target_values) <= min_leaf_size or depth >= max_depth or np.std(target_values) <= min_std:
        return TreeNode(value=np.mean(target_values))

    feature_index, threshold = find_best_split(features, target_values, min_leaf_size)
    if feature_index is None:
        return TreeNode(value=np.mean(target_values))

    f_left, t_left, f_right, t_right = split_data(features, target_values, feature_index, threshold)
    left_node = build_tree(f_left, t_left, depth + 1, max_depth, min_leaf_size, min_std)
    right_node = build_tree(f_right, t_right, depth + 1, max_depth, min_leaf_size, min_std)
    return TreeNode(feature_index, threshold, left_node, right_node)

# Post-prunes the tree by replacing nodes with leaves when it reduces error
# Controlled by alpha (regularization strength)
def prune_tree(node, features, target_values, alpha=0.01):
    if node.left is None and node.right is None:
        return node
    # Prune left and right subtrees
    if node.left:
        node.left = prune_tree(node.left, features, target_values, alpha)
    if node.right:
        node.right = prune_tree(node.right, features, target_values, alpha)
    #If both children are now leaves, evaluate whether to prune them
    if node.left.value is not None and node.right.value is not None:
        merged_value = (node.left.value * len(features) + node.right.value * len(features)) / (2 * len(features))
        #original_error = np.sum((target_values - predict_tree(node, features.T)) ** 2)
        original_error = np.sum((target_values - predict(node, features)) ** 2)

        merged_error = np.sum((target_values - merged_value) ** 2)

        if merged_error + alpha < original_error:
            return TreeNode(value=merged_value)

    return node

# Predict a value for a sample
def predict_tree(node, feature_row):
    if node.value is not None:
        return node.value
    if feature_row[node.feature] <= node.threshold:
        return predict_tree(node.left, feature_row)
    else:
        return predict_tree(node.right, feature_row)

# Predict values for all rows
def predict(model, features):
    return np.array([predict_tree(model, row) for row in features])

# Trains and prunes 
def train_model(training_data, target_col_values, max_depth=5, min_leaf_size=5, min_std=0.15, alpha=0.01):
    tree = build_tree(training_data, target_col_values, max_depth=max_depth, min_leaf_size=min_leaf_size, min_std=min_std)
    pruned_tree = prune_tree(tree, training_data, target_col_values, alpha=alpha)
    return pruned_tree

# Calculate
def calculate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # I'm editing this because accuracy (as far as I know) is a classification metric and =/= accuracy
    # I couldn't find a name for this metric in the lecture slides, the textbook, or online
    # However, it looks like maybe some people use it as a measure of "Goodness of fit"
    print(f"Goodness of fit: {(1 - rmse):.2%}")
    print("\nThe results of prediction are:")
    print(y_pred)
    print(f"\nRMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Adding code to plot the results of prediction to compare to true values
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual AUC (Normalized)")
    plt.ylabel("Predicted AUC (Normalized)")
    plt.title("Actual vs. Predicted AUC")
    plt.grid(True)
    plt.show()


