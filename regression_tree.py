
import numpy as np

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def mse(y):
    if len(y) == 0:
        return 0
    return np.mean((y - np.mean(y)) ** 2)

def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

def find_best_split(X, y, min_samples_leaf, num_thresholds=10):
    best_feature, best_threshold = None, None
    best_mse = float('inf')
    n_samples, n_features = X.shape

    for feature in range(n_features):
        values = X[:, feature]
        thresholds = np.unique(values)
        if len(thresholds) > num_thresholds:
            thresholds = np.linspace(values.min(), values.max(), num_thresholds)

        for t in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature, t)
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
                continue
            current_mse = (len(y_left) * mse(y_left) + len(y_right) * mse(y_right)) / n_samples
            if current_mse < best_mse:
                best_mse = current_mse
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=5, min_samples_leaf=5, min_std=0.15):
    if len(y) <= min_samples_leaf or depth >= max_depth or np.std(y) <= min_std:
        return TreeNode(value=np.mean(y))

    feature, threshold = find_best_split(X, y, min_samples_leaf)
    if feature is None:
        return TreeNode(value=np.mean(y))

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_leaf, min_std)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_leaf, min_std)
    return TreeNode(feature, threshold, left_subtree, right_subtree)

def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

def predict(model, X):
    return np.array([predict_tree(model, x) for x in X])
