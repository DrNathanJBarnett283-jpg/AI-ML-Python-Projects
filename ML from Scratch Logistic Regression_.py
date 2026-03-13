# -*- coding: utf-8 -*-
"""
LOGISTIC REGRESSION FROM SCRATCH
--------------------------------
This script implements a binary classifier (0 or 1) using Logistic Regression.
It uses the Breast Cancer dataset to predict whether a tumor is Malignant (0) or Benign (1).
"""

# Standard library and data manipulation imports
import numpy as np

# Visualization libraries (unused in this specific snippet but good to have)
import matplotlib.pyplot as plt

# Scikit-learn is used here ONLY to load data and split it.
# We are NOT using sklearn's LogisticRegression; we are building our own.
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LogisticRegression:
    """
    A custom implementation of Logistic Regression using Gradient Descent.
    """
    
    def __init__(self, learning_rate=0.001, n_iters=1000):
        # 1. Hyperparameters
        # ------------------
        # learning_rate (lr): Controls the step size when updating weights.
        #   - High lr: Fast learning, but might overshoot the best value.
        #   - Low lr: Stable, but takes a long time to converge.
        self.lr = learning_rate
        self.n_iters = n_iters # How many training loops to run.
        
        # 2. Parameters
        # -------------
        # weights: The importance of each input feature.
        # bias: The base value (y-intercept) added to the weighted sum.
        # initialized as None (will be set in fit())
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the model using Gradient Descent.
        X: Training features (Matrix of size: samples x features)
        y: Target labels (Vector of size: samples)
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize Parameters
        # We start with weights = 0. The model knows nothing yet.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Step 2: Gradient Descent Loop
        # We repeat the learning process 'n_iters' times.
        for _ in range(self.n_iters):
            
            # A. The Linear Model (f = wx + b)
            # This is the same as Linear Regression so far.
            # We calculate a "score" for each data point based on current weights.
            linear_model = np.dot(X, self.weights) + self.bias
            
            # B. The Activation Function (Sigmoid)
            # This transforms the "score" (which could be -Infinity to +Infinity)
            # into a probability between 0 and 1.
            y_predicted = self._sigmoid(linear_model)

            # C. Compute Gradients
            # We need to know: "If we change weights, how does the error change?"
            # Although the Cost Function (Log Loss) is different from Linear Regression,
            # the derivative formula ends up looking mathematically identical!
            
            # dw: The slope of the loss with respect to weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db: The slope of the loss with respect to bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # D. Update Parameters
            # Move the weights in the opposite direction of the gradient
            # to minimize the error.
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts class labels (0 or 1) for new data samples X.
        """
        # 1. Calculate the probability score using the trained weights
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        # 2. Decision Boundary
        # If the probability is > 0.5, we classify it as Class 1.
        # Otherwise, it is Class 0.
        # This list comprehension loops through every prediction 'i'.
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        """
        Helper function: The Sigmoid Curve.
        Formula: 1 / (1 + e^-x)
        Maps any input 'x' to a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))


# -----------------------------------------------------------
# Testing Block
# -----------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Define an accuracy metric
    # Counts how many predictions match the true labels perfectly.
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # 2. Load the Dataset
    # The Breast Cancer dataset is a classic binary classification problem.
    # Features (X): Radius, texture, perimeter, area, etc. of the cell.
    # Target (y): 0 (Malignant) or 1 (Benign).
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # 3. Split Data
    # 80% used for training, 20% reserved for testing validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # 4. Initialize and Train
    # Note: I corrected 'lr=' to 'learning_rate=' to match the __init__ method above.
    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    
    # 5. Predict and Evaluate
    predictions = regressor.predict(X_test)

    # Output the result
    print("LR classification accuracy:", accuracy(y_test, predictions))
