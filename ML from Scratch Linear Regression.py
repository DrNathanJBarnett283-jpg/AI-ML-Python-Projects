
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 22:36:19 2026

@author: Nathan
"""

#linear regression from scratch


import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.datasets import datasets

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 22:36:19 2026

@author: Nathan
"""

# LINEAR REGRESSION FROM SCRATCH
# ------------------------------
# We will build a model that finds the best fitting line (y = wx + b)
# for a set of data points.

import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn import datasets

# ---------------------------------------------------------
# Helper Function: R-Squared Score
# ---------------------------------------------------------
# This function measures how well our model performed.
# A score of 1.0 means perfect prediction.
# A score of 0.0 means the model is no better than just guessing the average.
def r2_score(y_true, y_pred):
    # Calculate the correlation matrix between true values and predicted values
    corr_matrix = np.corrcoef(y_true, y_pred)
    # Extract the specific correlation value
    corr = corr_matrix[0, 1]
    # R^2 is the square of the correlation coefficient
    return corr ** 2


# ---------------------------------------------------------
# Main Class: Linear Regression
# ---------------------------------------------------------
class LinearRegression:
    
    # __init__ is the "Constructor". It runs automatically when you create a new model.
    # We use it to store settings (Hyperparameters) for the model.
    def __init__(self, learning_rate=0.001, n_iters=1000):
        # 'self.lr' stores the learning rate. This controls how big of a step
        # we take when correcting our errors. 
        # Too high = we overshoot the target. Too low = it takes forever to learn.
        self.lr = learning_rate
        
        # 'self.n_iters' is how many times we will loop through the training data to learn.
        self.n_iters = n_iters
        
        # These are the model parameters we want to learn.
        # weights (w): The slope of the line (how important a feature is).
        # bias (b): The y-intercept (where the line crosses the y-axis).
        self.weights = None
        self.bias = None

    # The 'fit' method is where the training happens.
    # X: The input features (the data we learn from).
    # y: The target values (the correct answers).
    def fit(self, X, y):
        # n_samples = number of data points (rows)
        # n_features = number of input variables (columns)
        n_samples, n_features = X.shape

        # 1. Initialize Parameters
        # We start with weights as 0 because we don't know the answer yet.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        # We repeat this process 'n_iters' times to slowly get closer to the best line.
        for _ in range(self.n_iters):
            
            # Step A: Make a prediction using the current weights and linear equation:
            # y = wx + b
            # np.dot is a matrix multiplication that combines inputs (X) and weights.
            y_predicted = np.dot(X, self.weights) + self.bias

            # Step B: Calculate Gradients (Derivatives)
            # The gradient tells us the "direction" we need to move to reduce error.
            # We are asking: "If I change the weights slightly, how does the error change?"
            
            # dw: Derivative with respect to weights
            # Formula: (1/N) * X_transpose * (Predicted - Actual)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            
            # db: Derivative with respect to bias
            # Formula: (1/N) * Sum(Predicted - Actual)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Step C: Update Parameters
            # We move the weights in the opposite direction of the gradient to reduce error.
            # New_Weight = Old_Weight - (Learning_Rate * Gradient)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # The 'predict' method uses the learned weights to make guesses on new data.
    def predict(self, X):
        # Simply apply the formula: y = wx + b using the weights we found in fit()
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


# ---------------------------------------------------------
# Testing the Code
# ---------------------------------------------------------
# This block only runs if you run this file directly (not if you import it elsewhere).
if __name__ == "__main__":
    
    # Imports for testing
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # A simple function to calculate Mean Squared Error (Average squared difference)
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # 1. Generate synthetic data
    # We create 100 random data points with a linear relationship + some random noise.
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    # 2. Split data
    # We hide 20% of the data (test set) to see if the model can predict new data it hasn't seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # 3. Create and Train the Model
    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train) # The model "learns" here

    # 4. Make Predictions
    predictions = regressor.predict(X_test)

    # 5. Evaluate Performance
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse) # Lower is better

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu) # Closer to 1.0 is better

    # 6. Visualize Results
    # We plot the data points and the line our model drew.
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    
    # Plot training data points (Darker color)
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    # Plot testing data points (Lighter color)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    # Plot the best fit line (Black line)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    
    plt.show()
