# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 19:58:51 2026

@author: Nathan
"""

#kkn algorythm




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn as skl
from sklearn import datasets
from sklearn.model_selection import train_test_split

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train[0])


plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

#a = [1,1,1,12,2,2,2,1,2,2,2]
#from collections import Counter
#most_common = Counter(a).most_common(1)
#print(most_common)

from collections import Counter

import numpy as np


def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two data points.
    
    Formula: sqrt(sum((x1 - x2)^2))
    
    Args:
        x1 (numpy array): The first data point (feature vector).
        x2 (numpy array): The second data point (feature vector).
        
    Returns:
        float: The distance between x1 and x2.
    """
    # Calculate the element-wise difference, square it, sum the squares, and take the square root
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    """
    A K-Nearest Neighbors Classifier implemented from scratch.
    """
    def __init__(self, k=3):
        """
        Initializes the KNN classifier.
        
        Args:
            k (int): The number of nearest neighbors to consider for voting. Default is 3.
        """
        self.k = k

    def fit(self, X, y):
        """
        'Trains' the model. 
        
        Note: KNN is a lazy learner. It doesn't actually learn a discriminative function 
        but simply memorizes the training dataset to use during prediction.
        
        Args:
            X (numpy array): Training data features (samples x features).
            y (numpy array): Training labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts labels for the given input samples.
        
        Args:
            X (numpy array): The input samples to predict (samples x features).
            
        Returns:
            numpy array: Predicted labels for the input samples.
        """
        # Iterate over every sample in the input X and call the helper method _predict
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Helper method to predict the label for a single sample.
        
        Args:
            x (numpy array): A single data point.
            
        Returns:
            int/str: The predicted class label.
        """
        # 
        # 1. Compute distances between the input x and all examples in the training set
        # This creates a list of distances corresponding to each training point
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Sort the distances to find the nearest points
        # np.argsort returns the *indices* that would sort the array
        # [: self.k] slices the array to keep only the indices of the 'k' smallest distances
        k_idx = np.argsort(distances)[: self.k]
        
        # 3. Extract the labels of the k nearest neighbor training samples
        # We use the indices found in the previous step to look up the actual targets/labels
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        
        # 4. Return the most common class label (Majority Voting)
        # Counter creates a dictionary of label counts (e.g., {'ClassA': 2, 'ClassB': 1})
        # .most_common(1) returns a list with the top 1 element: [('ClassA', 2)]
        most_common = Counter(k_neighbor_labels).most_common(1)
        
        # Extract just the label itself (index [0][0]) from the tuple
        return most_common[0][0]


# Main execution block
if __name__ == "__main__":
    # Imports specific to this testing block
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # Define a custom color map for potential visualization (though not used directly in this script logic)
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        """
        Calculates the classification accuracy.
        
        Args:
            y_true (numpy array): The actual correct labels.
            y_pred (numpy array): The labels predicted by the model.
            
        Returns:
            float: The fraction of correctly classified samples (0.0 to 1.0).
        """
        # Compare arrays element-wise to get boolean array (True for match, False for mismatch)
        # Summing the boolean array counts the Trues (correct predictions)
        # Divide by total count to get percentage
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # 1. Load the Iris dataset (a classic dataset for classification)
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 2. Split the dataset into training and testing sets
    # 80% of data goes to training, 20% to testing
    # random_state ensures reproducibility (same split every time code runs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # 3. Initialize the KNN model with k=3 (looking at 3 nearest neighbors)
    k = 3
    clf = KNN(k=k)
    
    # 4. Fit the model (store the training data)
    clf.fit(X_train, y_train)
    
    # 5. Make predictions on the test set
    predictions = clf.predict(X_test)
    
    # 6. Calculate and print the accuracy of the model
    print("KNN classification accuracy", accuracy(y_test, predictions))
    
    # Validating the result manually (commented out in original code)
    # acc = np.sum(predictions == y_test) / len(y_test)
    #print(acc)
    
    
    
