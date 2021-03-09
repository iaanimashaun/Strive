import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):

        # Initialize empty list to hold labels
        y_pred = []

        # Loop through each data to be predicted
        for x in range(len(X)):

            # Calculate distances between x and all examples in the training set
            distances = [
                np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train
            ]

            # Sort by distance and return indices of the first k neighbors
            k_idx = np.argsort(distances)[: self.k]

            # Extract the labels of the k nearest neighbors
            k_neighbors_labels = [self.y_train[i] for i in k_idx]

            # Get the label/neighbor with majority vote(most common)
            most_common = Counter(k_neighbors_labels).most_common(1)[0][0]

            # Append to list of labels/predictions
            y_pred.append(most_common)

        return np.array(y_pred)


    def evaluate(self, y_pred, y_true):
        score = np.sum(y_true == y_pred) / len(y_true)
        return score * 100
