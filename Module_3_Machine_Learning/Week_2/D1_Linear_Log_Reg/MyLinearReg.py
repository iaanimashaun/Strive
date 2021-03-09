# import the minimum packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.intercept = None
        self.coef = None

    def variance(self, x):
        x_mean = np.mean(x)
        return sum([(val - x_mean) ** 2 for val in x])

    def covariance(self, x, y):
        covariance = 0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        for i in range(len(x)):
            covariance = covariance + (x[i] - x_mean) * (y[i] - y_mean)
        return covariance

    def fit(self, x, y):

        self.coef = self.covariance(x, y) / self.variance(x)
        self.intercept = np.mean(y) - self.coef * np.mean(x)
        return self

    def predict(self, x_test):
        return self.intercept + np.dot(x_test, self.coef)

    def rmse(self, x, y):
        mse = 0
        n = len(x)
        for i in range(n):
            y_pred = self.intercept + np.dot(x[i], self.coef)
            mse += (y_pred - y[i]) ** 2
        return np.sqrt(mse / n)

    def r_squared(self, x, y):
        sumofsquares = 0
        sumofresiduals = 0
        for i in range(len(x)):
            y_pred = self.intercept + np.dot(x[i], self.coef)
            sumofsquares += (y[i] - np.mean(y)) ** 2
            sumofresiduals += (y[i] - y_pred) ** 2

        score = 1 - (sumofresiduals / sumofsquares)
        return score
