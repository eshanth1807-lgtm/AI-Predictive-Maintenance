"""Interactions derived only from the six available operating measurements."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OperatingFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        result['Temperature gap'] = X['Process temperature [K]'] - X['Air temperature [K]']
        result['Mechanical power'] = X['Torque [Nm]'] * X['Rotational speed [rpm]'] * (2 * np.pi / 60)
        result['Wear torque'] = X['Tool wear [min]'] * X['Torque [Nm]']
        return result
