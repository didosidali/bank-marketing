from __future__ import division

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression())

    def fit(self, X, y):
        # return
        self.model.fit(X, y)

    def predict_proba(self, X):
        # return np.zeros(len(X)) 
        return self.model.predict_proba(X)
