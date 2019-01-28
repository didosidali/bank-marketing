from __future__ import division, print_function
import os
import datetime

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, recall_score, precision_score

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.feature_extractor import FeatureExtractor
from rampwf.workflows.classifier import Classifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

problem_title = 'Bank Marketing classification'



# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------



def get_data(path='data',filename='bank-full.csv'):
    fp = os.path.join(path, filename)
    data = pd.read_csv(fp , sep=";")
    data = data.loc[np.random.permutation(data.index)]
    labels = data.y.values
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)
    X = data.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)


