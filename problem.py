
from __future__ import division, print_function
import os
import datetime

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.feature_extractor import FeatureExtractor
from rampwf.workflows.classifier import Classifier


problem_title = 'Bank Marketing classification'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(label_names=[0, 1])

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

#-----------------------------------------------------------------------
# Define custom score metrics for the churner class
    
class w_FScore(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='weighted_fscore', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        weight = 5
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        score = ( (1 + weight) *p*r) / ((weight)*p + r)
        return score

class BaseScoreType(object):
    def check_y_pred_dimensions(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should have {} instances, '
                'instead it has {} instances'.format(len(y_true), len(y_pred)))

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths.y_pred[valid_indexes]
        y_pred = predictions.y_pred[valid_indexes]
        self.check_y_pred_dimensions(y_true, y_pred)
        return self.__call__(y_true, y_pred)


class ClassifierBaseScoreType(BaseScoreType):
    def score_function(self, ground_truths, predictions, valid_indexes=None):
        self.label_names = ground_truths.label_names
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
        y_true_label_index = ground_truths.y_pred_label_index[valid_indexes]
        self.check_y_pred_dimensions(y_true_label_index, y_pred_label_index)
        return self.__call__(y_true_label_index, y_pred_label_index)


class liftmetric(ClassifierBaseScoreType):
    def __init__(self, name='lift_metric', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == 1:
                TP += 1
        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y_true[i] != y_pred[i]:
                FP += 1
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == 0:
                TN += 1
        for i in range(len(y_pred)):
            if y_pred[i] == 0 and y_true[i] != y_pred[i]:
                FN += 1
        return (TP/(TP+FP))/((TP+FN)/(TP+TN+FP+FN))

score_types = [
    liftmetric(),
    rw.score_types.ROCAUC(name='roc_auc', precision=2),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=2),
    w_FScore(),
]

#-----------------------------------------------------------------------

def get_cv(X, y):
    """Returns stratified randomized folds."""
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X,y)

def _read_data(path, type_):
    fname = 'data-{}.csv'.format(type_)
    fp = os.path.join(path, 'data', fname)
    data = pd.read_csv(fp, sep=";")
    fname = 'label_{}.npy'.format(type_)
    fp = os.path.join(path, 'data', fname)
    labels = np.load(fp)
    return data, labels

def get_train_data(path='.'):
    f_name = 'train'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test'
    return _read_data(path, f_name)
