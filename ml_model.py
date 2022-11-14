import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import TransformerMixin
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.feature_selection import (SelectFdr, SelectFpr, SelectKBest,
                                       SelectPercentile, VarianceThreshold,
                                       chi2, f_classif, mutual_info_classif)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PolynomialFeatures, PowerTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.utils import check_array


class ValueCount(TransformerMixin):
    """
    Takes in a numpy array and returns the same numpy array but with two
    additional columns defined as the count of items equal to `value` in the
    array and the count of items which are not equal to `value`.
    """

    def __init__(self, value: float = 0):
        self.value = np.array(value)

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray, y=None):
        X = check_array(X)
        feature_count = X.shape[1]
        non_value = np.count_nonzero(X != self.value, axis=1)
        value = feature_count - non_value
        return np.c_[X, value, non_value]


class ValueOverUnder(TransformerMixin):
    """
    Takes in a numpy array and returns the same numpy array but with two
    additional columns defined as the count of items greater than `value` in the
    array and the count of items which are less than `value`.
    """

    def __init__(self, value: float = 0):
        self.value = np.array(value)

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray, y=None):
        X = check_array(X)
        over = np.count_nonzero(X > self.value, axis=1)
        under = np.count_nonzero(X < self.value, axis=1)
        return np.c_[X, over, under]


best_model = make_pipeline(
    VarianceThreshold(),
    MinMaxScaler(),
    ValueCount(value=0),
    MultinomialNB(alpha=1.0, fit_prior=True)
                        # ("76Neighbors",
                        #  KNeighborsClassifier(n_neighbors=76,
                        #                       p=2,
                        #                       weights="distance")),
                        # ("SGD",
                        #  SGDClassifier(alpha=0.0,
                        #                eta0=1.0,
                        #                fit_intercept=False,
                        #                l1_ratio=1.0,
                        #                learning_rate="invscaling",
                        #                loss="epsilon_insensitive",
                        #                penalty="elasticnet",
                        #                power_t=1.0))]),
)
