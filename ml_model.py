from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import (SelectFdr, SelectFpr, SelectKBest,
                                       SelectPercentile, VarianceThreshold,
                                       chi2, f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PolynomialFeatures, PowerTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.utils import check_array
from tpot.builtins import StackingEstimator


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


class StackAugmenter(BaseEstimator, TransformerMixin):
    """
    Allows the creation of new features by training and predicting on the same
    data. This is useful for stacking predictions to attempt to increase the
    accuracy of the model. Because the predictions are made on the same data
    that is being transformed, there is great risk of overfitting without
    using cross validation.

    Parameters
    ----------
    estimator : BaseEstimator
        The model to train and use to make predictions on the data.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        """
        Creates an instance of StackAugmenter.

        Parameters
        ----------
        estimator : BaseEstimator
            The model to train and use to make predictions on the data.
        """
        self.estimator = estimator
        self.is_classifier = is_classifier(estimator)
        self.use_proba = hasattr(estimator, 'predict_proba')
        self.fitted = False
        self.seen_data = set()

    def fit(self,
            X: ArrayLike,
            y: Optional[ArrayLike] = None,
            **kwargs) -> 'StackAugmenter':
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers
            in regression).
        kwargs : dict
            Additional keyword arguments to pass to the fit method of the
            estimator.
        """
        assert issubclass(
            type(self.estimator),
            BaseEstimator), 'estimator must inherit from BaseEstimator'
        assert hasattr(self.estimator,
                       'fit'), 'estimator must have a fit method'
        X = check_array(X)
        self.estimator.fit(X, y, **kwargs)  # type: ignore
        self.fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform the data by adding the predictions of the model as a new
        feature. If the model is a classifier, the class probabilities will be
        used if available.

        Parameters
        ----------
        X : ArrayLike
            The data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        if not self.fitted:
            raise NotFittedError('StackAugmenter must be fit before it can be '
                                 'used to transform data.')
        X_transformed = np.copy(X)
        if self.is_classifier and self.use_proba:
            prediction = self.estimator.predict_proba(X)
            if np.all(np.isfinite(prediction)):
                # If all finite, add predictions
                X_transformed = np.hstack((X_transformed, prediction))

        # Add predictions (possibly on top of probabilities)
        return np.hstack(
            (X_transformed, self.estimator.predict(X).reshape(-1, 1)))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures

# Average CV score on the training set was: 0.4422745479321074
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.005),
    StackAugmenter(estimator=RidgeClassifier(alpha=0.65, tol=0.01)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
    LinearDiscriminantAnalysis(solver="svd", tol=0.01)
)
best_model = exported_pipeline
# best_model = make_pipeline(
#     StackingEstimator(estimator=BernoulliNB(alpha=0.01, fit_prior=False)),
#     PCA(iterated_power=1, svd_solver="randomized"),
#     RobustScaler(),
#     LogisticRegression(C=5.0, dual=False, penalty="l2")
# )

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.feature_selection import SelectPercentile, f_classif
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures

# # Average CV score on the training set was: 0.39228978827510036
# exported_pipeline = make_pipeline(
#     PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
#     SelectPercentile(score_func=f_classif, percentile=66),
#     LinearDiscriminantAnalysis(solver="lsqr", tol=0.001)
# )