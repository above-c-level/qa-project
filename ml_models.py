from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import (SelectPercentile, f_classif)
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MaxAbsScaler, PolynomialFeatures,
                                   PowerTransformer, RobustScaler,
                                   StandardScaler)
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import LinearSVC
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


# Average CV score on the training set was: 0.7928506104371915
sentence_model = CalibratedClassifierCV(
    make_pipeline(
        StackAugmenter(estimator=GaussianNB(var_smoothing=1e-09)),
        StandardScaler(),
        SelectPercentile(score_func=f_classif, percentile=67),
        PolynomialFeatures(degree=2, include_bias=False,
                           interaction_only=True),
        LinearSVC(C=0.01,
                  class_weight="balanced",
                  dual=False,
                  loss="squared_hinge",
                  penalty="l1",
                  tol=0.01),
    ))

end_word_model = make_pipeline(
    RobustScaler(),
    LassoLarsCV(normalize=False)
)
start_word_model = make_pipeline(
    ValueCount(0),
    MaxAbsScaler(),
    LassoLarsCV(normalize=False)
)
