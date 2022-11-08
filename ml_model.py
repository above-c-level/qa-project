from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier
)

# Average CV score on the training set was: 0.09033330719573207
# model = make_pipeline(
#     Binarizer(threshold=0.75),
#     SelectPercentile(score_func=f_classif, percentile=51),
#     BernoulliNB(alpha=1.0, fit_prior=True)
# )
model = LogisticRegression(
    C=0.1,
    class_weight="balanced",
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    l1_ratio=None,
    max_iter=1000,
    multi_class="auto",
    penalty="l2",
)
