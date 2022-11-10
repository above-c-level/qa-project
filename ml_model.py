from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor,
    Lars, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,
    LinearRegression, LogisticRegression, LogisticRegressionCV,
    OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor, Perceptron, PoissonRegressor, Ridge, RidgeCV,
    SGDRegressor, TheilSenRegressor, TweedieRegressor)
from sklearn.naive_bayes import (BernoulliNB, GaussianNB)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

# Average CV score on the training set was: 0.09033330719573207
# model = make_pipeline(
#     Binarizer(threshold=0.75),
#     SelectPercentile(score_func=f_classif, percentile=51),
#     BernoulliNB(alpha=1.0, fit_prior=True)
# )
model = LinearRegression()
all_models = {
    # Discriminant Analysis
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    # Ensemble Methods
    "AdaBoostRegressor": AdaBoostRegressor(),
    "BaggingRegressor": BaggingRegressor(),
    "ExtraTreesRegressor": ExtraTreesRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
    # Kernel Ridge
    "KernelRidge": KernelRidge(),
    # Linear models, including baysesian methods and generalized linear models
    "ARDRegression": ARDRegression(),
    "BayesianRidge": BayesianRidge(),
    "ElasticNet": ElasticNet(),
    "ElasticNetCV": ElasticNetCV(),
    "HuberRegressor": HuberRegressor(max_iter=200),
    "Lars": Lars(),
    "Lasso": Lasso(),
    "LassoCV": LassoCV(),
    "LassoLars": LassoLars(),
    "LassoLarsCV": LassoLarsCV(),
    "LassoLarsIC": LassoLarsIC(),
    "LinearRegression": LinearRegression(),
    "LogisticRegression": LogisticRegression(),
    "LogisticRegressionCV": LogisticRegressionCV(),
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
    "OrthogonalMatchingPursuitCV": OrthogonalMatchingPursuitCV(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    "Perceptron": Perceptron(),
    "PoissonRegressor": PoissonRegressor(),
    "Ridge": Ridge(),
    "RidgeCV": RidgeCV(),
    "SGDRegressor": SGDRegressor(),
    "TheilSenRegressor": TheilSenRegressor(),
    "TweedieRegressor": TweedieRegressor(),
    # Naive Bayes
    "BernoulliNB": BernoulliNB(),
    "GaussianNB": GaussianNB(),
    # Nearest Neighbors
    "KNeighborsRegressor": KNeighborsRegressor(),
    "RadiusNeighborsRegressor": RadiusNeighborsRegressor(),
    # Multi-layer Perceptron
    "MLPRegressor": MLPRegressor(),
    # Support Vector Machines
    "LinearSVR": LinearSVR(),
    "NuSVR": NuSVR(),
    "SVR": SVR(),
    # Trees
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "ExtraTreeRegressor": ExtraTreeRegressor()
}
