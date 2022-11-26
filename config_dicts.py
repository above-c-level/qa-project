import numpy as np

classifier_config_dict = {
    # Things not in scikit-learn
    'catboost.CatBoostClassifier': {
        'boosting_type': ['Ordered', 'Plain'],
        'max_depth':
        range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'iterations':
        [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
        'min_data_in_leaf': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
        'subsample':
        np.arange(0.05, 1.01, 0.05),
        'rsm': [0.7, 0.9, 1.0],
    },
    'xgboost.XGBClassifier': {
        'n_estimators':
        [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
        'max_depth':
        range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample':
        np.arange(0.05, 1.01, 0.05),
        'min_child_weight':
        range(1, 21),
        'n_jobs': [1],
        'verbosity': [0]
    },

    # Preprocessors
    'ml_models.ValueCount': {
        'value': [0, 1]
    },
    'ml_models.ValueOverUnder': {
        'value': np.arange(-1.0, 1.01, 0.05)
    },
    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },
    'sklearn.cluster.KMeans': {
        'n_clusters': range(2, 21),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.decomposition.FactorAnalysis': {
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
    'sklearn.kernel_approximation.Nystroem': {
        'kernel': [
            'rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly',
            'linear', 'additive_chi2', 'sigmoid'
        ],
        'gamma':
        np.arange(0.0, 1.01, 0.05),
        'n_components':
        range(1, 11)
    },
    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.MaxAbsScaler': {},
    'sklearn.preprocessing.MinMaxScaler': {},
    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },
    'sklearn.preprocessing.OneHotEncoder': {
        'min_frequency': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'max_categories': [10]
    },
    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [True, False]
    },
    'sklearn.preprocessing.PowerTransformer': {},
    'sklearn.preprocessing.RobustScaler': {},
    'sklearn.preprocessing.StandardScaler': {},

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(50, 201, 50),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            },
            'sklearn.linear_model.ElasticNet': {
                'alpha': np.arange(0.0, 2.01, 0.05),
                'l1_ratio': np.arange(0.0, 1.01, 0.05)
            }
        }
    },
    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(50, 201, 50),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            },
            'sklearn.linear_model.ElasticNet': {
                'alpha': np.arange(0.0, 2.01, 0.05),
                'l1_ratio': np.arange(0.0, 1.01, 0.05)
            }
        }
    },

    # Classifiers
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': {
        'reg_param': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
        'criterion': ["gini", "entropy", "log_loss"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
        'criterion': ["gini", "entropy", "log_loss"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
        'criterion': ["friedman_mse", "squared_error", "mse"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2", "elasticnet", "none"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },
    'sklearn.linear_model.PassiveAggressiveClassifier': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'loss': ["hinge", "squared_hinge"],
    },
    'sklearn.linear_model.RidgeClassifier': {
        'alpha': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.linear_model.SGDClassifier': {
        'loss': [
            'hinge',
            'log_loss',
            'modified_huber',
            'squared_hinge',
            'perceptron',
            'squared_error',
            'huber',
            'epsilon_insensitive',
            'squared_epsilon_insensitive',
        ],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.],
        'learning_rate': ['invscaling', 'constant', 'optimal', 'adaptive'],
        'fit_intercept': [True, False],
        'l1_ratio':
        np.arange(0.0, 1.01, 0.05),
        'eta0': [1e-4, 1e-3, 1e-2, 1e-1, 0.5],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },
    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.ComplementNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False],
    },
    'sklearn.naive_bayes.GaussianNB': {
        'var_smoothing': [1e-9, 1e-10, 1e-8, 1e-11, 1e-7, 1e-6, 1e-5],
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors':
        range(1, 101),
        'weights': ["uniform", "distance"],
        'metric': [
            "euclidean", "cityblock", "chebyshev", "cosine", "canberra",
            "braycurtis", "sqeuclidean"
        ],
    },
    'sklearn.neighbors.RadiusNeighborsClassifier': {
        'radius': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'weights': ["uniform", "distance"],
        'metric': [
            "euclidean", "cityblock", "chebyshev", "cosine", "canberra",
            "braycurtis", "sqeuclidean"
        ],
    },
    'sklearn.neural_network.MLPClassifier': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'activation': ['relu', 'tanh', 'logistic'],
    },
    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'class_weight': ['balanced', None]
    },
    'sklearn.svm.NuSVC': {
        'nu': [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(2, 11),
        'class_weight': ['balanced', None]
    },
    'sklearn.svm.SVC': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(2, 11),
        'class_weight': ['balanced', None]
    },
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    },
    'sklearn.tree.ExtraTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    },
}

classifier_config_dict_fast = {
    # Things not in scikit-learn
    # 'catboost.CatBoostClassifier': {
    #     'boosting_type': ['Ordered', 'Plain'],
    #     'max_depth':
    #     range(1, 11),
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'iterations':
    #     [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
    #     'min_data_in_leaf': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
    #     'subsample':
    #     np.arange(0.05, 1.01, 0.05),
    #     'rsm': [0.7, 0.9, 1.0],
    # },
    # 'xgboost.XGBClassifier': {
    #     'n_estimators':
    #     [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
    #     'max_depth':
    #     range(1, 11),
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'subsample':
    #     np.arange(0.05, 1.01, 0.05),
    #     'min_child_weight':
    #     range(1, 21),
    #     'n_jobs': [1],
    #     'verbosity': [0]
    # },

    # Preprocessors
    'ml_models.ValueCount': {
        'value': [0, 1]
    },
    'ml_models.ValueOverUnder': {
        'value': np.arange(-1.0, 1.01, 0.05)
    },
    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },
    'sklearn.cluster.KMeans': {
        'n_clusters': range(2, 21),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.decomposition.FactorAnalysis': {
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
    'sklearn.kernel_approximation.Nystroem': {
        'kernel': [
            'rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly',
            'linear', 'additive_chi2', 'sigmoid'
        ],
        'gamma':
        np.arange(0.0, 1.01, 0.05),
        'n_components':
        range(1, 11)
    },
    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.MaxAbsScaler': {},
    'sklearn.preprocessing.MinMaxScaler': {},
    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },
    'sklearn.preprocessing.OneHotEncoder': {
        'min_frequency': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'max_categories': [10]
    },
    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [True, False]
    },
    'sklearn.preprocessing.PowerTransformer': {},
    'sklearn.preprocessing.RobustScaler': {},
    'sklearn.preprocessing.StandardScaler': {},

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            # 'sklearn.ensemble.ExtraTreesClassifier': {
            #     'n_estimators': range(50, 201, 50),
            #     'criterion': ['gini', 'entropy'],
            #     'max_features': np.arange(0.05, 1.01, 0.05)
            # },
            'sklearn.linear_model.ElasticNet': {
                'alpha': np.arange(0.0, 2.01, 0.05),
                'l1_ratio': np.arange(0.0, 1.01, 0.05)
            }
        }
    },
    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(50, 201, 50),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            },
            'sklearn.linear_model.ElasticNet': {
                'alpha': np.arange(0.0, 2.01, 0.05),
                'l1_ratio': np.arange(0.0, 1.01, 0.05)
            }
        }
    },

    # Classifiers
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': {
        'reg_param': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
        'criterion': ["gini", "entropy", "log_loss"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    # 'sklearn.ensemble.RandomForestClassifier': {
    #     'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
    #     'criterion': ["gini", "entropy", "log_loss"],
    #     'max_features': np.arange(0.05, 1.01, 0.05),
    #     'min_samples_split': range(2, 21),
    #     'min_samples_leaf': range(1, 21),
    #     'bootstrap': [True, False]
    # },
    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
        'criterion': ["friedman_mse", "squared_error", "mse"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2", "elasticnet", "none"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },
    'sklearn.linear_model.PassiveAggressiveClassifier': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'loss': ["hinge", "squared_hinge"],
    },
    'sklearn.linear_model.RidgeClassifier': {
        'alpha': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.linear_model.SGDClassifier': {
        'loss': [
            'hinge',
            'log_loss',
            'modified_huber',
            'squared_hinge',
            'perceptron',
            'squared_error',
            'huber',
            'epsilon_insensitive',
            'squared_epsilon_insensitive',
        ],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.],
        'learning_rate': ['invscaling', 'constant', 'optimal', 'adaptive'],
        'fit_intercept': [True, False],
        'l1_ratio':
        np.arange(0.0, 1.01, 0.05),
        'eta0': [1e-4, 1e-3, 1e-2, 1e-1, 0.5],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },
    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.ComplementNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False],
    },
    'sklearn.naive_bayes.GaussianNB': {
        'var_smoothing': [1e-9, 1e-10, 1e-8, 1e-11, 1e-7, 1e-6, 1e-5],
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors':
        range(1, 101),
        'weights': ["uniform", "distance"],
        'metric': [
            "euclidean", "cityblock", "chebyshev", "cosine", "canberra",
            "braycurtis", "sqeuclidean"
        ],
    },
    'sklearn.neighbors.RadiusNeighborsClassifier': {
        'radius': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'weights': ["uniform", "distance"],
        'metric': [
            "euclidean", "cityblock", "chebyshev", "cosine", "canberra",
            "braycurtis", "sqeuclidean"
        ],
    },
    'sklearn.neural_network.MLPClassifier': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'activation': ['relu', 'tanh', 'logistic'],
    },
    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'class_weight': ['balanced', None]
    },
    'sklearn.svm.NuSVC': {
        'nu': [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(2, 11),
        'class_weight': ['balanced', None]
    },
    'sklearn.svm.SVC': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(2, 11),
        'class_weight': ['balanced', None]
    },
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    },
    'sklearn.tree.ExtraTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    },
}

classifier_config_dict_extra_fast = {
    # Things not in scikit-learn
    # 'catboost.CatBoostClassifier': {
    #     'boosting_type': ['Ordered', 'Plain'],
    #     'max_depth':
    #     range(1, 11),
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'iterations':
    #     [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
    #     'min_data_in_leaf': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
    #     'subsample':
    #     np.arange(0.05, 1.01, 0.05),
    #     'rsm': [0.7, 0.9, 1.0],
    # },
    # 'xgboost.XGBClassifier': {
    #     'n_estimators':
    #     [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
    #     'max_depth':
    #     range(1, 11),
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'subsample':
    #     np.arange(0.05, 1.01, 0.05),
    #     'min_child_weight':
    #     range(1, 21),
    #     'n_jobs': [1],
    #     'verbosity': [0]
    # },

    # Preprocessors
    'ml_models.ValueCount': {
        'value': [0, 1]
    },
    'ml_models.ValueOverUnder': {
        'value': np.arange(-1.0, 1.01, 0.05)
    },
    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },
    'sklearn.cluster.KMeans': {
        'n_clusters': range(2, 21),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.decomposition.FactorAnalysis': {
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
    'sklearn.kernel_approximation.Nystroem': {
        'kernel': [
            'rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly',
            'linear', 'additive_chi2', 'sigmoid'
        ],
        'gamma':
        np.arange(0.0, 1.01, 0.05),
        'n_components':
        range(1, 11)
    },
    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.preprocessing.MaxAbsScaler': {},
    'sklearn.preprocessing.MinMaxScaler': {},
    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },
    'sklearn.preprocessing.OneHotEncoder': {
        'min_frequency': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'max_categories': [10]
    },
    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [True, False]
    },
    'sklearn.preprocessing.PowerTransformer': {},
    'sklearn.preprocessing.RobustScaler': {},
    'sklearn.preprocessing.StandardScaler': {},

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    # 'sklearn.feature_selection.RFE': {
    #     'step': np.arange(0.05, 1.01, 0.05),
    #     'estimator': {
    #         # 'sklearn.ensemble.ExtraTreesClassifier': {
    #         #     'n_estimators': range(50, 201, 50),
    #         #     'criterion': ['gini', 'entropy'],
    #         #     'max_features': np.arange(0.05, 1.01, 0.05)
    #         # },
    #         'sklearn.linear_model.ElasticNet': {
    #             'alpha': np.arange(0.0, 2.01, 0.05),
    #             'l1_ratio': np.arange(0.0, 1.01, 0.05)
    #         }
    #     }
    # },
    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(50, 201, 50),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            },
            'sklearn.linear_model.ElasticNet': {
                'alpha': np.arange(0.0, 2.01, 0.05),
                'l1_ratio': np.arange(0.0, 1.01, 0.05)
            }
        }
    },

    # Classifiers
    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': {
        'reg_param': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    # 'sklearn.ensemble.ExtraTreesClassifier': {
    #     'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
    #     'criterion': ["gini", "entropy", "log_loss"],
    #     'max_features': np.arange(0.05, 1.01, 0.05),
    #     'min_samples_split': range(2, 21),
    #     'min_samples_leaf': range(1, 21),
    #     'bootstrap': [True, False]
    # },
    # 'sklearn.ensemble.RandomForestClassifier': {
    #     'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
    #     'criterion': ["gini", "entropy", "log_loss"],
    #     'max_features': np.arange(0.05, 1.01, 0.05),
    #     'min_samples_split': range(2, 21),
    #     'min_samples_leaf': range(1, 21),
    #     'bootstrap': [True, False]
    # },
    # 'sklearn.ensemble.GradientBoostingClassifier': {
    #     'n_estimators': [5, 10, 25, 50, 100, 250, 500, 1000],
    #     'criterion': ["friedman_mse", "squared_error", "mse"],
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'max_depth': range(1, 11),
    #     'min_samples_split': range(2, 21),
    #     'min_samples_leaf': range(1, 21),
    #     'subsample': np.arange(0.05, 1.01, 0.05),
    #     'max_features': np.arange(0.05, 1.01, 0.05)
    # },
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2", "elasticnet", "none"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },
    'sklearn.linear_model.PassiveAggressiveClassifier': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'loss': ["hinge", "squared_hinge"],
    },
    'sklearn.linear_model.RidgeClassifier': {
        'alpha': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    'sklearn.linear_model.SGDClassifier': {
        'loss': [
            'hinge',
            'log_loss',
            'modified_huber',
            'squared_hinge',
            'perceptron',
            'squared_error',
            'huber',
            'epsilon_insensitive',
            'squared_epsilon_insensitive',
        ],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.],
        'learning_rate': ['invscaling', 'constant', 'optimal', 'adaptive'],
        'fit_intercept': [True, False],
        'l1_ratio':
        np.arange(0.0, 1.01, 0.05),
        'eta0': [1e-4, 1e-3, 1e-2, 1e-1, 0.5],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },
    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.ComplementNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False],
    },
    'sklearn.naive_bayes.GaussianNB': {
        'var_smoothing': [1e-9, 1e-10, 1e-8, 1e-11, 1e-7, 1e-6, 1e-5],
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors':
        range(1, 101),
        'weights': ["uniform", "distance"],
        'metric': [
            "euclidean", "cityblock", "chebyshev", "cosine", "canberra",
            "braycurtis", "sqeuclidean"
        ],
    },
    'sklearn.neighbors.RadiusNeighborsClassifier': {
        'radius': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'weights': ["uniform", "distance"],
        'metric': [
            "euclidean", "cityblock", "chebyshev", "cosine", "canberra",
            "braycurtis", "sqeuclidean"
        ],
    },
    # 'sklearn.neural_network.MLPClassifier': {
    #     'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    #     'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'activation': ['relu', 'tanh', 'logistic'],
    # },
    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'class_weight': ['balanced', None]
    },
    'sklearn.svm.NuSVC': {
        'nu': [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(2, 11),
        'class_weight': ['balanced', None]
    },
    # 'sklearn.svm.SVC': {
    #     'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'degree': range(2, 11),
    #     'class_weight': ['balanced', None]
    # },
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    },
    'sklearn.tree.ExtraTreeClassifier': {
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
    },
}