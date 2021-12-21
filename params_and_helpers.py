import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# models
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
import sklearn.gaussian_process as gp
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

# helpers etc.
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import combinations
import time

def powerset(l):
    S = []
    for size in range(0,len(l)+1):
        S.extend(list(combinations(l, size)))
    return S

def df_to_features_and_labels(df, only_positive_features=True, scale_features=True):
    r"""Separate df into features (X) and labels (Y). Convert weather to 0/1. Convert time to two continous variables ( https://stats.stackexchange.com/questions/245866/is-hour-of-day-a-categorical-variable )"""
    Y = df.iloc[:, 4]
    X = df.drop(columns=['date', df.columns[4]])
    X['weather'] = [0 if x == 'bad' else '1' for x in X['weather']]
    X['weather'] = X['weather'].astype(int)
    # https://stats.stackexchange.com/questions/245866/is-hour-of-day-a-categorical-variable
    X['time_x'] = np.sin(2*np.pi*X['time']/24)
    X['time_y'] = np.cos(2*np.pi*X['time']/24)
    X.drop(columns=['time'], inplace=True)

    
    if only_positive_features and scale_features: # use minmax scaler that scales the data to 0,1 interval
        feature_scaler = MinMaxScaler()
        X[:] = feature_scaler.fit_transform(X)        
    elif only_positive_features:
        X['time_x'] += 1
        X['time_y'] += 1 
    elif scale_features: 
        feature_scaler = StandardScaler()
        X[:] = feature_scaler.fit_transform(X)
        #X_train = feature_scaler.fit_transform(X_train)
        #X_test = feature_scaler.transform(X_test)
        
    return X, Y

only_positive_features = True
scale_features = False


# kernel ridge regression
candidate_kernels_kr = ['linear', 'rbf']
candidate_kernels_kr.append('chi2') if only_positive_features else None
parameters_kr = {'kernel': candidate_kernels_kr, 
                 "alpha": np.logspace(-5, 5, 11), 
                 "gamma": np.logspace(-5, 5, 11)}

# support vector regression
candidate_kernels_svr = ['rbf']
parameters_svr = {'kernel': candidate_kernels_svr, 
                  "C": [1e0, 1e1, 1e2, 1e3], 
                  "gamma": np.logspace(-5, 5, 11)}

# random forest regression
parameters_rfr = {"max_depth": np.arange(0,11)}


# Gaussian process regression
candidate_kernels_gpr = [RBF(l) for l in np.logspace(-1, 1, 4)]+ [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 4)]
pararameters_gpr = {'kernel':candidate_kernels_gpr, 
                    "alpha": np.logspace(-5, 5, 11)}

# Bayesian ridge regression
parameters_brr = {'alpha_init':np.logspace(-5, 5, 11),
                  'lambda_init': np.logspace(-5, 5, 11)}

# k-nearest neighbors regression
parameters_knn = {'n_neighbors': list(range(1, 15)),
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}

# elastic-net regression
parameters_enr = {"alpha": np.logspace(-5, 5, 11),
                  "l1_ratio": np.arange(0.0, 1.0, 0.1)}

# Lasso regression
parameters_lr =  {"alpha": np.logspace(-5, 5, 11)}

# Ada-Boost regression: https://educationalresearchtechniques.com/2019/01/07/adaboost-regression-with-python/
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py
parameters_ada = {'base_estimator': [KernelRidge(kernel="chi2"), KernelRidge(kernel="rbf"), None, KNeighborsRegressor()],
                  'n_estimators':[50,100,200],
                  'learning_rate':[.001,0.01,.1],
                  'random_state':[1]}

# bagging
# https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-ensemble-learning-bagging-and-random-forests
parameters_bag = {'base_estimator': [KernelRidge(kernel="chi2"), KernelRidge(kernel="rbf"), None, KNeighborsRegressor()], # zakaj KernelRidge ne dela s chi2 kernelom??
                  'n_estimators': [50,100,200],
                  #'max_samples': [0.5,1.0, n_samples//2,],
                  #'max_features': [0.5,1.0, n_features//2,],
                  #'bootstrap': [True, False],
                  #'bootstrap_features': [True, False]
                 }

# gradient boosting
parameters_gbr = {'n_estimators': [50,100,200],
                  'learning_rate':[.001,0.01,.1],
                  #'max_samples': [0.5,1.0, n_samples//2,],
                  #'max_features': [0.5,1.0, n_features//2,],
                  #'bootstrap': [True, False],
                  #'bootstrap_features': [True, False]
                 }

models = {'krr': KernelRidge(),
          'svr': SVR(),
          'rfr': RandomForestRegressor(n_estimators=100, oob_score=True),
          'gpr': GaussianProcessRegressor(),
          'bayes': BayesianRidge(),
          'knn': KNeighborsRegressor(),
          'elastic': ElasticNet(),
          'lasso': Lasso(),
          'ada':AdaBoostRegressor(),
          'bag':BaggingRegressor(),
          'gbr':GradientBoostingRegressor(),
         }


grids = {'krr': parameters_kr,
         'svr': parameters_svr,
         'rfr': parameters_rfr,
         'gpr': pararameters_gpr,
         'bayes': parameters_brr,
         'knn': parameters_knn,
         'elastic': parameters_enr,
         'lasso': parameters_lr,
         'ada': parameters_ada,
         'bag': parameters_bag,
         'gbr': parameters_gbr
        }