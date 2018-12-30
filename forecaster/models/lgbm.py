# -*- coding: utf-8 -*-
"""LightGBM Forecaster Module

This module goals is to handle forecasting using LightGBM
Forecasting can be done at granular level

Todo:
    * Integrate with all forecasters

References: 
    * https://www.kaggle.com/mlisovyi/beware-of-categorical-features-in-lgbm
    * https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
    * https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py

"""



# Native
import time

# External
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

# Custom
from ..model import Forecaster




class LGBMForecaster(Forecaster):
    def __init__(self): #,*args,**kwargs):

        # super().__init__(*args,**kwargs)
        pass


    def fit(self,X_train,X_test,y_train,y_test,categorical_vars = None,params = None
                ,num_boost_round = 100,early_stopping_rounds = 5,objective='regression_l2'):
        """Fit function of the LGBM forecaster
        Parameters available at https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        """

        # Prepare categorical variables
        if categorical_vars is None:
            categorical_vars = "auto"

        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_vars)
        test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_vars,reference = train_data)

        # Prepare hyperparams
        if params is None:
            params = {
                'boosting_type': 'gbdt',
                'objective': objective,
                'metric': {'l2', 'l1'},
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            }


        # Training pass
        print('... Starting training')
        self.model = lgb.train(params,
                        train_data,
                        num_boost_round=num_boost_round,
                        valid_sets=test_data,
                        early_stopping_rounds=early_stopping_rounds)




    def predict(self,X_train,X_test = None,y_train = None,y_test = None):

        if X_test is None:

            pred = self.model.predict(X_train)
            return pred

        else:

            pred_train =  self.model.predict(X_train)
            pred_test = self.model.predict(X_test)

            metrics_train = self._compute_all_metrics(y_train,pred_train)
            metrics_test = self._compute_all_metrics(y_test,pred_test)

            return metrics_train,metrics_test
            