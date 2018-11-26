
# Native
import time

# External
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom


import warnings
warnings.filterwarnings("ignore")


class Forecaster(object):
    """Wrapper for time series forecasting
    """
    def __init__(self,data):

        self.data = data


    def predict(self):
        pass



    # CLASSIC METRICS 
    @staticmethod
    def _scores_r2(y_true, y_pred):
        '''r squared coefficient of determination'''
        return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

    @staticmethod
    def _scores_mape(y_true, y_pred):
        '''mean absolute percent error'''
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def _scores_mae(y_true, y_pred):
        '''mean absolute error'''
        return np.mean(np.abs((y_true - y_pred)))


    def _compute_all_metrics(self,y_true,y_pred):

        metrics = {
            "r2":self._scores_r2(y_true,y_pred),
            "mape":self._scores_mape(y_true,y_pred),
            "mae":self._scores_mae(y_true,y_pred),
        }

        return metrics



    def train_test_predict(self,periods = 30,**kwargs):
        

        train,test = self.data.train_test_split(periods = periods,**kwargs)


        yhat_test,yhat_train,_ = self.predict(train = train,show = False)


        y_train,y_test = train[self.data._target],test[self.data._target]


        metrics_train = self._compute_all_metrics(y_train,yhat_train)
        metrics_test = self._compute_all_metrics(y_test,yhat_test)


        return metrics_train,metrics_test


        

class GranularForecaster(object):
    def __init__(self):
        pass