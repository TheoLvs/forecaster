# -*- coding: utf-8 -*-
"""ARIMA Forecaster

This module contains implementation for time series forecasting with ARIMA

Sample code

    # ARIMA example
    from statsmodels.tsa.arima_model import ARIMA
    from random import random
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data), typ='levels')
    print(yhat)


"""




# Native
import time

# External
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Custom
from ..model import Forecaster


class ArimaForecaster(Forecaster):
    """Wrapper for time series forecasting with ARIMA
    """
    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)


    def predict(self,periods = 30,show = True,train = None,order = (2,1,2),**kwargs):

        if train is None:
            train = self.data

        # fit model
        self.model = ARIMA(train[[self.data._target]], order=order)
        model_fit = self.model.fit(disp=False)

        # make prediction
        yhat = model_fit.predict(len(train), len(train)+periods, typ='levels')

        return yhat