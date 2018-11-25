# -*- coding: utf-8 -*-
"""Data Module

This module goals is to handle data either at granular or aggregated level
It provides time series functions and prepare data for the forecaster model


Todo:
    * Add create_time_features() method
    * Add Time Series cross validation
    * Add visualization for time series
    * Add time series clustering


"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import seaborn as sns
from statsmodels import api as sm





class Data(pd.DataFrame):

    def __init__(self,data,date_column,target,dayfirst = False):
        """Base data class for Forecaster library

        Args:
            data (pd.DataFrame): a Pandas dataframe with time series data.
            date_column (str): the date column to be used as index
            target (str): the target variable you want to predict
            dayfirst (bool): argument for ``pd.to_datetime`` function.

        """
        print("Reading input data for Forecaster model")

        # Convert date column to datetime and index
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column],dayfirst = dayfirst)
        data.set_index(date_column,inplace = True)

        # Init data
        super().__init__(data)

        # Set attributes
        assert target in data.columns
        self._target = target
        self._freq = self.index.inferred_freq
        print(f"Inferred time granularity is {self._freq}")



    @property
    def target(self):
        return self._target
    


    #--------------------------------------------------------------------------------
    # FEATURE ENGINEERING


    def create_time_features(self):
        pass


    def decompose_hp_filter(self,column,alpha = 6.25,show = False):
        """Decompose seasonality from a time series column with the Hodrick Prescott filter

        Args:
            column (str): a column of the dataframe.
            alpha (float): the Hodrick Prescott smoothing factor.
            show (bool): visualize decomposition with matplotlib
        
        """
        c, t = sm.tsa.filters.hpfilter(self[column], alpha)
        self[f"{column}_cycle"] = c
        self[f"{column}_trend"] = t

        # Visualize with matplotlib
        if show:
            self[[column,f"{column}_cycle",f"{column}_trend"]].plot(figsize = (15,4))
            plt.show()





    #--------------------------------------------------------------------------------
    # PERFORMANCE MEASUREMENT


    @staticmethod
    def _get_readable_freq(freq):
        if freq == "D":
            return "days"
        elif freq == "M":
            return "months"
        elif freq == "W":
            return "weeks"
    

    def train_test_split(self,periods = 30):
        print(f"Splitting data for testing on the last {periods} {self._get_readable_freq(self._freq)}")

        train = self.iloc[:-periods]
        test = self.iloc[-periods:]
        return train,test


    def CV_split(self):
        pass

        



# def train_test_split(ds, n_splits=3):
#     """Just a reimplementation of scikit-learn's TimeSeriesSplit that
#     outputs a TimeSeriesDataset instead of numpy arrays.
#     """
   
#     df = ds.build_dataframe()
#     df.sort_index(inplace=True)

#     tscv = TimeSeriesSplit(n_splits=n_splits)

#     for train_index, test_index in tscv.split(df.as_matrix()):

#         df_train, df_test = df.iloc[train_index, :], df.iloc[test_index]

#         ds_train = TimeSeriesDataset(dataframe=df_train, target=ds.target)
#         ds_test = TimeSeriesDataset(dataframe=df_test, target=ds.target)

#         yield (ds_train, ds_test)


# def sliding_split(ds, train_size, test_size, step=1):
#     """A cross validation with fixed-size, sliding train & test sets.
#     Is meant to test stability over time.
#     """
   
#     df = ds.build_dataframe()
#     df.sort_index(inplace=True)

#     i = 0
#     while i+train_size+test_size <= len(df):

#         df_train = df.iloc[i:i+train_size, :]
#         df_test = df.iloc[i+train_size:i+train_size+test_size, :]

#         ds_train = TimeSeriesDataset(dataframe=df_train, target=ds.target)
#         ds_test = TimeSeriesDataset(dataframe=df_test, target=ds.target)

#         yield (ds_train, ds_test)

#         i += step
