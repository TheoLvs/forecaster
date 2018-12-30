# -*- coding: utf-8 -*-
"""Data Module

This module goals is to handle data either at granular or aggregated level
It provides time series functions and prepare data for the forecaster model


Todo:
    * Add create_time_features() method
    * Add Time Series cross validation
    * Add visualization for time series
    * Add time series clustering
    * Tsfresh features


"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import seaborn as sns
from statsmodels import api as sm
from sklearn.model_selection import TimeSeriesSplit





class Data(pd.DataFrame):

    def __init__(self,data,date_column,target,dayfirst = False,categorical_vars = None):
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
        data.sort_index(inplace = True)
        

        # Init data
        super().__init__(data)

        # Set attributes
        assert target in data.columns
        self._target = target
        self._categorical_vars = []
        self._freq = self.index.inferred_freq
        print(f"... Inferred time granularity is {self._freq}")

        # Handling Categorical variables
        if categorical_vars is not None:
            print(f"... Categorical variables are {categorical_vars}")
            for col in categorical_vars:
                self._append_categorical_var(col)

    @property
    def target(self):
        return self._target


    def _append_categorical_var(self,col):
        assert col in self.columns
        self._categorical_vars.append(col)
        self[col] = self[col].astype("category")
    

    #--------------------------------------------------------------------------------
    # FEATURE ENGINEERING


    def create_time_features(self):

        # Select column with date
        date_col = self.index

        # Create time features
        self["year"] = date_col.year
        self["month"] = date_col.month
        self["week"] = date_col.week
        self["day"] = date_col.day
        self["weekday"] = date_col.weekday_name
        # self["weekday_number"] = date_col.dayofweek

        # All columns except for year can be considered as categorical variables
        for col in ["month","week","day","weekday"]:
            self._append_categorical_var(col)




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
        """Helper static method to convert frequencies to a readable format

        Args:
            freq (str): the input frequency inferred from Pandas DateTime indexes

        Returns:
            str: the readable frequency
        
        """
        if freq == "D":
            return "days"
        elif freq == "M":
            return "months"
        elif freq == "W":
            return "weeks"
        else:
            return freq
    

    def train_test_split(self,periods = 30):
        """Simple train test split function to separate the last moments of the time series

        Args:
            periods (int): the input frequency inferred from Pandas DateTime indexes

        Returns:
            str: the readable frequency
        """
        print(f"Splitting data for testing on the last {periods} {self._get_readable_freq(self._freq)}")

        train = self.iloc[:-periods]
        test = self.iloc[-periods:]
        return train,test


    def CV_split(self):
        pass

        





class GranularData(Data):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._all_dates = self.index.drop_duplicates()


    def train_test_split(self,periods = 30):
        
        # Split dates
        dates_train = self._all_dates[:-periods]
        dates_test = self._all_dates[-periods:]

        # Split real dataset
        train = self.loc[dates_train]
        test = self.loc[dates_test]

        return train,test


    def CV_split(self,n_splits = 5):

        # Prepare split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_index,test_index in tscv.split(self._all_dates):

            # Split dates
            dates_train = self._all_dates[train_index]
            dates_test = self._all_dates[test_index]

            # Split real dataset
            train = self.loc[dates_train]
            test = self.loc[dates_test]

            yield train,test










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
