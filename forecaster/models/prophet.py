

# Native
import time

# External
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

# Custom
from ..model import Forecaster


class ProphetForecaster(Forecaster):
    """Wrapper for time series forecasting
    """
    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)


    def predict(self,
            periods = 30,
            yearly_seasonality = "auto",
            show = True,
            changepoint_prior_scale=0.05,
            train = None,
            **kwargs):

        """Time Series forecasting using Facebook Prophet library
        """

        if train is None:
            train = self.data




        # Prepare the data
        df = pd.DataFrame(train[[self.data._target]]).reset_index()
        df.columns = ["ds","y"]

        # Prepare daily seasonality
        if self.data._freq == "D":
            daily_seasonality = True
        else:
            daily_seasonality = False


        # Prepare the model
        self.model = Prophet(
            yearly_seasonality=True,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            **kwargs)

        # Add seasonality
        if self.data._freq in ["D","M"]:
            self.model.add_seasonality(name='monthly', period=30, fourier_order=4)

        # Fit the model
        self.model.fit(df)

        # Prepare the predictions
        future = self.model.make_future_dataframe(periods=periods)

        # Prediction
        forecast = self.model.predict(future)


        if show:
            self.model.plot(forecast)
            plt.axvline(x = train.index[-1],color = 'red',linestyle = "-",linewidth=2,label = "now")
            plt.legend()
            plt.title("Prophet prediction")
            plt.show()

            try:
                print(">> Seasonality and trends")
                self.model.plot_components(forecast)
                plt.show()
            except Exception as e:
                print(e)


        prediction = forecast[["ds","yhat"]].copy().rename(columns = {"yhat":self.data._target,"ds":"dates"}).set_index("dates")[self.data._target]
        pred_train,pred = prediction.iloc[:-periods],prediction.iloc[-periods:]


        return pred,pred_train,forecast






    def train_test_predict(self,periods = 30,**kwargs):
        

        train,test = self.data.train_test_split(periods = periods,**kwargs)


        pred_test,pred_train,_ = self.predict(train = train,show = False)


        y_train,y_test = train[self.data._target],test[self.data._target]


        metrics_train = self._compute_all_metrics(y_train,pred_train)
        metrics_test = self._compute_all_metrics(y_test,pred_test)


        return metrics_train,metrics_test


