

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
            daily_seasonality = "auto",
            apply_log = True,
            show = True,
            changepoint_prior_scale=0.05,
            **kwargs):

        """Time Series forecasting using Facebook Prophet library
        """

        # Prepare the data
        df = pd.DataFrame(self.data[[self.data._target]]).reset_index()
        df.columns = ["ds","y"]

        # Prepare the model
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            **kwargs)

        # Add seasonality
        self.model.add_seasonality(name='monthly', period=30, fourier_order=4)

        # Fit the model
        self.model.fit(df)

        # Prepare the predictions
        future = self.model.make_future_dataframe(periods=periods)

        # Prediction
        forecast = self.model.predict(future)


        if show:
            self.model.plot(forecast)
            plt.axvline(x = self.data.index[-1],color = 'red',linestyle = "-",linewidth=2,label = "now")
            plt.legend()
            plt.title("Prophet prediction")
            plt.show()

            try:
                print(">> Seasonality and trends")
                self.model.plot_components(forecast)
                plt.show()
            except Exception as e:
                print(e)


        new_ts = forecast[["ds","yhat"]].copy().rename(columns = {"yhat":self.data._target,"ds":"dates"}).set_index("dates")[self.data._target]


        return forecast,new_ts
