





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




class GranularForecaster(object):
    def __init__(self):
        pass