class Forecasting_models:
    def __init__(self,train_df,test_df,forecast_days):
        """
        Class init to input train,test and nuber of days to forecast
        """
        self.train_df = train_df
        self.test_df = test_df
        self.forecast_days = forecast_days
        return None
