from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class ForecastingModels:
    def __init__(self, train_df, test_df, forecast_days):
        self.train_df = train_df
        self.test_df = test_df
        self.forecast_days = forecast_days

    def preprocess_data(self, data):
        """ Preprocess train and test data by adding lag and time-based features. """
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')

        # Replace NaNs and zeros in 'Call Volume' with the mean
        call_volume_mean = data['Call Volume'][data['Call Volume'] != 0].mean()
        data['Call Volume'] = data['Call Volume'].replace(0, call_volume_mean)
        data['Call Volume'] = data['Call Volume'].fillna(call_volume_mean)

        # Convert categorical columns to numerical encoding
        categorical_columns = ['U.S. Holiday Indicator', 'Call Volume Impact']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category').cat.codes

        # Add lag features (1 to 7 days)
        for lag in range(1, 8):
            data[f"Call Volume Lag {lag}"] = data['Call Volume'].shift(lag)

        # Add additional time-based features
        data['Day of Week'] = data['Date'].dt.weekday
        data['Day of Month'] = data['Date'].dt.day
        data['Day of Year'] = data['Date'].dt.dayofyear
        data['Week of Year'] = data['Date'].dt.isocalendar().week
        data['Month'] = data['Date'].dt.month
        data['Quarter'] = data['Date'].dt.quarter
        data['Is Weekend'] = (data['Date'].dt.weekday >= 5).astype(int)

        # Drop NaN rows due to lag features
        data.dropna(inplace=True)

        return data

    def forecast_acd_call_volume(self):
        """ Train the XGBoost model and forecast call volume for the next 14 days after training """

        # Preprocess both train and test datasets
        train_data = self.preprocess_data(self.train_df)
        test_data = self.preprocess_data(self.test_df)

        # Define feature columns
        features = [col for col in train_data.columns if col not in ['Date', 'Call Volume']]
        target = 'Call Volume'

        # Prepare training data
        X_train = train_data[features].apply(pd.to_numeric, errors='coerce')
        X_train.fillna(0, inplace=True)  # Handle missing values
        y_train = train_data[target]

        # Train the XGBoost model
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Select only the next 14 days after the last training date
        last_train_date = train_data['Date'].max()
        prediction_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=self.forecast_days, freq='D')
        test_subset = test_data[test_data['Date'].isin(prediction_dates)]

        # Prepare test data
        X_test = test_subset[features].apply(pd.to_numeric, errors='coerce')
        X_test.fillna(0, inplace=True)

        # Predict call volume for the next 14 days
        test_subset['Predicted Call Volume'] = xgb_model.predict(X_test)

        return test_subset[['Date', 'Predicted Call Volume']]
