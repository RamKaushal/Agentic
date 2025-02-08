from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class ForecastingModels:
    def __init__(self, train_df, forecast_days=14):
        self.train_df = train_df
        self.forecast_days = forecast_days

    def preprocess_data(self, data):
        """ Preprocess train data by adding lag and time-based features. """
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

        # Add lag features (1 to 7 days) only for features, not the target
        feature_columns = [col for col in data.columns if col not in ['Date', 'Call Volume']]
        lagged_features = []
        for lag in range(1, 8):
            for feature in feature_columns:
                lagged_features.append(data[feature].shift(lag).rename(f"{feature} Lag {lag}"))
        
        # Concatenate lagged features to original data
        data = pd.concat([data] + lagged_features, axis=1)

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
        """ Train the XGBoost model and forecast call volume for the next n days. """

        # Preprocess train dataset
        train_data = self.preprocess_data(self.train_df)

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

        # Generate next n days
        last_train_date = train_data['Date'].max()
        forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=self.forecast_days, freq='D')

        # Create empty DataFrame for future data
        future_data = pd.DataFrame({'Date': forecast_dates})

        # Add lag features for input features, not target
        for lag in range(1, 8):
            for feature in features:
                if feature not in ['Date', 'Call Volume']:
                    last_feature_values = train_data[[feature]].tail(lag).values.flatten()
                    if len(last_feature_values) < self.forecast_days:
                        last_feature_values = np.append(
                            np.full(self.forecast_days - len(last_feature_values), X_train[feature].mean()),
                            last_feature_values
                        )
                    future_data[f"{feature} Lag {lag}"] = last_feature_values[:self.forecast_days]

        # Add time-based features
        future_data['Day of Week'] = future_data['Date'].dt.weekday
        future_data['Day of Month'] = future_data['Date'].dt.day
        future_data['Day of Year'] = future_data['Date'].dt.dayofyear
        future_data['Week of Year'] = future_data['Date'].dt.isocalendar().week
        future_data['Month'] = future_data['Date'].dt.month
        future_data['Quarter'] = future_data['Date'].dt.quarter
        future_data['Is Weekend'] = (future_data['Date'].dt.weekday >= 5).astype(int)

        # Fill categorical columns with most frequent value from training data
        for col in ['U.S. Holiday Indicator', 'Call Volume Impact']:
            if col in train_data.columns:
                future_data[col] = train_data[col].mode()[0]

        # Prepare test data
        X_future = future_data[features].apply(pd.to_numeric, errors='coerce')
        X_future.fillna(0, inplace=True)

        # Predict call volume for the next n days
        future_data['Call Volume'] = xgb_model.predict(X_future)

        return future_data[['Date', 'Call Volume']]
