from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_percentage_error

class ForecastingModels:
    def __init__(self, train_df, forecast_days):
        self.train_df = train_df
        self.forecast_days = forecast_days
        self.model = None  # Model will be assigned after training

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

        # Add lag features (1 to 7 days)
        feature_columns = [col for col in data.columns if col not in ['Date', 'Call Volume']]
        for lag in range(1, 8):
            for feature in feature_columns:
                data[f"{feature} Lag {lag}"] = data[feature].shift(lag)

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

    def train_xgb_model(self,name):
        """ Train an XGBoost model and log results in MLflow. """

        # Preprocess training data
        train_data = self.preprocess_data(self.train_df)

        # Define feature columns
        features = [col for col in train_data.columns if col not in ['Date', 'Call Volume']]
        target = 'Call Volume'

        # Split data into train (except last 14 days) and test (last 14 days)
        test_size = 14
        train_data = train_data.sort_values("Date")
        train_set = train_data.iloc[:-test_size]
        test_set = train_data.iloc[-test_size:]

        X_train, y_train = train_set[features], train_set[target]
        X_test, y_test = test_set[features], test_set[target]

        # Train the XGBoost model
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        mlflow.set_experiment("Call_Volume_Forecasting")
        with mlflow.start_run(run_name=name):
            self.model.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = self.model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
           
            # Log parameters and metrics in MLflow
            mlflow.xgboost.log_model(self.model, "xgb_model")
            mlflow.log_params({
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5
            })
            mlflow.log_metrics({
                "MAPE": mape
            })
        mlflow.end_run()
        return self.model

    def forecast_xgb_model(self, trained_model):
        """ Forecast next `forecast_days` call volume using a trained XGBoost model. """

        # Get the last date from training data
        last_train_date = self.train_df['Date'].max()
        forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), 
                                       periods=self.forecast_days, freq='D')

        # Create empty DataFrame for future data
        future_data = pd.DataFrame({'Date': forecast_dates})

        # Get last available training data for feature extraction
        train_data = self.preprocess_data(self.train_df)
        feature_columns = [col for col in train_data.columns if col not in ['Date', 'Call Volume']]

        # Generate lagged features for prediction
        for lag in range(1, 8):
            for feature in feature_columns:
                if feature not in ['Date', 'Call Volume']:
                    last_feature_values = train_data[[feature]].tail(lag).values.flatten()
                    if len(last_feature_values) < self.forecast_days:
                        last_feature_values = np.append(
                            np.full(self.forecast_days - len(last_feature_values), train_data[feature].mean()),
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
                future_data[col] = train_data[col].mode()[0] if not train_data.empty else 0

        # Prepare test data
        X_future = future_data[feature_columns].apply(pd.to_numeric, errors='coerce')
        X_future.fillna(0, inplace=True)

        # Predict call volume for the next forecast_days
        future_data['Predicted_Call_Volume'] = trained_model.predict(X_future)

        return future_data[['Date', 'Predicted_Call_Volume']]
