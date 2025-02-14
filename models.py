from itertools import product
from xgboost import XGBRegressor
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

class ForecastingModels:
    def __init__(self, train_df, forecast_days):
        self.train_df = train_df
        self.forecast_days = forecast_days
        self.best_model = None
        self.best_params = None
        self.best_mape = float('inf')

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

    def train_xgb_model(self, experiment_name):
        """ Train an XGBoost model with hyperparameter tuning and log results in MLflow. """
        
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

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01],
            'max_depth': [3],
            'subsample': [0.7]
        }

        # Create all combinations of hyperparameters
        param_combinations = list(product(
            param_grid['n_estimators'], 
            param_grid['learning_rate'], 
            param_grid['max_depth'],
            param_grid['subsample']
        ))

        mlflow.set_experiment("BASE_NEW3")
        
        for params in param_combinations:
            n_estimators, learning_rate, max_depth, subsample = params

            with mlflow.start_run(run_name=experiment_name):
                model = XGBRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=42
                )

                model.fit(X_train, y_train)

                # Predictions and metrics
                y_pred = model.predict(X_test)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100

                # Log parameters and metrics in MLflow
                mlflow.log_params({
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "subsample": subsample
                })
                mlflow.log_metric("MAPE", mape)
                mlflow.xgboost.log_model(model, artifact_path="xgb_model")

                # Select the best model
                if mape < self.best_mape:
                    self.best_mape = mape
                    self.best_model = model
                    self.best_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "subsample": subsample
                    }

            mlflow.end_run()

        # Register the best model
        with mlflow.start_run(run_name="BEST_MODEL"):
            client = MlflowClient()
            mlflow.log_params(self.best_params)
            mlflow.log_metric("Best MAPE", self.best_mape)
            mlflow.xgboost.log_model(self.best_model, "best_xgb_model")
            mlflow.set_tag("Best Model", "True")
            model_uri = "runs:/{}/best_xgb_model".format(mlflow.active_run().info.run_id)
            client.create_registered_model("Best_XGB_Model")
            client.create_model_version(name="Best_XGB_Model", source=model_uri, run_id=mlflow.active_run().info.run_id)
            mlflow.end_run()

        return self.best_model


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
