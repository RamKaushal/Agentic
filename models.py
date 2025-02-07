from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

class Forecasting_models:
    def __init__(self, train_df, test_df, forecast_days):
        self.train_df = train_df
        self.test_df = test_df
        self.forecast_days = forecast_days
        return None

    def forecast_acd_call_volume(self, prediction_start_date, next_days):
        # Ensure the data is sorted by date
        data = self.train_df.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')

        # Replace NaNs and zeros in 'Call Volume' with the mean
        call_volume_mean = data['Call Volume'][data['Call Volume'] != 0].mean()
        data['Call Volume'] = data['Call Volume'].replace(0, call_volume_mean)
        data['Call Volume'] = data['Call Volume'].fillna(call_volume_mean)

        # Add lag features
        for lag in range(1, 8):
            data[f"Call Volume Lag {lag}"] = data['Call Volume'].shift(lag)

        # Include additional features from the screenshot
        additional_features = ['Day of Week', 'Day of Month', 'Day of Year', 'Week of Year', 'Month', 'Quarter', 'Is Weekend']
        features = [col for col in data.columns if col not in ['Date', 'Call Volume']] + additional_features

        # Drop rows with NaN values due to lag creation
        data.dropna(inplace=True)

        # Define target
        target = 'Call Volume'

        # Split into training and forecasting sets
        train_data = data[data['Date'] < prediction_start_date]
        forecast_start_date = pd.to_datetime(prediction_start_date)

        # Prepare training data
        X_train = train_data[features]
        y_train = train_data[target]

        # Train the XGBoost model
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Generate predictions for the specified number of days
        forecast_dates = [forecast_start_date + pd.Timedelta(days=i) for i in range(next_days)]
        forecast_data = pd.DataFrame({"Date": forecast_dates})

        # Add lag features dynamically for forecasting
        for lag in range(1, 8):
            forecast_data[f"Call Volume Lag {lag}"] = None

        predictions = []

        for date in forecast_dates:
            # Fill lag features based on the most recent data
            for lag in range(1, 8):
                lag_date = date - pd.Timedelta(days=lag)
                if lag_date in data['Date'].values:
                    lag_value = data.loc[data['Date'] == lag_date, 'Call Volume'].values[0]
                elif len(predictions) >= lag:
                    lag_value = predictions[-lag]
                else:
                    lag_value = call_volume_mean
                forecast_data.loc[forecast_data['Date'] == date, f"Call Volume Lag {lag}"] = lag_value

            # Include additional features dynamically for forecasting
            forecast_data['Day of Week'] = date.weekday()
            forecast_data['Day of Month'] = date.day
            forecast_data['Day of Year'] = date.timetuple().tm_yday
            forecast_data['Week of Year'] = date.isocalendar()[1]
            forecast_data['Month'] = date.month
            forecast_data['Quarter'] = (date.month - 1) // 3 + 1
            forecast_data['Is Weekend'] = 1 if date.weekday() >= 5 else 0

            # Prepare features for prediction
            current_features = forecast_data[forecast_data['Date'] == date][features]

            # Predict call volume
            predicted_value = xgb_model.predict(current_features)[0]
            predictions.append(predicted_value)

            # Update the data with the predicted value for lag calculations
            data = pd.concat([
                data,
                pd.DataFrame({"Date": [date], "Call Volume": [predicted_value]})
            ], ignore_index=True)

        # Combine results into a DataFrame
        forecast_data['Predicted Call Volume'] = predictions
        return forecast_data[['Date', 'Predicted Call Volume']]
