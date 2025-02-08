import pandas as pd
from utils import write_data_db, read_data_db, get_logger,plot_line_chart
import yaml
from models import ForecastingModels  # Ensure this imports the correct ForecastingModels class
import joblib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logger = get_logger()

# Load config
with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml", "r") as f:
    config = yaml.safe_load(f)

forecast_days = config['forecast_days']
train_date = config['train_date']

# Write data into DB
try:
    df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv")
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    write_data_db(df, "ACD_VOLUME","replace")
    logger.info(f"Data is pushed into DB")
except Exception as e:
    logger.error(f"Failed to push data into server because of {e}")

# Read data from DB
try:
    train_date = pd.to_datetime(train_date).strftime("%Y-%m-%d")
    query = f"""
        SELECT * FROM ACD_VOLUME 
        WHERE DATE(strftime('%Y-%m-%d', Date)) <= DATE('{train_date}')
    """
    df_read = read_data_db(query)
    df_read['Date'] = pd.to_datetime(df_read['Date'])
    logger.info("Data is read into DF FROM DB")
except Exception as e:
    logger.error(f"Failed to read data from DB because of {e}")

logger.info(f"Forecast days are read and set to {forecast_days}")
logger.info(f"Training data till {train_date}")

#SCENARIO BASE: CREATING AN XGB MODEL AND SAVING ITS WEIGHTS
try:
    forecast_obj = ForecastingModels(df_read, forecast_days)
    trained_model = forecast_obj.train_xgb_model()
    joblib.dump(trained_model, "xgb_model.pkl")
    logger.info(f"XGB model trained and saved successfully")
except Exception as e:
    logger.error(f"Model training or saving failed because of {e}")

#SCENARIO 1: Loading the XGB Model on Monday to make predicitons
try:
    XGB_LOADED = joblib.load("xgb_model.pkl")
    logger.info(f"XGB model successfully loaded")
    # Use the trained model to make future predictions
    forecast_df = forecast_obj.forecast_xgb_model(XGB_LOADED)
    logger.info(f"Forecasting for next {forecast_days} days completed")
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], format="%d-%m-%Y")
    forecast_df['Timestamp'] = datetime.now()
    df_read['Timestamp'] = datetime.now()
    df_read['Date'] = pd.to_datetime(df_read['Date'], format="%d-%m-%Y")
    write_data_db(df_read, "ACD_VOLUME_TRAIN","append")
    write_data_db(forecast_df, "ACD_VOLUME_FORECAST","append")
    logger.info(f"FORECAST Data is pushed into DB")
    logger.info(f"TRAIN Data is pushed into DB")


except Exception as e:
    logger.error(f"Prediction failed: {e}")

#SCENARIO 2: Next mondays run load the model, get the actual data retrained and forecast the model

try:
    #compare actuals vs forecast (as we are in next weeek)
    actual_query = query = f"""
        WITH cte1 AS (
        SELECT * FROM ACD_VOLUME_FORECAST 
        WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_FORECAST) 
        LIMIT 7
        ),
        cte2 AS (
            SELECT Date, "Call Volume" FROM ACD_VOLUME  -- Keep only required columns
        )
        SELECT a.Date,a.Predicted_Call_Volume, b."Call Volume"
        FROM cte1 a
        JOIN cte2 b
        ON a.Date = b.Date
    """
    df_actual = read_data_db(actual_query)
    plot_line_chart(df_actual,x='Date',y='Call Volume',df1=df_actual,x1='Date',x2='Predicted_Call_Volume',label1="Call Volume", label2="Predicted_Call_Volume")
except Exception as e:
    logger.error(f"plot failed: {e}")

    #retrain data
try:
    # XGB_LOADED = joblib.load("xgb_model.pkl")
    logger.info(f"XGB model successfully loaded")
    query = f"""
        SELECT * FROM ACD_VOLUME_TRAIN 
        WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN) 
    """
    df_train = read_data_db(query)
    df_train = df_train[['Date', 'Call Volume', 'U.S. Holiday Indicator', 'Call Volume Impact', 'Day of Week', 'Day of Month', 'Day of Year', 'Week of Year', 'Month', 'Quarter', 'Is Weekend']]
    df_train['Date'] = pd.to_datetime(df_train['Date'])

    df_actual_retrain_query =  f"""
       WITH cte1 AS (
        SELECT * FROM ACD_VOLUME_FORECAST 
        WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_FORECAST) 
        LIMIT 7
        ),
        cte2 AS (
            SELECT * FROM ACD_VOLUME 
        )
        SELECT b.*
        FROM cte1 a
        JOIN cte2 b
        ON a.Date = b.Date
    """
    df_actual_retrain = read_data_db(df_actual_retrain_query)
    df_retrain = pd.concat([df_train, df_actual_retrain], ignore_index=True)
    df_retrain['Date'] = pd.to_datetime(df_retrain['Date'])
    forecast_obj = ForecastingModels(df_retrain, forecast_days)
    retrained_model = forecast_obj.train_xgb_model()
    # Save the retrained model
    joblib.dump(retrained_model, "xgb_model_retrained.pkl")
    logger.info(f"XGB model retrained and saved successfully")

    # Forecast for the next `forecast_days`
    forecast_df = forecast_obj.forecast_xgb_model(retrained_model)
    logger.info(f"Forecasting for next {forecast_days} days completed")
    
    # Add timestamp
    forecast_df['Timestamp'] = datetime.now()
    
    # Store the forecast results into DB
    write_data_db(forecast_df, "ACD_VOLUME_FORECAST","append")
    logger.info(f"Updated FORECAST Data pushed into DB")
    write_data_db(df_retrain, "ACD_VOLUME_TRAIN","append")
    logger.info(f"Updated Train data Data pushed into DB")
except Exception as e:
    logger.error(f"retrain failed: {e}")



