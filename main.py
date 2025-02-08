import pandas as pd
from utils import write_data_db,read_data_db,get_logger
import yaml 
from models import ForecastingModels
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


logger = get_logger()

with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml","r") as f:
    config = yaml.safe_load(f)

forecast_days = config['forecast_days']
train_date = config['train_date']

# Write data into server
try:
    df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv")
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
    write_data_db(df,"ACD_VOLUME")
    logger.info(f"Data is pushed into DB")
except Exception as e:
    logger.error(f"Failed to pushed data into server becasue of {e}")


#read data into server
try:
    train_date = pd.to_datetime(train_date).strftime("%Y-%m-%d")  # Ensure correct format
    query = f"""
        SELECT * FROM ACD_VOLUME 
        WHERE DATE(strftime('%Y-%m-%d', Date)) <= DATE('{train_date}')
    """
    df_read = read_data_db(query)

    # Ensure Date is in datetime format in Pandas
    df_read['Date'] = pd.to_datetime(df_read['Date'], format="%Y-%m-%d")  # Change format


    logger.info("Data is read into DF FROM DB")
except Exception as e:
    logger.error(f"Failed to read data from DB because of {e}")



logger.info(f"Forecast days are read and set to {forecast_days}")
logger.info(f"Training data till {train_date}")


#SCENARIO BASE: CREATE A MODEL TRAIN TILL NOV 3 and SAVE Weights

forecast_obj =  ForecastingModels(df_read,forecast_days)
try:
    XGB_PRED = forecast_obj.forecast_acd_call_volume()
    XGB_PRED['Date'] = pd.to_datetime(XGB_PRED['Date'], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
    XGB_PRED['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_data_db(XGB_PRED,"ACD_VOLUME_PREDICTION")
    logger.info(f"Prediciton data is pushed into DB")
    logger.info(f"XGB predcition is {XGB_PRED}")
except Exception as e:
    logger.error(f"Model building failed becasue of {e}")


try:
    model_file = "xgb_model.pkl"
    joblib.dump(XGB_PRED,model_file)
    logger.info(f"XGB model saved to {model_file}")
except Exception as e:
    logger.error(f"Model saving failed becasue of {e}")


# #Sceanrio 1: Load the model after 7 days(Monday) and retrain on last 7 days and predict next 14 days
# try:
#     model_file = "xgb_model.pkl"
#     XGB_LOADED = joblib.load(model_file)
#     logger.info(f"XGB model loaded")
# except Exception as e:
#     logger.error(f"Model loading failed becasue of {e}")




