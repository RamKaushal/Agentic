import pandas as pd
from utils import write_data_db,read_data_db,get_logger
import yaml 
from models import ForecastingModels
import joblib

logger = get_logger()

# Write data into server
try:
    df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv")
    write_data_db(df,"ACD_VOLUME")
    logger.info(f"Data is pushed into DB")
except Exception as e:
    logger.error(f"Failed to pushed data into server becasue of {e}")


#read data into server
try:
    query = "SELECT * FROM ACD_VOLUME"
    df = read_data_db(query)
    df['Date'] = pd.to_datetime(df['Date'],format="%d-%m-%Y")
    logger.info(f"Data is read into DF FROM  DB")
except Exception as e:
    logger.error(f"Failed to read data from DB becasue of {e}")

with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml","r") as f:
    config = yaml.safe_load(f)
    

forecast_days = config['forecast_days']
train_date = config['train_date']

logger.info(f"Forecast days are read and set to {forecast_days}")
logger.info(f"Training data till {train_date}")


#SCENARIO BASE: CREATE A MODEL TRAIN TILL NOV 3 and SAVE Weights
train_df = df[df['Date']<= train_date ]
test_df  = df[df['Date']> train_date]


forecast_obj =  ForecastingModels(train_df,test_df,forecast_days)
try:
    XGB_PRED = forecast_obj.forecast_acd_call_volume()
    logger.info(f"XGB predcition is {XGB_PRED}")
except Exception as e:
    logger.error(f"Model building failed becasue of {e}")


try:
    model_file = "xgb_model.pkl"
    joblib.dump(XGB_PRED,model_file)
    logger.info(f"XGB model saved to {model_file}")
except Exception as e:
    logger.error(f"Model saving failed becasue of {e}")


#Sceanrio 1: Load the model and 

