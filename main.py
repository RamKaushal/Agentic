import pandas as pd
from utils import write_data_db,read_data_db,get_logger
import yaml 
from models import Forecasting_models

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
print(forecast_days)

#SCENARIO BASE: CREATE A MODEL TRAIN TILL NOV 3 and SAVE Weights

train_date = '2024-11-03'
test_date = '2024-11-04'
train_df = df[df['Date']<= train_date ]
test_df  = df[df['Date']> train_date]


forecast_obj =  Forecasting_models(train_df,test_df,forecast_days)
XGB_PRED = forecast_obj.forecast_acd_call_volume(test_date)

