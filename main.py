import pandas as pd
from utils import write_data_db,read_data_db,get_logger
import yaml 

logger = get_logger()

#Write data into server
try:
    df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv")
    write_data_db(df,"ACD_VOLUME")
    logger.info(f"Data is pushed into DB")
except Exception as e:
    logger.error(f"Failed to pushed data into server becasue of {e}")


#read data into server
try:
    query = "SELECT * FROM ACD_VOLUME"
    read_df = read_data_db(query)
    logger.info(f"Data is read into DF FROM  DB")
except Exception as e:
    logger.error(f"Failed to read data from DB becasue of {e}")

with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml","r") as f:
    config = yaml.safe_load(f)
    

forecast_days = config['forecast_days']
logger.info(f"Forecast days are read and set to {forecast_days}")
print(forecast_days)
