import pandas as pd
from utils import write_data_db,read_data_db
import yaml 


# df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv")
# write_data_db(df,"ACD_VOLUME")

query = "SELECT * FROM ACD_VOLUME"
read_df = read_data_db(query)

with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml","r") as f:
    config = yaml.safe_load(f)
    

forecast_days = config['forecast_days']
print(forecast_days)

