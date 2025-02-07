import pandas as pd

from utils import write_data_db,read_data_db

df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv")
write_data_db(df,"ACD_VOLUME")

query = "SELECT * FROM ACD_VOLUME"
read_df = read_data_db(query)

print(read_df.head())