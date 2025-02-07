import pandas as pd
import sqlite3

def write_data_db(df,table_name):
    conn = sqlite3.connect("calls.db")
    df.to_sql(table_name, conn, if_exists="append", index=False)
    return None 

def read_data_db(query):
    query = query
    conn = sqlite3.connect("calls.db")
    df_from_db = pd.read_sql(query, conn)
    return df_from_db


