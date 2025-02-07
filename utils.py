import pandas as pd
import sqlite3
import logging

def write_data_db(df,table_name):
    conn = sqlite3.connect("calls.db")
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return None 

def read_data_db(query):
    query = query
    conn = sqlite3.connect("calls.db")
    df_from_db = pd.read_sql(query, conn)
    return df_from_db

import logging

def get_logger():
    logger = logging.getLogger("AGENTIC_PROJECT")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler("logfile.txt", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger





