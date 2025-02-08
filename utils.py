import pandas as pd
import sqlite3
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def write_data_db(df,table_name,type):
    conn = sqlite3.connect("calls.db")
    df.to_sql(table_name, conn, if_exists=type, index=False)
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



import matplotlib.pyplot as plt
import seaborn as sns

def plot_line_chart(df, x, y, df1, x1, x2, 
                    label1="Dataset 1", label2="Dataset 2", 
                    title="Line Plot", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Plots a line chart with two datasets and custom labels for each dataset.
    
    Parameters:
    df (DataFrame): First dataset
    x (str): X-axis column for first dataset
    y (str): Y-axis column for first dataset
    df1 (DataFrame): Second dataset
    x1 (str): X-axis column for second dataset
    x2 (str): Y-axis column for second dataset
    label1 (str): Label for first dataset
    label2 (str): Label for second dataset
    title (str): Chart title
    xlabel (str): X-axis label
    ylabel (str): Y-axis label
    """
    plt.figure(figsize=(12, 6))

    # Plot the first line with a custom label
    sns.lineplot(data=df, x=x, y=y, label=label1, marker="o")

    # Plot the second line with a custom label
    sns.lineplot(data=df1, x=x1, y=x2, label=label2, marker="s")

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    
    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()







