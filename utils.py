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




def plot_line_chart(df, x, y, df1=None, x1=None, x2=None, 
                    label1="Dataset 1", label2="Dataset 2", 
                    title="Line Plot", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Plots a line chart with one or two datasets and custom labels.

    Parameters:
    df (DataFrame): First dataset (required)
    x (str): X-axis column for first dataset
    y (str): Y-axis column for first dataset
    df1 (DataFrame, optional): Second dataset (default: None)
    x1 (str, optional): X-axis column for second dataset (default: None)
    x2 (str, optional): Y-axis column for second dataset (default: None)
    label1 (str, optional): Label for first dataset (default: "Dataset 1")
    label2 (str, optional): Label for second dataset (default: "Dataset 2")
    title (str, optional): Chart title (default: "Line Plot")
    xlabel (str, optional): X-axis label (default: "X-Axis")
    ylabel (str, optional): Y-axis label (default: "Y-Axis")
    """
    plt.figure(figsize=(12, 6))

    # Plot first dataset
    sns.lineplot(data=df, x=x, y=y, label=label1, marker="o")

    # Plot second dataset only if provided
    if df1 is not None and x1 is not None and x2 is not None:
        sns.lineplot(data=df1, x=x1, y=x2, label=label2, marker="o", linestyle="-")

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    
    # Show grid for better readability
    plt.grid(True)

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_weekday_call_volume_distribution(df, day_column, volume_column):
    """
    Generates subplots for the distribution of call volumes for each day of the week.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    day_column (str): Column name representing the day of the week (1 for Monday, 7 for Sunday).
    volume_column (str): Column name representing the call volume.

    Returns:
    None (Displays the subplots)
    """
    # Map numeric days to names
    day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                   5: 'Friday', 6: 'Saturday', 0: 'Sunday'}
    df['day_name'] = df[day_column].map(day_mapping)

    # List of days in order
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot each day's distribution
    for i, day in enumerate(days):
        filtered_data = df[df['day_name'] == day]
        ax = axes[i]
        if not filtered_data.empty:
            sns.histplot(filtered_data[volume_column], kde=True, ax=ax)
        ax.set_title(f"Call Volume Distribution - {day}")

    # Remove empty subplot if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
