#creating APIs 
from langchain_google_genai import ChatGoogleGenerativeAI
import yaml



def llm_call(input,AGENT):
    with open(r"/content/Agentic/config.yaml", "r") as f: #opeining config file to pull params
        config = yaml.safe_load(f)

    lang_chain = config['lang_chain'] 
    gemini = config['gemini'] 
    langchain_project = config['langchain_project'] 

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=gemini  
    )


    AGENT_INSIGHTS = '''
Context:
The forecast represents call volume for a bank, predicting the number of customer calls received. The goal is to analyze training data, forecast data and last 1 week forecast and actual data to find insights 

Data Provided:
Last 100 Days of Actual Call Volume Data → df_actual_latest
Previous 7 Days: Actual vs. Forecasted Call Volume → df_actual_retrain
Next 28 Days Forecast of Call Volume → df_forecast_latest

Tasks:
1. Data Preparation:
Compute the following metrics for each dataset:
Mean (Average Call Volume)
Median (Middle Value of Call Volume Distribution)
Standard Deviation (Std) (Variability of Call Volume)
Min/Max (Lowest & Highest Call Volume)
Interquartile Range (IQR) (Spread of middle 50 of data)

Once these 3 are done then find if there is any reason on why my forecast is off, take training and forecast data and tell me the reasons
*DONT PRINT PYTHON CODE IN OUTPUT*

    '''

    AGENT_WEEKLY_ANALYSIS = '''
Objective:
Analyze the distribution of call volumes across weekdays using the following datasets:

Last 100 Days of Actual Call Volume (df_actual_latest)
Previous 7 Days: Actual vs. Forecasted Call Volume (df_actual_retrain)
Next 28 Days Forecasted Call Volume (df_forecast_latest)
The goal is to evaluate whether the forecasted call volume follows historical weekday distribution patterns, identify deviations, and provide insights for improving forecast accuracy.

Tasks:
1. Data Preparation:
Extract the weekday from the date column in all datasets.
Aggregate call volume by weekday for each dataset.
Compute the following metrics for each weekday:
Mean (Average Call Volume)
Median (Middle Value of Call Volume Distribution)
Standard Deviation (Std) (Variability of Call Volume)
Min/Max (Lowest & Highest Call Volume)
Interquartile Range (IQR) (Spread of middle 50 of data)

compare all 3 agaist each and tell if my forecast is off on any day and what might be the reason for that,create us holiday list and check if there is any holiday of USA in forecasted period that might has increased or decread forecast volumne
*DONT PRINT PYTHON CODE IN OUTPUT*
'''    
    AGENT_ANOMALY = '''
You are a highly experienced data scientist specializing in banking analytics, time series forecasting, Identify any days where the call volume 
is significantly higher or lower than usual, excluding U.S. holidays and the two days following each holiday. 
Given my training data Prepare a report highlighting these anomalies, including details such as the date, call volume,
 and the extent of deviation from the normal volume, *DOnt generate python code just give dates and anomlay*
 *DONT PRINT PYTHON CODE IN OUTPUT*
'''

    AGENT_NEWS = '''
Analyze the provided news data and identify any news that could impact the call volume forecast for a bank like Citi. 
*DONT PRINT PYTHON CODE IN OUTPUT*
'''

    AGENT_REPORT = '''
You are a highly experienced data analyst specializing in banking analytics, report summary generation
Determine any key insights I might be missing, such as seasonality, sudden spikes, or trends that may impact future performance. 
Present your findings in a structured format rather than paragraphs, making it easy to interpret key insights, variance explanations, and actionable recommendations.
This report should be suitable for sharing with management to support data-driven decision-making.
*DONT PRINT PYTHON CODE IN OUTPUT*
'''

    full_prompt = f"{AGENT}\n{input}"

    response = llm.invoke(full_prompt)
    output = response.content

    return output

