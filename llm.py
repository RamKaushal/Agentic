#creating APIs 
from langchain_google_genai import ChatGoogleGenerativeAI
import yaml



def llm_call(input,AGENT):
    with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml", "r") as f: #opeining config file to pull params
        config = yaml.safe_load(f)

    lang_chain = config['lang_chain'] 
    gemini = config['gemini'] 
    langchain_project = config['langchain_project'] 

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-preview-02-05",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=gemini  
    )


    AGENT_INSIGHTS = '''
    Role & Expertise:
    You are a highly experienced data scientist specializing in banking analytics, time series forecasting, and insight generation.
    Given the actuals and forecast data for the past week, along with forecasted values for the next N weeks, conduct an in-depth analysis.
    Additionally, I will provide the last 100 days of historical training data.  
    '''

    AGENT_WEEKLY_ANALYSIS = '''
You are a highly experienced data scientist specializing in banking analytics, time series forecasting,
Analyze the call distribution of training data,forecasted data across weekdays and get those numbers in the report
and then compare  values. Identify any patterns, deviations, or anomalies in weekday trends and highlight any unexpected variations.
'''    
    AGENT_ANOMALY = '''
You are a highly experienced data scientist specializing in banking analytics, time series forecasting, Identify any days where the call volume 
is significantly higher or lower than usual, excluding U.S. holidays and the two days following each holiday. 
Given my training data Prepare a report highlighting these anomalies, including details such as the date, call volume,
 and the extent of deviation from the normal volume, *DOnt generate python code just give dates and anomlay*
'''

    AGENT_NEWS = '''
Analyze the provided news data and identify any news that could impact the call volume forecast for a bank like Citi. 
b  Exclude news that has no impact. For relevant news, determine whether the impact is positive or negative.
'''

    AGENT_REPORT = '''
You are a highly experienced data analyst specializing in banking analytics, report summary generation
Determine any key insights I might be missing, such as seasonality, sudden spikes, or trends that may impact future performance. 
Present your findings in a structured format rather than paragraphs, making it easy to interpret key insights, variance explanations, and actionable recommendations.
This report should be suitable for sharing with management to support data-driven decision-making.
'''

    full_prompt = f"{AGENT}\n{input}"

    response = llm.invoke(full_prompt)
    output = response.content

    return output

