#creating APIs 
from langchain_google_genai import ChatGoogleGenerativeAI
import yaml


def llm_call(input):
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
    template_text = '''
    Role & Expertise:
    You are a highly experienced data scientist specializing in banking analytics and operational analytics. You have a deep understanding of how external world events impact the bank's call volume. Based on past forecasting, you can now identify the reasons behind fluctuations in call volume.

    Thinking Approach:

    Data-Driven Analysis: Compare the given forecasted call volume (provided as user input) with actual data and identify any deviations.
    External Factors Consideration: Analyze external macroeconomic factors such as interest rate changes (e.g., repo rate in the USA), economic conditions, regulatory updates, or any current events that may influence customer behavior and call volume.
    Task:

    Compare the given forecasted call volume with actual data.
    Identify and explain any differences (increase or decrease in volume).
    List all possible internal (operational) and external (market-driven) factors that could have contributed to these changes by searching the internet.
'''
    full_prompt = f"{template_text}\n{input}"

    response = llm.invoke(full_prompt)
    output = response.content

    return output

def llm_call2(input):
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
    template_text = '''
You are a highly experienced data scientist specializing in banking analytics, time series forecasting, and insight generation.
Given the actuals and forecast data for the past week, along with forecasted values for the next N weeks, conduct an in-depth analysis.
Additionally, I will provide the last 100 days of historical training data.  

Analyze the call distribution of training data,forecasted data across weekdays and get those numbers in the report
and then compare  values. Identify any patterns, deviations, or anomalies in weekday trends and highlight any unexpected variations.  

Identify any days where the call volume is significantly higher or lower than usual, excluding U.S. holidays and the two days following each holiday. 
Prepare a report highlighting these anomalies, including details such as the date, call volume, and the extent of deviation from the normal volume.  
Compose an email to the operations team inquiring if there were any issues on the identified dates that could explain the unusual call volume. 
The email should include key details such as the specific date, the call volume recorded, and how significantly it deviates from the normal trend.

Determine any key insights I might be missing, such as seasonality, sudden spikes, or trends that may impact future performance. 
Present your findings in a structured format rather than paragraphs, making it easy to interpret key insights, variance explanations, and actionable recommendations.
This report should be suitable for sharing with management to support data-driven decision-making.
'''
    full_prompt = f"{template_text}\n{input}"

    response = llm.invoke(full_prompt)
    output = response.content

    return output

