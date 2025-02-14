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
        model="gemini-2.0-flash",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=gemini  
    )


    AGENT_INSIGHTS = '''
Context:
The forecast represents call volume for a bank, predicting the number of customer calls received. The goal is to analyze forecast accuracy and determine if any manual adjustments are needed for upcoming forecasts

Data Provided:
Last 100 Days of Actual Call Volume Data → df_actual_latest
Previous 7 Days: Actual vs. Forecasted Call Volume → df_actual_retrain
Next 28 Days Forecast of Call Volume → df_forecast_latest
Tasks:

Trend & Seasonality Analysis (100 Days):
Identify long-term trends in call volume.
Detect seasonality, recurring patterns, and any anomalies.
Check for external factors (e.g., weekends, holidays, or news events) affecting call volume.

Forecast Accuracy Analysis (Last 7 Days):
Compare actual vs. forecasted call volume and compute error metrics (MAPE, RMSE, Bias).
Identify consistent underestimation or overestimation trends.

Upcoming 28-Day Forecast Validation:
Compare forecasted call volume against historical trends.
Detect any sudden shifts or unrealistic variations.
Check if forecasted trends align with past patterns.
Forecast Anomaly Detection

Analyze discrepancies across all datasets (actuals, past forecasts, and upcoming forecasts).
Identify recurring errors (e.g., consistent over/under-forecasting on specific days).
U.S. Holiday Impact Analysis

Generate a list of upcoming U.S. holidays.
Assess historical impact on call volume for similar holidays.
Determine if manual adjustments are needed for upcoming forecasts based on past deviations.
Expected Output:
Key insights and trends in historical call volume.
Accuracy assessment of previous forecasts.
Potential forecast risks and anomalies.
Recommendations for manual forecast adjustments (if necessary).

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
2. Compare & Validate:
Compare the forecasted weekday distribution (df_forecast_latest) against:
Last 100 days of actuals (df_actual_latest)
Recent 7 days of actual vs. forecast (df_actual_retrain)
Check whether the forecast aligns with historical weekday patterns.
Identify significant deviations, where the forecast is consistently higher or lower than expected.
3. Insights & Interpretation:
Evaluate whether weekday seasonality is captured (e.g., higher call volumes on Mondays, lower on Sundays).
Analyze any recent trends in actual data (df_actual_retrain) that might be impacting forecast accuracy.
Identify possible external factors (e.g., holidays, promotions, system outages) that may explain deviations.
4. Validate & Recommend:
If the forecast does not align with historical weekday patterns, diagnose potential reasons such as:
Model underfitting or overfitting
Lack of seasonality adjustments
Influence of external shocks (e.g., events, news, policy changes)
Recommend improvements such as:
Incorporating exogenous variables
Enhancing seasonality detection
Adjusting retraining frequency
Final Output (Data Analyst Report):
The report should include:

Historical Weekday Call Distribution (Last 100 Days)

Summary statistics for each weekday
Key trends in call volume variation
Forecasted Weekday Call Distribution (Next 28 Days)

Comparison with historical patterns
Any unexpected spikes or drops
Deviation Analysis & Root Cause Assessment

How much the forecast deviates from actual trends
Possible reasons for deviations
Recommendations for Forecast Improvement

Suggested adjustments to improve forecast accuracy
Actions to align forecasts with real-world patterns
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

