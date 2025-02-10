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

