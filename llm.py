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
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=gemini  
    )

    response = llm.invoke(input)
    output = response.content

    return output

