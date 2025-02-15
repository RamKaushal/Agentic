import pandas as pd
import numpy as np
from utils import write_data_db, read_data_db, get_logger,plot_line_chart,get_news
import yaml
from models import ForecastingModels  
import joblib
import time
from datetime import datetime
import warnings
from sklearn.metrics import mean_absolute_percentage_error
from llm import llm_call
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_percentage_error
warnings.filterwarnings("ignore")



def total_data_push(train_date,forecast_days,lag_days):
    # Write data into DB
    try:
        df = pd.read_csv(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\ACD call volume.csv") #reading entire data        
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y") #converting it to date month year format to push to sql lite db
        max_date = df['Date'].max() #getting date max to maintain log
        min_date = df['Date'].min() #getting date min to maintain log
        write_data_db(df, "ACD_VOLUME","replace") #function to write back data to db,replaces the table
        logger.info(f"Data is pushed into DB from {min_date} to {max_date}")
    except Exception as e:
        logger.error(f"Failed to push data into server because of {e}")

    # Read data from DB
    try:
        train_date = pd.to_datetime(train_date).strftime("%Y-%m-%d") #tain date param is read from config file and passed into function in main currently set to 14
        query = f"""
            SELECT * FROM ACD_VOLUME 
            WHERE DATE(strftime('%Y-%m-%d', Date)) <= DATE('{train_date}')
        """
        df_read = read_data_db(query) #above query run data till Nov 3 as we will train our base model till here
        df_read['Date'] = pd.to_datetime(df_read['Date']) #converting to datetime
        max_date = df_read['Date'].max() #getting max of date for logs
        min_date = df_read['Date'].min() #getting min of date for logs

        logger.info(f"Data is read into DF from {min_date} to {max_date}") 
    except Exception as e:
        logger.error(f"Failed to read data from DB because of {e}")



    #SCENARIO BASE: CREATING AN XGB MODEL AND SAVING ITS WEIGHTS, This model gets trained will Nov 3
    try:
        forecast_obj = ForecastingModels(df_read, forecast_days) #creating an object from the forecasting models class, passing the df_read which have data till nov3
        forecast_obj.train_xgb_model("BASE") #This will train our XGB model till NOv 3
        logger.info(f"XGB model trained and saved successfully")
    except Exception as e:
        logger.error(f"Model training or saving failed because of {e}")

    # #SCENARIO 1: Loading the XGB Model on Monday to make predicitons for next 14 days
    try:
        model_name = "Best_XGB_Model"
        model_version = "latest"

        # Load from MLflow registry
        XGB_LOADED = mlflow.xgboost.load_model(model_uri=f"models:/{model_name}/{model_version}")
        logger.info(f"XGB model successfully loaded from ML_FLOW")

        # Use the trained model to make future predictions
        #  #creating an object from the forecasting models class, passing the df_read which have data till nov3
        forecast_df = forecast_obj.forecast_xgb_model(XGB_LOADED) #passing the model to the class
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], format="%d-%m-%Y") #converting to datetime
        forecast_df = forecast_df.iloc[lag_days:]
        max_date = forecast_df['Date'].max() #getting max of date for logs
        min_date = forecast_df['Date'].min() #getting min of date for logs
        forecast_df['Timestamp'] = datetime.now() #creating the timestamp column this will help us if we rerun model on same day to pick the latest forecast
        df_read['Timestamp'] = datetime.now() #creating the timestamp column this will help us if we rerun model on same day to pick the latest forecast
        df_read['Date'] = pd.to_datetime(df_read['Date'], format="%d-%m-%Y") #converting to datetime
        max_date_r = df_read['Date'].max()  #getting max of date for logs
        min_date_r = df_read['Date'].min() #getting min of date for logs
        write_data_db(df_read, "ACD_VOLUME_TRAIN","append") #wring the train data back to DB
        write_data_db(forecast_df, "ACD_VOLUME_FORECAST","append") #writng the forecast data back to DB
        plot_line_chart(df_read,x='Date',y='Call Volume',label1="Call Volume Train",df1 = forecast_df,x1='Date',x2='Predicted_Call_Volume',label2="Call Volume Forecasted") #linechart of train and predicted
        logger.info(f"Forecasting for next {forecast_days-lag_days} days completed from {min_date} to {max_date}")
        logger.info(f"FORECAST Data is pushed into DB and forecast is {forecast_df}")
        logger.info(f"TRAIN Data is pushed into DB from {min_date_r} to {max_date_r}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
    return None
     
#SCenario 2: 1 week later re-train on last 7 days
def scenario2(forecast_days):
    try:   
        logger.info(f"SCENARIO 2 of retraining last 7 days started")
        query = f"""
            SELECT * FROM ACD_VOLUME_TRAIN 
            WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN) 
        """
        df_train = read_data_db(query) #query to pull the training data, to append the next 7 days of actual data
        df_train['Date'] = pd.to_datetime(df_train['Date']) #converting to datetime
        df_train = df_train[['Date', 'Call Volume', 'U.S. Holiday Indicator', 'Call Volume Impact', 'Day of Week', 'Day of Month', 'Day of Year', 'Week of Year', 'Month', 'Quarter', 'Is Weekend']] #getting all the columns other than timestamp column
        df_train['Date'] = pd.to_datetime(df_train['Date']) #converting to datetime
        df_actual_retrain_query =  f"""
            WITH cte1 AS (
                SELECT * 
                FROM ACD_VOLUME 
                WHERE Date <= (
                    SELECT DATE(MAX(Date), '+8 days')
                    FROM ACD_VOLUME_TRAIN 
                    WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN)
                )
            )
            SELECT * FROM cte1 order by Date Desc limit 7;
        """
        df_actual_retrain = read_data_db(df_actual_retrain_query) 
        df_actual_retrain['Date'] = pd.to_datetime(df_actual_retrain['Date'])
        df_actual_retrain = df_actual_retrain.sort_values(by=['Date']).reset_index(drop=True)
        max_date = df_actual_retrain['Date'].max() #getting max of date for logs
        min_date = df_actual_retrain['Date'].min() #getting min of date for logs
        logger.info(f"values of  {min_date} to {max_date} are added and are being retrained")
        df_retrain = pd.concat([df_train, df_actual_retrain], ignore_index=True)
        df_retrain['Date'] = pd.to_datetime(df_retrain['Date'], format="%d-%m-%Y") #converting to datetime
        ###########
        model_name = "Best_XGB_Model"  
        client = MlflowClient()
        latest_version = max([int(v.version) for v in client.get_latest_versions(model_name)])
        model_version_details = client.get_model_version(model_name, str(latest_version))
        run_id = model_version_details.run_id  
        run_details = client.get_run(run_id)
        logged_params = run_details.data.params 
        forecast_obj = ForecastingModels(df_retrain, forecast_days)
        XGB_TRAINED = forecast_obj.train_xgb_model_same_param(logged_params)
        forecast_df = forecast_obj.forecast_xgb_model(XGB_TRAINED) #passing the model to the class
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], format="%d-%m-%Y") #converting to datetime
        df_retrain['Date'] = pd.to_datetime(df_retrain['Date'], format="%d-%m-%Y") #converting to datetime
        forecast_df = forecast_df.iloc[lag_days:]
        max_date = forecast_df['Date'].max() #getting max of date for logs
        min_date = forecast_df['Date'].min() #getting min of date for logs
        min_date_r = df_retrain['Date'].min()
        max_date_r = df_retrain['Date'].max()
        df_retrain['Timestamp'] = datetime.now() 
        forecast_df['Timestamp'] = datetime.now() 
        try:
            write_data_db(df_retrain, "ACD_VOLUME_TRAIN","append") #wring the train data back to DB
            write_data_db(forecast_df, "ACD_VOLUME_FORECAST","append") #writng the forecast data back to DB
        except:
            print("error")
        plot_line_chart(df_retrain,x='Date',y='Call Volume',label1="Call Volume Train",df1 = forecast_df,x1='Date',x2='Predicted_Call_Volume',label2="Call Volume Forecasted") #linechart of train and predicted
        logger.info(f"Forecasting for next {forecast_days-lag_days} days completed from {min_date} to {max_date}")
        logger.info(f"FORECAST Data is pushed into DB and forecast is {forecast_df}")
        logger.info(f"TRAIN Data is pushed into DB from {min_date_r} to {max_date_r}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None




def retrain_actuals(forecast_days):
    try:
        mape_query  = '''
        SELECT * FROM ACD_VOLUME_MAPE order by Date Desc limit 4
        '''
        df_mape_pull = read_data_db(mape_query) 
        if (df_mape_pull.shape[0] > 4): 
            df_mape_pull['Flag'] = np.where(df_mape_pull['MAPE'] > 15, 1, 0)
            if df_mape_pull['Flag'].sum() ==4:
                df_latest_data_q = '''
                SELECT * FROM ACD_VOLUME_TRAIN WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN) order by DATE DESC limit 100
                '''
                df_latest_data_df = read_data_db(df_latest_data_q) 
                forecast_obj = ForecastingModels(df_latest_data_df, forecast_days) #creating an object from the forecasting models class, passing the df_read which have data till nov3
                forecast_obj.train_xgb_model("BASE") #This will train our XGB model till NOv 3
                logger.info(f"XGB model trained and saved successfully as MAPE value was above threshold")
    except:
        pass
    else:
        pass
    scenario2(forecast_days)
    actuals_query = '''
    with cte1 as(
    SELECT * FROM ACD_VOLUME_TRAIN 
    WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN) 
    order by Date Desc limit 14
    ),
    cte2 as (
    select * from cte1 order by Date limit 7
    ),
    cte3 as (
    SELECT a.* FROM ACD_VOLUME_FORECAST a 
    join cte2 b
    on a.Date = b.Date  
    )
    ,cte4 as (
    select a.Date,a.Predicted_Call_Volume,b."Call Volume"
    from cte3 a
    join cte2 b
    on a.Date = b.Date
    where a.Timestamp = (SELECT MAX(Timestamp) FROM cte3)
    )
    select * from cte4
    '''
    df_actual_retrain = read_data_db(actuals_query) 
    plot_line_chart(df_actual_retrain,x='Date',y='Call Volume',label1="Call Volume Train",df1 = df_actual_retrain,x1='Date',x2='Predicted_Call_Volume',label2="Call Volume Forecasted")
    df_actual_retrain_mape = df_actual_retrain.copy()
    actual_retrain_mape_val = mean_absolute_percentage_error(df_actual_retrain_mape['Call Volume'],df_actual_retrain_mape['Predicted_Call_Volume']) * 100
    df_actual_retrain_mape_val = pd.DataFrame()
    df_actual_retrain_mape_val['Date'] =  datetime.now() 
    df_actual_retrain_mape_val['MAPE'] =  actual_retrain_mape_val
    df_actual_retrain_mape_val['Date'] = pd.to_datetime(df_actual_retrain_mape_val['Date'], format="%d-%m-%Y") #converting to datetime
    write_data_db(df_actual_retrain_mape_val, "ACD_VOLUME_MAPE","append") #wring the train data back to DB
    logger.info(f"MAPE IS {actual_retrain_mape_val}")
    max_date = df_actual_retrain['Date'].max() #getting max of date for logs
    min_date = df_actual_retrain['Date'].min() #getting min of date for logs
    logger.info(f" Compared actual vs predicted of {min_date} to {max_date} is done")

    df_forecast_latest_q = '''
    SELECT * FROM ACD_VOLUME_FORECAST WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_FORECAST) 
    '''
    df_forecast_latest = read_data_db(df_forecast_latest_q) 

    df_actual_latest_q = '''
    SELECT * FROM ACD_VOLUME_TRAIN WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN) order by DATE DESC limit 100
    '''
    df_actual_latest = read_data_db(df_actual_latest_q) 
    llm_response_append = ''

    news_df = get_news()

    llm_input = f'''Analyze the provided news dataset {news_df} from the last 14 days of {df_actual_latest} and identify events or trends that could impact the 28-day call volume forecast {df_forecast_latest} for citi bank and make sure you use news of the date present in df_actual_latest only.
                '''
    response = llm_call(llm_input,"AGENT_NEWS")
    logger.info(f"--------------------------------------AGENT_NEWS-------------------------------------")
    logger.info(f"{response}")
    llm_response_append += "--------------------------------------AGENT_NEWS-------------------------------------"
    llm_response_append += response
    time.sleep(5)

    llm_input = f"Last 100 Days of Training Data: {df_actual_latest[['Date','Call Volume']]}, Actual and Forecasted Data for previous 7 Days: {df_actual_retrain}, Next 28 Days Forecast {df_forecast_latest} "
    response = llm_call(llm_input,"AGENT_INSIGHTS")
    logger.info(f"--------------------------------------AGENT_INSIGHTS-------------------------------------")
    logger.info(f"{response}")
    llm_response_append += "--------------------------------------AGENT_INSIGHTS-------------------------------------"
    llm_response_append += response
    time.sleep(5)



    llm_input = f"Last 100 Days of Training Data: {df_actual_latest}, Actual and Forecasted Data for previous 7 Days: {df_actual_retrain}, Next 28 Days Forecast {df_forecast_latest} Give week day analysis of each in table format"
    response = llm_call(llm_input,"AGENT_WEEKLY_ANALYSIS")
    logger.info(f"--------------------------------------AGENT_WEEKLY_ANALYSIS-------------------------------------")
    logger.info(f"{response}")
    llm_response_append += "--------------------------------------AGENT_WEEKLY_ANALYSIS-------------------------------------"
    llm_response_append += response
    time.sleep(5)

    llm_input = f"This is my actual  volume {df_actual_retrain} check for last 4 weeks and give date and call volume if its an anomaly not on an USA holiday along with its standard deviation,generate usa holidays by urself *Dont generate python code just give dates and anomlay*"
    response = llm_call(llm_input,"AGENT_ANOMALY")
    logger.info(f"--------------------------------------AGENT_ANOMALY-------------------------------------")
    logger.info(f"{response}")
    llm_response_append += "--------------------------------------AGENT_ANOMALY-------------------------------------"
    llm_response_append += response
    time.sleep(5)

    llm_input = f" This is the output generated by diff LLM agents can you make a good summarized report of it {llm_response_append}, Get news agents summary, Insights agent summary, Weekly analysis agent summary and dont forget to include day of week tables,agent anomaly as last and add anonamly dates and their deviation in table"
    response = llm_call(llm_input,"AGENT_REPORT")
    logger.info(f"--------------------------------------AGENT_SUMMARY-------------------------------------")
    logger.info(f"{response}")
    time.sleep(5)

    return None

if __name__ == "__main__":
    logger = get_logger()
    # Load config
    with open(r"C:\Users\ramka\Downloads\Agentic-main\Agentic\config.yaml", "r") as f: #opeining config file to pull params
        config = yaml.safe_load(f)

    lag_days = config['lag_days'] #No of days to leave before starting the forecsat
    forecast_days = config['forecast_days'] #gets forecast param from config file,change in config file if we need to increase forecast days
    forecast_days = forecast_days+lag_days
    train_date = config['train_date'] #gets train param from config file,change in train date is we want to train base model from a diff date
    logger.info(f"Forecast days are read and set to {forecast_days}")
    logger.info(f"Training data till {train_date}")
    # total_data_push(train_date,forecast_days,lag_days) #this function needs to run one time (create XGB model and trains it and generates forecast for 14 days)
    # scenario2(forecast_days)
    # scenario2(forecast_days)
    retrain_actuals(forecast_days) #This function needs to run in loop to simulates sub sequent weeks
    