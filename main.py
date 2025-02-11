import pandas as pd
from utils import write_data_db, read_data_db, get_logger,plot_line_chart,plot_weekday_call_volume_distribution
import yaml
from models import ForecastingModels  
import joblib
from datetime import datetime
import warnings
from sklearn.metrics import mean_absolute_percentage_error
from llm import llm_call
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
        trained_model = forecast_obj.train_xgb_model() #This will train our XGB model till NOv 3
        joblib.dump(trained_model, "xgb_model.pkl") #Dumping our model so that we can pull on monday and forecast it
        logger.info(f"XGB model trained and saved successfully")
    except Exception as e:
        logger.error(f"Model training or saving failed because of {e}")

    #SCENARIO 1: Loading the XGB Model on Monday to make predicitons for next 14 days
    try:
        XGB_LOADED = joblib.load("xgb_model.pkl") #pulling the model weights that we saved previously
        logger.info(f"XGB model successfully loaded")

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



def retrain_actuals(forecast_days):
    # #SCENARIO 2: On next monday we will 
        # Retrain our model by adding last 7 days of actuals
        # Predict for next 14 days 
        # compare actuals vs foreacated of last 7 days
        #save the retrained model
    try:
        query = f"""
            SELECT * FROM ACD_VOLUME_TRAIN 
            WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_TRAIN) 
        """
        df_train = read_data_db(query) #query to pull the training data, to append the next 7 days of actual data
        df_train = df_train[['Date', 'Call Volume', 'U.S. Holiday Indicator', 'Call Volume Impact', 'Day of Week', 'Day of Month', 'Day of Year', 'Week of Year', 'Month', 'Quarter', 'Is Weekend']] #getting all the columns other than timestamp column
        df_train['Date'] = pd.to_datetime(df_train['Date']) #converting to datetime
        
        df_actual_retrain_query =  f"""
        WITH cte1 AS (
            SELECT * FROM ACD_VOLUME_FORECAST 
            WHERE Timestamp = (SELECT MAX(Timestamp) FROM ACD_VOLUME_FORECAST) 
            LIMIT 7
            ),
            cte2 AS (
                SELECT * FROM ACD_VOLUME 
            )
            SELECT b.*,a.Predicted_Call_Volume
            FROM cte1 a
            JOIN cte2 b
            ON a.Date = b.Date
        """
        df_actual_retrain = read_data_db(df_actual_retrain_query) #ACD_VOLUME_FORECAST will have 14 days of forecast as we are 7 days into functure we will join it with actuals table to comapre our forecast with actuals
        max_date = df_actual_retrain['Date'].max() #getting max of date for logs
        min_date = df_actual_retrain['Date'].min() #getting min of date for logs
        logger.info(f"Actuals of  {min_date} to {max_date} are added and are being retrained")

        act_pred_df = df_actual_retrain[['Date','Day of Week','Call Volume','Predicted_Call_Volume']] #created a new subset df which helps in creating the line chart of actual vs pred of last 7 days
        act_pred_df_metric = act_pred_df.copy() #created a new df copy which helps in calculating MAPE of actual vs pred of last 7 days
        act_pred_df_metric = act_pred_df_metric.set_index('Date') #setting index to date 
        act_pred_df_metric_val = mean_absolute_percentage_error(act_pred_df_metric['Call Volume'],act_pred_df_metric['Predicted_Call_Volume']) * 100 #calcualting MAPE
        logger.info(f"MAPE for {min_date} to {max_date} is {act_pred_df_metric_val}") 
        act_pred_df_metric_df = pd.DataFrame() #creating an df to push mape values
        act_pred_df_metric_df['START_DATE'] = min_date #creating  start date col
        act_pred_df_metric_df['END_DATE'] = max_date #creating end datecol
        act_pred_df_metric_df['MAPE'] = act_pred_df_metric_val #MAPE value
        act_pred_df_metric_df['START_DATE'] = pd.to_datetime(act_pred_df_metric_df['START_DATE'], format="%d-%m-%Y") #converting to dateitme
        act_pred_df_metric_df['END_DATE'] = pd.to_datetime(act_pred_df_metric_df['END_DATE'], format="%d-%m-%Y") #converting to dateitme
        write_data_db(act_pred_df_metric_df, "ACD_VOLUME_MAPE","append") #function to write back data to db,replaces the table


        # plot_weekday_call_volume_distribution(act_pred_df,'Day of Week','Predicted_Call_Volume')
       

        plot_line_chart(act_pred_df,x='Date',y='Call Volume',df1=act_pred_df,x1='Date',x2='Predicted_Call_Volume',label1="Call Volume last 7 days", label2="Predicted_Call_Volume last 7 days") #line chart for comaprision
        response = llm_call(f"Hey can u tell me if anything has changed today that can affect my forecast and this is my forecast {act_pred_df}")
        logger.info(f"{response}")
        print(response)

        df_actual_retrain = df_actual_retrain.drop(columns=['Predicted_Call_Volume']) #droping the Predicted_Call_Volume as we dont we it for retraining approach
        df_retrain = pd.concat([df_train, df_actual_retrain], ignore_index=True) #adding last 7 days of data to the training data set
        df_retrain_viz = df_retrain.copy() #creating a copy df for creating the dist plot
        plot_weekday_call_volume_distribution(df_retrain_viz,'Day of Week','Call Volume') #function is from utils that generates distubution at DAY X CALL volume level
    
        df_retrain['Date'] = pd.to_datetime(df_retrain['Date']) #convert to datetime
        max_date_r = df_retrain['Date'].max() #getting max of date for logs
        min_date_r = df_retrain['Date'].min() #getting min of date for logs
        logger.info(f"New training is happening on {min_date_r} to {max_date_r}")
        forecast_obj = ForecastingModels(df_retrain, forecast_days) #creting obj of the model with the latest data
        retrained_model = forecast_obj.train_xgb_model() #call the function to train data on the latest data
        # Save the retrained model
        joblib.dump(retrained_model, "xgb_model_retrained.pkl") #dumping the mdeol into pickle file
        logger.info(f"XGB model retrained and saved successfully")

        # Forecast for the next `forecast_days`
        forecast_df = forecast_obj.forecast_xgb_model(retrained_model) #forecasting data for next 14 days ( we already retained model on last 7 days actuals so we will forecast for next 14 days)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']) #converting to datetime

        plot_line_chart(df_retrain,x='Date',y='Call Volume',label1="Call Volume Train",df1 = forecast_df,x1='Date',x2='Predicted_Call_Volume',label2="Call Volume Forecasted") #creating a line chart of traina nd pred
        max_date = forecast_df['Date'].max() #getting max of date for logs
        min_date = forecast_df['Date'].min() #getting max of date for logs

        logger.info(f"Forecasting for next {forecast_days} days completed and forecasted for {min_date} to {max_date} and forecast is {forecast_df}")
        
        
        # Add timestamp
        forecast_df['Timestamp'] = datetime.now() #adding timestamp
        
        # Store the forecast results into DB
        write_data_db(forecast_df, "ACD_VOLUME_FORECAST","append") #pushing  forecasting data to DB
        logger.info(f"Updated FORECAST Data pushed into DB from  {min_date} to {max_date}")
        write_data_db(df_retrain, "ACD_VOLUME_TRAIN","append") #pushing train data to DB
        logger.info(f"Updated Train data pushed into DB from  {min_date_r} to {max_date_r}")
    except Exception as e: 
        logger.error(f"retrain failed: {e}")
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
    total_data_push(train_date,forecast_days,lag_days) #this function needs to run one time (create XGB model and trains it and generates forecast for 14 days)
    # retrain_actuals(forecast_days) #This function needs to run in loop to simulates sub sequent weeks

    