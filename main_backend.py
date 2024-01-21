from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()


class DeleteEntryRequest(BaseModel): #common pydantic model for all features
    date: str
#-----------------------------------------------------------------------------------------------------------------------------------------#
# Define models for different categories, this is for PE signals 
class PEEntry(BaseModel):
    date: str
    pe_value: float
    # Fields for PE entry
# Multipliers for each financial year
multipliers = {
    "2021-2022": 1.19,
    "2022-2023": 1.20,
    "2023-2024": 1.19,
    # Add more years as needed
}
def determine_financial_year(date):
    year = date.year
    if date.month < 4:
        return f"{year-1}-{year}"
    else:
        return f"{year}-{year+1}"

def apply_multiplier(pe_value, date):
    financial_year = determine_financial_year(date)
    return pe_value * multipliers.get(financial_year, 1)  # Default to 1 if no multiplier

def add_pe_signal_logic(data):
    import pandas as pd
    new_df=data
    # taking 252 days in an year as average
    new_df['SMA_5_Years']=new_df['P/E'].rolling(window=1260).mean()
    new_df["sma_7_5_years"] = new_df['P/E'].rolling(window=1890).mean()
    new_df['sma_10_years'] = new_df['P/E'].rolling(window=2520).mean()
    final_df=new_df.copy()
    # Calculate the moving averages
    sma_5_years = final_df['P/E'].rolling(1260).mean()  # 5 years = 1260 days

    # Calculate standard deviations corresponding to the 5-year SMA positive direction
    std_5_years=final_df['P/E'].rolling(1260).std()  # Standard deviation
    final_df['1.5sd_5SMA_POS'] = sma_5_years + 1.5 * std_5_years  # 1.5 standard deviations above SMA
    final_df['2sd_5SMA_POS'] = sma_5_years + 2 * std_5_years  # 2 standard deviations above SMA
    final_df['3sd_5SMA_POS']= sma_5_years + 3 * std_5_years  # 3 standard deviations above SMA

    # Calculate standard deviations corresponding to the 5-year SMA negative direction

    final_df['1.5sd_5SMA_NEG'] = sma_5_years - 1.5 * std_5_years  # 1.5 standard deviations above SMA
    final_df['2sd_5SMA_NEG'] = sma_5_years - 2 * std_5_years  # 2 standard deviations above SMA
    final_df['3sd_5SMA_NEG']= sma_5_years - 3 * std_5_years  # 3 standard deviations above SMA
    # Calculate the moving averages 7.5 yrs
    sma_7_5_years = final_df['P/E'].rolling(1890).mean()  # 7.5 years = 1890 days

    # Calculate standard deviations corresponding to the 7.5-year SMA positive direction
    std_7_5_years=final_df['P/E'].rolling(1890).std()  # Standard deviation
    final_df['1.5sd_7_5_SMA_POS'] = sma_7_5_years + 1.5 * std_7_5_years  # 1.5 standard deviations above SMA
    final_df['2sd_7_5SMA_POS'] = sma_7_5_years + 2 * std_7_5_years  # 2 standard deviations above SMA
    final_df['3sd_7_5SMA_POS']= sma_7_5_years + 3 * std_7_5_years  # 3 standard deviations above SMA

    # Calculate standard deviations corresponding to the 7.5-year SMA negative direction

    final_df['1.5sd_7_5SMA_NEG'] = sma_7_5_years - 1.5 * std_7_5_years  # 1.5 standard deviations above SMA
    final_df['2sd_7_5SMA_NEG'] = sma_7_5_years - 2 * std_7_5_years  # 2 standard deviations above SMA
    final_df['3sd_7_5SMA_NEG']= sma_7_5_years - 3 * std_7_5_years  # 3 standard deviations above SMA

    # Calculate the moving averages
    sma_10_years = final_df['P/E'].rolling(2520).mean()  # 10 years = 2520 days

    # Calculate standard deviations corresponding to the 5-year SMA positive direction
    std_10_years=final_df['P/E'].rolling(2520).std()  # Standard deviation
    final_df['1.5sd_10_SMA_POS'] = sma_10_years + 1.5 * std_10_years  # 1.5 standard deviations above SMA
    final_df['2sd_10SMA_POS'] = sma_10_years + 2 * std_10_years  # 2 standard deviations above SMA
    final_df['3sd_10SMA_POS']= sma_10_years + 3 * std_10_years  # 3 standard deviations above SMA

    # Calculate standard deviations corresponding to the 5-year SMA negative direction

    final_df['1.5sd_10SMA_NEG'] = sma_10_years - 1.5 * std_10_years  # 1.5 standard deviations above SMA
    final_df['2sd_10SMA_NEG'] = sma_10_years - 2 * std_10_years  # 2 standard deviations above SMA
    final_df['3sd_10SMA_NEG']= sma_10_years - 3 * std_10_years  # 3 standard deviations above SMA

    plot_df=final_df.copy()

    # signal generation for 5sma


    import numpy as np

    # Initializing an empty 'signals' column with NaNs for plot_df
    plot_df['signals'] = np.nan

    # Looping through the rows from the second to the end (since we'll be looking back one row each time)
    for i in range(1, len(plot_df)):
    
        # Check if SMA value is NaN
        if pd.isna(plot_df.loc[i, 'SMA_5_Years']):
            plot_df.loc[i, 'signals'] = np.nan
            continue
    
        # Sell Signals
        if plot_df.loc[i-1, 'P/E'] > plot_df.loc[i-1, '1.5sd_5SMA_POS'] and plot_df.loc[i, 'P/E'] < plot_df.loc[i, '1.5sd_5SMA_POS']:
            plot_df.loc[i, 'signals'] = '5S1'
        elif plot_df.loc[i-1, 'P/E'] > plot_df.loc[i-1, '2sd_5SMA_POS'] and plot_df.loc[i, 'P/E'] < plot_df.loc[i, '2sd_5SMA_POS']:
            plot_df.loc[i, 'signals'] = '5S2'
        elif plot_df.loc[i-1, 'P/E'] > plot_df.loc[i-1, '3sd_5SMA_POS'] and plot_df.loc[i, 'P/E'] < plot_df.loc[i, '3sd_5SMA_POS']:
            plot_df.loc[i, 'signals'] = '5S3'
    
        # Buy Signals
        elif plot_df.loc[i-1, 'P/E'] < plot_df.loc[i-1, '1.5sd_5SMA_NEG'] and plot_df.loc[i, 'P/E'] > plot_df.loc[i, '1.5sd_5SMA_NEG']:
            plot_df.loc[i, 'signals'] = '5B1'
        elif plot_df.loc[i-1, 'P/E'] < plot_df.loc[i-1, '2sd_5SMA_NEG'] and plot_df.loc[i, 'P/E'] > plot_df.loc[i, '2sd_5SMA_NEG']:
            plot_df.loc[i, 'signals'] = '5B2'
        elif plot_df.loc[i-1, 'P/E'] < plot_df.loc[i-1, '3sd_5SMA_NEG'] and plot_df.loc[i, 'P/E'] > plot_df.loc[i, '3sd_5SMA_NEG']:
            plot_df.loc[i, 'signals'] = '5B3'
        else:
            plot_df.loc[i, 'signals'] = 0

    df5 = plot_df.copy()


    # signal generation for 7.5
    df7=plot_df.copy()
    import pandas as pd
    import numpy as np

    # Initializing an empty 'signals' column with NaNs for plot_df
    df7['signals'] = np.nan

    # Looping through the rows from the second to the end (since we'll be looking back one row each time)
    for i in range(1, len(df7)):
        
        # Check if SMA value is NaN
        if pd.isna(df7.loc[i, 'sma_7_5_years']):
            df7.loc[i, 'signals'] = np.nan
            continue
        
        # Sell Signals
        if df7.loc[i-1, 'P/E'] > df7.loc[i-1, '1.5sd_7_5_SMA_POS'] and df7.loc[i, 'P/E'] < df7.loc[i, '1.5sd_7_5_SMA_POS']:
            df7.loc[i, 'signals'] = '7.5S1'
        elif df7.loc[i-1, 'P/E'] > df7.loc[i-1, '2sd_7_5SMA_POS'] and df7.loc[i, 'P/E'] < df7.loc[i, '2sd_7_5SMA_POS']:
            df7.loc[i, 'signals'] = '7.5S2'
        elif df7.loc[i-1, 'P/E'] > df7.loc[i-1, '3sd_7_5SMA_POS'] and df7.loc[i, 'P/E'] < df7.loc[i, '3sd_7_5SMA_POS']:
            df7.loc[i, 'signals'] = '7.5S3'
        
        # Buy Signals
        elif df7.loc[i-1, 'P/E'] < df7.loc[i-1, '1.5sd_7_5SMA_NEG'] and df7.loc[i, 'P/E'] > df7.loc[i, '1.5sd_7_5SMA_NEG']:
            df7.loc[i, 'signals'] = '7.5B1'
        elif df7.loc[i-1, 'P/E'] < df7.loc[i-1, '2sd_7_5SMA_NEG'] and df7.loc[i, 'P/E'] > df7.loc[i, '2sd_7_5SMA_NEG']:
            df7.loc[i, 'signals'] = '7.5B2'
        elif df7.loc[i-1, 'P/E'] < df7.loc[i-1, '3sd_7_5SMA_NEG'] and df7.loc[i, 'P/E'] > df7.loc[i, '3sd_7_5SMA_NEG']:
            df7.loc[i, 'signals'] = '7.5B3'
        else:
            df7.loc[i, 'signals'] = 0


    # Signal generation for 10 years

    df10=df7.copy()

    import pandas as pd
    import numpy as np

    # Initializing an empty 'signals' column with NaNs for plot_df
    df10['signals'] = np.nan

    # Looping through the rows from the second to the end (since we'll be looking back one row each time)
    for i in range(1, len(df10)):
        
        # Check if SMA value is NaN
        if pd.isna(df10.loc[i, 'sma_10_years']):
            df10.loc[i, 'signals'] = np.nan
            continue
        
        # Sell Signals
        if df10.loc[i-1, 'P/E'] > df10.loc[i-1, '1.5sd_10_SMA_POS'] and df10.loc[i, 'P/E'] < df10.loc[i, '1.5sd_10_SMA_POS']:
            df10.loc[i, 'signals'] = '10S1'
        elif df10.loc[i-1, 'P/E'] > df10.loc[i-1, '2sd_10SMA_POS'] and df10.loc[i, 'P/E'] < df10.loc[i, '2sd_10SMA_POS']:
            df10.loc[i, 'signals'] = '10S2'
        elif df10.loc[i-1, 'P/E'] > df10.loc[i-1, '3sd_10SMA_POS'] and df10.loc[i, 'P/E'] < df10.loc[i, '3sd_10SMA_POS']:
            df10.loc[i, 'signals'] = '10S3'
        
        # Buy Signals
        elif df10.loc[i-1, 'P/E'] < df10.loc[i-1, '1.5sd_10SMA_NEG'] and df10.loc[i, 'P/E'] > df10.loc[i, '1.5sd10SMA_NEG']:
            df10.loc[i, 'signals'] = '10B1'
        elif df10.loc[i-1, 'P/E'] < df10.loc[i-1, '2sd_10SMA_NEG'] and df10.loc[i, 'P/E'] > df10.loc[i, '2sd_10SMA_NEG']:
            df10.loc[i, 'signals'] = '10B2'
        elif df10.loc[i-1, 'P/E'] < df10.loc[i-1, '3sd_10SMA_NEG'] and df10.loc[i, 'P/E'] > df10.loc[i, '3sd_10SMA_NEG']:
            df10.loc[i, 'signals'] = '10B3'
        else:
            df10.loc[i, 'signals'] = 0


    
    
    return df5, df7 ,df10

@app.post("/pe/add_entry/")
async def add_entry(entry: PEEntry):
    # Path to your CSV file
    FILE_PATH = 'data_pe.csv'
    try:
        # Read the existing data, ensuring date parsing
        data = pd.read_csv(FILE_PATH, parse_dates=['Date'])

        # Convert the entry date to a datetime object
        entry_date = pd.to_datetime(entry.date)

        # Apply the multiplier to the PE value
        adjusted_pe_value = apply_multiplier(entry.pe_value, entry_date)

        # Create a new row and append it to the dataframe
        new_row = pd.DataFrame({'Date': [entry_date], 'P/E': [adjusted_pe_value]})
        data = pd.concat([data, new_row], ignore_index=True)

        # Sort the data by date
        data.sort_values(by='Date', inplace=True)

        # Reset the index after sorting
        data.reset_index(drop=True, inplace=True)

        # Apply the signal logic
        df5, df7, df10 = add_pe_signal_logic(data)

        # Save the updated dataframes
        data.to_csv(FILE_PATH, index=False)
        df5.to_csv('df5.csv', index=False)
        df7.to_csv('df7.csv', index=False)
        df10.to_csv('df10.csv', index=False)

        return {"message": "Entry added successfully and data processed!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Other endpoints remain the same

@app.post("/pe/delete_entry/")
async def delete_entry(request: DeleteEntryRequest):
    # Path to your CSV file
    FILE_PATH = 'data_pe.csv'
    try:
        # Load the main dataset and the three signal datasets
        main_data = pd.read_csv(FILE_PATH, parse_dates=['Date'])
        df5 = pd.read_csv('df5.csv', parse_dates=['Date'])
        df7 = pd.read_csv('df7.csv', parse_dates=['Date'])
        df10 = pd.read_csv('df10.csv', parse_dates=['Date'])

        # Convert the input date string to a datetime object
        target_date = pd.to_datetime(request.date)

        # Remove the entry from all datasets
        main_data = main_data[main_data['Date'] != target_date]
        df5 = df5[df5['Date'] != target_date]
        df7 = df7[df7['Date'] != target_date]
        df10 = df10[df10['Date'] != target_date]

        # Reapply the signal logic on the main dataset
        # Assuming add_pe_signal_logic returns updated df5, df7, df10
        df5, df7, df10 = add_pe_signal_logic(main_data)

        # Save the updated datasets
        main_data.to_csv(FILE_PATH, index=False, date_format='%Y-%m-%d')
        df5.to_csv('df5.csv', index=False, date_format='%Y-%m-%d')
        df7.to_csv('df7.csv', index=False, date_format='%Y-%m-%d')
        df10.to_csv('df10.csv', index=False, date_format='%Y-%m-%d')

        return {"message": "Entry deleted successfully and datasets updated!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints for PE Signals
# Endpoint to fetch last 5 entries for each signal dataframe
@app.get("/pe/last_entries/{signal_type}")
async def last_entries(signal_type: str):
    # Path to your CSV file
    FILE_PATH = 'data_pe.csv'
    try:
        file_map = {'5SMA': 'df5.csv', '7.5SMA': 'df7.csv', '10SMA': 'df10.csv'}
        if signal_type not in file_map:
            raise HTTPException(status_code=404, detail="Invalid signal type")

        # Load the appropriate DataFrame
        df = pd.read_csv(file_map[signal_type])

        # Select only the 'Date', 'P/E', and 'Signals' columns
        df = df[['Date', 'P/E', 'signals']]

        # Get the last 5 entries
        last_entries = df.tail(5).to_dict(orient='records')
        return last_entries
    except Exception as e:
        return {"error": str(e)}


@app.get("/pe/full_data/{signal_type}")
async def full_data(signal_type: str):
    # Path to your CSV file
    FILE_PATH = 'data_pe.csv'
    try:
        file_map = {'5SMA': 'df5.csv', '7.5SMA': 'df7.csv', '10SMA': 'df10.csv'}

        if signal_type not in file_map:
            raise HTTPException(status_code=404, detail=f"Signal type {signal_type} not found")

        # Load the appropriate CSV file based on signal_type
        file_path = file_map[signal_type]
        data = pd.read_csv(file_path)

        # Convert the data to a suitable format for JSON response
        # Here, converting it to a list of dictionaries
        response_data = data.to_dict(orient='records')
        return response_data
    except Exception as e:
        return {"error": str(e)}

#-----------------------------------------------------------------------------------------------------------------------------------------#
#Momentum
# Define models for different categories
class MomentumEntry(BaseModel):
    # Fields for Market Momentum entry
    pass

# Add more models as needed for other features




# Endpoints for Market Momentum
@app.post("/momentum/add_entry/")
async def add_entry(date: str, close_price: float):
    #FILE_PATH = 'current_momentum.csv'
    # Load data
    data = pd.read_csv("current_momentum.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    new_row = pd.DataFrame({'Date': [pd.to_datetime(date)], 'Close': [close_price]})
    data = pd.concat([data, new_row]).reset_index(drop=True)

    # Compute momentum logic
    data['3EMA_Close'] = data['Close'].ewm(span=66, adjust=False).mean()
    data['6EMA_Close'] = data['Close'].ewm(span=132, adjust=False).mean()
    data['9EMA_Close'] = data['Close'].ewm(span=198, adjust=False).mean()

    #data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype('Int64')
    prev_momentum = data['EMA_Actual Momentum'].iloc[-2] if data.shape[0] > 1 else None

    close_value = data['Close'].iloc[-1]
    ema_values = data.iloc[-1][['3EMA_Close', '6EMA_Close', '9EMA_Close']]

    if ema_values.isna().any():
        new_momentum = None
    elif all(close_value > ema_values):
        new_momentum = 1
    elif all(close_value < ema_values):
        new_momentum = -1
    else:
        new_momentum = 0

    if prev_momentum is not None:
        if new_momentum != prev_momentum:
            buffer = 0.99 if new_momentum < prev_momentum else 1.01
            new_ema_values = ema_values * buffer

            if all(close_value > new_ema_values):
                new_momentum = 1
            elif all(close_value < new_ema_values):
                new_momentum = -1
            else:
                new_momentum = 0

    data.at[data.index[-1], 'EMA_Actual Momentum'] = int(new_momentum) if new_momentum is not None else None

    # Save updated data back to CSV
    columns_to_round = ['Close', '3EMA_Close', '6EMA_Close', '9EMA_Close']
    for col in columns_to_round:
        data[col] = data[col].round(2)

    data.to_csv('current_momentum.csv', index=False)
    return {"message": "Added successfully!"}

@app.post("/momentum/delete_entry/")
async def delete_entry(date: str):
    #FILE_PATH = 'current_momentum.csv'
    # Load data
    data = pd.read_csv("current_momentum.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Delete logic
    if pd.to_datetime(date) in data['Date'].values:
        data = data[data['Date'] != pd.to_datetime(date)]
        data.to_csv('current_momentum.csv', index=False, float_format='%.2f')
        return {"message": f"Entry for {date} has been deleted and the dataset has been updated."}
    else:
        raise HTTPException(status_code=404, detail=f"No entry found for the date {date}.")

@app.get("/momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 'current_momentum.csv'
    data = pd.read_csv("current_momentum.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/nifty/contratrend/last_entries/")
async def last_entries_contratrend():
    #FILE_PATH = ''
    
    # Format the columns to remove trailing zeros

    #for data till 2023
    import calendar
    import pandas as pd
    from datetime import datetime,timedelta
    import calendar
    import pandas as pd

    def last_thursday(year, month):
        # Find the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        # Starting from the last day, find the last Thursday
        for day in range(last_day, 0, -1):
            if calendar.weekday(year, month, day) == calendar.THURSDAY:
                return year, month, day

    start_year = 2004
    end_year = 2024

    dates = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 12:  # Stop after December 2023
                break
            date = last_thursday(year, month)
            dates.append(f"{date[2]}-{date[1]:02}-{date[0]}")

    df_date = pd.DataFrame(dates, columns=['Last_Thursday'])

    df_momentum=pd.read_csv('current_momentum.csv')
    def reformat_date(date_string):
        date_obj = datetime.strptime(date_string, '%d-%m-%Y')
        return date_obj.strftime('%Y-%m-%d')

    # Assuming df['Last_Thursday'] contains the dates in "dd-mm-yyyy" format
    # Convert 'Last_Thursday' to string if it's in datetime format
    df_date['Last_Thursday'] = df_date['Last_Thursday'].astype(str)

    # Apply the reformat_date function
    #df_date['Last_Thursday'] = df_date['Last_Thursday'].apply(reformat_date)


    # This function checks and returns the actual expiry date
    def get_actual_expiry(theoretical_date):
        date_obj = datetime.strptime(theoretical_date, '%Y-%m-%d')
        while True:
            if date_obj.strftime('%Y-%m-%d') in df_momentum['Date'].values:
                return date_obj.strftime('%Y-%m-%d')
            date_obj -= timedelta(days=1)

    # Generate actual expiry dates
    df_date['Last_Thursday'] = pd.to_datetime(df_date['Last_Thursday'], dayfirst=True)
    thursdays_df=df_date[df_date["Last_Thursday"]>'2023-12-31']
    df_date['Last_Thursday'] = df_date['Last_Thursday'].dt.strftime('%Y-%m-%d')

    df_date['Actual_Expiry'] = df_date['Last_Thursday'].apply(get_actual_expiry)
    df_date['Last_Thursday'] = pd.to_datetime(df_date['Last_Thursday'], dayfirst=True)
    df_date1 = df_date[df_date["Last_Thursday"]<'2024-01-01'].copy()
    #df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])

    thursdays_df = thursdays_df.rename(columns={"Last_Thursday": 'Date'})

    #for data from 2024
    df_date2=df_date[df_date["Last_Thursday"]<'2024-01-01']
    main_df=df_momentum[df_momentum['Date']>'2023-12-31']
    holiday_df=pd.read_excel('nifty_2024_holidays.xlsx')
    import pandas as pd


    import pandas as pd

    import pandas as pd

    def calculate_actual_expiry(thursdays_df, holiday_df):
        # Convert the holiday dates to datetime for comparison
        holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], format='%d-%b-%y')
        thursdays_df.loc[:, 'Date'] = pd.to_datetime(thursdays_df['Date'])
        thursdays_df.loc[:, 'Actual_Expiry'] = pd.NaT

        # Create a new column for Actual_Expiry
        thursdays_df['Actual_Expiry'] = pd.NaT

        for index, row in thursdays_df.iterrows():
            expiry = row['Date']

            # Check if the Thursday is a holiday
            while expiry in holiday_df['Date'].values:
                # Move to the previous day
                expiry -= pd.Timedelta(days=1)

            # Assign the nearest non-holiday weekday as the Actual_Expiry
            thursdays_df.at[index, 'Actual_Expiry'] = expiry

        return thursdays_df

    # Example usage:
    # thursdays_df = pd.read_csv('last_thursdays.csv')
    # holiday_df = pd.read_csv('holidays.csv')

    # Update thursdays_df with the Actual_Expiry column
    updated_thursdays_df = calculate_actual_expiry(thursdays_df, holiday_df)

    df_date2=updated_thursdays_df.copy()
    df_date1.rename(columns={"Last_Thursday":'Date'},inplace=True)
    df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])
    df_date2['Date'] = pd.to_datetime(df_date2['Date'])
    df_date2['Actual_Expiry'] = pd.to_datetime(df_date2['Actual_Expiry'])

    combined_df = pd.concat([df_date1, df_date2], ignore_index=True)


    #Making Expiry window for dates till 


    closing_prices_df=pd.read_csv('current_momentum.csv')
    closing_prices_df.drop(["3EMA_Close","6EMA_Close","9EMA_Close","EMA_Actual Momentum"],inplace=True,axis=1)
    expiry_dates_df = combined_df.copy()

    # Convert date columns to datetime objects
    closing_prices_df['Date'] = pd.to_datetime(closing_prices_df['Date'])
    expiry_dates_df['Actual_Expiry'] = pd.to_datetime(expiry_dates_df['Actual_Expiry'])

    # Sort the expiry dates in case they are not in order
    expiry_dates_df = expiry_dates_df.sort_values(by='Actual_Expiry')

    # Identify the start date of the first window
    start_date = closing_prices_df['Date'].iloc[0]

    # Find the first expiry date after the start date
    first_expiry_date = expiry_dates_df[expiry_dates_df['Actual_Expiry'] >= start_date].iloc[0]['Actual_Expiry']

    # Function to find the expiry window for a given date
    def find_expiry_window(date, expiry_dates):
        # Special case for the first window
        if date < first_expiry_date:
            return start_date.strftime('%Y-%m-%d'), first_expiry_date.strftime('%Y-%m-%d')
        
        for i in range(len(expiry_dates) - 1):
            if date > expiry_dates[i] and date <= expiry_dates[i + 1]:
                start_window = (expiry_dates[i] + timedelta(days=1)).strftime('%Y-%m-%d')
                end_window = expiry_dates[i + 1].strftime('%Y-%m-%d')
                return start_window, end_window
        return None

    # Create a list of expiry dates
    expiry_dates = expiry_dates_df['Actual_Expiry'].tolist()

    # Map each date in the closing price dataset to its expiry window
    closing_prices_df['ExpiryWindow'] = closing_prices_df['Date'].apply(lambda date: find_expiry_window(date, expiry_dates))

    # Now your closing_prices_df has a new column 'ExpiryWindow'

    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].astype(str)

    # Removing parentheses and quotes from the 'ExpiryWindow' column
    # Removing parentheses and single quotes from the 'ExpiryWindow' column
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].str.replace("[()']", "", regex=True)


    import pandas as pd
    import numpy as np

    # Read the dataset
    df = closing_prices_df.copy()

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['ExpiryWindow'] = df['ExpiryWindow'].astype(str).str.replace("[()']", "", regex=True)

    # Split 'ExpiryWindow' into 'StartExpiry' and 'EndExpiry' and convert to datetime
    split_columns = df['ExpiryWindow'].str.split(', ', expand=True)
    # Convert to datetime, coerce errors to NaT
    df['StartExpiry'] = pd.to_datetime(split_columns[0], errors='coerce')
    df['EndExpiry'] = pd.to_datetime(split_columns[1], errors='coerce')

    # Calculate the rolling 6-month low for 'Close', considering 21 trading days in a month
    rolling_window = 6 * 21
    df['Normal_6m_low'] = df['Close'].rolling(window=rolling_window, min_periods=rolling_window).min()

    # Initialize 'Current_6m_low' with the values of 'Normal_6m_low'
    df['Current_6m_low'] = df['Normal_6m_low']

    # Initialize the 'Signals' column
    df['Signals'] = np.nan

    # Define a helper function to find the next expiry window
    def find_next_expiry(date, expiry_windows):
        for start, end in expiry_windows:
            if pd.notnull(start) and start > date:
                return start, end
        return None, None

    # Prepare a list of unique expiry windows
    unique_expiry_windows = df[['StartExpiry', 'EndExpiry']].drop_duplicates().values

    # Initialize the 'last_signal_date'
    last_signal_date = None

    # Iterate through the DataFrame to generate signals
    for index, row in df.iterrows():
        # Check if a new expiry window has started
        if last_signal_date is None or row['Date'] >= row['EndExpiry']:
            signal_allowed = True
        
        # If we are within an expiry window and signals are allowed, check for the signal condition
        if signal_allowed and (last_signal_date is None or (row['Date'] - last_signal_date).days >= 7):
            if row['Close'] <= 1.01 * row['Current_6m_low']:
                # Mark the signal
                df.at[index, 'Signals'] = '6L'
                signal_allowed = False  # Disallow further signals in this window
                last_signal_date = row['Date']
                
                # Find the end of the next expiry window and update 'Current_6m_low' accordingly
                _, next_window_end = find_next_expiry(row['Date'], unique_expiry_windows)
                if next_window_end:
                    df.loc[(df['Date'] >= row['Date']) & (df['Date'] <= next_window_end), 'Current_6m_low'] = row['Close']

        # At the end of the expiry window, if no new signal has been generated, reset 'Current_6m_low' to 'Normal_6m_low'
        if row['Date'] == row['EndExpiry'] and pd.isnull(df.at[index, 'Signals']):
            next_window_start = df.loc[index + 1, 'StartExpiry'] if index + 1 < len(df) else None
            if next_window_start:
                df.loc[(df['Date'] >= next_window_start), 'Current_6m_low'] = df.loc[(df['Date'] >= next_window_start), 'Normal_6m_low']

    # Display the first few rows of the dataframe with signals
    print(df)
    print(df["Signals"].value_counts())
    df.to_csv("nifty_contra_signals.csv",index=False)
















    out = df[["Date", 'Close', 'Current_6m_low', "Signals"]]
    out=out.copy()
    out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string
    new = out.fillna(0)
    new.rename(columns={"Current_6m_low":"Contratrend_Value"},inplace=True)
    last_rows = new.tail(5).to_dict(orient='records')
    print(last_rows)
    return last_rows

#-----------------------------------------------------------------------------------------------------------------------------------------#
#Volatility code backend for nifty50,nifty200,nifty500







class VolatilityEntry(BaseModel):
    date: str
    price: float
    index: str  # 'nifty50', 'nifty200', or 'nifty500'

def calculate_volatility(returns_series, window_size):
    return returns_series.rolling(window=window_size).std()

def calculate_mean_volatility(df):
    df['Group'] = df.index // 5
    df['mean_Volatility'] = df.groupby('Group')['volatility_ratio'].transform(lambda x: x.iloc[-1])
    return df

def update_dataframe(file_path, new_date, new_nifty_price):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    last_price = df.iloc[-1][df.columns[1]]
    daily_return = (new_nifty_price - last_price) / last_price
    new_data = {'date': pd.to_datetime(new_date), df.columns[1]: new_nifty_price, 'daily_return': daily_return}
    df = df.append(new_data, ignore_index=True)
    df['volatility_21_days'] = calculate_volatility(df['daily_return'], 21)
    df['volatility_252_days'] = calculate_volatility(df['daily_return'], 252)
    df['volatility_ratio'] = df['volatility_21_days'] / df['volatility_252_days']
    df = calculate_mean_volatility(df)
    df.to_csv(file_path, index=False)
    return df

@app.post("/volatility/add_entry/")
async def add_volatility_entry(entry: VolatilityEntry):
    file_path = f"{entry.index}_volatility.csv"
    try:
        update_dataframe(file_path, entry.date, entry.price)
        return {"message": f"Entry added successfully for {entry.index.upper()}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DeleteEntryRequest(BaseModel):  #seperate pydantic model for deletion, if we use a common model it will show error, the model expects all the field listed when there is a request
    date: str
    index: str

@app.post("/volatility/delete_entry/")
async def delete_volatility_entry(entry: DeleteEntryRequest):
    file_path = f"{entry.index}_volatility.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if pd.to_datetime(entry.date) not in df['date'].values:
            return {"message": f"No entry found for the given date in {entry.index.upper()}."}

        df = df[df['date'] != pd.to_datetime(entry.date)]
        df.to_csv(file_path, index=False)
        return {"message": f"Entry deleted successfully for {entry.index.upper()}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/volatility/last_entries/{dataset}")
async def get_last_volatility_entries(dataset: str):
    file_path = f"{dataset}_volatility.csv"
    try:
        df = pd.read_csv(file_path)
        df.sort_values(by='date', ascending=False, inplace=True)
        last_entries = df.tail().to_dict(orient='records')
        return last_entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#-----------------------------------------------------------------------------------------------------------------------------------------#

#S&P Momentum
# Define models for different categories
class MomentumEntry(BaseModel):
    # Fields for Market Momentum entry
    pass

# Add more models as needed for other features




# Endpoints for Market Momentum
@app.post("/snp/momentum/add_entry/")
async def add_entry(date: str, close_price: float):
    #FILE_PATH = 's&p_data.csv'
    # Load data
    data = pd.read_csv("golden_dragon.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    new_row = pd.DataFrame({'Date': [pd.to_datetime(date)], 'Close': [close_price]})
    data = pd.concat([data, new_row]).reset_index(drop=True)

    # Compute momentum logic
    data['3EMA_Close'] = data['Close'].ewm(span=66, adjust=False).mean()
    data['6EMA_Close'] = data['Close'].ewm(span=132, adjust=False).mean()
    data['9EMA_Close'] = data['Close'].ewm(span=198, adjust=False).mean()

    data['EMA_Actual Momentum'] = np.nan
    prev_momentum = np.nan
    for index, row in data.iterrows():
        close_value = row['Close']
        ema_values = row[['3EMA_Close', '6EMA_Close', '9EMA_Close']]

        if ema_values.isna().any():
            new_momentum = np.nan
        elif all(close_value > ema_values):
            new_momentum = 1
        elif all(close_value < ema_values):
            new_momentum = -1
        else:
            new_momentum = 0

        if not np.isnan(prev_momentum):
            if new_momentum != prev_momentum:
                buffer = 0.99 if new_momentum < prev_momentum else 1.01
                new_ema_values = ema_values * buffer

                if all(close_value > new_ema_values):
                    new_momentum = 1
                elif all(close_value < new_ema_values):
                    new_momentum = -1
                else:
                    new_momentum = 0

        data.at[index, 'EMA_Actual Momentum'] = int(new_momentum)
        prev_momentum = new_momentum

    data.to_csv('golden_dragon.csv', index=False)
    return {"message": "Added successfully!"}

@app.post("/snp/momentum/delete_entry/")
async def delete_entry(date: str):
    #FILE_PATH = 's&p_data.csv'
    # Load data
    data = pd.read_csv("golden_dragon.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Delete logic
    if pd.to_datetime(date) in data['Date'].values:
        data = data[data['Date'] != pd.to_datetime(date)]
        data.to_csv('golden_dragon.csv', index=False, float_format='%.2f')
        return {"message": f"Entry for {date} has been deleted and the dataset has been updated."}
    else:
        raise HTTPException(status_code=404, detail=f"No entry found for the date {date}.")

@app.get("/snp/momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 's&p_data.csv'
    data = pd.read_csv("golden_dragon.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/gold_momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 'gold_data.csv'
    data = pd.read_csv("gold_data.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/snp/contratrend/last_entries/")
async def last_entries_contratrend():
        #for data till 2023
    import calendar
    import pandas as pd
    from datetime import datetime,timedelta
    import calendar
    import pandas as pd

    
    import calendar
    import pandas as pd
    from datetime import datetime, timedelta

    def last_friday_of_month(year, month):
        # Find the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_day_date = datetime(year, month, last_day)

        # If the last day is not Friday, adjust to the previous Friday
        if last_day_date.weekday() != 4:  # 4 represents Friday
            last_day_date -= timedelta(days=(last_day_date.weekday() - 4) % 7)

        return last_day_date.strftime('%Y-%m-%d')

    start_year = 2012
    end_year = 2024
    dates = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 12:  # Stop after December 2023
                break
            date = last_friday_of_month(year, month)
            dates.append(date)

    df = pd.DataFrame(dates, columns=['Last_Day'])
    df_date=df.copy()

    df_close=pd.read_csv('s&p_data.csv')
    def reformat_date(date_string):
        date_obj = datetime.strptime(date_string, '%d-%m-%Y')
        return date_obj.strftime('%Y-%m-%d')

    # Assuming df['Last_Thursday'] contains the dates in "dd-mm-yyyy" format
    #df_date['Last_Day'] = df_date['Last_Day'].apply(reformat_date)
    # Convert 'Last_Thursday' to string if it's in datetime format
    df_date['Last_Day'] = df_date['Last_Day'].astype(str)

    # Apply the reformat_date function
    #df_date['Last_Thursday'] = df_date['Last_Thursday'].apply(reformat_date)


    # This function checks and returns the actual expiry date
    from datetime import datetime, timedelta
    import pandas as pd

# Assuming df and df_close are already loaded and their date columns are in datetime format

# Function to find the actual expiry date
    from datetime import datetime, timedelta

    def get_actual_expiry(expiry_date_str):
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
        max_attempts = 10  # Set a limit to prevent infinite looping
        attempts = 0
        while expiry_date.strftime('%Y-%m-%d') not in df_close['Date'].values:
            expiry_date -= timedelta(days=1)  # Go back one day
            attempts += 1
            if attempts >= max_attempts:
                return None  # Or handle the missing date as appropriate
        return expiry_date.strftime('%Y-%m-%d')

    # Apply the function to the 'Last_Day' column
    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)

    # Generate actual expiry dates
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    lastday_df=df_date[df_date["Last_Day"]>'2023-12-31']
    df_date['Last_Day'] = df_date['Last_Day'].dt.strftime('%Y-%m-%d')

    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    df_date1 = df_date[df_date["Last_Day"]<'2024-01-01'].copy()
    #df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])

    lastday_df = lastday_df.rename(columns={"Last_Day": 'Date'})

    #for data from 2024
    df_date2=df_date[df_date["Last_Day"]<'2024-01-01']
    main_df=df_close[df_close['Date']>'2023-12-31']
    holiday_df=pd.read_excel('gold_2024_holidays.xlsx')
    import pandas as pd


    import pandas as pd

    import pandas as pd

    def calculate_actual_expiry(lastday_df, holiday_df):
        # Convert the holiday dates to datetime for comparison
        holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%d-%b-%y')
        lastday_df.loc[:, 'Date'] = pd.to_datetime(lastday_df['Date'])
        lastday_df.loc[:, 'Actual_Expiry'] = pd.NaT

        # Create a new column for Actual_Expiry
        lastday_df['Actual_Expiry'] = pd.NaT

        for index, row in lastday_df.iterrows():
            expiry = row['Date']

            # Check if the Thursday is a holiday
            while expiry in holiday_df['date'].values:
                # Move to the previous day
                expiry -= pd.Timedelta(days=1)

            # Assign the nearest non-holiday weekday as the Actual_Expiry
            lastday_df.at[index, 'Actual_Expiry'] = expiry

        return lastday_df

    # Example usage:
    # thursdays_df = pd.read_csv('last_thursdays.csv')
    # holiday_df = pd.read_csv('holidays.csv')

    # Update thursdays_df with the Actual_Expiry column
    updated_thursdays_df = calculate_actual_expiry(lastday_df, holiday_df)

    df_date2=updated_thursdays_df.copy()
    df_date1.rename(columns={"Last_Day":'Date'},inplace=True)
    df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])
    df_date2['Date'] = pd.to_datetime(df_date2['Date'])
    df_date2['Actual_Expiry'] = pd.to_datetime(df_date2['Actual_Expiry'])

    combined_df = pd.concat([df_date1, df_date2], ignore_index=True)


    #Making Expiry window for dates till 


    import pandas as pd
    from datetime import datetime, timedelta

    # Load your data
    closing_prices_df = pd.read_csv('s&p_data.csv')
    closing_prices_df.drop(["3EMA_Close", "6EMA_Close", "9EMA_Close", "EMA_Actual Momentum"], inplace=True, axis=1)
    expiry_dates_df = combined_df.copy()  # Assuming 'df' is defined and contains the correct expiry dates

    # Convert date columns to datetime objects
    closing_prices_df['Date'] = pd.to_datetime(closing_prices_df['Date'])
    expiry_dates_df['Actual_Expiry'] = pd.to_datetime(expiry_dates_df['Actual_Expiry'])

    # Sort the expiry dates in case they are not in order
    expiry_dates_df = expiry_dates_df.sort_values(by='Actual_Expiry')

    def find_expiry_window(date, expiry_dates, start_date):
        # Special case for the first window
        if date < expiry_dates[0]:
            return start_date.strftime('%Y-%m-%d'), (expiry_dates[0] - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for i in range(1, len(expiry_dates)):
            # The start of the window is the expiry of the previous month
            # The end of the window is one day before the expiry of the current month
            if date < expiry_dates[i]:
                start_window = expiry_dates[i - 1].strftime('%Y-%m-%d')
                end_window = (expiry_dates[i] - timedelta(days=1)).strftime('%Y-%m-%d')
                return start_window, end_window

        # For dates after the last expiry date, the window starts with the last expiry date
        return expiry_dates[-1].strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')

    # Create a list of expiry dates
    expiry_dates = expiry_dates_df['Actual_Expiry'].tolist()

    # Identify the start date of the dataset for the first window
    start_date = closing_prices_df['Date'].min()

    # Map each date in the closing price dataset to its expiry window
    closing_prices_df['ExpiryWindow'] = closing_prices_df['Date'].apply(lambda date: find_expiry_window(date, expiry_dates, start_date))

    # Cleaning 'ExpiryWindow' column
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].astype(str)
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].str.replace("[()']", "", regex=True)



    import pandas as pd
    import numpy as np

    # Read the dataset
    df = closing_prices_df.copy()

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['ExpiryWindow'] = df['ExpiryWindow'].astype(str).str.replace("[()']", "", regex=True)

    # Split 'ExpiryWindow' into 'StartExpiry' and 'EndExpiry' and convert to datetime
    split_columns = df['ExpiryWindow'].str.split(', ', expand=True)
    # Convert to datetime, coerce errors to NaT
    df['StartExpiry'] = pd.to_datetime(split_columns[0], errors='coerce')
    df['EndExpiry'] = pd.to_datetime(split_columns[1], errors='coerce')

    # Calculate the rolling 6-month low for 'Close', considering 21 trading days in a month
    rolling_window = 6 * 21
    df['Normal_6m_low'] = df['Close'].rolling(window=rolling_window, min_periods=rolling_window).min()

    # Initialize 'Current_6m_low' with the values of 'Normal_6m_low'
    df['Current_6m_low'] = df['Normal_6m_low']

    # Initialize the 'Signals' column
    df['Signals'] = np.nan

    # Define a helper function to find the next expiry window
    def find_next_expiry(date, expiry_windows):
        for start, end in expiry_windows:
            if pd.notnull(start) and start > date:
                return start, end
        return None, None

    # Prepare a list of unique expiry windows
    unique_expiry_windows = df[['StartExpiry', 'EndExpiry']].drop_duplicates().values

    # Initialize the 'last_signal_date'
    last_signal_date = None

    # Iterate through the DataFrame to generate signals
    for index, row in df.iterrows():
        # Check if a new expiry window has started
        if last_signal_date is None or row['Date'] >= row['EndExpiry']:
            signal_allowed = True
        
        # If we are within an expiry window and signals are allowed, check for the signal condition
        if signal_allowed and (last_signal_date is None or (row['Date'] - last_signal_date).days >= 7):
            if row['Close'] <= 1.01 * row['Current_6m_low']:
                # Mark the signal
                df.at[index, 'Signals'] = '6L'
                signal_allowed = False  # Disallow further signals in this window
                last_signal_date = row['Date']
                
                # Find the end of the next expiry window and update 'Current_6m_low' accordingly
                _, next_window_end = find_next_expiry(row['Date'], unique_expiry_windows)
                if next_window_end:
                    df.loc[(df['Date'] >= row['Date']) & (df['Date'] <= next_window_end), 'Current_6m_low'] = row['Close']

        # At the end of the expiry window, if no new signal has been generated, reset 'Current_6m_low' to 'Normal_6m_low'
        if row['Date'] == row['EndExpiry'] and pd.isnull(df.at[index, 'Signals']):
            next_window_start = df.loc[index + 1, 'StartExpiry'] if index + 1 < len(df) else None
            if next_window_start:
                df.loc[(df['Date'] >= next_window_start), 'Current_6m_low'] = df.loc[(df['Date'] >= next_window_start), 'Normal_6m_low']

    # Display the first few rows of the dataframe with signals
    print(df)
    print(df["Signals"].value_counts())
















    out = df[["Date", 'Close', 'Current_6m_low', "Signals"]]
    out=out.copy()
    out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string
    new = out.fillna(0)
    new.rename(columns={"Current_6m_low":"Contratrend_Value"},inplace=True)
    last_rows = new.tail(5).to_dict(orient='records')
    print(last_rows)
    return last_rows

#-----------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------#

#Gold Momentum
# Define models for different categories
class MomentumEntry(BaseModel):
    # Fields for Market Momentum entry
    pass

# Add more models as needed for other features




# Endpoints for Market Momentum
@app.post("/gold_momentum/add_entry/")
async def add_entry(date: str, close_price: float):
    #FILE_PATH = 'gold_data.csv'
    # Load data
    data = pd.read_csv("gold_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    new_row = pd.DataFrame({'Date': [pd.to_datetime(date)], 'Close': [close_price]})
    data = pd.concat([data, new_row]).reset_index(drop=True)

    # Compute momentum logic
    data['3EMA_Close'] = data['Close'].ewm(span=66, adjust=False).mean()
    data['6EMA_Close'] = data['Close'].ewm(span=132, adjust=False).mean()
    data['9EMA_Close'] = data['Close'].ewm(span=198, adjust=False).mean()

    data['EMA_Actual Momentum'] = np.nan
    prev_momentum = np.nan
    for index, row in data.iterrows():
        close_value = row['Close']
        ema_values = row[['3EMA_Close', '6EMA_Close', '9EMA_Close']]

        if ema_values.isna().any():
            new_momentum = np.nan
        elif all(close_value > ema_values):
            new_momentum = 1
        elif all(close_value < ema_values):
            new_momentum = -1
        else:
            new_momentum = 0

        if not np.isnan(prev_momentum):
            if new_momentum != prev_momentum:
                buffer = 0.99 if new_momentum < prev_momentum else 1.01
                new_ema_values = ema_values * buffer

                if all(close_value > new_ema_values):
                    new_momentum = 1
                elif all(close_value < new_ema_values):
                    new_momentum = -1
                else:
                    new_momentum = 0

        data.at[index, 'EMA_Actual Momentum'] = int(new_momentum)
        prev_momentum = new_momentum

    data.to_csv('gold_data.csv', index=False)
    return {"message": "Added successfully!"}

@app.post("/gold_momentum/delete_entry/")
async def delete_entry(date: str):
    #FILE_PATH = 'gold_data.csv'
    # Load data
    data = pd.read_csv("gold_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Delete logic
    if pd.to_datetime(date) in data['Date'].values:
        data = data[data['Date'] != pd.to_datetime(date)]
        data.to_csv('gold_data.csv', index=False, float_format='%.2f')
        return {"message": f"Entry for {date} has been deleted and the dataset has been updated."}
    else:
        raise HTTPException(status_code=404, detail=f"No entry found for the date {date}.")

@app.get("/gold_momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 'gold_data.csv'
    data = pd.read_csv("gold_data.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/gold/contratrend/last_entries/")
async def last_entries_contratrend():
        #for data till 2023
    import calendar
    import pandas as pd
    from datetime import datetime,timedelta
    import calendar
    import pandas as pd

    
    def last_day_of_month(year, month):
        # Find the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        return year, month, last_day

    start_year = 2012
    end_year = 2024

    dates = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 12:  # Stop after December 2023
                break
            date = last_day_of_month(year, month)
            dates.append(f"{date[2]}-{date[1]:02}-{date[0]}")

    df = pd.DataFrame(dates, columns=['Last_Day'])
    df_date=df.copy()

    df_close=pd.read_csv('gold_data.csv')
    def reformat_date(date_string):
        date_obj = datetime.strptime(date_string, '%d-%m-%Y')
        return date_obj.strftime('%Y-%m-%d')

    # Assuming df['Last_Thursday'] contains the dates in "dd-mm-yyyy" format
    df_date['Last_Day'] = df_date['Last_Day'].apply(reformat_date)
    # Convert 'Last_Thursday' to string if it's in datetime format
    df_date['Last_Day'] = df_date['Last_Day'].astype(str)

    # Apply the reformat_date function
    #df_date['Last_Thursday'] = df_date['Last_Thursday'].apply(reformat_date)


    # This function checks and returns the actual expiry date
    from datetime import datetime, timedelta
    import pandas as pd

# Assuming df and df_close are already loaded and their date columns are in datetime format

# Function to find the actual expiry date
    from datetime import datetime, timedelta

    def get_actual_expiry(expiry_date_str):
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
        max_attempts = 10  # Set a limit to prevent infinite looping
        attempts = 0
        while expiry_date.strftime('%Y-%m-%d') not in df_close['Date'].values:
            expiry_date -= timedelta(days=1)  # Go back one day
            attempts += 1
            if attempts >= max_attempts:
                return None  # Or handle the missing date as appropriate
        return expiry_date.strftime('%Y-%m-%d')

    # Apply the function to the 'Last_Day' column
    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)

    # Generate actual expiry dates
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    lastday_df=df_date[df_date["Last_Day"]>'2023-12-31']
    df_date['Last_Day'] = df_date['Last_Day'].dt.strftime('%Y-%m-%d')

    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    df_date1 = df_date[df_date["Last_Day"]<'2024-01-01'].copy()
    #df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])

    lastday_df = lastday_df.rename(columns={"Last_Day": 'Date'})

    #for data from 2024
    df_date2=df_date[df_date["Last_Day"]<'2024-01-01']
    main_df=df_close[df_close['Date']>'2023-12-31']
    holiday_df=pd.read_excel('gold_2024_holidays.xlsx')
    import pandas as pd


    import pandas as pd

    import pandas as pd

    def calculate_actual_expiry(lastday_df, holiday_df):
        # Convert the holiday dates to datetime for comparison
        holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%d-%b-%y')
        lastday_df.loc[:, 'Date'] = pd.to_datetime(lastday_df['Date'])
        lastday_df.loc[:, 'Actual_Expiry'] = pd.NaT

        # Create a new column for Actual_Expiry
        lastday_df['Actual_Expiry'] = pd.NaT

        for index, row in lastday_df.iterrows():
            expiry = row['Date']

            # Check if the Thursday is a holiday
            while expiry in holiday_df['date'].values:
                # Move to the previous day
                expiry -= pd.Timedelta(days=1)

            # Assign the nearest non-holiday weekday as the Actual_Expiry
            lastday_df.at[index, 'Actual_Expiry'] = expiry

        return lastday_df

    # Example usage:
    # thursdays_df = pd.read_csv('last_thursdays.csv')
    # holiday_df = pd.read_csv('holidays.csv')

    # Update thursdays_df with the Actual_Expiry column
    updated_thursdays_df = calculate_actual_expiry(lastday_df, holiday_df)

    df_date2=updated_thursdays_df.copy()
    df_date1.rename(columns={"Last_Day":'Date'},inplace=True)
    df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])
    df_date2['Date'] = pd.to_datetime(df_date2['Date'])
    df_date2['Actual_Expiry'] = pd.to_datetime(df_date2['Actual_Expiry'])

    combined_df = pd.concat([df_date1, df_date2], ignore_index=True)


    #Making Expiry window for dates till 


    import pandas as pd
    from datetime import datetime, timedelta

    # Load your data
    closing_prices_df = pd.read_csv('gold_data.csv')
    closing_prices_df.drop(["3EMA_Close", "6EMA_Close", "9EMA_Close", "EMA_Actual Momentum"], inplace=True, axis=1)
    expiry_dates_df = combined_df.copy()  # Assuming 'df' is defined and contains the correct expiry dates

    # Convert date columns to datetime objects
    closing_prices_df['Date'] = pd.to_datetime(closing_prices_df['Date'])
    expiry_dates_df['Actual_Expiry'] = pd.to_datetime(expiry_dates_df['Actual_Expiry'])

    # Sort the expiry dates in case they are not in order
    expiry_dates_df = expiry_dates_df.sort_values(by='Actual_Expiry')

    def find_expiry_window(date, expiry_dates, start_date):
        # Special case for the first window
        if date < expiry_dates[0]:
            return start_date.strftime('%Y-%m-%d'), (expiry_dates[0] - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for i in range(1, len(expiry_dates)):
            # The start of the window is the expiry of the previous month
            # The end of the window is one day before the expiry of the current month
            if date < expiry_dates[i]:
                start_window = expiry_dates[i - 1].strftime('%Y-%m-%d')
                end_window = (expiry_dates[i] - timedelta(days=1)).strftime('%Y-%m-%d')
                return start_window, end_window

        # For dates after the last expiry date, the window starts with the last expiry date
        return expiry_dates[-1].strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')

    # Create a list of expiry dates
    expiry_dates = expiry_dates_df['Actual_Expiry'].tolist()

    # Identify the start date of the dataset for the first window
    start_date = closing_prices_df['Date'].min()

    # Map each date in the closing price dataset to its expiry window
    closing_prices_df['ExpiryWindow'] = closing_prices_df['Date'].apply(lambda date: find_expiry_window(date, expiry_dates, start_date))

    # Cleaning 'ExpiryWindow' column
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].astype(str)
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].str.replace("[()']", "", regex=True)



    import pandas as pd
    import numpy as np

    # Read the dataset
    df = closing_prices_df.copy()

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['ExpiryWindow'] = df['ExpiryWindow'].astype(str).str.replace("[()']", "", regex=True)

    # Split 'ExpiryWindow' into 'StartExpiry' and 'EndExpiry' and convert to datetime
    split_columns = df['ExpiryWindow'].str.split(', ', expand=True)
    # Convert to datetime, coerce errors to NaT
    df['StartExpiry'] = pd.to_datetime(split_columns[0], errors='coerce')
    df['EndExpiry'] = pd.to_datetime(split_columns[1], errors='coerce')

    # Calculate the rolling 6-month low for 'Close', considering 21 trading days in a month
    rolling_window = 6 * 21
    df['Normal_6m_low'] = df['Close'].rolling(window=rolling_window, min_periods=rolling_window).min()

    # Initialize 'Current_6m_low' with the values of 'Normal_6m_low'
    df['Current_6m_low'] = df['Normal_6m_low']

    # Initialize the 'Signals' column
    df['Signals'] = np.nan

    # Define a helper function to find the next expiry window
    def find_next_expiry(date, expiry_windows):
        for start, end in expiry_windows:
            if pd.notnull(start) and start > date:
                return start, end
        return None, None

    # Prepare a list of unique expiry windows
    unique_expiry_windows = df[['StartExpiry', 'EndExpiry']].drop_duplicates().values

    # Initialize the 'last_signal_date'
    last_signal_date = None

    # Iterate through the DataFrame to generate signals
    for index, row in df.iterrows():
        # Check if a new expiry window has started
        if last_signal_date is None or row['Date'] >= row['EndExpiry']:
            signal_allowed = True
        
        # If we are within an expiry window and signals are allowed, check for the signal condition
        if signal_allowed and (last_signal_date is None or (row['Date'] - last_signal_date).days >= 7):
            if row['Close'] <= 1.01 * row['Current_6m_low']:
                # Mark the signal
                df.at[index, 'Signals'] = '6L'
                signal_allowed = False  # Disallow further signals in this window
                last_signal_date = row['Date']
                
                # Find the end of the next expiry window and update 'Current_6m_low' accordingly
                _, next_window_end = find_next_expiry(row['Date'], unique_expiry_windows)
                if next_window_end:
                    df.loc[(df['Date'] >= row['Date']) & (df['Date'] <= next_window_end), 'Current_6m_low'] = row['Close']

        # At the end of the expiry window, if no new signal has been generated, reset 'Current_6m_low' to 'Normal_6m_low'
        if row['Date'] == row['EndExpiry'] and pd.isnull(df.at[index, 'Signals']):
            next_window_start = df.loc[index + 1, 'StartExpiry'] if index + 1 < len(df) else None
            if next_window_start:
                df.loc[(df['Date'] >= next_window_start), 'Current_6m_low'] = df.loc[(df['Date'] >= next_window_start), 'Normal_6m_low']

    # Display the first few rows of the dataframe with signals
    print(df)
    print(df["Signals"].value_counts())
















    out = df[["Date", 'Close', 'Current_6m_low', "Signals"]]
    out=out.copy()
    out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string
    new = out.fillna(0)
    new.rename(columns={"Current_6m_low":"Contratrend_Value"},inplace=True)
    last_rows = new.tail(5).to_dict(orient='records')
    print(last_rows)
    return last_rows

#-----------------------------------------------------------------------------------------------------------------------------------------#
#Nasdaq Momentum
# Define models for different categories
class MomentumEntry(BaseModel):
    # Fields for Market Momentum entry
    pass

# Add more models as needed for other features




# Endpoints for Market Momentum
@app.post("/nasdaq_momentum/add_entry/")
async def add_entry(date: str, close_price: float):
    #FILE_PATH = 'nasdaq_data.csv'
    # Load data
    data = pd.read_csv("nasdaq_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    new_row = pd.DataFrame({'Date': [pd.to_datetime(date)], 'Close': [close_price]})
    data = pd.concat([data, new_row]).reset_index(drop=True)

    # Compute momentum logic
    data['3EMA_Close'] = data['Close'].ewm(span=66, adjust=False).mean()
    data['6EMA_Close'] = data['Close'].ewm(span=132, adjust=False).mean()
    data['9EMA_Close'] = data['Close'].ewm(span=198, adjust=False).mean()

    data['EMA_Actual Momentum'] = np.nan
    prev_momentum = np.nan
    for index, row in data.iterrows():
        close_value = row['Close']
        ema_values = row[['3EMA_Close', '6EMA_Close', '9EMA_Close']]

        if ema_values.isna().any():
            new_momentum = np.nan
        elif all(close_value > ema_values):
            new_momentum = 1
        elif all(close_value < ema_values):
            new_momentum = -1
        else:
            new_momentum = 0

        if not np.isnan(prev_momentum):
            if new_momentum != prev_momentum:
                buffer = 0.99 if new_momentum < prev_momentum else 1.01
                new_ema_values = ema_values * buffer

                if all(close_value > new_ema_values):
                    new_momentum = 1
                elif all(close_value < new_ema_values):
                    new_momentum = -1
                else:
                    new_momentum = 0

        data.at[index, 'EMA_Actual Momentum'] = int(new_momentum)
        prev_momentum = new_momentum

    data.to_csv('nasdaq_data.csv', index=False)
    return {"message": "Added successfully!"}

@app.post("/nasdaq_momentum/delete_entry/")
async def delete_entry(date: str):
    #FILE_PATH = 'nasdaq_data.csv'
    # Load data
    data = pd.read_csv("nasdaq_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Delete logic
    if pd.to_datetime(date) in data['Date'].values:
        data = data[data['Date'] != pd.to_datetime(date)]
        data.to_csv('nasdaq_data.csv', index=False, float_format='%.2f')
        return {"message": f"Entry for {date} has been deleted and the dataset has been updated."}
    else:
        raise HTTPException(status_code=404, detail=f"No entry found for the date {date}.")

@app.get("/nasdaq_momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 'nasdaq_data.csv'
    data = pd.read_csv("nasdaq_data.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/nasdaq/contratrend/last_entries/")
async def last_entries_contratrend():
        #for data till 2023
    import calendar
    import pandas as pd
    from datetime import datetime,timedelta
    import calendar
    import pandas as pd

    
    import calendar
    import pandas as pd
    from datetime import datetime, timedelta

    def last_friday_of_month(year, month):
        # Find the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_day_date = datetime(year, month, last_day)

        # If the last day is not Friday, adjust to the previous Friday
        if last_day_date.weekday() != 4:  # 4 represents Friday
            last_day_date -= timedelta(days=(last_day_date.weekday() - 4) % 7)

        return last_day_date.strftime('%Y-%m-%d')

    start_year = 2012
    end_year = 2024
    dates = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 12:  # Stop after December 2023
                break
            date = last_friday_of_month(year, month)
            dates.append(date)

    df = pd.DataFrame(dates, columns=['Last_Day'])
    df_date=df.copy()

    df_close=pd.read_csv('nasdaq_data.csv')
    def reformat_date(date_string):
        date_obj = datetime.strptime(date_string, '%d-%m-%Y')
        return date_obj.strftime('%Y-%m-%d')

    # Assuming df['Last_Thursday'] contains the dates in "dd-mm-yyyy" format
    #df_date['Last_Day'] = df_date['Last_Day'].apply(reformat_date)
    # Convert 'Last_Thursday' to string if it's in datetime format
    df_date['Last_Day'] = df_date['Last_Day'].astype(str)

    # Apply the reformat_date function
    #df_date['Last_Thursday'] = df_date['Last_Thursday'].apply(reformat_date)


    # This function checks and returns the actual expiry date
    from datetime import datetime, timedelta
    import pandas as pd

# Assuming df and df_close are already loaded and their date columns are in datetime format

# Function to find the actual expiry date
    from datetime import datetime, timedelta

    def get_actual_expiry(expiry_date_str):
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
        max_attempts = 10  # Set a limit to prevent infinite looping
        attempts = 0
        while expiry_date.strftime('%Y-%m-%d') not in df_close['Date'].values:
            expiry_date -= timedelta(days=1)  # Go back one day
            attempts += 1
            if attempts >= max_attempts:
                return None  # Or handle the missing date as appropriate
        return expiry_date.strftime('%Y-%m-%d')

    # Apply the function to the 'Last_Day' column
    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)

    # Generate actual expiry dates
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    lastday_df=df_date[df_date["Last_Day"]>'2023-12-31']
    df_date['Last_Day'] = df_date['Last_Day'].dt.strftime('%Y-%m-%d')

    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    df_date1 = df_date[df_date["Last_Day"]<'2024-01-01'].copy()
    #df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])

    lastday_df = lastday_df.rename(columns={"Last_Day": 'Date'})

    #for data from 2024
    df_date2=df_date[df_date["Last_Day"]<'2024-01-01']
    main_df=df_close[df_close['Date']>'2023-12-31']
    holiday_df=pd.read_excel('gold_2024_holidays.xlsx')
    import pandas as pd


    import pandas as pd

    import pandas as pd

    def calculate_actual_expiry(lastday_df, holiday_df):
        # Convert the holiday dates to datetime for comparison
        holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%d-%b-%y')
        lastday_df.loc[:, 'Date'] = pd.to_datetime(lastday_df['Date'])
        lastday_df.loc[:, 'Actual_Expiry'] = pd.NaT

        # Create a new column for Actual_Expiry
        lastday_df['Actual_Expiry'] = pd.NaT

        for index, row in lastday_df.iterrows():
            expiry = row['Date']

            # Check if the Thursday is a holiday
            while expiry in holiday_df['date'].values:
                # Move to the previous day
                expiry -= pd.Timedelta(days=1)

            # Assign the nearest non-holiday weekday as the Actual_Expiry
            lastday_df.at[index, 'Actual_Expiry'] = expiry

        return lastday_df

    # Example usage:
    # thursdays_df = pd.read_csv('last_thursdays.csv')
    # holiday_df = pd.read_csv('holidays.csv')

    # Update thursdays_df with the Actual_Expiry column
    updated_thursdays_df = calculate_actual_expiry(lastday_df, holiday_df)

    df_date2=updated_thursdays_df.copy()
    df_date1.rename(columns={"Last_Day":'Date'},inplace=True)
    df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])
    df_date2['Date'] = pd.to_datetime(df_date2['Date'])
    df_date2['Actual_Expiry'] = pd.to_datetime(df_date2['Actual_Expiry'])

    combined_df = pd.concat([df_date1, df_date2], ignore_index=True)


    #Making Expiry window for dates till 


    import pandas as pd
    from datetime import datetime, timedelta

    # Load your data
    closing_prices_df = pd.read_csv('nasdaq_data.csv')
    closing_prices_df.drop(["3EMA_Close", "6EMA_Close", "9EMA_Close", "EMA_Actual Momentum"], inplace=True, axis=1)
    expiry_dates_df = combined_df.copy()  # Assuming 'df' is defined and contains the correct expiry dates

    # Convert date columns to datetime objects
    closing_prices_df['Date'] = pd.to_datetime(closing_prices_df['Date'])
    expiry_dates_df['Actual_Expiry'] = pd.to_datetime(expiry_dates_df['Actual_Expiry'])

    # Sort the expiry dates in case they are not in order
    expiry_dates_df = expiry_dates_df.sort_values(by='Actual_Expiry')

    def find_expiry_window(date, expiry_dates, start_date):
        # Special case for the first window
        if date < expiry_dates[0]:
            return start_date.strftime('%Y-%m-%d'), (expiry_dates[0] - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for i in range(1, len(expiry_dates)):
            # The start of the window is the expiry of the previous month
            # The end of the window is one day before the expiry of the current month
            if date < expiry_dates[i]:
                start_window = expiry_dates[i - 1].strftime('%Y-%m-%d')
                end_window = (expiry_dates[i] - timedelta(days=1)).strftime('%Y-%m-%d')
                return start_window, end_window

        # For dates after the last expiry date, the window starts with the last expiry date
        return expiry_dates[-1].strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')

    # Create a list of expiry dates
    expiry_dates = expiry_dates_df['Actual_Expiry'].tolist()

    # Identify the start date of the dataset for the first window
    start_date = closing_prices_df['Date'].min()

    # Map each date in the closing price dataset to its expiry window
    closing_prices_df['ExpiryWindow'] = closing_prices_df['Date'].apply(lambda date: find_expiry_window(date, expiry_dates, start_date))

    # Cleaning 'ExpiryWindow' column
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].astype(str)
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].str.replace("[()']", "", regex=True)



    import pandas as pd
    import numpy as np

    # Read the dataset
    df = closing_prices_df.copy()

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['ExpiryWindow'] = df['ExpiryWindow'].astype(str).str.replace("[()']", "", regex=True)

    # Split 'ExpiryWindow' into 'StartExpiry' and 'EndExpiry' and convert to datetime
    split_columns = df['ExpiryWindow'].str.split(', ', expand=True)
    # Convert to datetime, coerce errors to NaT
    df['StartExpiry'] = pd.to_datetime(split_columns[0], errors='coerce')
    df['EndExpiry'] = pd.to_datetime(split_columns[1], errors='coerce')

    # Calculate the rolling 6-month low for 'Close', considering 21 trading days in a month
    rolling_window = 6 * 21
    df['Normal_6m_low'] = df['Close'].rolling(window=rolling_window, min_periods=rolling_window).min()

    # Initialize 'Current_6m_low' with the values of 'Normal_6m_low'
    df['Current_6m_low'] = df['Normal_6m_low']

    # Initialize the 'Signals' column
    df['Signals'] = np.nan

    # Define a helper function to find the next expiry window
    def find_next_expiry(date, expiry_windows):
        for start, end in expiry_windows:
            if pd.notnull(start) and start > date:
                return start, end
        return None, None

    # Prepare a list of unique expiry windows
    unique_expiry_windows = df[['StartExpiry', 'EndExpiry']].drop_duplicates().values

    # Initialize the 'last_signal_date'
    last_signal_date = None

    # Iterate through the DataFrame to generate signals
    for index, row in df.iterrows():
        # Check if a new expiry window has started
        if last_signal_date is None or row['Date'] >= row['EndExpiry']:
            signal_allowed = True
        
        # If we are within an expiry window and signals are allowed, check for the signal condition
        if signal_allowed and (last_signal_date is None or (row['Date'] - last_signal_date).days >= 7):
            if row['Close'] <= 1.01 * row['Current_6m_low']:
                # Mark the signal
                df.at[index, 'Signals'] = '6L'
                signal_allowed = False  # Disallow further signals in this window
                last_signal_date = row['Date']
                
                # Find the end of the next expiry window and update 'Current_6m_low' accordingly
                _, next_window_end = find_next_expiry(row['Date'], unique_expiry_windows)
                if next_window_end:
                    df.loc[(df['Date'] >= row['Date']) & (df['Date'] <= next_window_end), 'Current_6m_low'] = row['Close']

        # At the end of the expiry window, if no new signal has been generated, reset 'Current_6m_low' to 'Normal_6m_low'
        if row['Date'] == row['EndExpiry'] and pd.isnull(df.at[index, 'Signals']):
            next_window_start = df.loc[index + 1, 'StartExpiry'] if index + 1 < len(df) else None
            if next_window_start:
                df.loc[(df['Date'] >= next_window_start), 'Current_6m_low'] = df.loc[(df['Date'] >= next_window_start), 'Normal_6m_low']

    # Display the first few rows of the dataframe with signals
    print(df)
    print(df["Signals"].value_counts())
















    out = df[["Date", 'Close', 'Current_6m_low', "Signals"]]
    out=out.copy()
    out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string
    new = out.fillna(0)
    new.rename(columns={"Current_6m_low":"Contratrend_Value"},inplace=True)
    last_rows = new.tail(5).to_dict(orient='records')
    print(last_rows)
    return last_rows

#-----------------------------------------------------------------------------------------------------------------------------------------#

#Golden_dagon Momentum
# Define models for different categories
class MomentumEntry(BaseModel):
    # Fields for Market Momentum entry
    pass

# Add more models as needed for other features




# Endpoints for Market Momentum
@app.post("/golden_momentum/add_entry/")
async def add_entry(date: str, close_price: float):
    #FILE_PATH = 'nasdaq_data.csv'
    # Load data
    data = pd.read_csv("golden_dragon.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    new_row = pd.DataFrame({'Date': [pd.to_datetime(date)], 'Close': [close_price]})
    data = pd.concat([data, new_row]).reset_index(drop=True)

    # Compute momentum logic
    data['3EMA_Close'] = data['Close'].ewm(span=66, adjust=False).mean()
    data['6EMA_Close'] = data['Close'].ewm(span=132, adjust=False).mean()
    data['9EMA_Close'] = data['Close'].ewm(span=198, adjust=False).mean()

    data['EMA_Actual Momentum'] = np.nan
    prev_momentum = np.nan
    for index, row in data.iterrows():
        close_value = row['Close']
        ema_values = row[['3EMA_Close', '6EMA_Close', '9EMA_Close']]

        if ema_values.isna().any():
            new_momentum = np.nan
        elif all(close_value > ema_values):
            new_momentum = 1
        elif all(close_value < ema_values):
            new_momentum = -1
        else:
            new_momentum = 0

        if not np.isnan(prev_momentum):
            if new_momentum != prev_momentum:
                buffer = 0.99 if new_momentum < prev_momentum else 1.01
                new_ema_values = ema_values * buffer

                if all(close_value > new_ema_values):
                    new_momentum = 1
                elif all(close_value < new_ema_values):
                    new_momentum = -1
                else:
                    new_momentum = 0

        data.at[index, 'EMA_Actual Momentum'] = int(new_momentum)
        prev_momentum = new_momentum

    data.to_csv('golden_dragon.csv', index=False)
    return {"message": "Added successfully!"}

@app.post("/golden_momentum/delete_entry/")
async def delete_entry(date: str):
    #FILE_PATH = 'nasdaq_data.csv'
    # Load data
    data = pd.read_csv("golden_dragon.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Delete logic
    if pd.to_datetime(date) in data['Date'].values:
        data = data[data['Date'] != pd.to_datetime(date)]
        data.to_csv('golden_dragon.csv', index=False, float_format='%.2f')
        return {"message": f"Entry for {date} has been deleted and the dataset has been updated."}
    else:
        raise HTTPException(status_code=404, detail=f"No entry found for the date {date}.")

@app.get("/golden_momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 
    data = pd.read_csv("golden_dragon.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/gold_momentum/last_entries/")
async def last_entries():
    #FILE_PATH = 'gold_data.csv'
    data = pd.read_csv("gold_data.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}

@app.get("/goldendragon/contratrend/last_entries/")
async def last_entries_contratrend():
        #for data till 2023
    import calendar
    import pandas as pd
    from datetime import datetime,timedelta
    import calendar
    import pandas as pd

    
    def last_day_of_month(year, month):
        # Find the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        return year, month, last_day

    start_year = 2012
    end_year = 2024

    dates = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 12:  # Stop after December 2023
                break
            date = last_day_of_month(year, month)
            dates.append(f"{date[2]}-{date[1]:02}-{date[0]}")

    df = pd.DataFrame(dates, columns=['Last_Day'])
    df_date=df.copy()

    df_close=pd.read_csv('golden_dragon.csv')
    def reformat_date(date_string):
        date_obj = datetime.strptime(date_string, '%d-%m-%Y')
        return date_obj.strftime('%Y-%m-%d')

    # Assuming df['Last_Thursday'] contains the dates in "dd-mm-yyyy" format
    df_date['Last_Day'] = df_date['Last_Day'].apply(reformat_date)
    # Convert 'Last_Thursday' to string if it's in datetime format
    df_date['Last_Day'] = df_date['Last_Day'].astype(str)

    # Apply the reformat_date function
    #df_date['Last_Thursday'] = df_date['Last_Thursday'].apply(reformat_date)


    # This function checks and returns the actual expiry date
    from datetime import datetime, timedelta
    import pandas as pd

# Assuming df and df_close are already loaded and their date columns are in datetime format

# Function to find the actual expiry date
    from datetime import datetime, timedelta

    def get_actual_expiry(expiry_date_str):
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
        max_attempts = 10  # Set a limit to prevent infinite looping
        attempts = 0
        while expiry_date.strftime('%Y-%m-%d') not in df_close['Date'].values:
            expiry_date -= timedelta(days=1)  # Go back one day
            attempts += 1
            if attempts >= max_attempts:
                return None  # Or handle the missing date as appropriate
        return expiry_date.strftime('%Y-%m-%d')

    # Apply the function to the 'Last_Day' column
    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)

    # Generate actual expiry dates
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    lastday_df=df_date[df_date["Last_Day"]>'2023-12-31']
    df_date['Last_Day'] = df_date['Last_Day'].dt.strftime('%Y-%m-%d')

    df_date['Actual_Expiry'] = df_date['Last_Day'].apply(get_actual_expiry)
    df_date['Last_Day'] = pd.to_datetime(df_date['Last_Day'], dayfirst=True)
    df_date1 = df_date[df_date["Last_Day"]<'2024-01-01'].copy()
    #df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])

    lastday_df = lastday_df.rename(columns={"Last_Day": 'Date'})

    #for data from 2024
    df_date2=df_date[df_date["Last_Day"]<'2024-01-01']
    main_df=df_close[df_close['Date']>'2023-12-31']
    holiday_df=pd.read_excel('gold_2024_holidays.xlsx')
    import pandas as pd


    import pandas as pd

    import pandas as pd

    def calculate_actual_expiry(lastday_df, holiday_df):
        # Convert the holiday dates to datetime for comparison
        holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%d-%b-%y')
        lastday_df.loc[:, 'Date'] = pd.to_datetime(lastday_df['Date'])
        lastday_df.loc[:, 'Actual_Expiry'] = pd.NaT

        # Create a new column for Actual_Expiry
        lastday_df['Actual_Expiry'] = pd.NaT

        for index, row in lastday_df.iterrows():
            expiry = row['Date']

            # Check if the Thursday is a holiday
            while expiry in holiday_df['date'].values:
                # Move to the previous day
                expiry -= pd.Timedelta(days=1)

            # Assign the nearest non-holiday weekday as the Actual_Expiry
            lastday_df.at[index, 'Actual_Expiry'] = expiry

        return lastday_df

    # Example usage:
    # thursdays_df = pd.read_csv('last_thursdays.csv')
    # holiday_df = pd.read_csv('holidays.csv')

    # Update thursdays_df with the Actual_Expiry column
    updated_thursdays_df = calculate_actual_expiry(lastday_df, holiday_df)

    df_date2=updated_thursdays_df.copy()
    df_date1.rename(columns={"Last_Day":'Date'},inplace=True)
    df_date1['Date'] = pd.to_datetime(df_date1['Date'])
    df_date1['Actual_Expiry'] = pd.to_datetime(df_date1['Actual_Expiry'])
    df_date2['Date'] = pd.to_datetime(df_date2['Date'])
    df_date2['Actual_Expiry'] = pd.to_datetime(df_date2['Actual_Expiry'])

    combined_df = pd.concat([df_date1, df_date2], ignore_index=True)


    #Making Expiry window for dates till 


    import pandas as pd
    from datetime import datetime, timedelta

    # Load your data
    closing_prices_df = pd.read_csv('gold_data.csv')
    closing_prices_df.drop(["3EMA_Close", "6EMA_Close", "9EMA_Close", "EMA_Actual Momentum"], inplace=True, axis=1)
    expiry_dates_df = combined_df.copy()  # Assuming 'df' is defined and contains the correct expiry dates

    # Convert date columns to datetime objects
    closing_prices_df['Date'] = pd.to_datetime(closing_prices_df['Date'])
    expiry_dates_df['Actual_Expiry'] = pd.to_datetime(expiry_dates_df['Actual_Expiry'])

    # Sort the expiry dates in case they are not in order
    expiry_dates_df = expiry_dates_df.sort_values(by='Actual_Expiry')

    def find_expiry_window(date, expiry_dates, start_date):
        # Special case for the first window
        if date < expiry_dates[0]:
            return start_date.strftime('%Y-%m-%d'), (expiry_dates[0] - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for i in range(1, len(expiry_dates)):
            # The start of the window is the expiry of the previous month
            # The end of the window is one day before the expiry of the current month
            if date < expiry_dates[i]:
                start_window = expiry_dates[i - 1].strftime('%Y-%m-%d')
                end_window = (expiry_dates[i] - timedelta(days=1)).strftime('%Y-%m-%d')
                return start_window, end_window

        # For dates after the last expiry date, the window starts with the last expiry date
        return expiry_dates[-1].strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')

    # Create a list of expiry dates
    expiry_dates = expiry_dates_df['Actual_Expiry'].tolist()

    # Identify the start date of the dataset for the first window
    start_date = closing_prices_df['Date'].min()

    # Map each date in the closing price dataset to its expiry window
    closing_prices_df['ExpiryWindow'] = closing_prices_df['Date'].apply(lambda date: find_expiry_window(date, expiry_dates, start_date))

    # Cleaning 'ExpiryWindow' column
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].astype(str)
    closing_prices_df['ExpiryWindow'] = closing_prices_df['ExpiryWindow'].str.replace("[()']", "", regex=True)



    import pandas as pd
    import numpy as np

    # Read the dataset
    df = closing_prices_df.copy()

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['ExpiryWindow'] = df['ExpiryWindow'].astype(str).str.replace("[()']", "", regex=True)

    # Split 'ExpiryWindow' into 'StartExpiry' and 'EndExpiry' and convert to datetime
    split_columns = df['ExpiryWindow'].str.split(', ', expand=True)
    # Convert to datetime, coerce errors to NaT
    df['StartExpiry'] = pd.to_datetime(split_columns[0], errors='coerce')
    df['EndExpiry'] = pd.to_datetime(split_columns[1], errors='coerce')

    # Calculate the rolling 6-month low for 'Close', considering 21 trading days in a month
    rolling_window = 6 * 21
    df['Normal_6m_low'] = df['Close'].rolling(window=rolling_window, min_periods=rolling_window).min()

    # Initialize 'Current_6m_low' with the values of 'Normal_6m_low'
    df['Current_6m_low'] = df['Normal_6m_low']

    # Initialize the 'Signals' column
    df['Signals'] = np.nan

    # Define a helper function to find the next expiry window
    def find_next_expiry(date, expiry_windows):
        for start, end in expiry_windows:
            if pd.notnull(start) and start > date:
                return start, end
        return None, None

    # Prepare a list of unique expiry windows
    unique_expiry_windows = df[['StartExpiry', 'EndExpiry']].drop_duplicates().values

    # Initialize the 'last_signal_date'
    last_signal_date = None

    # Iterate through the DataFrame to generate signals
    for index, row in df.iterrows():
        # Check if a new expiry window has started
        if last_signal_date is None or row['Date'] >= row['EndExpiry']:
            signal_allowed = True
        
        # If we are within an expiry window and signals are allowed, check for the signal condition
        if signal_allowed and (last_signal_date is None or (row['Date'] - last_signal_date).days >= 7):
            if row['Close'] <= 1.01 * row['Current_6m_low']:
                # Mark the signal
                df.at[index, 'Signals'] = '6L'
                signal_allowed = False  # Disallow further signals in this window
                last_signal_date = row['Date']
                
                # Find the end of the next expiry window and update 'Current_6m_low' accordingly
                _, next_window_end = find_next_expiry(row['Date'], unique_expiry_windows)
                if next_window_end:
                    df.loc[(df['Date'] >= row['Date']) & (df['Date'] <= next_window_end), 'Current_6m_low'] = row['Close']

        # At the end of the expiry window, if no new signal has been generated, reset 'Current_6m_low' to 'Normal_6m_low'
        if row['Date'] == row['EndExpiry'] and pd.isnull(df.at[index, 'Signals']):
            next_window_start = df.loc[index + 1, 'StartExpiry'] if index + 1 < len(df) else None
            if next_window_start:
                df.loc[(df['Date'] >= next_window_start), 'Current_6m_low'] = df.loc[(df['Date'] >= next_window_start), 'Normal_6m_low']

    # Display the first few rows of the dataframe with signals
    print(df)
    print(df["Signals"].value_counts())
















    out = df[["Date", 'Close', 'Current_6m_low', "Signals"]]
    out=out.copy()
    out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string
    new = out.fillna(0)
    new.rename(columns={"Current_6m_low":"Contratrend_Value"},inplace=True)
    last_rows = new.tail(5).to_dict(orient='records')
    print(last_rows)
    return last_rows

#-----------------------------------------------------------------------------------------------------------------------------------------#

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



    #things to do
    # - seperete nifty, s&p and gold
    # - nifty price action vs moving averages for PE
    # - 3 year in graph for PE
    # - momentum and volatility 1 yr rolling
    # - keep the till date data and graphs somehwere else that can be accessed 
    # - one combined signal chart once done with contratrend
