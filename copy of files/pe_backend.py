from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


app = FastAPI()

# Path to your CSV file
FILE_PATH = 'data_pe.csv'

class PEEntry(BaseModel):
    date: str
    pe_value: float

class DeleteEntryRequest(BaseModel):
    date: str

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


# Endpoint to fetch last 5 entries for each signal dataframe
@app.get("/pe/last_entries/{signal_type}")
async def last_entries(signal_type: str):
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

