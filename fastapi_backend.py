from fastapi import FastAPI, HTTPException, Form
import pandas as pd

app = FastAPI()

FILE_PATH = 'current_momentum.csv'

@app.post("/add_entry/")
async def add_entry(date: str, close_price: float):
    # Load data
    data = pd.read_csv("current_momentum.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    new_row = pd.DataFrame({'Date': [pd.to_datetime(date)], 'Close': [close_price]})
    data = pd.concat([data, new_row]).reset_index(drop=True)

    # Compute momentum logic
    data['3EMA_Close'] = data['Close'].ewm(span=66, adjust=False).mean()
    data['6EMA_Close'] = data['Close'].ewm(span=132, adjust=False).mean()
    data['9EMA_Close'] = data['Close'].ewm(span=198, adjust=False).mean()

    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype('Int64')
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

    data.to_csv(FILE_PATH, index=False)
    return {"message": "Added successfully!"}

@app.post("/delete_entry/")
async def delete_entry(date: str):
    # Load data
    data = pd.read_csv("current_momentum.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # Delete logic
    if pd.to_datetime(date) in data['Date'].values:
        data = data[data['Date'] != pd.to_datetime(date)]
        data.to_csv(FILE_PATH, index=False, float_format='%.2f')
        return {"message": f"Entry for {date} has been deleted and the dataset has been updated."}
    else:
        raise HTTPException(status_code=404, detail=f"No entry found for the date {date}.")

@app.get("/last_entries/")
async def last_entries():
    data = pd.read_csv("current_momentum.csv")
    # Format the columns to remove trailing zeros
    data['Close'] = data['Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['3EMA_Close'] = data['3EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['6EMA_Close'] = data['6EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['9EMA_Close'] = data['9EMA_Close'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
    data['EMA_Actual Momentum'] = data['EMA_Actual Momentum'].astype(int)  # Convert momentum to integer
    last_rows = data.tail(5).to_dict(orient='records')
    return {"data": last_rows}
