import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

def pe_signals():
    # Code for PE Signals feature
    def plot_df5(df5):
    # Plotting for df5
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd

        # Assuming df5 is your DataFrame and it's already loaded
        # Plotting for df5
        df5['Date'] = pd.to_datetime(df5['Date'])
        plt.figure(figsize=(15, 8))
        plt.plot(df5['Date'], df5['P/E'], label='P/E', color='blue')
        plt.plot(df5['Date'], df5['SMA_5_Years'], label='5Y SMA', color='black')
        plt.plot(df5['Date'], df5['1.5sd_5SMA_POS'], label='+1.5 SD', color='red', linestyle='--')
        plt.plot(df5['Date'], df5['2sd_5SMA_POS'], label='+2 SD', color='red', linestyle='--')
        plt.plot(df5['Date'], df5['3sd_5SMA_POS'], label='+3 SD', color='red', linestyle='--')
        plt.plot(df5['Date'], df5['1.5sd_5SMA_NEG'], label='-1.5 SD', color='green', linestyle='--')
        plt.plot(df5['Date'], df5['2sd_5SMA_NEG'], label='-2 SD', color='green', linestyle='--')
        plt.plot(df5['Date'], df5['3sd_5SMA_NEG'], label='-3 SD', color='green', linestyle='--')

        # Highlighting buy signals with an up triangle marker
        buy_signals = df5[df5['signals'].str.contains('B', na=False)]
        plt.scatter(buy_signals['Date'], buy_signals['P/E'], marker='^', color='green', s=100, label='Buy Signal')

        # Highlighting sell signals with a down triangle marker
        sell_signals = df5[df5['signals'].str.contains('S', na=False)]
        plt.scatter(sell_signals['Date'], sell_signals['P/E'], marker='v', color='red', s=100, label='Sell Signal')
        min_date = df5['Date'].min()
        max_date = df5['Date'].max()

        # Set x-axis to display years with a specific interval
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().set_xlim(min_date, max_date)

        plt.title('P/E Ratio with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('P/E')
        plt.legend()
        plt.grid(True)
        #plt.show()


        return plt

# Similarly, plot_df7 and plot_df10 for the other datasets


    def plot_df7(df7):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        # Convert 'Date' to datetime format if it's not already
        df7['Date'] = pd.to_datetime(df7['Date'])

        plt.figure(figsize=(15, 8))
        plt.plot(df7['Date'], df7['P/E'], label='P/E', color='blue')
        plt.plot(df7['Date'], df7['sma_7_5_years'], label='7.5Y SMA', color='black')
        plt.plot(df7['Date'], df7['1.5sd_7_5_SMA_POS'], label='+1.5 SD', color='red', linestyle='--')
        plt.plot(df7['Date'], df7['2sd_7_5SMA_POS'], label='+2 SD', color='red', linestyle='--')
        plt.plot(df7['Date'], df7['3sd_7_5SMA_POS'], label='+3 SD', color='red', linestyle='--')
        plt.plot(df7['Date'], df7['1.5sd_7_5SMA_NEG'], label='-1.5 SD', color='green', linestyle='--')
        plt.plot(df7['Date'], df7['2sd_7_5SMA_NEG'], label='-2 SD', color='green', linestyle='--')
        plt.plot(df7['Date'], df7['3sd_7_5SMA_NEG'], label='-3 SD', color='green', linestyle='--')

        # Highlighting buy signals with an up triangle marker
        buy_signals = df7[df7['signals'].str.contains('B', na=False)]
        plt.scatter(buy_signals['Date'], buy_signals['P/E'], marker='^', color='green', s=100, label='Buy Signal')

        # Highlighting sell signals with a down triangle marker
        sell_signals = df7[df7['signals'].str.contains('S', na=False)]
        plt.scatter(sell_signals['Date'], sell_signals['P/E'], marker='v', color='red', s=100, label='Sell Signal')

        # Set x-axis to display years with a specific interval
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().set_xlim(df7['Date'].min(), df7['Date'].max())

        plt.title('P/E Ratio with Buy/Sell Signals for df7')
        plt.xlabel('Date')
        plt.ylabel('P/E')
        plt.legend()
        plt.grid(True)
        #plt.show()

        return plt



    def plot_df10(df10):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        # Convert 'Date' to datetime format if it's not already
        df10['Date'] = pd.to_datetime(df10['Date'])

        plt.figure(figsize=(15, 8))
        plt.plot(df10['Date'], df10['P/E'], label='P/E', color='blue')
        plt.plot(df10['Date'], df10['sma_10_years'], label='10Y SMA', color='black')
        plt.plot(df10['Date'], df10['1.5sd_10_SMA_POS'], label='+1.5 SD', color='red', linestyle='--')
        plt.plot(df10['Date'], df10['2sd_10SMA_POS'], label='+2 SD', color='red', linestyle='--')
        plt.plot(df10['Date'], df10['3sd_10SMA_POS'], label='+3 SD', color='red', linestyle='--')
        plt.plot(df10['Date'], df10['1.5sd_10SMA_NEG'], label='-1.5 SD', color='green', linestyle='--')
        plt.plot(df10['Date'], df10['2sd_10SMA_NEG'], label='-2 SD', color='green', linestyle='--')
        plt.plot(df10['Date'], df10['3sd_10SMA_NEG'], label='-3 SD', color='green', linestyle='--')

        # Highlighting buy signals with an up triangle marker
        buy_signals = df10[df10['signals'].str.contains('B', na=False)]
        plt.scatter(buy_signals['Date'], buy_signals['P/E'], marker='^', color='green', s=100, label='Buy Signal')

        # Highlighting sell signals with a down triangle marker
        sell_signals = df10[df10['signals'].str.contains('S', na=False)]
        plt.scatter(sell_signals['Date'], sell_signals['P/E'], marker='v', color='red', s=100, label='Sell Signal')

        # Set x-axis to display years with a specific interval
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().set_xlim(df10['Date'].min(), df10['Date'].max())

        plt.title('P/E Ratio with Buy/Sell Signals for df10')
        plt.xlabel('Date')
        plt.ylabel('P/E')
        plt.legend()
        plt.grid(True)
        #plt.show()

        return plt
    # Function to fetch last entries for a given signal type
    def fetch_last_entries(signal_type):
        response = requests.get(f"http://127.0.0.1:8000/pe/last_entries/{signal_type}")
        return response.json() if response.status_code == 200 else []

    

    # Input new PE entry
    st.subheader("Add New PE Entry")
    pe_date = st.text_input("Date (YYYY-MM-DD)", key="pe_date")
    pe_value_str = st.text_input("PE Value", key="pe_value")
    pe_value = float(pe_value_str) if pe_value_str else None

    if st.button("Add PE Entry"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/pe/add_entry/",
                json={"date": pe_date, "pe_value": pe_value}
            )
            if response.status_code == 200:
                st.success("Entry added successfully!")
                st.experimental_rerun()
            else:
                st.error(f"Failed to add entry: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Delete PE entry
    st.subheader("Delete PE Entry")
    delete_date = st.text_input("Date to Delete (YYYY-MM-DD)", key="delete_date")
    if st.button("Delete Entry"):
        response = requests.post("http://127.0.0.1:8000/pe/delete_entry/", json={"date": delete_date})
        if response.status_code == 200:
            st.success("Entry deleted successfully!")
            st.experimental_rerun()
        else:
            st.error("Failed to delete entry.")

    # Display sections for each signal type
    for signal_type, file_name in [('5SMA', 'df5.csv'), ('7.5SMA', 'df7.csv'), ('10SMA', 'df10.csv')]:
        st.header(f"{signal_type} Signals")

        # Fetch and display the last few entries
        signal_data = fetch_last_entries(signal_type)
        st.table(signal_data)

        # Read the data directly from the file
        df = pd.read_csv(file_name, parse_dates=['Date'])

        # Plot the data
        if signal_type == '5SMA':
            plt = plot_df5(df)
        elif signal_type == '7.5SMA':
            plt = plot_df7(df)
        elif signal_type == '10SMA':
            plt = plot_df10(df)

        st.pyplot(plt)
    pass

def market_momentum():
    # Code for Market Momentum feature
    def fetch_last_5_entries():
        response = requests.get("http://127.0.0.1:8000/momentum/last_entries/")
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.write("Error retrieving data!")
            return []
    st.title("Market Momentum App")

    # Check if 'data' is already in the session state
    if 'data' not in st.session_state:
        st.session_state.data = fetch_last_5_entries()

    # Display the data table
    st.write("### Last 5 Entries")
    st.table(st.session_state.data)

    # Input new entry
    st.write("### Add New Entry")
    date = st.text_input("Date (YYYY-MM-DD)")
    close_price_str = st.text_input("Closing Price")
    close_price = float(close_price_str) if close_price_str else None

    if st.button("Add Entry"):
        response = requests.post(f"http://127.0.0.1:8000/momentum/add_entry/?date={date}&close_price={close_price}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data = fetch_last_5_entries()
            
            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")
            error_details = response.json().get("detail", [])
            for error in error_details:
                st.write(error)

    # Delete entry
    st.write("### Delete Entry")
    delete_date = st.text_input("Date to Delete (YYYY-MM-DD)")
    if st.button("Delete"):
        response = requests.post(f"http://127.0.0.1:8000/momentum/delete_entry/?date={delete_date}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data = fetch_last_5_entries()

            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")

        # Code for contratrend
    # Code for contratrend
    def nifty_contratrend():
        response = requests.get("http://127.0.0.1:8000/nifty/contratrend/last_entries/")
        if response.status_code == 200:
            return response.json()  # Directly return the JSON response
        else:
            st.write("Error retrieving data!")
            return []

    # Check if 'contra_nifty' is already in the session state
    if 'contra_nifty' not in st.session_state:
        st.session_state.contra_nifty = nifty_contratrend()

    # Display the data table
    st.write("### Nifty Contratrend Signals")
    st.table(st.session_state.contra_nifty)


    #plot price action against moving averages
    st.header('Nift price action and moving averages')
    # Filtering the DataFrame for dates starting from 2023
    df=pd.read_csv('current_momentum.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Date'] >= pd.Timestamp('2023-01-01')]

    # Plotting Close price and the moving averages for the filtered date range
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price', color='blue')
    plt.plot(df_filtered['Date'], df_filtered['3EMA_Close'], label='3-Months EMA', color='red', linestyle='--')
    plt.plot(df_filtered['Date'], df_filtered['6EMA_Close'], label='6-Months EMA', color='green', linestyle='-.')
    plt.plot(df_filtered['Date'], df_filtered['9EMA_Close'], label='9-Months EMA', color='purple', linestyle=':')

    # Adding title and labels
    plt.title('Close Price and Moving Averages (2023 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    st.pyplot(plt)

def volatility():   # Function to send add entry request

    
    def fetch_last_5_entries(dataset):
        response = requests.get(f"http://127.0.0.1:8000/volatility/last_entries/{dataset}")
        return response.json() if response.status_code == 200 else []

    def add_entry(dataset, date, price):
        response = requests.post("http://127.0.0.1:8000/volatility/add_entry/", 
                         json={"index": dataset, "date": date, "price": price})

        return response

    # Function to send delete entry request
    def delete_entry(dataset, date):
        response = requests.post("http://127.0.0.1:8000/volatility/delete_entry/", 
                             json={"index": dataset, "date": date})
        return response


        

    def plot_volatility(dataset):
        file_path = f"{dataset}_volatility.csv"
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df[df['date'].dt.year >= 2022]  # Filter the DataFrame for dates starting from 2023
        df = df.dropna(subset=['mean_Volatility'])

        # Apply a rolling mean with a window size to smooth the data
        window_size = 30  # Adjust the window size as needed
        df['smoothed_mean_Volatility'] = df['mean_Volatility'].rolling(window=window_size, min_periods=1, center=True).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['mean_Volatility'], label='Mean Volatility', color='blue', alpha=0.5)  # Original data
        plt.plot(df['date'], df['smoothed_mean_Volatility'], label='Smoothed Mean Volatility', color='orange')  # Smoothed data
        plt.axhline(y=1.5, color='r', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Mean Volatility')
        plt.title(f'NIFTY {dataset.upper()} Mean Volatility Over Time')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)


    def display_volatility_section(dataset_name):
        st.subheader(f"NIFTY {dataset_name.upper()} Volatility Analysis")

        # Function to read the last 5 entries from the CSV file
        def read_last_5_entries(dataset):
            file_path = f"{dataset}_volatility.csv"
            try:
                df = pd.read_csv(file_path)
                df.sort_values(by='date', ascending=False, inplace=True)
                return df.head(5)
            except Exception as e:
                st.error(f"Failed to read data: {e}")
                return pd.DataFrame()

        # Initialize or update the reload flag
        reload_key = f'reload_{dataset_name}'
        if reload_key not in st.session_state:
            st.session_state[reload_key] = True

        # If data needs to be reloaded, read and display the last 5 entries
        if st.session_state[reload_key]:
            last_entries = read_last_5_entries(dataset_name)
            st.write("### Last 5 Entries")
            st.table(last_entries)

        # Input fields for adding new entries
        date = st.text_input(f"Enter Date for NIFTY {dataset_name.upper()} (YYYY-MM-DD)", key=f"date_{dataset_name}")
        price = st.text_input(f"Enter NIFTY {dataset_name.upper()} Closing Price", key=f"price_{dataset_name}")

        # Button for adding new entries
        if st.button(f"Add NIFTY {dataset_name.upper()} Entry", key=f'add_entry_{dataset_name}'):
            response = add_entry(dataset_name, date, price)
            if response.status_code == 200:
                st.success(response.json().get("message", "Success!"))
                st.session_state[reload_key] = not st.session_state[reload_key]  # Toggle reload flag
            else:
                st.error("Failed to add entry.")

        # Section for deleting entries
        date_delete = st.text_input(f"Enter Date for Deleting NIFTY {dataset_name.upper()} Entry (YYYY-MM-DD)", key=f"date_delete_{dataset_name}")
        if st.button(f"Delete NIFTY {dataset_name.upper()} Entry", key=f'delete_entry_{dataset_name}'):
            response = delete_entry(dataset_name, date_delete)
            if response.status_code == 200:
                st.success(response.json().get("message", "Deleted successfully."))
                st.session_state[reload_key] = not st.session_state[reload_key]  # Toggle reload flag
            else:
                st.error("Failed to delete entry.")

        

        plot_volatility(dataset_name)


# ... rest of your volatility function ...


    st.title("Volatility Analysis for NIFTY Indices")
    for dataset in ['nifty_50', 'nifty_200', 'nifty_500']:
        display_volatility_section(dataset)

def snp():
    # Code for Market Momentum feature
    def fetch_last_5_entries():
        response = requests.get("http://127.0.0.1:8000/snp/momentum/last_entries/")
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'data_snp' not in st.session_state:
        st.session_state.data_snp = fetch_last_5_entries()

    # Display the data table
    st.write("### Last 5 Entries")
    st.table(st.session_state.data_snp)

    # Input new entry
    st.write("### Add New Entry")
    date = st.text_input("Date (YYYY-MM-DD)")
    close_price_str = st.text_input("Closing Price")
    close_price = float(close_price_str) if close_price_str else None

    if st.button("Add Entry"):
        response = requests.post(f"http://127.0.0.1:8000/snp/momentum/add_entry/?date={date}&close_price={close_price}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_snp = fetch_last_5_entries()
            
            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")
            error_details = response.json().get("detail", [])
            for error in error_details:
                st.write(error)

    # Delete entry
    st.write("### Delete Entry")
    delete_date = st.text_input("Date to Delete (YYYY-MM-DD)")
    if st.button("Delete"):
        response = requests.post(f"http://127.0.0.1:8000/snp/momentum/delete_entry/?date={delete_date}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_snp = fetch_last_5_entries()

            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")

    #plot price action against moving averages
    st.header('S&P price action and moving averages')
    # Filtering the DataFrame for dates starting from 2023
    df=pd.read_csv('golden_dragon.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Date'] >= pd.Timestamp('2023-01-01')]

    # Plotting Close price and the moving averages for the filtered date range
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price', color='blue')
    plt.plot(df_filtered['Date'], df_filtered['3EMA_Close'], label='3-Months EMA', color='red', linestyle='--')
    plt.plot(df_filtered['Date'], df_filtered['6EMA_Close'], label='6-Months EMA', color='green', linestyle='-.')
    plt.plot(df_filtered['Date'], df_filtered['9EMA_Close'], label='9-Months EMA', color='purple', linestyle=':')

    # Adding title and labels
    plt.title('Close Price and Moving Averages (2023 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    st.pyplot(plt)

    
    # Code for contratrend
    def contra_snp():
        response = requests.get("http://127.0.0.1:8000/snp/contratrend/last_entries/")
        if response.status_code == 200:
            return response.json()
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'contra_snp' not in st.session_state:
        st.session_state.contra_snp = contra_snp()

    # Display the data table
    st.write("### SnP Contratrend Signals")
    st.table(st.session_state.contra_snp)




def gold():
    # Code for Market Momentum feature
    def fetch_last_5_entries():
        response = requests.get("http://127.0.0.1:8000/gold_momentum/last_entries/")
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'data_gold' not in st.session_state:
        st.session_state.data_gold = fetch_last_5_entries()

    # Display the data table
    st.write("### Last 5 Entries")
    st.table(st.session_state.data_gold)

    # Input new entry
    st.write("### Add New Entry")
    date = st.text_input("Date (YYYY-MM-DD)")
    close_price_str = st.text_input("Closing Price")
    close_price = float(close_price_str) if close_price_str else None

    if st.button("Add Entry"):
        response = requests.post(f"http://127.0.0.1:8000/gold_momentum/add_entry/?date={date}&close_price={close_price}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_gold = fetch_last_5_entries()
            
            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")
            error_details = response.json().get("detail", [])
            for error in error_details:
                st.write(error)

    # Delete entry
    st.write("### Delete Entry")
    delete_date = st.text_input("Date to Delete (YYYY-MM-DD)")
    if st.button("Delete"):
        response = requests.post(f"http://127.0.0.1:8000/gold_momentum/delete_entry/?date={delete_date}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_gold = fetch_last_5_entries()

            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")

    def gold_contratrend():
        response = requests.get("http://127.0.0.1:8000/gold/contratrend/last_entries/")
        if response.status_code == 200:
            return response.json()
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'contra_gold' not in st.session_state:
        st.session_state.contra_gold = gold_contratrend()

    # Display the data table
    st.write("### Gold Contratrend Signals")
    st.table(st.session_state.contra_gold)

    #plot price action against moving averages
    st.header('Gold price action and moving averages')
    # Filtering the DataFrame for dates starting from 2023
    df=pd.read_csv('gold_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Date'] >= pd.Timestamp('2023-01-01')]

    # Plotting Close price and the moving averages for the filtered date range
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price', color='blue')
    plt.plot(df_filtered['Date'], df_filtered['3EMA_Close'], label='3-Months EMA', color='red', linestyle='--')
    plt.plot(df_filtered['Date'], df_filtered['6EMA_Close'], label='6-Months EMA', color='green', linestyle='-.')
    plt.plot(df_filtered['Date'], df_filtered['9EMA_Close'], label='9-Months EMA', color='purple', linestyle=':')

    # Adding title and labels
    plt.title('Close Price and Moving Averages (2023 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    st.pyplot(plt)


def nasdaq():
    # Code for nasdaq Market Momentum feature
    def fetch_last_5_entries():
        response = requests.get("http://127.0.0.1:8000/nasdaq_momentum/last_entries/")
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'data_nasdaq' not in st.session_state:
        st.session_state.data_nasdaq = fetch_last_5_entries()

    # Display the data table
    st.write("### Last 5 Entries")
    st.table(st.session_state.data_nasdaq)

    # Input new entry
    st.write("### Add New Entry")
    date = st.text_input("Date (YYYY-MM-DD)")
    close_price_str = st.text_input("Closing Price")
    close_price = float(close_price_str) if close_price_str else None

    if st.button("Add Entry"):
        response = requests.post(f"http://127.0.0.1:8000/nasdaq_momentum/add_entry/?date={date}&close_price={close_price}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_nasdaq = fetch_last_5_entries()
            
            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")
            error_details = response.json().get("detail", [])
            for error in error_details:
                st.write(error)

    # Delete entry
    st.write("### Delete Entry")
    delete_date = st.text_input("Date to Delete (YYYY-MM-DD)")
    if st.button("Delete"):
        response = requests.post(f"http://127.0.0.1:8000/nasdaq_momentum/delete_entry/?date={delete_date}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_nasdaq = fetch_last_5_entries()

            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")

    def nasdaq_contratrend():
        response = requests.get("http://127.0.0.1:8000/nasdaq/contratrend/last_entries/")
        if response.status_code == 200:
            return response.json()
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'contra_nasdaq' not in st.session_state:
        st.session_state.contra_nasdaq = nasdaq_contratrend()

    # Display the data table
    st.write("### Nasdaq Contratrend Signals")
    st.table(st.session_state.contra_nasdaq)

    #plot price action against moving averages
    st.header('Nasdaq price action and moving averages')
    # Filtering the DataFrame for dates starting from 2023
    df=pd.read_csv('nasdaq_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Date'] >= pd.Timestamp('2023-01-01')]

    # Plotting Close price and the moving averages for the filtered date range
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price', color='blue')
    plt.plot(df_filtered['Date'], df_filtered['3EMA_Close'], label='3-Months EMA', color='red', linestyle='--')
    plt.plot(df_filtered['Date'], df_filtered['6EMA_Close'], label='6-Months EMA', color='green', linestyle='-.')
    plt.plot(df_filtered['Date'], df_filtered['9EMA_Close'], label='9-Months EMA', color='purple', linestyle=':')

    # Adding title and labels
    plt.title('Close Price and Moving Averages (2023 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    st.pyplot(plt)



def golden_dragon():
    # Code for nasdaq Market Momentum feature
    def fetch_last_5_entries():
        response = requests.get("http://127.0.0.1:8000/golden_momentum/last_entries/")
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'data_golden' not in st.session_state:
        st.session_state.data_golden = fetch_last_5_entries()

    # Display the data table
    st.write("### Last 5 Entries")
    st.table(st.session_state.data_golden)

    # Input new entry
    st.write("### Add New Entry")
    date = st.text_input("Date (YYYY-MM-DD)")
    close_price_str = st.text_input("Closing Price")
    close_price = float(close_price_str) if close_price_str else None

    if st.button("Add Entry"):
        response = requests.post(f"http://127.0.0.1:8000/golden_momentum/add_entry/?date={date}&close_price={close_price}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_golden = fetch_last_5_entries()
            
            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")
            error_details = response.json().get("detail", [])
            for error in error_details:
                st.write(error)

    # Delete entry
    st.write("### Delete Entry")
    delete_date = st.text_input("Date to Delete (YYYY-MM-DD)")
    if st.button("Delete"):
        response = requests.post(f"http://127.0.0.1:8000/golden_momentum/delete_entry/?date={delete_date}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data_golden = fetch_last_5_entries()

            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")

    def goldendragon_contratrend():
        response = requests.get("http://127.0.0.1:8000/goldendragon/contratrend/last_entries/")
        if response.status_code == 200:
            return response.json()
        else:
            st.write("Error retrieving data!")
            return []
    

    # Check if 'data' is already in the session state
    if 'contra_goldendragon' not in st.session_state:
        st.session_state.contra_goldendragon = goldendragon_contratrend()

    # Display the data table
    st.write("### Golden Dragon Contratrend Signals")
    st.table(st.session_state.contra_goldendragon)

    #plot price action against moving averages
    st.header('Golden Dragon price action and moving averages')
    # Filtering the DataFrame for dates starting from 2023
    df=pd.read_csv('golden_dragon.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Date'] >= pd.Timestamp('2023-01-01')]

    # Plotting Close price and the moving averages for the filtered date range
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price', color='blue')
    plt.plot(df_filtered['Date'], df_filtered['3EMA_Close'], label='3-Months EMA', color='red', linestyle='--')
    plt.plot(df_filtered['Date'], df_filtered['6EMA_Close'], label='6-Months EMA', color='green', linestyle='-.')
    plt.plot(df_filtered['Date'], df_filtered['9EMA_Close'], label='9-Months EMA', color='purple', linestyle=':')

    # Adding title and labels
    plt.title('Close Price and Moving Averages (2023 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    st.pyplot(plt)
#------------------------------------------------------------------------------------------------------------------------------------------>
#contratrend logic for all index/instruments

    
        


    

def main():
    st.sidebar.title("Index/instrument")

    # Buttons for each main category
    if st.sidebar.button("Nifty"):
        st.session_state['current_page'] = 'Nifty'
    if st.sidebar.button("s&p"):
        st.session_state['current_page'] = 's&p'
    if st.sidebar.button("gold"):
        st.session_state['current_page'] = 'gold'
    if st.sidebar.button("Nasdaq"):
        st.session_state['current_page'] = 'Nasdaq'
    if st.sidebar.button("Golden Dragon"):
        st.session_state['current_page'] = 'Golden'
        

    # Display the selected page
    if st.session_state.get('current_page') == 'Nifty':
        st.title(" Nifty PE Signals")
        pe_signals()
        st.title("Nifty Market Momentum")
        market_momentum()
        st.title("Nifty Volatility")
        volatility()
    elif st.session_state.get('current_page') == 's&p':
        st.title("s&p Market Momentum")
        snp()
    elif st.session_state.get('current_page') == 'gold':
        st.title("Gold Market Momentum")
        gold()
    elif st.session_state.get('current_page') == 'Nasdaq':
        st.title("Nasdaq Market Momentum")
        nasdaq()
    elif st.session_state.get('current_page') == 'Golden':
        st.title("Golden dragon Market Momentum")
        golden_dragon()

if __name__ == "__main__":
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Nifty'
    main()
