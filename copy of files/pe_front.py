import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# Function to fetch last entries for a given signal type
def fetch_last_entries(signal_type):
    response = requests.get(f"http://127.0.0.1:8000/pe/last_entries/{signal_type}")
    return response.json() if response.status_code == 200 else []


#functions for plotting


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

def main():
    st.title("Financial Signals App")

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

if __name__ == "__main__":
    main()
