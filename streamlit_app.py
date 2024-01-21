import streamlit as st
import requests

def fetch_last_5_entries():
    response = requests.get("http://127.0.0.1:8000/last_entries/")
    if response.status_code == 200:
        return response.json()["data"]
    else:
        st.write("Error retrieving data!")
        return []

def main():
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
        response = requests.post(f"http://127.0.0.1:8000/add_entry/?date={date}&close_price={close_price}")
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
        response = requests.post(f"http://127.0.0.1:8000/delete_entry/?date={delete_date}")
        if response.status_code == 200:
            message = response.json().get("message", "No message returned from server.")
            st.write(message)

            # Update the data in session state
            st.session_state.data = fetch_last_5_entries()

            # Use st.experimental_rerun to refresh the page
            st.experimental_rerun()
        else:
            st.write(f"Error: Received status code {response.status_code} from server.")

    st.write("### PE Signals")

if __name__ == '__main__':
    main()
