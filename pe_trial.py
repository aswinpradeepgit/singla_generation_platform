import pandas as pd
new_df=pd.read_csv("data_pe.csv")
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

import pandas as pd
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


df10.to_csv("pe_signal.csv",index=False)