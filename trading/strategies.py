#subscript to implement trading strategies 

import alpaca_trade_api as trade_api
from alpaca_trade_api import REST
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

#import necessary libraries to plot data

from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from conf import config
from datetime import datetime, timedelta
import json as json

#import needed packages for LSTM strategy
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

api = trade_api.REST(config.API_Key, config.Secret_key, config.alpaca_base_URL)

#authorize stock fetching through API
client = StockHistoricalDataClient(config.API_Key, config.Secret_key)

#parametrize needed variables for requesting market data
#start_date = "2020-01-01"
#shorter time frame in order to get momentum accounted in trading strategies
start_date = datetime.today() - timedelta(days=500)
end_date = datetime.today().strftime('%Y-%m-%d')
tickers = ["AAPL", "TSLA", "SPY"]

#specify stock data request
request_params = StockBarsRequest(
  symbol_or_symbols=tickers,
  timeframe=TimeFrame.Day,
  start=start_date,
  end=end_date
)

#save fetched stock market data
response = client.get_stock_bars(request_params)

#changing the wrapped market data into dataframe 
df2 = response.df

#print (df2)

df = pd.DataFrame(df2)

df = df.reset_index()

#changing the YYYY-MM-DD HH:MM:SS date format to YYYY-MM-DD
df['timestamp'] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
df['timestamp'] = df['timestamp'].dt.date
#print(df)

#dropping unnecessary columns and calculating changes in daily stock prices
df_small = df.drop(['open','high','low','volume','trade_count','vwap'], axis=1)
df_small["pct_change"] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)


#1. trading strategy is defined by simple moving average of 50 days. If price is over the moving average, the stock should be bought and vice versa. 

# 50-day moving average calculation 
df_small["SMA_50"] = df_small['close'].rolling(window=50).mean()
df_small.dropna(inplace=True)

#flag to determine long=True or short=False stock
cond1 = df_small.close > df_small.SMA_50

df_small["long_short"] = np.where(cond1, 'buy' , 'sell')

#separate dataset to two different df

df_aapl = df_small.query('symbol == "AAPL"')
df_tsla = df_small.query('symbol == "TSLA"')

#df_aapl.plot(x="timestamp", y=["SMA_50", "close"],
#        kind="line", figsize=(10, 10))
#plt.title("Apple SMA_50 vs closing price")
#plt.xlabel("Dates")
#plt.ylabel("Price")

#plt.show()

#df_tsla.plot(x="timestamp", y=["SMA_50", "close"],
#        kind="line", figsize=(10, 10))
#plt.title("Tesla SMA_50 vs closing price")
#plt.xlabel("Dates")
#plt.ylabel("Price")

#plt.show()

df_buysell= df_small.sort_values(by='timestamp', ascending=False).head(2)

#the buy or sell order to be executed based on SMA50 for each stock
print(df_buysell[['symbol', 'long_short']])

#2. trading startegy is Markov chain. It only predicts the probability of market moves in a simple way. 
# more info: https://medium.com/analytics-vidhya/how-to-build-a-market-simulator-using-markov-chains-and-python-b925a106b1c4
#and https://unofficed.com/courses/markov-model-application-of-markov-chain-in-stock-market/lessons/markov-chains-in-stock-market-using-python-getting-transition-matrix/

#Loop through each stock ticker
#tickers = ['TSLA', 'AAPL']
tickers = ['AAPL']  # Adjust this list as needed
markov_matrices = {}
datasets_of_patterns = {}

for ticker in tickers:
    df_ticker_all = df[df['symbol'] == ticker]

    # Creation of several changes in time series
    Close_Gap = df_ticker_all['close'].pct_change()
    #Volume_Gap = df_ticker_all['volume'].pct_change()
    #Daily_Change = (df_ticker_all['close'] - df_ticker_all['open']) / df_ticker_all['open']

    # Create the DataFrame directly
    new_dataset_df = pd.DataFrame({
        'Close_Gap': Close_Gap
        #'Volume_Gap': Volume_Gap,
        #'Daily_Change': Daily_Change,
    })

    # Ensure no NaN values after percentage change
    new_dataset_df.dropna(inplace=True)

    # Classify states
    new_dataset_df['state_close'] = new_dataset_df['Close_Gap'].apply(lambda x: 'Up' if x > 0.001 else ('Steady' if 0.000 < x <= 0.001 else 'Down'))

    # Shift states and drop NA
    new_dataset_df['priorstate_close'] = new_dataset_df['state_close'].shift(1)
    new_dataset_df.dropna(inplace=True)

    # Create transition matrices
    states_close = new_dataset_df[['priorstate_close', 'state_close']].dropna()
    states_matrix_close = states_close.groupby(['priorstate_close', 'state_close']).size().unstack().fillna(0)

    #printing the transition matrix (frequencies of events in dataframe)
    #print(states_matrix_close)

    # Calculate Markov matrices
    markov_matrix_close = states_matrix_close.apply(lambda x: x / float(x.sum()) if x.sum() > 0 else 0, axis=1)
    print(markov_matrix_close)
    # Combine matrices into one for output
    combined_matrix = pd.concat([markov_matrix_close], axis=1)

    combined_matrix.columns = [
        'state_close_down', 'state_close_steady','state_close_up'
    ]
    combined_matrix.index = ['priorstate_down', 'priorstate_steady','priorstate_up']

    markov_matrices[ticker] = combined_matrix

# Print Markov matrices
for ticker in tickers:
    print(f"{ticker} Markov Matrix:")
    print(markov_matrices[ticker])
    print("\n")

# Prepare patterns DataFrame for probabilities
patterns_df = pd.DataFrame(new_dataset_df.iloc[-1]).T  # Use the last row of new_dataset_df

# Debugging: Check the contents of patterns_df
print("Patterns DataFrame:")
print(patterns_df)

def get_probability(row, variable):
    state = row.get(f'state_{variable}', None)
    prior_state = row.get(f'priorstate_{variable}', None)

    prior_state_key = f'priorstate_{prior_state.lower()}' if prior_state else None

    if state and prior_state_key:
        lookup_col = f'state_{variable}_{state.strip().lower()}'
        print(f"Lookup Column: {lookup_col}, Prior State Key: {prior_state_key}")

        if prior_state_key in markov_matrices[ticker].index:
            if lookup_col in markov_matrices[ticker].columns:
                try:
                    return markov_matrices[ticker].loc[prior_state_key, lookup_col]
                except KeyError as e:
                    print(f"KeyError: {e} - Check if '{prior_state_key}' exists in the index.")
    return None

patterns_df['probability_close'] = patterns_df.apply(lambda row: get_probability(row, 'close'), axis=1)

markov_buy_sell_hold = patterns_df[['state_close', 'priorstate_close', 'probability_close']]

markov_buy_sell_hold['buy_sell_hold'] = markov_buy_sell_hold['state_close'].apply(
    lambda x: 'buy' if x == 'Up' else ('hold' if x == 'Steady' else 'sell')
)

#3. trading startegy is Long Short-Term Memory (LSTM). 
#LSTM is useful for recognizing patterns in time-series especially in nonlinear data.

#firstly we test and fit the model 
#only AAPL closing prices for analysis
lstm_df = pd.DataFrame(df_aapl['close'])

#change prices to returns so that the min max scaling is useful
lstm_df["returns"] = (lstm_df['close'] - lstm_df['close'].shift(1)) / lstm_df['close'].shift(1)
lstm_df = lstm_df.drop(['close'], axis=1)
lstm_df.dropna(inplace=True)

#Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
returns_scaled = scaler.fit_transform(lstm_df)

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(returns_scaled, time_step=60)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(returns_scaled) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lstm_model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(X_train, y_train, epochs=100, batch_size=32)

train_loss = lstm_model.evaluate(X_train, y_train, verbose=0)
test_loss = lstm_model.evaluate(X_test, y_test, verbose=0)

train_predictions = lstm_model.predict(X_train)
test_predictions = lstm_model.predict(X_test)

train_predictions = scaler.inverse_transform(train_predictions)
y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))

plt.plot(range(len(y_train_unscaled)), y_train_unscaled, color='blue', label='Actual Train Returns')
plt.plot(range(len(train_predictions)), train_predictions, color='red', linestyle='--', label='Predicted Train Returns')

plt.plot(range(len(y_train_unscaled), len(y_train_unscaled) + len(y_test_unscaled)), y_test_unscaled, color='green', label='Actual Test Returns')
plt.plot(range(len(train_predictions), len(train_predictions) + len(test_predictions)), test_predictions, color='orange', linestyle='--', label='Predicted Test Returns')

plt.title('Actual vs Predicted Closing Returns (LSTM)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Returns (%)', fontsize=14)
plt.legend()
plt.grid(True)

#secondly, the actual prediction for tomorrows return
last_60_days = lstm_df.values[-60:]
last_60_days = last_60_days.reshape(-1, 1)

last_60_days_scaled = scaler.transform(last_60_days)

X_test_input = last_60_days_scaled.reshape(1, 60, 1)

predicted_return = lstm_model.predict(X_test_input)

predicted_return_unscaled = scaler.inverse_transform(predicted_return)

today_return = lstm_df['returns'].values[-1]

print(f"Today's return: ${today_return:.2f}")
print(f"Predicted Tomorrow's return: ${predicted_return_unscaled[0][0]:.2f}")

if predicted_return_unscaled[0][0] > today_return:
    print("buy")
else:
    print("sell")

# Step 1: Collect signals from each strategy
sma_signal = df_buysell[['symbol', 'long_short']].iloc[0]['long_short']

markov_signal = markov_buy_sell_hold['buy_sell_hold'].iloc[0]

lstm_signal = "buy" if predicted_return_unscaled[0][0] > today_return else "sell"

signals = [sma_signal, markov_signal, lstm_signal]

buy_signals = signals.count('buy')
sell_signals = signals.count('sell')

if buy_signals >= 2:
    final_decision = "buy"
elif sell_signals >= 2:
    final_decision = "sell"
else:
    final_decision = "hold"

print(f"SMA Signal: {sma_signal}")
print(f"Markov Signal: {markov_signal}")
print(f"LSTM Signal: {lstm_signal}")
print(f"Final Decision: {final_decision}")

# ===== EXPORT VARIABLES FOR OTHER FILES =====

# Use AAPL as main trading asset
returns_series = df_aapl["pct_change"].dropna()

# Align states with returns
state_series = new_dataset_df['state_close']

# Export signals and matrices
signals = signals
markov_matrices = markov_matrices
