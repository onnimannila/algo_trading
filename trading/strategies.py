# Trading strategies module: implements 3 strategies (SMA, Markov, LSTM) for AAPL, TSLA, NVDA
# Generates trading signals independently per stock and aggregates them

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np
from conf import config
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Authorize stock fetching through API
client = StockHistoricalDataClient(config.API_Key, config.Secret_key)

# Fetch 500 days of historical data for AAPL, TSLA, NVDA
start_date = datetime.today() - timedelta(days=500)
end_date = datetime.today().strftime('%Y-%m-%d')
tickers = ["AAPL", "TSLA", "NVDA"]

request_params = StockBarsRequest(
    symbol_or_symbols=tickers,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date
)

# Fetch and process data
response = client.get_stock_bars(request_params)
df = pd.DataFrame(response.df).reset_index()
df['timestamp'] = pd.to_datetime(df.timestamp).dt.tz_localize(None).dt.date
df_small = df.drop(['open', 'high', 'low', 'volume', 'trade_count', 'vwap'], axis=1)
df_small["pct_change"] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

# ===== STRATEGY 1: Simple Moving Average (SMA-50) =====
df_small["SMA_50"] = df_small['close'].rolling(window=50).mean()
df_small.dropna(inplace=True)
df_small["long_short"] = np.where(df_small.close > df_small.SMA_50, 'buy', 'sell')

# Get latest SMA signals per stock
df_buysell = df_small.sort_values(by='timestamp').groupby('symbol').tail(1)
print("\n[STRATEGY 1] SMA-50 Signals:")
print(df_buysell[['symbol', 'long_short']])

# ===== STRATEGY 2: Markov Chain =====
markov_matrices = {}
markov_signals = {}
state_series_dict = {}

print("\n[STRATEGY 2] Markov Chain Matrices:")
for ticker in tickers:
    df_ticker_all = df[df['symbol'] == ticker]
    Close_Gap = df_ticker_all['close'].pct_change()
    
    new_dataset_df = pd.DataFrame({'Close_Gap': Close_Gap})
    new_dataset_df.dropna(inplace=True)
    
    # Classify market states
    new_dataset_df['state_close'] = new_dataset_df['Close_Gap'].apply(
        lambda x: 'Up' if x > 0.005 else ('Steady' if 0.000 < x <= 0.005 else 'Down')
    )
    new_dataset_df['priorstate_close'] = new_dataset_df['state_close'].shift(1)
    new_dataset_df.dropna(inplace=True)
    
    # Build transition matrix
    states_close = new_dataset_df[['priorstate_close', 'state_close']]
    states_matrix_close = states_close.groupby(['priorstate_close', 'state_close']).size().unstack().fillna(0)
    markov_matrix_close = states_matrix_close.apply(lambda x: x / float(x.sum()) if x.sum() > 0 else 0, axis=1)
    
    combined_matrix = pd.concat([markov_matrix_close], axis=1)
    combined_matrix.columns = ['state_close_down', 'state_close_steady', 'state_close_up']
    combined_matrix.index = ['priorstate_down', 'priorstate_steady', 'priorstate_up']
    
    print(f"{ticker}:\n{combined_matrix}\n")
    markov_matrices[ticker] = combined_matrix
    
    # Get signal from latest state
    patterns_df = pd.DataFrame(new_dataset_df.iloc[-1]).T
    patterns_df['probability_close'] = patterns_df.apply(
        lambda row: combined_matrix.loc[f"priorstate_{row.get('priorstate_close', '').lower()}", 
                                       f"state_close_{row.get('state_close', '').lower()}"]
        if row.get('state_close') and row.get('priorstate_close') else None, axis=1
    )
    
    markov_buy_sell_hold = patterns_df['state_close'].apply(
        lambda x: 'buy' if x == 'Up' else ('hold' if x == 'Steady' else 'sell')
    )
    markov_signals[ticker] = markov_buy_sell_hold.iloc[0]
    state_series_dict[ticker] = new_dataset_df['state_close']

# ===== STRATEGY 3: LSTM Neural Network =====
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

lstm_signals = {}
returns_series_dict = {}

print("\n[STRATEGY 3] LSTM Training (3 stocks)...")
for ticker in tickers:
    df_ticker = df_small[df_small['symbol'] == ticker]
    lstm_df = pd.DataFrame(df_ticker['close'])
    lstm_df["returns"] = (lstm_df['close'] - lstm_df['close'].shift(1)) / lstm_df['close'].shift(1)
    lstm_df = lstm_df.drop(['close'], axis=1)
    lstm_df.dropna(inplace=True)
    
    returns_series_dict[ticker] = lstm_df["returns"]
    
    # Normalize and prepare sequences
    scaler = MinMaxScaler(feature_range=(0, 1))
    returns_scaled = scaler.fit_transform(lstm_df)
    X, y = create_sequences(returns_scaled, time_step=60)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    train_size = int(len(returns_scaled) * 0.75)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train LSTM model
    lstm_model = Sequential([
        LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(units=1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    # Predict tomorrow's return
    last_60_days = lstm_df.values[-60:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days).reshape(1, 60, 1)
    predicted_return = scaler.inverse_transform(lstm_model.predict(last_60_days_scaled, verbose=0))
    today_return = lstm_df['returns'].values[-1]
    
    lstm_signals[ticker] = "buy" if predicted_return[0][0] > today_return else "sell"
    print(f"{ticker} - LSTM signal: {lstm_signals[ticker]}")

# ===== SIGNAL AGGREGATION =====
print("\n[AGGREGATION] Final Signals (majority vote: 2/3):")
signals_dict = {}

for ticker in tickers:
    sma_signal = df_buysell[df_buysell['symbol'] == ticker]['long_short'].iloc[0]
    markov_signal = markov_signals[ticker]
    lstm_signal = lstm_signals[ticker]
    
    signals_list = [sma_signal, markov_signal, lstm_signal]
    buy_signals = signals_list.count('buy')
    sell_signals = signals_list.count('sell')
    
    if buy_signals >= 2:
        final_decision = "buy"
    elif sell_signals >= 2:
        final_decision = "sell"
    else:
        final_decision = "hold"
    
    signals_dict[ticker] = {
        'sma': sma_signal,
        'markov': markov_signal,
        'lstm': lstm_signal,
        'final': final_decision
    }
    
    print(f"{ticker}: SMA={sma_signal} + Markov={markov_signal} + LSTM={lstm_signal} -> {final_decision.upper()}")

# ===== EXPORT VARIABLES FOR OTHER FILES =====
signals = signals_dict
returns_series = returns_series_dict
state_series = state_series_dict
price_data = df_small
