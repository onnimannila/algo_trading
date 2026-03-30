#sub script to check current position and risk before executing a trade

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
import sys
import os
from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from conf import config
from datetime import datetime, timedelta
import pickle
import json as json
import trading as strategies


api = trade_api.REST(config.API_Key, config.Secret_key, config.alpaca_base_URL)

# --- FIX PATH (since execution.py is in subfolder) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_positions(trading_client):
    return trading_client.get_all_positions()

def analyze_portfolio_weights(positions, threshold=0.3):
    # total portfolio market value
    total_market_value = sum(float(p.market_value) for p in positions)

    results = []

    for p in positions:
        symbol = p.symbol
        market_value = float(p.market_value)

        weight = market_value / total_market_value if total_market_value > 0 else 0

        warning = weight > threshold

        results.append({
            "symbol": symbol,
            "market_value": market_value,
            "weight": weight,
            "warning": warning
        })

    return results
