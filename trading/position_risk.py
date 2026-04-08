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

def analyze_portfolio_weights(positions, threshold=0.8):
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio: (mean return - risk_free_rate) / std dev of returns
    """
    if len(returns) == 0:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

def calculate_simple_var(returns, confidence=0.95):
    """
    Calculate simple VaR: -z * std_dev (assuming normal distribution)
    z for 95% = 1.645
    Returns positive value as loss amount
    """
    if len(returns) == 0:
        return 0.0
    z = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
    std_return = np.std(returns)
    var = z * std_return
    return var  # positive, as potential loss

def analyze_portfolio_risk(positions, returns_dict):
    """
    Analyze risk metrics for portfolio and individual positions
    """
    total_market_value = sum(float(p.market_value) for p in positions)
    portfolio_returns = []

    # Aggregate portfolio returns (weighted)
    for p in positions:
        symbol = p.symbol
        if symbol in returns_dict:
            weight = float(p.market_value) / total_market_value
            portfolio_returns.extend(returns_dict[symbol] * weight)

    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
    portfolio_var = calculate_simple_var(portfolio_returns)

    individual_risks = {}
    for p in positions:
        symbol = p.symbol
        if symbol in returns_dict:
            sharpe = calculate_sharpe_ratio(returns_dict[symbol])
            var = calculate_simple_var(returns_dict[symbol])
            individual_risks[symbol] = {'sharpe': sharpe, 'var': var}

    return {
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_var': portfolio_var,
        'individual_risks': individual_risks
    }
