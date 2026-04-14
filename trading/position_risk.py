# Position and risk analysis module
# Calculates portfolio weights, Sharpe ratio, VaR, and other risk metrics

import sys
import os
import pandas as pd
import numpy as np
from conf import config

# --- FIX PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_positions(trading_client):
    """Fetch all open positions from trading client"""
    return trading_client.get_all_positions()

def analyze_portfolio_weights(positions, threshold=0.8):
    """
    Analyze portfolio weights per position
    Flags positions exceeding 80% threshold
    """
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
    
    daily_risk_free = risk_free_rate / 252
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - daily_risk_free) / std_return
    return sharpe

def calculate_simple_var(returns, confidence=0.95):
    """
    Calculate Value at Risk (VaR) using normal distribution
    Returns positive value as percentage loss
    """
    if len(returns) == 0:
        return 0.0
    
    z_scores = {0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(confidence, 1.645)
    std_return = np.std(returns)
    var_pct = z * std_return
    
    return var_pct

def analyze_portfolio_risk(positions, returns_dict):
    """
    Comprehensive risk analysis for entire portfolio and individual positions
    """
    if not positions:
        return {'portfolio_sharpe': 0, 'portfolio_var': 0, 'individual_risks': {}}
    
    total_market_value = sum(float(p.market_value) for p in positions)
    portfolio_returns_list = []
    
    # Aggregate portfolio returns (weighted)
    for p in positions:
        symbol = p.symbol
        if symbol in returns_dict:
            weight = float(p.market_value) / total_market_value if total_market_value > 0 else 0
            weighted_returns = returns_dict[symbol] * weight
            portfolio_returns_list.append(weighted_returns)
    
    if portfolio_returns_list:
        portfolio_returns = pd.concat(portfolio_returns_list) if isinstance(portfolio_returns_list[0], pd.Series) else np.array(portfolio_returns_list).flatten()
    else:
        portfolio_returns = np.array([])
    
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
    portfolio_var = calculate_simple_var(portfolio_returns)
    
    # Individual risk metrics
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
