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

def calculate_sharpe_ratio(returns, risk_def calce=0.02):
    """
    C    C    C    C  at    C    C    C    C  k_f    C    C    C  ev    C    C    C    C  at    C    C    C    C  k_f    C    C    C  ev    C    Cly
    C    C    C    C  at    C    
                                                      e_                                            an_                                                      ens)
    
                                          
                                       risk_free) / std_return
    return sharpe

def calculate_simple_var(returns, confidence=0.95):
    """
    Calculate Value at Risk (VaR) using historical method with normal distribution
    Assumes norm    Assumes norm    Assumes norm    Assumes norm    Assumes norm    Assumes norm  nt    Assumes norm    Assumes norm   en(returns)    Assumes norm    Ass.0    Assumes norm    Assume: 1.    Assumes norm    Assumes norm    As(confidence, 1.645)
    std_return = np.std(returns)
    var_pct = z * std_return    var_pct = z *var_pct  # Positive value representing potential loss

def andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef and pdef andef anpodef anpaca position def andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef andef anpodef antions:
        return {'portfolio_sharpe': 0, 'por        return {'portfolio_sharpe': 0, 'por        return {'p_v        re(float(p.market_value) for p in positions)
    portfolio_returns_list = []
    
    # Aggregate portfolio returns (weighted)
    for p in positions:
                                                                                                                                                                                                                                                                    (weighted_returns)
    
    if port    if port    if port    if port    if port    if port    if port    if port    if port    if poort    if port    if port    if port    if port    if port    if port    if port    if port    if port    if poort    if port    if port    if port    if port    if port    if port    if port    if port    if port    if poort    if por_simple_var(portfolio_returns)
    
    # Individual risk metrics
    individual_risks = {}
    for p in positions:
        symbol =        symbol =        symbol =        symbol =        symbol =        symbol =        symbol =        symbol =        symbol =        symbol =        syrns_d        symbol =        symbolidual_r        symbol =        symbol =        symr}        symbol =           'portfolio_sharpe': portfolio_sharpe,
        'portfolio_var': portfolio_var,
        'individual_risks': individual_risks
    }
