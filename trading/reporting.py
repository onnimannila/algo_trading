#reporting module: summarize strategies, calculate metrics, generate visualizations, export PDF

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from conf import config
import trading.strategies as strategies
from trading.position_risk import calculate_sharpe_ratio, calculate_simple_var, analyze_portfolio_weights
from alpaca.trading.client import TradingClient
from trading.position_risk import get_positions

# --- FIX PATH (since reporting.py is in subfolder) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def gather_strategy_data():
    """Collect all strategy outputs"""
    tickers = ['AAPL', 'TSLA', 'NVDA']
    
    strategy_summary = {}
    for ticker in tickers:
        signals = strategies.signals[ticker]
        returns = strategies.returns_series[ticker]
        
        strategy_summary[ticker] = {
            'sma_signal': signals['sma'],
            'markov_signal': signals['markov'],
            'lstm_signal': signals['lstm'],
            'final_signal': signals['final'],
            'sharpe': calculate_sharpe_ratio(returns),
            'var': calculate_simple_var(returns),
            'mean_return': returns.mean(),
            'std_return': returns.std()
        }
    
    return strategy_summary

def generate_visualizations(strategy_summary, trade_decisions, portfolio_value):
    """Generate visualization plots: time series + summary stats"""
    tickers = ['AAPL', 'TSLA', 'NVDA']
    price_data = strategies.price_data
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Trading Algorithm Report - {datetime.now().strftime("%Y-%m-%d")}', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme for stocks
    colors_map = {'AAPL': '#1f77b4', 'TSLA': '#ff7f0e', 'NVDA': '#2ca02c'}
    
    # --- PLOT 1: Historical Price Time Series ---
    ax1 = plt.subplot(2, 1, 1)
    for ticker in tickers:
        ticker_data = price_data[price_data['symbol'] == ticker].sort_values('timestamp')
        ax1.plot(range(len(ticker_data)), ticker_data['close'].values, label=ticker, color=colors_map[ticker], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Days (Last 500)')
    ax1.set_ylabel('Price (USD)')
    ax1.set_title('Historical Stock Prices (500-Day Window)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # --- PLOT 2: Historical Returns Time Series ---
    ax2 = plt.subplot(2, 1, 2)
    for ticker in tickers:
        returns = strategies.returns_series[ticker]
        ax2.plot(range(len(returns)), returns.values * 100, label=ticker, color=colors_map[ticker], linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Days (Last 500)')
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_title('Historical Daily Returns')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def export_pdf_report(fig, strategy_summary, trade_decisions, portfolio_value):
    """Export analytics to PDF with timestamp in /results folder"""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, f"algo_results_{timestamp}.pdf")
    
    try:
        fig.savefig(filename, format='pdf', dpi=150, bbox_inches='tight')
        print(f"\n[OK] PDF Report exported: results/algo_results_{timestamp}.pdf")
    except Exception as e:
        print(f"[ERROR] Error exporting PDF: {e}")
        # Fallback to PNG if PDF fails
        filename = os.path.join(results_dir, f"algo_results_{timestamp}.png")
        fig.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        print(f"[OK] Fallback: PNG exported: results/algo_results_{timestamp}.png")
    
    return filename

def generate_and_export():
    """Main reporting function: collect data, generate plots, export PDF"""
    print("\n===== REPORTING ANALYSIS =====")
    
    # Gather strategy data
    strategy_summary = gather_strategy_data()
    
    # Generate visualizations (includes time series plots)
    fig = generate_visualizations(strategy_summary, {}, 0)
    
    # Add text summary on figure
    fig.text(0.05, 0.95, 'RISK METRICS & SIGNALS SUMMARY', fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    tickers = ['AAPL', 'TSLA', 'NVDA']
    y_pos = 0.90
    
    for ticker in tickers:
        data = strategy_summary[ticker]
        signal_text = f"{ticker}: {data['final_signal'].upper()}"
        metrics_text = f"  Vol: {data['std_return']*100:.2f}% | VaR (95%): {data['var']*100:.2f}% | Mean Return: {data['mean_return']*100:.3f}%"
        
        fig.text(0.05, y_pos, signal_text, fontsize=10, fontweight='bold', transform=fig.transFigure, color='darkgreen' if data['final_signal'] == 'buy' else 'darkred' if data['final_signal'] == 'sell' else 'gray')
        fig.text(0.05, y_pos - 0.03, metrics_text, fontsize=9, transform=fig.transFigure)
        y_pos -= 0.08
    
    # Print summary to console
    print("\nStrategy Summary:")
    for ticker, data in strategy_summary.items():
        print(f"\n{ticker}:")
        print(f"  Signal: {data['final_signal'].upper()}")
        print(f"  Volatility: {data['std_return']*100:.2f}%")
        print(f"  VaR (95%): {data['var']*100:.2f}%")
        print(f"  Mean Return: {data['mean_return']*100:.3f}%")
    
    # Get current positions for portfolio view
    trading_client = TradingClient(config.API_Key, config.Secret_key, paper=True)
    positions = get_positions(trading_client)
    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    
    # Analyze portfolio weights
    weights = analyze_portfolio_weights(positions)
    print(f"\nPortfolio Value: ${portfolio_value:.2f}")
    for w in weights:
        print(f"  {w['symbol']}: {w['weight']*100:.2f}%")
    
    # Export PDF
    filename = export_pdf_report(fig, strategy_summary, {}, portfolio_value)
    
    plt.close(fig)
    
    return filename

# If run directly (not imported), execute
if __name__ == "__main__":
    generate_and_export()
