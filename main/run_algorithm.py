# Main orchestrator: runs the complete trading algorithm pipeline
# Pipeline: strategies -> execution -> reporting
# PAPER TRADING ONLY - no real trades executed

import sys
import os
from datetime import datetime

print("=" * 60)
print("ALPACA TRADING ALGORITHM - PAPER TRADING MODE")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# --- Fix imports path ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Step 1: Import and run strategies
    print("\n[STEP 1/3] Running Strategy Analysis...")
    print("-" * 60)
    from trading import strategies
    print("[OK] Strategies executed: SMA, Markov Chain, LSTM (all 3 stocks)")
    
    # Step 2: Run execution logic
    print("\n[STEP 2/3] Running Execution Analysis...")
    print("-" * 60)
    from trading.execution import run_execution
    trade_decisions, simulated_positions, portfolio_value = run_execution()
    print("[OK] Execution completed: simulated trades, portfolio balancing (80% limit)")
    
    # Step 3: Generate reporting
    print("\n[STEP 3/3] Generating Report...")
    print("-" * 60)
    from trading.reporting import generate_and_export
    report_filename = generate_and_export()
    print("[OK] Report generated and exported")
    
    # Final summary
    print("\n" + "=" * 60)
    print("ALGORITHM EXECUTION SUMMARY")
    print("=" * 60)
    print("[SUCCESS] All pipeline stages completed successfully")
    print(f"[REPORT] {report_filename}")
    print(f"[HOLDINGS] Simulated (paper): {sum(1 for s in simulated_positions.values() if s > 0)} positions")
    print(f"[PORTFOLIO] Value: ${portfolio_value:.2f}")
    print("[NOTICE] All trades are SIMULATED (paper trading) - NO REAL EXECUTION")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
