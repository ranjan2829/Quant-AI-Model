import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quant_strategy import DataPreprocessor, ModelTrainer, SignalGenerator, PositionSizer, TradingStrategy
import joblib
from datetime import datetime

# Create output directory for results
os.makedirs('results', exist_ok=True)

# Initialize the trading strategy
strategy = TradingStrategy(data_dir='data', models_dir='models')

# Train the strategy models
print("Training strategy models...")
training_results = strategy.train_strategy()

# List of symbols to backtest
indices = ['NSEI', 'BSESN', 'NSEBANK', 'CNXIT', 'CNXPHARMA', 'CNXAUTO']
stocks = [f.replace('.csv', '') for f in os.listdir('data') if f.endswith('.csv') and '_NS' in f]

# Backtest parameters
initial_capital = 1000000  # 1 million INR

# Results storage
backtest_results = {}
performance_metrics = {}

# Backtest indices
print("\nBacktesting indices...")
for symbol in indices:
    print(f"Backtesting {symbol}...")
    backtest, performance = strategy.backtest_strategy(symbol, capital=initial_capital)
    
    if backtest is not None:
        backtest_results[symbol] = backtest
        performance_metrics[symbol] = performance
        
        # Save backtest results
        backtest.to_csv(f"results/{symbol}_backtest.csv")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(backtest['Equity'])
        plt.title(f"Equity Curve for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Equity (INR)")
        plt.grid(True)
        plt.savefig(f"results/{symbol}_equity_curve.png")
        plt.close()
        
        print(f"  Total Return: {performance['total_return']:.2%}")
        print(f"  Annual Return: {performance['annual_return']:.2%}")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"  Win Rate: {performance['win_rate']:.2%}")

# Backtest stocks
print("\nBacktesting stocks...")
for symbol in stocks:
    print(f"Backtesting {symbol}...")
    backtest, performance = strategy.backtest_strategy(symbol, capital=initial_capital)
    
    if backtest is not None:
        backtest_results[symbol] = backtest
        performance_metrics[symbol] = performance
        
        # Save backtest results
        backtest.to_csv(f"results/{symbol}_backtest.csv")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(backtest['Equity'])
        plt.title(f"Equity Curve for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Equity (INR)")
        plt.grid(True)
        plt.savefig(f"results/{symbol}_equity_curve.png")
        plt.close()
        
        print(f"  Total Return: {performance['total_return']:.2%}")
        print(f"  Annual Return: {performance['annual_return']:.2%}")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"  Win Rate: {performance['win_rate']:.2%}")

# Create summary report
summary = pd.DataFrame(columns=[
    'Symbol', 'Total Return', 'Annual Return', 
    'Sharpe Ratio', 'Max Drawdown', 'Win Rate'
])

for i, (symbol, perf) in enumerate(performance_metrics.items()):
    summary.loc[i] = [
        symbol,
        f"{perf['total_return']:.2%}",
        f"{perf['annual_return']:.2%}",
        f"{perf['sharpe_ratio']:.2f}",
        f"{perf['max_drawdown']:.2%}",
        f"{perf['win_rate']:.2%}"
    ]

# Save summary report
summary.to_csv("results/backtest_summary.csv", index=False)

# Generate trading plans for next day
print("\nGenerating trading plans for next day...")
trading_plans = {}

for symbol in indices + stocks:
    plan = strategy.generate_trading_plan(symbol, capital=initial_capital)
    if plan is not None:
        trading_plans[symbol] = plan
        print(f"Trading plan for {symbol}: {'BUY' if plan['signal'] == 1 else 'HOLD/SELL'} (Confidence: {plan['confidence']:.2f})")

# Save trading plans
with open("results/trading_plans.txt", "w") as f:
    f.write(f"Trading Plans for {datetime.now().strftime('%Y-%m-%d')}\n")
    f.write("="*50 + "\n\n")
    
    for symbol, plan in trading_plans.items():
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Date: {plan['date']}\n")
        f.write(f"Current Price: {plan['current_price']:.2f}\n")
        f.write(f"Signal: {'BUY' if plan['signal'] == 1 else 'HOLD/SELL'}\n")
        f.write(f"Confidence: {plan['confidence']:.2f}\n")
        f.write(f"Stop Loss: {plan['stop_loss']:.2f}\n")
        f.write(f"Position Size (Shares): {plan['position']['shares']}\n")
        f.write(f"Position Value: {plan['position']['position_value']:.2f}\n")
        f.write(f"Risk Amount: {plan['position']['risk_amount']:.2f}\n")
        f.write("\nModel Signals:\n")
        for model, signal in plan['model_signals'].items():
            f.write(f"  {model}: {'BUY' if signal == 1 else 'SELL'} (Prob: {plan['model_probabilities'][model]:.2f})\n")
        f.write("\n" + "-"*30 + "\n\n")

print("\nBacktesting complete. Results saved to 'results' directory.")
