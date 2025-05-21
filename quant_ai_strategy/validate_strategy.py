import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# Create directory for validation results
os.makedirs('validation', exist_ok=True)

# Load backtest summary
summary_file = "results/backtest_summary.csv"
if os.path.exists(summary_file):
    summary = pd.read_csv(summary_file)
    print("Loaded backtest summary with results for", len(summary), "symbols")
else:
    print("Backtest summary file not found")
    summary = None

# Load individual backtest results
backtest_files = glob.glob("results/*_backtest.csv")
backtest_results = {}

for file in backtest_files:
    symbol = file.split('/')[-1].replace('_backtest.csv', '')
    backtest_results[symbol] = pd.read_csv(file)
    backtest_results[symbol]['Date'] = pd.to_datetime(backtest_results[symbol]['Date'])
    backtest_results[symbol].set_index('Date', inplace=True)
    print(f"Loaded backtest results for {symbol}")

# Validation functions
def calculate_risk_metrics(returns):
    """Calculate advanced risk metrics for a return series"""
    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    
    # Downside deviation (semi-deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    # Sortino ratio (using 0 as minimum acceptable return)
    sortino_ratio = returns.mean() * 252 / downside_deviation if downside_deviation != 0 else 0
    
    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(returns, 5)
    
    # Conditional VaR (CVaR) / Expected Shortfall - 95% confidence
    cvar_95 = returns[returns <= var_95].mean()
    
    # Calmar ratio (annualized return / maximum drawdown)
    calmar_ratio = returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Positive periods ratio
    positive_periods = len(returns[returns > 0]) / len(returns)
    
    return {
        'annual_volatility': annual_vol,
        'max_drawdown': max_drawdown,
        'downside_deviation': downside_deviation,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'calmar_ratio': calmar_ratio,
        'positive_periods': positive_periods
    }

def analyze_robustness(backtest_results):
    """Analyze strategy robustness across different market conditions"""
    # Identify market regimes (bull, bear, sideways)
    market_regimes = {}
    
    for symbol, df in backtest_results.items():
        # Calculate 20-day returns for regime identification
        df['Market_20d_Return'] = df['Close'].pct_change(20)
        
        # Define regimes
        df['Market_Regime'] = 'Sideways'
        df.loc[df['Market_20d_Return'] > 0.05, 'Market_Regime'] = 'Bull'
        df.loc[df['Market_20d_Return'] < -0.05, 'Market_Regime'] = 'Bear'
        
        # Calculate strategy returns in each regime
        bull_returns = df[df['Market_Regime'] == 'Bull']['Returns'].mean()
        bear_returns = df[df['Market_Regime'] == 'Bear']['Returns'].mean()
        sideways_returns = df[df['Market_Regime'] == 'Sideways']['Returns'].mean()
        
        market_regimes[symbol] = {
            'bull_returns': bull_returns,
            'bear_returns': bear_returns,
            'sideways_returns': sideways_returns,
            'bull_days': len(df[df['Market_Regime'] == 'Bull']),
            'bear_days': len(df[df['Market_Regime'] == 'Bear']),
            'sideways_days': len(df[df['Market_Regime'] == 'Sideways'])
        }
    
    return market_regimes

def analyze_sector_performance(backtest_results, summary):
    """Analyze strategy performance by sector"""
    # Define sectors for stocks
    sectors = {
        'RELIANCE_NS': 'Energy',
        'TCS_NS': 'IT',
        'HDFCBANK_NS': 'Banking',
        'INFY_NS': 'IT',
        'HINDUNILVR_NS': 'FMCG',
        'ICICIBANK_NS': 'Banking',
        'SBIN_NS': 'Banking',
        'BAJFINANCE_NS': 'Financial Services',
        'BHARTIARTL_NS': 'Telecom',
        'KOTAKBANK_NS': 'Banking',
        'WIPRO_NS': 'IT',
        'AXISBANK_NS': 'Banking',
        'MARUTI_NS': 'Auto',
        'SUNPHARMA_NS': 'Pharma',
        'TATAMOTORS_NS': 'Auto'
    }
    
    # Group performance by sector
    sector_performance = {}
    
    for symbol in backtest_results.keys():
        if symbol in sectors:
            sector = sectors[symbol]
            if sector not in sector_performance:
                sector_performance[sector] = []
            
            # Get performance metrics from summary
            if summary is not None:
                symbol_summary = summary[summary['Symbol'] == symbol]
                if not symbol_summary.empty:
                    total_return = symbol_summary['Total Return'].values[0]
                    # Remove % sign and convert to float
                    if isinstance(total_return, str):
                        total_return = float(total_return.strip('%')) / 100
                    
                    sector_performance[sector].append(total_return)
    
    # Calculate average performance by sector
    sector_avg_performance = {}
    for sector, returns in sector_performance.items():
        if returns:
            sector_avg_performance[sector] = sum(returns) / len(returns)
    
    return sector_avg_performance

def analyze_correlation(backtest_results):
    """Analyze correlation between strategy returns"""
    # Extract returns for each symbol
    returns_data = {}
    for symbol, df in backtest_results.items():
        if 'Returns' in df.columns:
            returns_data[symbol] = df['Returns']
    
    # Create a DataFrame with all returns
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix

def create_validation_report(summary, backtest_results, risk_metrics, market_regimes, sector_performance, correlation_matrix):
    """Create a comprehensive validation report"""
    report = []
    
    # Report header
    report.append("# Quant AI Strategy Validation Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overall performance summary
    report.append("## 1. Overall Performance Summary")
    if summary is not None:
        # Calculate average metrics
        avg_total_return = summary['Total Return'].str.rstrip('%').astype(float).mean()
        avg_annual_return = summary['Annual Return'].str.rstrip('%').astype(float).mean()
        avg_sharpe = summary['Sharpe Ratio'].astype(float).mean()
        avg_max_dd = summary['Max Drawdown'].str.rstrip('%').astype(float).mean()
        avg_win_rate = summary['Win Rate'].str.rstrip('%').astype(float).mean()
        
        report.append(f"- Average Total Return: {avg_total_return:.2f}%")
        report.append(f"- Average Annual Return: {avg_annual_return:.2f}%")
        report.append(f"- Average Sharpe Ratio: {avg_sharpe:.2f}")
        report.append(f"- Average Maximum Drawdown: {avg_max_dd:.2f}%")
        report.append(f"- Average Win Rate: {avg_win_rate:.2f}%\n")
        
        # Top 3 performing symbols
        top_performers = summary.sort_values(by='Annual Return', ascending=False).head(3)
        report.append("### Top 3 Performing Symbols:")
        for _, row in top_performers.iterrows():
            report.append(f"- {row['Symbol']}: Annual Return {row['Annual Return']}, Sharpe Ratio {row['Sharpe Ratio']}")
        report.append("")
    
    # Risk metrics
    report.append("## 2. Advanced Risk Metrics")
    for symbol, metrics in risk_metrics.items():
        report.append(f"### {symbol}")
        report.append(f"- Annual Volatility: {metrics['annual_volatility']:.4f}")
        report.append(f"- Maximum Drawdown: {metrics['max_drawdown']:.4f}")
        report.append(f"- Downside Deviation: {metrics['downside_deviation']:.4f}")
        report.append(f"- Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        report.append(f"- Value at Risk (95%): {metrics['var_95']:.4f}")
        report.append(f"- Conditional VaR (95%): {metrics['cvar_95']:.4f}")
        report.append(f"- Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        report.append(f"- Positive Periods Ratio: {metrics['positive_periods']:.4f}")
        report.append("")
    
    # Market regime analysis
    report.append("## 3. Market Regime Analysis")
    for symbol, regimes in market_regimes.items():
        report.append(f"### {symbol}")
        report.append(f"- Bull Market Performance: {regimes['bull_returns']:.4f} (Days: {regimes['bull_days']})")
        report.append(f"- Bear Market Performance: {regimes['bear_returns']:.4f} (Days: {regimes['bear_days']})")
        report.append(f"- Sideways Market Performance: {regimes['sideways_returns']:.4f} (Days: {regimes['sideways_days']})")
        report.append("")
    
    # Sector performance
    report.append("## 4. Sector Performance")
    for sector, performance in sector_performance.items():
        report.append(f"- {sector}: {performance:.4f}")
    report.append("")
    
    # Correlation analysis
    report.append("## 5. Strategy Correlation Analysis")
    report.append("The correlation matrix between strategy returns across different symbols indicates the diversification benefit.")
    report.append("Lower correlation values suggest better diversification and potentially more stable overall portfolio performance.")
    report.append("")
    
    # Strategy robustness
    report.append("## 6. Strategy Robustness Assessment")
    
    # Calculate average performance across market regimes
    avg_bull = np.mean([regimes['bull_returns'] for regimes in market_regimes.values()])
    avg_bear = np.mean([regimes['bear_returns'] for regimes in market_regimes.values()])
    avg_sideways = np.mean([regimes['sideways_returns'] for regimes in market_regimes.values()])
    
    report.append(f"- Average Bull Market Performance: {avg_bull:.4f}")
    report.append(f"- Average Bear Market Performance: {avg_bear:.4f}")
    report.append(f"- Average Sideways Market Performance: {avg_sideways:.4f}")
    report.append("")
    
    # Robustness score (simple heuristic)
    robustness_score = 0
    if avg_bull > 0: robustness_score += 1
    if avg_bear > -0.001: robustness_score += 2  # Higher weight for bear market performance
    if avg_sideways > 0: robustness_score += 1
    
    robustness_rating = "Low"
    if robustness_score >= 3:
        robustness_rating = "High"
    elif robustness_score >= 2:
        robustness_rating = "Medium"
    
    report.append(f"- Overall Robustness Rating: {robustness_rating}")
    report.append("")
    
    # Conclusion and recommendations
    report.append("## 7. Conclusion and Recommendations")
    
    # Generate conclusion based on results
    if avg_annual_return > 20 and avg_sharpe > 1.0 and robustness_rating != "Low":
        conclusion = "The Quant AI Strategy demonstrates strong performance with good risk-adjusted returns and acceptable robustness across different market conditions. The strategy appears suitable for implementation with appropriate risk management controls."
    elif avg_annual_return > 10 and avg_sharpe > 0.5:
        conclusion = "The Quant AI Strategy shows moderate performance with reasonable risk-adjusted returns. Further optimization may be beneficial before full-scale implementation, particularly to improve performance in certain market regimes."
    else:
        conclusion = "The Quant AI Strategy requires further refinement before implementation. The current performance metrics indicate potential issues with risk-adjusted returns or robustness across market conditions."
    
    report.append(conclusion)
    report.append("")
    
    # Recommendations
    report.append("### Recommendations:")
    
    # Generate recommendations based on results
    recommendations = []
    
    if avg_max_dd > 15:
        recommendations.append("Implement tighter stop-loss mechanisms to reduce maximum drawdown")
    
    if avg_win_rate < 40:
        recommendations.append("Refine entry signals to improve win rate")
    
    if avg_bear < 0:
        recommendations.append("Enhance bear market performance through defensive features or market regime detection")
    
    if len(recommendations) == 0:
        recommendations.append("Proceed with implementation using the current strategy parameters")
        recommendations.append("Consider portfolio allocation across multiple symbols to benefit from diversification")
        recommendations.append("Implement real-time monitoring system to track strategy performance")
    
    for rec in recommendations:
        report.append(f"- {rec}")
    
    return "\n".join(report)

# Calculate risk metrics for each symbol
risk_metrics = {}
for symbol, df in backtest_results.items():
    if 'Returns' in df.columns:
        risk_metrics[symbol] = calculate_risk_metrics(df['Returns'])
        print(f"Calculated risk metrics for {symbol}")

# Analyze strategy robustness
market_regimes = analyze_robustness(backtest_results)
print("Analyzed strategy robustness across market regimes")

# Analyze sector performance
sector_performance = analyze_sector_performance(backtest_results, summary)
print("Analyzed performance by sector")

# Analyze correlation
correlation_matrix = analyze_correlation(backtest_results)
print("Calculated correlation matrix")

# Save correlation matrix
correlation_matrix.to_csv("validation/correlation_matrix.csv")
plt.figure(figsize=(12, 10))
plt.matshow(correlation_matrix, fignum=1)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Strategy Returns Correlation Matrix")
plt.savefig("validation/correlation_heatmap.png")
plt.close()

# Create validation report
validation_report = create_validation_report(
    summary, 
    backtest_results, 
    risk_metrics, 
    market_regimes, 
    sector_performance, 
    correlation_matrix
)

# Save validation report
with open("validation/validation_report.md", "w") as f:
    f.write(validation_report)

print("Validation report created and saved to validation/validation_report.md")

# Create summary visualizations
plt.figure(figsize=(12, 8))
if summary is not None:
    # Extract numeric values from percentage strings
    summary['Total_Return_Numeric'] = summary['Total Return'].str.rstrip('%').astype(float)
    summary['Annual_Return_Numeric'] = summary['Annual Return'].str.rstrip('%').astype(float)
    
    # Sort by annual return
    summary_sorted = summary.sort_values('Annual_Return_Numeric', ascending=False)
    
    # Plot annual returns
    plt.bar(summary_sorted['Symbol'], summary_sorted['Annual_Return_Numeric'])
    plt.title('Annual Returns by Symbol')
    plt.xlabel('Symbol')
    plt.ylabel('Annual Return (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("validation/annual_returns.png")
    plt.close()
    
    # Plot Sharpe ratios
    plt.figure(figsize=(12, 8))
    plt.bar(summary_sorted['Symbol'], summary_sorted['Sharpe Ratio'])
    plt.title('Sharpe Ratios by Symbol')
    plt.xlabel('Symbol')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("validation/sharpe_ratios.png")
    plt.close()

# Plot sector performance
if sector_performance:
    plt.figure(figsize=(12, 8))
    sectors = list(sector_performance.keys())
    performances = list(sector_performance.values())
    
    # Sort by performance
    sorted_indices = np.argsort(performances)[::-1]
    sorted_sectors = [sectors[i] for i in sorted_indices]
    sorted_performances = [performances[i] for i in sorted_indices]
    
    plt.bar(sorted_sectors, [p * 100 for p in sorted_performances])
    plt.title('Average Performance by Sector')
    plt.xlabel('Sector')
    plt.ylabel('Average Return (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("validation/sector_performance.png")
    plt.close()

print("Validation visualizations created")
print("Strategy validation complete")
