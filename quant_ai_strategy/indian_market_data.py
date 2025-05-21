import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import json
from datetime import datetime
import os

# Create client
client = ApiClient()

# Define key Indian indices
indices = {
    "^NSEI": "Nifty 50",
    "^BSESN": "BSE SENSEX",
    "^NSEBANK": "Nifty Bank",
    "^CNXIT": "CNX IT",
    "^CNXPHARMA": "CNX Pharma",
    "^CNXAUTO": "CNX Auto"
}

# Define key Indian stocks (top companies from various sectors)
stocks = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "WIPRO.NS": "Wipro",
    "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TATAMOTORS.NS": "Tata Motors"
}

# Create directory for data
os.makedirs("data", exist_ok=True)

# Function to fetch and save data
def fetch_and_save_data(symbol, name, data_type):
    print(f"Fetching data for {name} ({symbol})...")
    
    # Fetch historical data (1 year with daily interval)
    data = client.call_api('YahooFinance/get_stock_chart', query={
        'symbol': symbol,
        'region': 'IN',
        'interval': '1d',
        'range': '1y',
        'includeAdjustedClose': True
    })
    
    # Save raw data
    with open(f"data/{symbol.replace('^', '').replace('.', '_')}_raw.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    # Process and save as CSV if data is valid
    if data and 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
        result = data['chart']['result'][0]
        
        # Extract timestamps and convert to dates
        timestamps = result['timestamp']
        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
        
        # Extract price data
        quotes = result['indicators']['quote'][0]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': quotes.get('open', []),
            'High': quotes.get('high', []),
            'Low': quotes.get('low', []),
            'Close': quotes.get('close', []),
            'Volume': quotes.get('volume', [])
        })
        
        # Add adjusted close if available
        if 'adjclose' in result['indicators'] and result['indicators']['adjclose']:
            df['Adj_Close'] = result['indicators']['adjclose'][0].get('adjclose', [])
        
        # Save to CSV
        csv_file = f"data/{symbol.replace('^', '').replace('.', '_')}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved data to {csv_file}")
        
        return df
    else:
        print(f"No valid data found for {symbol}")
        return None

# Fetch data for indices
print("Fetching data for Indian market indices...")
indices_data = {}
for symbol, name in indices.items():
    indices_data[symbol] = fetch_and_save_data(symbol, name, "index")

# Fetch data for stocks
print("Fetching data for Indian stocks...")
stocks_data = {}
for symbol, name in stocks.items():
    stocks_data[symbol] = fetch_and_save_data(symbol, name, "stock")

# Create a summary file
with open("data/market_data_summary.txt", 'w') as f:
    f.write("# Indian Market Data Summary\n\n")
    
    f.write("## Indices\n")
    for symbol, name in indices.items():
        f.write(f"- {name} ({symbol})\n")
    
    f.write("\n## Stocks\n")
    for symbol, name in stocks.items():
        f.write(f"- {name} ({symbol})\n")
    
    f.write("\n## Data Collection Details\n")
    f.write("- Time Period: 1 year\n")
    f.write("- Interval: Daily\n")
    f.write("- Data Source: Yahoo Finance API\n")
    f.write(f"- Collection Date: {datetime.now().strftime('%Y-%m-%d')}\n")
    f.write("- Data Fields: Date, Open, High, Low, Close, Volume, Adjusted Close\n")

print("Data collection complete. Summary saved to data/market_data_summary.txt")
