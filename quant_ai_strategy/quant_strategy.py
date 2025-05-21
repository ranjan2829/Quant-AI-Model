import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class DataPreprocessor:
    """
    Class for preprocessing financial data for the Quant AI strategy
    """
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def load_data(self, symbol):
        """Load data from CSV file"""
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            file_path = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            print(f"Data file for {symbol} not found")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for feature engineering"""
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rs = rs.fillna(0)  # Replace NaN with 0
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        # Avoid division by zero
        df['BB_Middle'] = df['BB_Middle'].replace(0, np.nan)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Width'] = df['BB_Width'].fillna(0)  # Replace NaN with 0
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)
        df['Volume_Change'] = df['Volume_Change'].fillna(0)  # Replace NaN with 0
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        # Avoid division by zero
        df['Volume_MA10'] = df['Volume_MA10'].replace(0, np.nan)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA10']
        df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1)  # Replace NaN with 1
        
        # Price momentum
        df['ROC5'] = df['Close'].pct_change(periods=5, fill_method=None) * 100
        df['ROC10'] = df['Close'].pct_change(periods=10, fill_method=None) * 100
        df['ROC20'] = df['Close'].pct_change(periods=20, fill_method=None) * 100
        # Fill NaN values
        df['ROC5'] = df['ROC5'].fillna(0)
        df['ROC10'] = df['ROC10'].fillna(0)
        df['ROC20'] = df['ROC20'].fillna(0)
        
        # Price volatility
        df['Close_MA20'] = df['Close'].rolling(window=20).mean()
        # Avoid division by zero
        df['Close_MA20'] = df['Close_MA20'].replace(0, np.nan)
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close_MA20']
        df['Volatility'] = df['Volatility'].fillna(0)  # Replace NaN with 0
        
        # Target variable: Price direction (1 if price goes up in next day, 0 otherwise)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target for model training"""
        # Features to use
        feature_columns = [
            'MA5', 'MA10', 'MA20', 'MA50', 
            'EMA5', 'EMA10', 'EMA20',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI', 
            'BB_Width', 
            'ATR',
            'Volume_Change', 'Volume_Ratio',
            'ROC5', 'ROC10', 'ROC20',
            'Volatility'
        ]
        
        # Add price-based features
        # Avoid division by zero
        df['MA20'] = df['MA20'].replace(0, np.nan)
        df['Close_Ratio'] = df['Close'] / df['MA20']
        df['Close_Ratio'] = df['Close_Ratio'].fillna(1)  # Replace NaN with 1
        
        df['Low'] = df['Low'].replace(0, np.nan)
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['High_Low_Ratio'] = df['High_Low_Ratio'].fillna(1)  # Replace NaN with 1
        
        df['Close'] = df['Close'].replace(0, np.nan)
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Open_Close_Ratio'] = df['Open_Close_Ratio'].fillna(1)  # Replace NaN with 1
        
        feature_columns.extend(['Close_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio'])
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['Target'].copy()
        
        # Replace inf and -inf with large/small values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill remaining NaN values with 0
        X.fillna(0, inplace=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_columns
    
    def process_all_data(self):
        """Process all data files in the data directory"""
        processed_data = {}
        
        # Process indices
        indices = ['NSEI', 'BSESN', 'NSEBANK', 'CNXIT', 'CNXPHARMA', 'CNXAUTO']
        for idx in indices:
            df = self.load_data(idx)
            if df is not None:
                processed_df = self.calculate_technical_indicators(df)
                processed_data[idx] = processed_df
                print(f"Processed index data: {idx}")
        
        # Process stocks
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and '_NS.csv' in f]
        for stock_file in stock_files:
            symbol = stock_file.replace('.csv', '')
            df = self.load_data(symbol)
            if df is not None:
                processed_df = self.calculate_technical_indicators(df)
                processed_data[symbol] = processed_df
                print(f"Processed stock data: {symbol}")
        
        return processed_data


class ModelTrainer:
    """
    Class for training and evaluating AI models for the Quant strategy
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
    def train_models(self, X, y, symbol):
        """Train multiple models on the data"""
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        for name, model in self.models.items():
            print(f"Training {name} for {symbol}...")
            
            # Train with time series cross-validation
            cv_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores.append(accuracy)
            
            # Train on full dataset
            model.fit(X, y)
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{symbol}_{name}.joblib")
            joblib.dump(model, model_path)
            
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores)
            }
            
            print(f"  {name} CV accuracy: {np.mean(cv_scores):.4f}")
        
        return results
    
    def evaluate_model(self, model, X, y):
        """Evaluate model performance"""
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_ensemble(self, models_results):
        """Create an ensemble of the best models"""
        # Implement a simple voting ensemble
        pass


class SignalGenerator:
    """
    Class for generating trading signals based on model predictions
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        
    def load_models(self, symbol):
        """Load trained models for a symbol"""
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith(f"{symbol}_")]
        
        for model_file in model_files:
            model_name = model_file.replace(f"{symbol}_", "").replace(".joblib", "")
            model_path = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
        
        return self.models
    
    def generate_signals(self, X, models=None):
        """Generate trading signals based on model predictions"""
        if models is None:
            models = self.models
        
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X)
            
        # Ensemble prediction (majority voting)
        ensemble_pred = np.zeros(X.shape[0])
        for pred in predictions.values():
            ensemble_pred += pred
        
        # If more than half of the models predict 1, the ensemble prediction is 1
        ensemble_pred = (ensemble_pred > len(predictions) / 2).astype(int)
        
        return {
            'individual': predictions,
            'ensemble': ensemble_pred
        }


class PositionSizer:
    """
    Class for determining position sizes based on risk management rules
    """
    def __init__(self, risk_per_trade=0.02, max_position_size=0.1):
        self.risk_per_trade = risk_per_trade  # Risk 2% of capital per trade
        self.max_position_size = max_position_size  # Maximum position size is 10% of capital
    
    def calculate_position_size(self, capital, entry_price, stop_loss, confidence=0.5):
        """Calculate position size based on risk management rules"""
        # Risk amount in currency
        risk_amount = capital * self.risk_per_trade
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Avoid division by zero
        if risk_per_share == 0:
            risk_per_share = 0.01 * entry_price  # Use 1% of entry price as default risk
        
        # Calculate number of shares
        shares = risk_amount / risk_per_share
        
        # Calculate position value
        position_value = shares * entry_price
        
        # Adjust based on model confidence
        position_value = position_value * confidence
        
        # Ensure position size doesn't exceed maximum
        max_value = capital * self.max_position_size
        if position_value > max_value:
            position_value = max_value
            shares = position_value / entry_price
        
        return {
            'shares': int(shares),
            'position_value': position_value,
            'risk_amount': risk_amount
        }


class TradingStrategy:
    """
    Main class for the Quant AI Trading Strategy
    """
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_dir)
        self.model_trainer = ModelTrainer(models_dir)
        self.signal_generator = SignalGenerator(models_dir)
        self.position_sizer = PositionSizer()
        
    def train_strategy(self):
        """Train the strategy on all available data"""
        # Process all data
        processed_data = self.preprocessor.process_all_data()
        
        # Train models for each symbol
        all_results = {}
        for symbol, df in processed_data.items():
            print(f"\nTraining models for {symbol}...")
            
            # Prepare features and target
            X, y, feature_columns = self.preprocessor.prepare_features(df)
            
            # Train models
            results = self.model_trainer.train_models(X, y, symbol)
            all_results[symbol] = results
        
        return all_results
    
    def backtest_strategy(self, symbol, capital=1000000):
        """Backtest the strategy on historical data"""
        # Load data
        df = self.preprocessor.load_data(symbol)
        if df is None:
            return None, None
        
        # Calculate technical indicators
        df = self.preprocessor.calculate_technical_indicators(df)
        
        # Prepare features
        X, y, feature_columns = self.preprocessor.prepare_features(df)
        
        # Load models
        models = self.signal_generator.load_models(symbol)
        
        # If no models found, return None
        if not models:
            print(f"No trained models found for {symbol}")
            return None, None
        
        # Generate signals
        signals = self.signal_generator.generate_signals(X, models)
        
        # Add signals to dataframe
        df = df.iloc[len(df) - len(signals['ensemble']):]  # Align indices
        df['Signal'] = signals['ensemble']
        
        # Initialize backtest results
        backtest = df.copy()
        backtest['Position'] = 0
        backtest['Cash'] = capital
        backtest['Holdings'] = 0
        backtest['Equity'] = capital
        
        # Run backtest
        position = 0
        for i in range(1, len(backtest)):
            # Previous day's position and equity
            prev_position = backtest.iloc[i-1]['Position']
            prev_cash = backtest.iloc[i-1]['Cash']
            prev_holdings = backtest.iloc[i-1]['Holdings']
            
            # Current day's signal
            signal = backtest.iloc[i-1]['Signal']  # Signal from previous day
            
            # Current day's price
            price = backtest.iloc[i]['Close']
            
            # Update position based on signal
            if signal == 1 and prev_position == 0:
                # Buy signal
                shares_to_buy = int(prev_cash * 0.95 / price)  # Use 95% of cash
                new_position = shares_to_buy
                new_cash = prev_cash - (shares_to_buy * price)
                new_holdings = shares_to_buy * price
            elif signal == 0 and prev_position > 0:
                # Sell signal
                new_position = 0
                new_cash = prev_cash + prev_holdings
                new_holdings = 0
            else:
                # Hold
                new_position = prev_position
                new_cash = prev_cash
                new_holdings = prev_position * price
            
            # Update backtest dataframe
            backtest.iloc[i, backtest.columns.get_loc('Position')] = new_position
            backtest.iloc[i, backtest.columns.get_loc('Cash')] = new_cash
            backtest.iloc[i, backtest.columns.get_loc('Holdings')] = new_holdings
            backtest.iloc[i, backtest.columns.get_loc('Equity')] = new_cash + new_holdings
        
        # Calculate returns
        backtest['Returns'] = backtest['Equity'].pct_change()
        backtest['Returns'] = backtest['Returns'].fillna(0)  # Replace NaN with 0
        backtest['Cumulative_Returns'] = (1 + backtest['Returns']).cumprod() - 1
        
        # Calculate performance metrics
        total_return = backtest['Cumulative_Returns'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(backtest)) - 1
        daily_returns = backtest['Returns'].dropna()
        
        # Avoid division by zero
        if daily_returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        max_drawdown = (backtest['Equity'] / backtest['Equity'].cummax() - 1).min()
        
        # Calculate win rate
        if len(daily_returns) > 0:
            win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
        else:
            win_rate = 0
        
        performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
        return backtest, performance
    
    def generate_trading_plan(self, symbol, capital=1000000):
        """Generate a trading plan for the next day"""
        # Load data
        df = self.preprocessor.load_data(symbol)
        if df is None:
            return None
        
        # Calculate technical indicators
        df = self.preprocessor.calculate_technical_indicators(df)
        
        # Prepare features
        X, y, feature_columns = self.preprocessor.prepare_features(df)
        
        # Load models
        models = self.signal_generator.load_models(symbol)
        
        # If no models found, return None
        if not models:
            print(f"No trained models found for {symbol}")
            return None
        
        # Generate signals for the latest data point
        latest_features = X[-1].reshape(1, -1)
        signals = {}
        for name, model in models.items():
            signals[name] = model.predict(latest_features)[0]
            signals[f"{name}_prob"] = model.predict_proba(latest_features)[0][1]
        
        # Ensemble signal
        signals['ensemble'] = 1 if sum(signals[m] for m in models.keys()) > len(models) / 2 else 0
        
        # Calculate confidence
        confidence = np.mean([signals[f"{m}_prob"] for m in models.keys()])
        
        # Current price and ATR for stop loss
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Calculate stop loss
        stop_loss = current_price - (2 * atr)
        
        # Calculate position size
        position = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=current_price,
            stop_loss=stop_loss,
            confidence=confidence
        )
        
        # Generate trading plan
        trading_plan = {
            'symbol': symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': current_price,
            'signal': signals['ensemble'],
            'confidence': confidence,
            'stop_loss': stop_loss,
            'position': position,
            'model_signals': {m: signals[m] for m in models.keys()},
            'model_probabilities': {m: signals[f"{m}_prob"] for m in models.keys()}
        }
        
        return trading_plan
