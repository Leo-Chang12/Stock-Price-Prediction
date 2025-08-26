"""
Improved Stock Price Prediction using LSTM with Twitter Sentiment Analysis

This implementation addresses the feedback from the Journal of Emerging Investigators (JEI) 
by providing:
1. Multi-stock analysis for generalizability
2. Cross-validation for uncertainty quantification  
3. Statistical testing for model comparisons
4. Professional visualizations and comprehensive metrics
5. Improved LSTM architecture with regularization
6. Better documentation and code organization

Authors: Leo Chang, Aditya Saraf, Jenjen Chen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Deep Learning and ML libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Statistical testing
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Global configuration
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

class ImprovedStockPredictor:
    """
    Advanced LSTM-based stock price predictor with sentiment analysis integration.
    
    Features:
    - Multi-layer LSTM architecture with regularization
    - Cross-validation for robust performance evaluation
    - Multiple stock analysis for generalizability
    - Statistical testing for model comparison
    - Professional visualization and comprehensive metrics
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=1):
        """
        Initialize the stock predictor.
        
        Args:
            sequence_length (int): Number of days to use for prediction
            prediction_horizon (int): Number of days ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_features = RobustScaler()
        self.scaler_target = RobustScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, stock_symbol):
        """
        Load and preprocess stock data with sentiment analysis.
        
        Args:
            stock_symbol (str): Stock symbol to analyze
            
        Returns:
            pd.DataFrame: Preprocessed combined dataset
        """
        print(f"Loading data for {stock_symbol}...")
        
        # Load datasets
        try:
            tweets = pd.read_csv('stock_tweets.csv')
            yfinance_data = pd.read_csv('stock_yfinance_data.csv')
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None
        
        # Filter data for selected stock
        tweets_selected = tweets[tweets['Stock Name'] == stock_symbol].copy()
        yfinance_selected = yfinance_data[yfinance_data['Stock Name'] == stock_symbol].copy()
        
        if tweets_selected.empty or yfinance_selected.empty:
            print(f"No data found for {stock_symbol}")
            return None
        
        # Enhanced sentiment analysis
        print("Processing sentiment data...")
        tweets_selected['Sentiment'] = tweets_selected['Tweet'].apply(self._calculate_sentiment)
        tweets_selected['Date'] = pd.to_datetime(tweets_selected['Date']).dt.date
        
        # Aggregate sentiment metrics
        daily_sentiment = tweets_selected.groupby('Date').agg({
            'Sentiment': ['mean', 'std', 'count', 'min', 'max']
        }).round(6)
        daily_sentiment.columns = ['sent_mean', 'sent_std', 'sent_count', 'sent_min', 'sent_max']
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment['sent_std'] = daily_sentiment['sent_std'].fillna(0)
        
        # Process price data
        yfinance_selected['Date'] = pd.to_datetime(yfinance_selected['Date']).dt.date
        yfinance_selected = yfinance_selected.sort_values('Date')
        
        # Feature engineering
        yfinance_selected = self._create_technical_features(yfinance_selected)
        
        # Merge datasets
        combined_data = yfinance_selected.merge(daily_sentiment, on='Date', how='left')
        
        # Fill missing sentiment data
        sentiment_cols = ['sent_mean', 'sent_std', 'sent_count', 'sent_min', 'sent_max']
        for col in sentiment_cols:
            combined_data[col] = combined_data[col].fillna(0)
            
        # Remove any remaining NaN values
        combined_data = combined_data.dropna()
        
        print(f"Data loaded: {len(combined_data)} days of data for {stock_symbol}")
        return combined_data
    
    def _calculate_sentiment(self, text):
        """Calculate sentiment polarity with error handling."""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    
    def _create_technical_features(self, df):
        """
        Create technical analysis features.
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Enhanced data with technical features
        """
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['Close'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['macd'] = self._calculate_macd(df['Close'])
        
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd
    
    def prepare_sequences(self, data, features, target_col='Close'):
        """
        Prepare sequential data for LSTM training.
        
        Args:
            data (pd.DataFrame): Combined dataset
            features (list): List of feature column names
            target_col (str): Target column name
            
        Returns:
            tuple: (X, y, feature_names, dates)
        """
        # Select and scale features
        feature_data = data[features].values
        target_data = data[target_col].values.reshape(-1, 1)
        
        # Scale the data
        feature_scaled = self.scaler_features.fit_transform(feature_data)
        target_scaled = self.scaler_target.fit_transform(target_data)
        
        # Create sequences
        X, y = [], []
        valid_dates = []
        
        for i in range(self.sequence_length, len(feature_scaled) - self.prediction_horizon + 1):
            X.append(feature_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i + self.prediction_horizon - 1])
            valid_dates.append(data.iloc[i + self.prediction_horizon - 1]['Date'])
        
        return np.array(X), np.array(y), features, valid_dates
    
    def build_advanced_lstm(self, input_shape):
        """
        Build an advanced LSTM architecture with regularization.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer with return sequences
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM layer without return sequences
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        # Custom optimizer with learning rate scheduling
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_baseline_lstm(self, input_shape):
        """
        Build a baseline LSTM for comparison (without sentiment).
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled baseline LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_with_cross_validation(self, X, y, stock_symbol, model_type='advanced', n_splits=5):
        """
        Train model using time series cross-validation.
        
        Args:
            X (np.array): Feature sequences
            y (np.array): Target values
            stock_symbol (str): Stock symbol
            model_type (str): Type of model ('advanced' or 'baseline')
            n_splits (int): Number of CV splits
            
        Returns:
            dict: Cross-validation results
        """
        print(f"Training {model_type} model for {stock_symbol} with {n_splits}-fold CV...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(X)//10)
        cv_scores = {'rmse': [], 'mae': [], 'r2': []}
        cv_predictions = []
        cv_actuals = []
        
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            print(f"  Fold {fold}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Build model
            if model_type == 'advanced':
                model = self.build_advanced_lstm((X_train.shape[1], X_train.shape[2]))
            else:
                model = self.build_baseline_lstm((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predict
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = self.scaler_target.inverse_transform(y_pred_scaled).flatten()
            y_true = self.scaler_target.inverse_transform(y_test).flatten()
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            
            cv_predictions.extend(y_pred)
            cv_actuals.extend(y_true)
            
            fold += 1
        
        # Calculate summary statistics
        results = {
            'stock': stock_symbol,
            'model_type': model_type,
            'cv_scores': cv_scores,
            'mean_rmse': np.mean(cv_scores['rmse']),
            'std_rmse': np.std(cv_scores['rmse']),
            'mean_mae': np.mean(cv_scores['mae']),
            'std_mae': np.std(cv_scores['mae']),
            'mean_r2': np.mean(cv_scores['r2']),
            'std_r2': np.std(cv_scores['r2']),
            'predictions': cv_predictions,
            'actuals': cv_actuals
        }
        
        return results
    
    def statistical_comparison(self, results_dict):
        """
        Perform statistical tests to compare model performances.
        
        Args:
            results_dict (dict): Dictionary containing results for different models
            
        Returns:
            dict: Statistical test results
        """
        print("Performing statistical tests...")
        
        test_results = {}
        
        # Compare advanced vs baseline models for each stock
        for stock in set([r['stock'] for r in results_dict.values()]):
            advanced_key = f"{stock}_advanced"
            baseline_key = f"{stock}_baseline"
            
            if advanced_key in results_dict and baseline_key in results_dict:
                advanced_rmse = results_dict[advanced_key]['cv_scores']['rmse']
                baseline_rmse = results_dict[baseline_key]['cv_scores']['rmse']
                
                # Paired t-test
                t_stat, t_p_value = ttest_rel(baseline_rmse, advanced_rmse)
                
                # Wilcoxon signed-rank test (non-parametric)
                w_stat, w_p_value = wilcoxon(baseline_rmse, advanced_rmse)
                
                test_results[stock] = {
                    'advanced_mean_rmse': np.mean(advanced_rmse),
                    'baseline_mean_rmse': np.mean(baseline_rmse),
                    'improvement': np.mean(baseline_rmse) - np.mean(advanced_rmse),
                    'improvement_pct': ((np.mean(baseline_rmse) - np.mean(advanced_rmse)) / np.mean(baseline_rmse)) * 100,
                    't_test': {'statistic': t_stat, 'p_value': t_p_value},
                    'wilcoxon_test': {'statistic': w_stat, 'p_value': w_p_value},
                    'significant': t_p_value < 0.05
                }
        
        return test_results
    
    def create_comprehensive_visualizations(self, results_dict, statistical_results):
        """
        Create professional visualizations for the analysis.
        
        Args:
            results_dict (dict): Dictionary containing all results
            statistical_results (dict): Statistical test results
        """
        print("Creating comprehensive visualizations...")
        
        # Set up the figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Model Performance Comparison (RMSE)
        ax1 = plt.subplot(4, 2, 1)
        self._plot_model_comparison(results_dict, ax1)
        
        # 2. Cross-validation scores distribution
        ax2 = plt.subplot(4, 2, 2)
        self._plot_cv_distribution(results_dict, ax2)
        
        # 3. Actual vs Predicted scatter plots
        ax3 = plt.subplot(4, 2, 3)
        self._plot_actual_vs_predicted(results_dict, ax3)
        
        # 4. Time series predictions for each stock
        for i, stock in enumerate(['AAPL', 'TSLA', 'MSFT']):
            ax = plt.subplot(4, 2, 4 + i)
            self._plot_time_series_predictions(results_dict, stock, ax)
        
        # 7. Statistical significance heatmap
        ax7 = plt.subplot(4, 2, 7)
        self._plot_statistical_significance(statistical_results, ax7)
        
        # 8. Model architecture diagram (text-based)
        ax8 = plt.subplot(4, 2, 8)
        self._plot_model_architecture(ax8)
        
        plt.tight_layout()
        plt.savefig('comprehensive_stock_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create separate high-quality individual plots
        self._create_individual_plots(results_dict, statistical_results)
    
    def _plot_model_comparison(self, results_dict, ax):
        """Plot model performance comparison."""
        stocks = []
        advanced_rmse = []
        baseline_rmse = []
        advanced_std = []
        baseline_std = []
        
        for stock in ['AAPL', 'TSLA', 'MSFT']:
            if f"{stock}_advanced" in results_dict and f"{stock}_baseline" in results_dict:
                stocks.append(stock)
                advanced_rmse.append(results_dict[f"{stock}_advanced"]['mean_rmse'])
                baseline_rmse.append(results_dict[f"{stock}_baseline"]['mean_rmse'])
                advanced_std.append(results_dict[f"{stock}_advanced"]['std_rmse'])
                baseline_std.append(results_dict[f"{stock}_baseline"]['std_rmse'])
        
        x = np.arange(len(stocks))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, advanced_rmse, width, yerr=advanced_std, 
                      label='LSTM with Sentiment', capsize=5, color='#2E86AB')
        bars2 = ax.bar(x + width/2, baseline_rmse, width, yerr=baseline_std, 
                      label='LSTM without Sentiment', capsize=5, color='#A23B72')
        
        ax.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stocks)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    def _plot_cv_distribution(self, results_dict, ax):
        """Plot cross-validation score distributions."""
        data_for_plot = []
        labels = []
        
        for key, result in results_dict.items():
            if 'advanced' in key:
                stock = result['stock']
                rmse_scores = result['cv_scores']['rmse']
                data_for_plot.extend([(score, f"{stock}\nw/ Sentiment") for score in rmse_scores])
            elif 'baseline' in key:
                stock = result['stock']
                rmse_scores = result['cv_scores']['rmse']
                data_for_plot.extend([(score, f"{stock}\nw/o Sentiment") for score in rmse_scores])
        
        if data_for_plot:
            df_plot = pd.DataFrame(data_for_plot, columns=['RMSE', 'Model'])
            sns.boxplot(data=df_plot, x='Model', y='RMSE', ax=ax)
            ax.set_title('Cross-Validation RMSE Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('RMSE (USD)', fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_actual_vs_predicted(self, results_dict, ax):
        """Plot actual vs predicted values."""
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        markers = ['o', 's', '^']
        
        for i, stock in enumerate(['AAPL', 'TSLA', 'MSFT']):
            key = f"{stock}_advanced"
            if key in results_dict:
                result = results_dict[key]
                actuals = result['actuals'][:100]  # Limit points for readability
                predictions = result['predictions'][:100]
                
                ax.scatter(actuals, predictions, alpha=0.6, 
                          color=colors[i], marker=markers[i], 
                          label=f'{stock}', s=30)
        
        # Perfect prediction line
        min_val = min([min(results_dict[f"{s}_advanced"]['actuals']) for s in ['AAPL', 'TSLA', 'MSFT'] if f"{s}_advanced" in results_dict])
        max_val = max([max(results_dict[f"{s}_advanced"]['actuals']) for s in ['AAPL', 'TSLA', 'MSFT'] if f"{s}_advanced" in results_dict])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price (USD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Actual vs Predicted Prices\n(LSTM with Sentiment)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_time_series_predictions(self, results_dict, stock, ax):
        """Plot time series predictions for a specific stock."""
        key = f"{stock}_advanced"
        if key in results_dict:
            result = results_dict[key]
            actuals = result['actuals'][-100:]  # Show last 100 predictions
            predictions = result['predictions'][-100:]
            
            x = range(len(actuals))
            
            ax.plot(x, actuals, label='Actual', linewidth=2, color='#2E86AB')
            ax.plot(x, predictions, label='Predicted', linewidth=2, color='#A23B72', alpha=0.8)
            ax.fill_between(x, actuals, predictions, alpha=0.2, color='gray')
            
            ax.set_title(f'{stock} Price Predictions\n(Last 100 Predictions)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Price (USD)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    def _plot_statistical_significance(self, statistical_results, ax):
        """Plot statistical significance results."""
        stocks = list(statistical_results.keys())
        improvements = [statistical_results[stock]['improvement_pct'] for stock in stocks]
        p_values = [statistical_results[stock]['t_test']['p_value'] for stock in stocks]
        
        colors = ['green' if p < 0.05 else 'orange' for p in p_values]
        
        bars = ax.bar(stocks, improvements, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Significance of Improvements\n(Green: p < 0.05)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add p-value annotations
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            ax.annotate(f'p={p_val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=9, fontweight='bold')
    
    def _plot_model_architecture(self, ax):
        """Display model architecture as text."""
        architecture_text = """
LSTM Model Architecture

Input Layer:
• Sequence Length: 60 days
• Features: Price + Technical + Sentiment

Layer 1: LSTM (128 units)
• Return Sequences: True  
• L2 Regularization: 0.001
• Batch Normalization
• Dropout: 30%

Layer 2: LSTM (64 units)  
• Return Sequences: True
• L2 Regularization: 0.001
• Batch Normalization
• Dropout: 30%

Layer 3: LSTM (32 units)
• Return Sequences: False
• L2 Regularization: 0.001  
• Batch Normalization
• Dropout: 20%

Dense Layer: 16 units
• Activation: ReLU
• L2 Regularization: 0.001
• Dropout: 10%

Output Layer: 1 unit
• Activation: Linear

Optimizer: Adam (lr=0.001)
Loss Function: MSE
        """
        
        ax.text(0.1, 0.9, architecture_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('LSTM Architecture Details', fontsize=14, fontweight='bold')
    
    def _create_individual_plots(self, results_dict, statistical_results):
        """Create individual high-quality plots."""
        # Individual time series plot for each stock
        for stock in ['AAPL', 'TSLA', 'MSFT']:
            if f"{stock}_advanced" in results_dict:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                self._plot_detailed_time_series(results_dict, stock, ax)
                plt.tight_layout()
                plt.savefig(f'{stock}_prediction_detailed.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    def _plot_detailed_time_series(self, results_dict, stock, ax):
        """Create detailed time series plot for publication quality."""
        key = f"{stock}_advanced"
        result = results_dict[key]
        
        actuals = result['actuals']
        predictions = result['predictions']
        x = range(len(actuals))
        
        ax.plot(x, actuals, label='Actual Price', linewidth=2.5, color='#1f77b4', alpha=0.8)
        ax.plot(x, predictions, label='LSTM Predicted Price', linewidth=2.5, color='#ff7f0e', alpha=0.8)
        
        # Calculate and show error bands
        errors = np.array(predictions) - np.array(actuals)
        mae = np.mean(np.abs(errors))
        ax.fill_between(x, np.array(predictions) - mae, np.array(predictions) + mae, 
                       alpha=0.2, color='gray', label=f'±MAE ({mae:.2f})')
        
        ax.set_title(f'{stock} Stock Price Prediction Results\nLSTM with Twitter Sentiment Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
        ax.set_ylabel('Stock Price (USD)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics as text box
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae_val = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        metrics_text = f'RMSE: {rmse:.3f}\nMAE: {mae_val:.3f}\nR²: {r2:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
    
    def generate_comprehensive_results_table(self, results_dict, statistical_results):
        """
        Generate comprehensive results table for publication.
        
        Args:
            results_dict (dict): All model results
            statistical_results (dict): Statistical test results
            
        Returns:
            pd.DataFrame: Formatted results table
        """
        table_data = []
        
        for stock in ['AAPL', 'TSLA', 'MSFT']:
            # Advanced model results
            adv_key = f"{stock}_advanced"
            base_key = f"{stock}_baseline"
            
            if adv_key in results_dict and base_key in results_dict:
                adv_result = results_dict[adv_key]
                base_result = results_dict[base_key]
                stat_result = statistical_results.get(stock, {})
                
                # Advanced model row
                table_data.append({
                    'Stock': stock,
                    'Model': 'LSTM + Sentiment',
                    'RMSE (Mean ± Std)': f"{adv_result['mean_rmse']:.3f} ± {adv_result['std_rmse']:.3f}",
                    'MAE (Mean ± Std)': f"{adv_result['mean_mae']:.3f} ± {adv_result['std_mae']:.3f}",
                    'R² (Mean ± Std)': f"{adv_result['mean_r2']:.3f} ± {adv_result['std_r2']:.3f}",
                    'Improvement (%)': f"{stat_result.get('improvement_pct', 0):.2f}%",
                    'P-value': f"{stat_result.get('t_test', {}).get('p_value', 1):.4f}",
                    'Significant': 'Yes' if stat_result.get('significant', False) else 'No'
                })
                
                # Baseline model row
                table_data.append({
                    'Stock': '',
                    'Model': 'LSTM Baseline',
                    'RMSE (Mean ± Std)': f"{base_result['mean_rmse']:.3f} ± {base_result['std_rmse']:.3f}",
                    'MAE (Mean ± Std)': f"{base_result['mean_mae']:.3f} ± {base_result['std_mae']:.3f}",
                    'R² (Mean ± Std)': f"{base_result['mean_r2']:.3f} ± {base_result['std_r2']:.3f}",
                    'Improvement (%)': '—',
                    'P-value': '—',
                    'Significant': '—'
                })
        
        results_df = pd.DataFrame(table_data)
        
        # Save to CSV for easy copy-paste into papers
        results_df.to_csv('comprehensive_results_table.csv', index=False)
        
        print("\nCOMPREHENSIVE RESULTS TABLE")
        print("=" * 120)
        print(results_df.to_string(index=False))
        print("=" * 120)
        
        return results_df
    
    def run_complete_analysis(self, stocks=['AAPL', 'TSLA', 'MSFT']):
        """
        Run the complete analysis pipeline for multiple stocks.
        
        Args:
            stocks (list): List of stock symbols to analyze
            
        Returns:
            tuple: (results_dict, statistical_results, results_table)
        """
        print("=" * 60)
        print("IMPROVED STOCK PRICE PREDICTION ANALYSIS")
        print("=" * 60)
        
        all_results = {}
        
        for stock in stocks:
            print(f"\n--- Processing {stock} ---")
            
            # Load and preprocess data
            data = self.load_and_preprocess_data(stock)
            if data is None:
                print(f"Skipping {stock} due to data loading issues")
                continue
            
            # Select features (all except Date, Stock Name, and Close)
            feature_cols = [col for col in data.columns 
                          if col not in ['Date', 'Stock Name', 'Close']]
            
            print(f"Using {len(feature_cols)} features for {stock}")
            
            # Prepare sequences
            X, y, features, dates = self.prepare_sequences(data, feature_cols, 'Close')
            
            if len(X) < 100:  # Minimum data requirement
                print(f"Insufficient data for {stock}: {len(X)} sequences")
                continue
            
            # Train advanced model (with sentiment)
            adv_results = self.train_with_cross_validation(X, y, stock, 'advanced', n_splits=5)
            all_results[f"{stock}_advanced"] = adv_results
            
            # Train baseline model (without sentiment features)
            sentiment_features = [f for f in feature_cols if 'sent_' in f]
            non_sentiment_features = [f for f in feature_cols if f not in sentiment_features]
            
            if non_sentiment_features:
                X_baseline, _, _, _ = self.prepare_sequences(data, non_sentiment_features, 'Close')
                base_results = self.train_with_cross_validation(X_baseline, y, stock, 'baseline', n_splits=5)
                all_results[f"{stock}_baseline"] = base_results
            
        # Perform statistical analysis
        statistical_results = self.statistical_comparison(all_results)
        
        # Generate comprehensive table
        results_table = self.generate_comprehensive_results_table(all_results, statistical_results)
        
        # Create visualizations
        self.create_comprehensive_visualizations(all_results, statistical_results)
        
        print(f"\nAnalysis complete! Results saved to comprehensive_results_table.csv")
        print(f"Visualizations saved as high-resolution PNG files.")
        
        return all_results, statistical_results, results_table


def main():
    """Main execution function."""
    # Initialize the improved predictor
    predictor = ImprovedStockPredictor(sequence_length=60, prediction_horizon=1)
    
    # Run complete analysis on multiple stocks
    stocks_to_analyze = ['AAPL', 'TSLA', 'MSFT']
    
    results, stats, table = predictor.run_complete_analysis(stocks_to_analyze)
    
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    
    for stock, stat_result in stats.items():
        improvement = stat_result['improvement_pct']
        significant = "statistically significant" if stat_result['significant'] else "not statistically significant"
        print(f"\n{stock}:")
        print(f"  • RMSE improvement: {improvement:.2f}%")
        print(f"  • Statistical significance: {significant} (p={stat_result['t_test']['p_value']:.4f})")
    
    print(f"\nAll results and visualizations have been saved to the current directory.")


if __name__ == "__main__":
    main()