import os
import json
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import numpy as np
import logging
import traceback
from dotenv import load_dotenv
import yfinance as yf
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Updated CORS configuration
CORS(app, 
     resources={
         r"/*": {  # Allow all routes
             "origins": [
                 "https://stockmarketai.netlify.app",
                 "http://localhost:3000",
                 "https://www.stockmarketai.netlify.app",
                 "http://stockmarketai.netlify.app"
             ],
             "methods": ["GET", "POST", "OPTIONS", "HEAD"],
             "allow_headers": [
                 "Content-Type", 
                 "Authorization", 
                 "Access-Control-Allow-Headers",
                 "Access-Control-Allow-Origin",
                 "Access-Control-Allow-Methods",
                 "Accept",
                 "Origin"
             ],
             "expose_headers": [
                 "Content-Type",
                 "Authorization"
             ],
             "supports_credentials": True,
             "max_age": 86400  # Cache preflight requests for 24 hours
         }
     })

# Get API keys from environment variables or use fallback for development
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Google Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY not set in environment variables")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy"})

def generate_mock_stock_data(symbol, days=100):
    """Generate mock stock data for demonstration when API fails"""
    logger.warning(f"Generating mock data for {symbol} as fallback")
    
    np.random.seed(hash(symbol) % 10000)  # Use symbol as seed for consistency
    
    # Start with a base price based on the symbol
    base_price = 100 + hash(symbol) % 900  # Between 100 and 1000
    
    # Generate dates (starting from today and going back)
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create price movements with some randomness but trends
    price_changes = np.random.normal(0, 2, size=len(dates))
    trend = np.linspace(0, 5, len(dates))  # Add an upward trend
    
    # Oscillation for more realistic price movements
    oscillation = 5 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    
    # Combine all factors
    cumulative_changes = np.cumsum(price_changes) + trend + oscillation
    
    # Generate OHLC data
    closes = base_price + cumulative_changes
    daily_volatility = np.random.uniform(0.5, 2.0, size=len(dates))
    
    # Create dataframe
    df = pd.DataFrame({
        'open': closes - np.random.uniform(0, daily_volatility, size=len(dates)),
        'high': closes + np.random.uniform(1, daily_volatility * 2, size=len(dates)),
        'low': closes - np.random.uniform(1, daily_volatility * 2, size=len(dates)),
        'close': closes,
        'volume': np.random.randint(100000, 10000000, size=len(dates))
    }, index=dates)
    
    # Ensure high is the highest price of the day
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    
    # Ensure low is the lowest price of the day
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Sort by date descending (newest first) to match the real API
    return df.sort_index(ascending=False)

def fetch_stock_data(symbol):
    """Fetch stock data using yfinance"""
    try:
        logger.info(f"Fetching data for {symbol} using yfinance")
        
        # Remove $ prefix if present (yfinance doesn't accept $ in ticker symbols)
        if symbol.startswith('$'):
            symbol = symbol[1:]
        
        # Try with direct ticker first
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo")
        
        # If no data returned, try with .{exchange} suffixes
        if df.empty:
            logger.warning(f"No data returned for {symbol}, trying alternative formats")
            
            # Common exchanges to try
            exchanges = ["", ".NS", ".BO", ".L", ".PA", ".DE", ".TO"]
            
            for exchange in exchanges:
                try:
                    alternate_symbol = f"{symbol}{exchange}"
                    logger.debug(f"Trying alternate symbol: {alternate_symbol}")
                    ticker = yf.Ticker(alternate_symbol)
                    df = ticker.history(period="3mo")
                    if not df.empty:
                        logger.info(f"Found data using alternate symbol: {alternate_symbol}")
                        break
                except Exception as ex:
                    logger.debug(f"Failed with {alternate_symbol}: {str(ex)}")
                    continue
        
        # If still no data, try downloading directly
        if df.empty:
            logger.warning(f"Still no data for {symbol}, trying yf.download directly")
            df = yf.download(symbol, period="3mo")
        
        # If all real data fetch attempts fail, use mock data
        if df.empty:
            logger.warning(f"All attempts to fetch real data for {symbol} failed, using mock data")
            df = generate_mock_stock_data(symbol)
            logger.info(f"Generated mock data for {symbol}. Shape: {df.shape}")
            return df, None
        
        # Rename columns for consistency with the rest of the code
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Convert index to datetime if it's not already
        df.index = pd.to_datetime(df.index)
        
        # Sort data by date descending (newest first)
        df = df.sort_index(ascending=False)
        
        # Keep only the columns we need
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Limit to 100 days
        df = df.head(100)
        
        logger.info(f"Successfully fetched data for {symbol}. Shape: {df.shape}")
        return df, None
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Use mock data as a last resort after any exception
        try:
            logger.warning(f"Exception occurred, falling back to mock data for {symbol}")
            df = generate_mock_stock_data(symbol)
            logger.info(f"Generated mock data for {symbol} after exception. Shape: {df.shape}")
            return df, None
        except Exception as mock_error:
            logger.error(f"Even mock data generation failed: {str(mock_error)}")
            return None, {'error': f"Error fetching data for {symbol}: {str(e)}"}

def create_candlestick_chart(df, symbol):
    """Create a candlestick chart for stock data"""
    try:
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert date column to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date (ascending for charting)
        df = df.sort_values('Date')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Format the x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Plot the candlesticks
        width = 0.6
        width2 = width * 0.8
        
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        
        # Up candles
        ax.bar(up.Date, up.close - up.open, width, bottom=up.open, color='green')
        ax.bar(up.Date, up.high - up.close, width2, bottom=up.close, color='green')
        ax.bar(up.Date, up.open - up.low, width2, bottom=up.low, color='green')
        
        # Down candles
        ax.bar(down.Date, down.close - down.open, width, bottom=down.open, color='red')
        ax.bar(down.Date, down.high - down.open, width2, bottom=down.open, color='red')
        ax.bar(down.Date, down.close - down.low, width2, bottom=down.low, color='red')
        
        # Set the title and labels
        ax.set_title(f'{symbol} Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Tight layout
        fig.tight_layout()
        
        # Convert plot to base64 encoded image
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Close the figure to free memory
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def analyze_stock_with_gemini(symbol, df):
    """Analyze stock data using Google Gemini"""
    try:
        if not GEMINI_API_KEY:
            return "API key for Google Gemini not configured. Cannot perform analysis."
        
        # Sort dataframe by date ascending for calculation
        df = df.sort_index(ascending=True)
        
        # Calculate technical indicators
        # 1. Moving Averages
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # 2. RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Get the most recent data point
        latest_data = df.iloc[-1]
        
        # Format the technical indicators, handling NaN values
        rsi = latest_data['RSI']
        rsi_formatted = f"{rsi:.2f}" if not pd.isna(rsi) else "N/A"
        
        macd = latest_data['MACD']
        macd_formatted = f"{macd:.4f}" if not pd.isna(macd) else "N/A"
        
        signal = latest_data['Signal_Line']
        signal_formatted = f"{signal:.4f}" if not pd.isna(signal) else "N/A"
        
        ma20 = latest_data['MA20']
        ma20_formatted = f"${ma20:.2f}" if not pd.isna(ma20) else "N/A"
        
        ma50 = latest_data['MA50']
        ma50_formatted = f"${ma50:.2f}" if not pd.isna(ma50) else "N/A"
        
        current_price = latest_data['close']
        current_price_formatted = f"${current_price:.2f}"
        
        # Calculate price change
        first_price = df['close'].iloc[0]
        price_change = ((current_price - first_price) / first_price) * 100
        price_change_formatted = f"{price_change:.2f}%"
        
        # Prepare a prompt for Gemini
        prompt = f"""
        You are a professional stock market analyst. Analyze the stock {symbol} with the following technical indicators:
        
        Current Price: {current_price_formatted}
        Period Price Change: {price_change_formatted}
        RSI (14-day): {rsi_formatted}
        MACD: {macd_formatted}
        Signal Line: {signal_formatted}
        20-day Moving Average: {ma20_formatted}
        50-day Moving Average: {ma50_formatted}
        
        Provide a comprehensive analysis including:
        1. Technical analysis based on the indicators above
        2. Current trend identification (bullish, bearish, or neutral)
        3. Possible support and resistance levels
        4. Trading volume significance
        5. Potential future movement based on these indicators
        
        Format your response in a clear, professional manner with sections and bullet points where appropriate.
        Keep your analysis concise but thorough, with approximately 300-400 words.
        """
        
        # Generate content with Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error analyzing stock: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error performing analysis: {str(e)}"

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data():
    """Get stock data for a given symbol"""
    try:
        symbol = request.args.get('symbol', 'AAPL')
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
            
        logger.info(f"Processing request for symbol: {symbol}")
        df, error = fetch_stock_data(symbol)
        
        if error:
            logger.error(f"Error fetching data for {symbol}: {error}")
            return jsonify(error), 500
            
        if df is None or df.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        # Create candlestick chart
        chart_image = create_candlestick_chart(df, symbol)
        
        if not chart_image:
            return jsonify({'error': 'Failed to create chart'}), 500
        
        # Prepare data for response
        logger.debug(f"Successfully processed data for {symbol}")
        return jsonify({
            'symbol': symbol,
            'chart': chart_image,
            'data': json.loads(df.head(10).to_json(orient='records', date_format='iso'))
        })
    except Exception as e:
        logger.error(f"Unexpected error in get_stock_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    """Analyze stock data with AI"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
            
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
            
        # Fetch fresh data for analysis
        df, error = fetch_stock_data(symbol)
        
        if error:
            return jsonify({'error': f'Error fetching data: {error}'}), 500
            
        if df is None:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
            
        # Perform AI analysis
        analysis = analyze_stock_with_gemini(symbol, df)
        
        return jsonify({
            'symbol': symbol,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005) 