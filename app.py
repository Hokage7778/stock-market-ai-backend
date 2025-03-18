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
import yfinance as yf

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

def fetch_stock_data(symbol):
    """Fetch stock data using yfinance"""
    try:
        # Get historical data for last 100 days
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo")
        
        if df.empty:
            logger.error(f"No data returned from yfinance for {symbol}")
            return None, {'error': f"No data available for symbol {symbol}"}
        
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
    symbol = request.args.get('symbol', 'AAPL')
    
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
        
    df, error = fetch_stock_data(symbol)
    
    if error:
        logger.error(f"Error fetching data for {symbol}: {error}")
        return jsonify(error), 500
        
    if df is None:
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