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

# API Keys
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Validate API keys
if not ALPHA_VANTAGE_API_KEY:
    logger.error("Alpha Vantage API key not found in environment variables")
if not GEMINI_API_KEY:
    logger.error("Gemini API key not found in environment variables")

# Initialize Google Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def fetch_stock_data(symbol, interval='daily', outputsize='compact'):
    """Fetch stock data from Alpha Vantage API"""
    if not ALPHA_VANTAGE_API_KEY:
        return None, "Alpha Vantage API key not configured"
    
    function = 'TIME_SERIES_DAILY' if interval == 'daily' else 'TIME_SERIES_INTRADAY'
    
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
    if interval != 'daily':
        url += f'&interval={interval}'
    
    logger.info(f"Fetching data for {symbol} from Alpha Vantage")
    
    try:
        response = requests.get(url, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # Print the response for debugging
        logger.debug(f"API Response keys: {data.keys()}")
        
        # Check for API rate limit message
        if 'Information' in data and 'rate limit' in data['Information'].lower():
            logger.warning(f"Alpha Vantage rate limit reached: {data['Information']}")
            return None, {
                'error': 'API rate limit reached. Please try again later or contact support for premium access.',
                'details': data['Information']
            }
        
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return None, {'error': data['Error Message']}
        
        if 'Note' in data:
            logger.warning(f"Alpha Vantage note for {symbol}: {data['Note']}")
            return None, {'error': data['Note']}
        
        if 'Time Series (Daily)' not in data:
            logger.error(f"Unexpected API response for {symbol}: {data}")
            return None, {'error': 'Invalid API response format'}
        
        # Extract time series data
        time_series_key = f'Time Series ({interval})' if interval != 'daily' else 'Time Series (Daily)'
        
        if time_series_key not in data:
            logger.error(f"No time series data found for {symbol}. Available keys: {data.keys()}")
            return None, "No time series data found"
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        logger.info(f"Successfully fetched data for {symbol}. Shape: {df.shape}")
        return df, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {symbol}: {e}")
        return None, f"Network error: {str(e)}"
    except Exception as e:
        logger.error(f"Exception in fetch_stock_data for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return None, f"Error fetching stock data: {str(e)}"

def create_candlestick_chart(df):
    """Create a candlestick chart using Matplotlib"""
    try:
        # Get the most recent 30 days of data
        df_recent = df.tail(30)
        
        # Create figure and axis with Agg backend
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Calculate width of candlestick elements
        width = 0.6
        width2 = 0.1
        
        # Define up and down days
        up = df_recent[df_recent.close >= df_recent.open]
        down = df_recent[df_recent.close < df_recent.open]
        
        # Plot up days
        ax.bar(up.index, up.close-up.open, width, bottom=up.open, color='green')
        ax.bar(up.index, up.high-up.close, width2, bottom=up.close, color='green')
        ax.bar(up.index, up.low-up.open, width2, bottom=up.open, color='green')
        
        # Plot down days
        ax.bar(down.index, down.close-down.open, width, bottom=down.open, color='red')
        ax.bar(down.index, down.high-down.open, width2, bottom=down.open, color='red')
        ax.bar(down.index, down.low-down.close, width2, bottom=down.close, color='red')
        
        # Add 20-day moving average if available
        if 'MA20' in df_recent.columns and not df_recent['MA20'].isna().all():
            ax.plot(df_recent.index, df_recent['MA20'], color='blue', linewidth=1.5, label='20-day MA')
        
        # Add 50-day moving average if available
        if 'MA50' in df_recent.columns and not df_recent['MA50'].isna().all():
            ax.plot(df_recent.index, df_recent['MA50'], color='orange', linewidth=1.5, label='50-day MA')
        
        # Add legend if we have moving averages
        if ('MA20' in df_recent.columns and not df_recent['MA20'].isna().all()) or \
           ('MA50' in df_recent.columns and not df_recent['MA50'].isna().all()):
            ax.legend()
        
        # Set title and labels
        ax.set_title(f'Candlestick Chart (Last 30 Days)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encode as base64
        image_png = buffer.getvalue()
        buffer.close()
        encoded_image = base64.b64encode(image_png).decode('utf-8')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return encoded_image
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        logger.error(traceback.format_exc())
        return ""

def analyze_stock_with_gemini(stock_symbol, chart_base64, stock_data):
    """Analyze stock using Google Gemini"""
    
    # Calculate some basic technical indicators
    df = stock_data.copy()
    
    # Calculate 20-day and 50-day moving averages
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Get the latest data
    latest_data = df.iloc[-1].to_dict()
    
    # Get the last 10 days of price data for trend analysis
    recent_prices = df.tail(10)['close'].tolist()
    recent_dates = [date.strftime('%Y-%m-%d') for date in df.tail(10).index.tolist()]
    
    # Format the technical indicators with proper handling of NaN values
    rsi_value = latest_data.get('RSI')
    rsi_formatted = f"{rsi_value:.2f}" if pd.notna(rsi_value) else "N/A"
    
    macd_value = latest_data.get('MACD')
    macd_formatted = f"{macd_value:.4f}" if pd.notna(macd_value) else "N/A"
    
    signal_value = latest_data.get('Signal_Line')
    signal_formatted = f"{signal_value:.4f}" if pd.notna(signal_value) else "N/A"
    
    ma20_value = latest_data.get('MA20')
    ma20_formatted = f"${ma20_value:.2f}" if pd.notna(ma20_value) else "N/A"
    
    ma50_value = latest_data.get('MA50')
    ma50_formatted = f"${ma50_value:.2f}" if pd.notna(ma50_value) else "N/A"
    
    # Prepare the prompt for Gemini
    prompt = f"""
    You are a stock market analyst. Analyze the following stock data for {stock_symbol} and provide investment recommendations.
    
    Recent Price Data (last 10 trading days):
    {', '.join([f"{date}: ${price:.2f}" for date, price in zip(recent_dates, recent_prices)])}
    
    Latest Technical Indicators:
    - Close Price: ${latest_data['close']:.2f}
    - 20-Day Moving Average: {ma20_formatted}
    - 50-Day Moving Average: {ma50_formatted}
    - RSI (14-day): {rsi_formatted}
    - MACD: {macd_formatted}
    - MACD Signal Line: {signal_formatted}
    """
    
    # Create a model instance
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )
    
    try:
        # If we have a chart, include it in the analysis
        if chart_base64 and len(chart_base64) > 0:
            # Update prompt to mention the chart
            prompt += """
            
            Based on the candlestick chart and technical indicators:
            1. Analyze the current price action and trend
            2. Interpret the technical indicators (MA, RSI, MACD)
            3. Identify potential support and resistance levels
            4. Provide a short-term outlook (1-2 weeks)
            5. Provide a medium-term outlook (1-3 months)
            6. Give a clear buy, sell, or hold recommendation with reasoning
            
            Format your response in a clear, structured manner with headings for each section.
            """
            
            # Decode the base64 image
            image_data = base64.b64decode(chart_base64)
            
            # Generate content with text and image
            response = model.generate_content(
                contents=[
                    prompt,
                    {"mime_type": "image/png", "data": image_data}
                ]
            )
        else:
            # Add instructions without mentioning the chart
            prompt += """
            
            Based on the price data and technical indicators:
            1. Analyze the current price action and trend
            2. Interpret the technical indicators (MA, RSI, MACD)
            3. Identify potential support and resistance levels
            4. Provide a short-term outlook (1-2 weeks)
            5. Provide a medium-term outlook (1-3 months)
            6. Give a clear buy, sell, or hold recommendation with reasoning
            
            Format your response in a clear, structured manner with headings for each section.
            """
            
            # Generate content with text only
            response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return f"Error analyzing stock: {e}"

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data():
    try:
        symbol = request.args.get('symbol', '').upper()
        logger.debug(f"Received request for symbol: {symbol}")
        
        if not symbol:
            logger.error("No symbol provided")
            return jsonify({'error': 'Symbol is required'}), 400
            
        logger.debug("Fetching stock data from Alpha Vantage")
        df, error = fetch_stock_data(symbol)
        
        if error:
            logger.error(f"Error fetching data for {symbol}: {error}")
            return jsonify(error), 500
            
        if df is None:
            logger.error(f"No data returned for symbol: {symbol}")
            return jsonify({'error': 'Failed to fetch stock data'}), 500
            
        # Convert DataFrame to JSON-serializable format
        data = {
            'symbol': symbol,
            'data': df.reset_index().to_dict(orient='records'),
            'chart': create_candlestick_chart(df)
        }
        
        logger.debug(f"Successfully processed data for {symbol}")
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        symbol = data.get('symbol', 'AAPL')
        chart_base64 = data.get('chart', '')
        
        logger.info(f"Received analysis request for symbol: {symbol}")
        
        # Fetch stock data
        df, error = fetch_stock_data(symbol, outputsize='full')  # Use full data for analysis
        
        if error:
            logger.error(f"Error fetching data for analysis of {symbol}: {error}")
            return jsonify({"error": error}), 400
        
        # Analyze with Gemini
        analysis = analyze_stock_with_gemini(symbol, chart_base64, df)
        
        return jsonify({
            "symbol": symbol,
            "analysis": analysis
        })
    except Exception as e:
        logger.error(f"Exception in analyze_stock route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5005) 