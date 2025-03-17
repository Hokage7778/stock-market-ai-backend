import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Get API key
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

def test_alpha_vantage():
    """Test Alpha Vantage API functionality"""
    print("Testing Alpha Vantage API...")
    print(f"API Key exists: {bool(ALPHA_VANTAGE_API_KEY)}")
    
    if not ALPHA_VANTAGE_API_KEY:
        print("Error: Alpha Vantage API key not found in environment variables.")
        return
    
    # Test symbols
    symbols = ["MSFT", "AAPL", "GOOGL"]
    
    for symbol in symbols:
        print(f"\nTesting symbol: {symbol}")
        
        # Test daily data
        function = "TIME_SERIES_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
        
        print(f"Requesting data from: {url}")
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Print response keys for debugging
            print(f"Response keys: {data.keys()}")
            
            # Check for error messages
            if 'Error Message' in data:
                print(f"Error from Alpha Vantage: {data['Error Message']}")
                continue
            
            # Check for information messages (like API limit reached)
            if 'Information' in data:
                print(f"Information from Alpha Vantage: {data['Information']}")
                continue
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            
            if time_series_key not in data:
                print(f"No time series data found. Available keys: {data.keys()}")
                continue
            
            time_series = data[time_series_key]
            
            # Print first data point
            first_date = list(time_series.keys())[0]
            print(f"First data point ({first_date}): {time_series[first_date]}")
            
            # Convert to DataFrame
            df = pd.DataFrame(time_series).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Print DataFrame info
            print(f"DataFrame shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Create a simple plot
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['close'])
            plt.title(f"{symbol} Stock Price")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.grid(True)
            
            # Save to BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Encode as base64
            image_png = buffer.getvalue()
            buffer.close()
            encoded_image = base64.b64encode(image_png).decode('utf-8')
            
            # Close the figure to free memory
            plt.close()
            
            print(f"Successfully created chart for {symbol}")
            print(f"Base64 image length: {len(encoded_image)}")
            
        except Exception as e:
            print(f"Error testing {symbol}: {e}")

if __name__ == "__main__":
    test_alpha_vantage()
    print("\nTest completed!") 