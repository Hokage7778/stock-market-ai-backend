import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

def test_alpha_vantage_api():
    """Test the Alpha Vantage API"""
    symbol = "AAPL"
    function = "TIME_SERIES_DAILY"
    
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}'
    
    print(f"Testing Alpha Vantage API with URL: {url}")
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            print(f"Error: {data['Error Message']}")
            return False
        
        # Check if we have time series data
        if 'Time Series (Daily)' not in data:
            print("Error: No time series data found")
            return False
        
        # Print some data
        time_series = data['Time Series (Daily)']
        dates = list(time_series.keys())
        
        if len(dates) > 0:
            latest_date = dates[0]
            print(f"Latest date: {latest_date}")
            print(f"Data: {time_series[latest_date]}")
            return True
        else:
            print("Error: No dates found in time series data")
            return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print(f"Using Alpha Vantage API key: {ALPHA_VANTAGE_API_KEY}")
    result = test_alpha_vantage_api()
    
    if result:
        print("Alpha Vantage API test successful!")
    else:
        print("Alpha Vantage API test failed!") 