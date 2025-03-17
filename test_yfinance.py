import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time

def test_yfinance():
    """Test yfinance functionality"""
    print("Testing yfinance...")
    
    # Test symbols
    symbols = ["MSFT", "AAPL", "GOOGL"]
    
    # Try different periods
    periods = ["1d", "5d", "1mo", "3mo"]
    
    for symbol in symbols:
        print(f"\nTesting symbol: {symbol}")
        
        for period in periods:
            print(f"  Testing period: {period}")
            
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                
                # Print info
                print(f"  Ticker info available: {bool(stock.info)}")
                
                # Get history with retry
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        df = stock.history(period=period)
                        break
                    except Exception as e:
                        print(f"  Retry {retry+1}/{max_retries}: Error: {e}")
                        if retry < max_retries - 1:
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise
                
                if df.empty:
                    print(f"  No data returned for {symbol} with period {period}")
                    continue
                
                # Print basic info
                print(f"  Data shape: {df.shape}")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Latest price: ${df['Close'].iloc[-1]:.2f}")
                
                # Create a simple plot for the last successful period
                if period == periods[-1] or periods.index(period) == len(periods) - 1:
                    plt.figure(figsize=(10, 5))
                    plt.plot(df.index, df['Close'])
                    plt.title(f"{symbol} Stock Price - {period}")
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
                    
                    print(f"  Successfully created chart for {symbol}")
                    print(f"  Base64 image length: {len(encoded_image)}")
                
            except Exception as e:
                print(f"  Error testing {symbol} with period {period}: {e}")

if __name__ == "__main__":
    test_yfinance()
    print("\nTest completed!") 