import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import numpy as np

# Create a simple dataframe with sample data
data = {
    'date': pd.date_range(start='2023-01-01', periods=30),
    'open': np.random.uniform(150, 160, 30),
    'high': np.random.uniform(155, 165, 30),
    'low': np.random.uniform(145, 155, 30),
    'close': np.random.uniform(150, 160, 30)
}

df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Ensure high is the highest value and low is the lowest
for i in range(len(df)):
    values = [df['open'].iloc[i], df['close'].iloc[i], df['high'].iloc[i], df['low'].iloc[i]]
    df['high'].iloc[i] = max(values)
    df['low'].iloc[i] = min(values)

# Calculate moving averages
df['MA20'] = df['close'].rolling(window=20).mean()
df['MA50'] = df['close'].rolling(window=20).mean()  # Using 20 instead of 50 for this small dataset

try:
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Calculate width of candlestick elements
    width = 0.6
    width2 = 0.1
    
    # Define up and down days
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    # Plot up days
    ax.bar(up.index, up.close-up.open, width, bottom=up.open, color='green')
    ax.bar(up.index, up.high-up.close, width2, bottom=up.close, color='green')
    ax.bar(up.index, up.low-up.open, width2, bottom=up.open, color='green')
    
    # Plot down days
    ax.bar(down.index, down.close-down.open, width, bottom=down.open, color='red')
    ax.bar(down.index, down.high-down.open, width2, bottom=down.open, color='red')
    ax.bar(down.index, down.low-down.close, width2, bottom=down.close, color='red')
    
    # Add 20-day moving average if available
    if 'MA20' in df.columns and not df['MA20'].isna().all():
        ax.plot(df.index, df['MA20'], color='blue', linewidth=1.5, label='20-day MA')
    
    # Add 50-day moving average if available
    if 'MA50' in df.columns and not df['MA50'].isna().all():
        ax.plot(df.index, df['MA50'], color='orange', linewidth=1.5, label='50-day MA')
    
    # Add legend if we have moving averages
    if ('MA20' in df.columns and not df['MA20'].isna().all()) or \
       ('MA50' in df.columns and not df['MA50'].isna().all()):
        ax.legend()
    
    # Set title and labels
    ax.set_title(f'Test Candlestick Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode as base64
    image_png = buffer.getvalue()
    buffer.close()
    encoded_image = base64.b64encode(image_png).decode('utf-8')
    
    # Close the figure to free memory
    plt.close(fig)
    
    print("Chart generation successful!")
    print(f"Base64 image length: {len(encoded_image)}")
    
    # Save to a file as well for visual inspection
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title('Simple Line Chart of Close Prices')
    plt.savefig('test_matplotlib.png')
    plt.close()
    
    print("Test chart saved to file: test_matplotlib.png")
    
except Exception as e:
    print(f"Error generating chart: {e}") 