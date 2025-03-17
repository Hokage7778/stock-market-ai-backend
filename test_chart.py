import pandas as pd
import plotly.graph_objects as go
import base64

# Create a simple dataframe with sample data
data = {
    'date': pd.date_range(start='2023-01-01', periods=10),
    'open': [150, 151, 149, 152, 153, 155, 156, 153, 152, 151],
    'high': [155, 156, 153, 157, 158, 159, 160, 158, 157, 155],
    'low': [148, 149, 147, 150, 151, 153, 154, 151, 150, 149],
    'close': [151, 149, 152, 153, 155, 156, 153, 152, 151, 150]
}

df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

fig.update_layout(
    title='Test Stock Price Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Try to save the chart as a PNG image
try:
    img_bytes = fig.to_image(format="png")
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    print("Chart generation successful!")
    print(f"Base64 image length: {len(encoded_image)}")
except Exception as e:
    print(f"Error generating chart: {e}")

# Save the chart to a file as a test
try:
    fig.write_image("test_chart.png")
    print("Chart saved to file successfully!")
except Exception as e:
    print(f"Error saving chart to file: {e}") 