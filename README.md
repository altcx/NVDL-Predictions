# NVDL Stock Predictor

A stock price prediction system for NVDL using machine learning and time series analysis.

## Features

- Historical stock data collection via Alpaca Markets API
- Data validation and preprocessing
- Multiple prediction models (LSTM, ARIMA)
- Performance evaluation and backtesting
- Visualization of predictions and results

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - `ALPACA_API_KEY`: Your Alpaca Markets API key
   - `ALPACA_SECRET_KEY`: Your Alpaca Markets secret key

3. Run tests:
   ```
   python -m pytest
   ```

## Project Structure

- `data/`: Data collection and preprocessing
- `models/`: Prediction models (LSTM, ARIMA)
- `utils/`: Utility functions and logging
- `tests/`: Unit tests
- `visualization/`: Data visualization tools