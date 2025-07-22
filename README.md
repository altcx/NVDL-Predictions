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

## Usage

### Running the Complete Pipeline

To execute the entire prediction pipeline from data collection to visualization:

```bash
python main.py --save-results
```

This will:
1. Collect historical NVDL stock data from Alpaca Markets
2. Preprocess and engineer features from the data
3. Train both LSTM and ARIMA prediction models
4. Evaluate model performance with various metrics
5. Simulate trading strategies based on model predictions
6. Generate interactive visualizations and a comprehensive dashboard
7. Save all results to the `results/` directory

### Configuration

Adjust parameters in `config.py` to customize:
- Data collection period
- Model hyperparameters
- Trading simulation settings
- Visualization options

### Visualizations

After running the pipeline, the following visualizations will be available in the `results/` directory:
- Price charts with buy/sell signals
- Equity curve comparisons between models
- Performance metric comparisons
- Confusion matrices
- Comprehensive dashboard with all results

## Project Structure

- `data/`: Data collection and preprocessing
- `models/`: Prediction models (LSTM, ARIMA)
- `utils/`: Utility functions and logging
- `tests/`: Unit tests
- `visualization/`: Data visualization tools
- `main.py`: Main execution pipeline
- `config.py`: Configuration parameters
- `results/`: Generated visualizations and results
- `checkpoints/`: Saved model checkpoints