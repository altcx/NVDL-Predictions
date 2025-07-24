# NVDL Stock Predictor

An AI-powered stock prediction system that uses machine learning models to predict stock price movements and generate trading signals for NVDL (GraniteShares 1.5x Long NVDA Daily ETF).

## Features

- **Dual Model Approach**: Combines LSTM neural networks and ARIMA time series models
- **Real-time Data**: Fetches live market data from Alpaca Markets API
- **Advanced Analytics**: Comprehensive performance metrics and statistical analysis
- **Interactive Visualizations**: Dynamic charts and dashboards using Plotly
- **Backtesting Engine**: Simulates trading strategies with realistic transaction costs
- **Risk Management**: Includes drawdown analysis and risk-adjusted returns
- **Automated Reporting**: Generates detailed performance reports in multiple formats

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your Alpaca Markets API keys
   ```

3. **Run the predictor:**
   ```bash
   python main.py
   ```

4. **View results:**
   Open `results/model_comparison_dashboard.html` in your browser

## Architecture

The system follows a modular architecture with the following components:

### Data Layer
- **Data Collector**: Fetches historical and real-time market data
- **Data Preprocessor**: Cleans, normalizes, and prepares data for model training

### Model Layer
- **LSTM Predictor**: Deep learning model for capturing complex temporal patterns
- **ARIMA Predictor**: Statistical model for time series forecasting
- **Model Evaluator**: Comprehensive evaluation and comparison framework

### Trading Layer
- **Trading Simulator**: Backtesting engine with realistic transaction costs
- **Performance Analyzer**: Risk metrics and portfolio analytics

### Visualization Layer
- **Visualization Engine**: Interactive charts and dashboards
- **Report Generator**: Automated report generation in multiple formats

## Installation

### Prerequisites
- Python 3.8 or higher
- Alpaca Markets API account (free at [alpaca.markets](https://alpaca.markets/))

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nvdl-stock-predictor.git
   cd nvdl-stock-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Alpaca API credentials:
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ```

4. **Optional: Install additional dependencies:**
   ```bash
   # For LSTM model (recommended)
   pip install tensorflow

   # For Excel export
   pip install openpyxl
   ```

## Usage

### Basic Usage

Run the complete prediction pipeline:
```bash
python main.py
```

This will:
1. Fetch historical data for NVDL
2. Train both LSTM and ARIMA models
3. Generate predictions and trading signals
4. Create comprehensive analysis reports
5. Generate interactive visualizations

### Command Line Options

```bash
# Predict specific symbol
python main.py --symbol NVDL

# Custom date range
python main.py --start-date 2023-01-01 --end-date 2024-01-01

# Use specific models only
python main.py --models lstm
python main.py --models arima

# Show all options
python main.py --help
```

### Programmatic Usage

```python
from main import NVDLPredictorPipeline

# Initialize pipeline
pipeline = NVDLPredictorPipeline()

# Run complete analysis
results = pipeline.run_pipeline()

# Access results
lstm_results = results['lstm_results']
arima_results = results['arima_results']
analysis_results = results['analysis_results']
```

## Output Files

The system generates comprehensive output in the `results/` directory:

### Interactive Dashboards
- `model_comparison_dashboard.html` - Main interactive dashboard
- `lstm_price_chart.html` - LSTM predictions visualization
- `arima_price_chart.html` - ARIMA predictions visualization
- `equity_comparison.html` - Strategy performance comparison

### Performance Reports
- `reports/performance_report_*.html` - Detailed HTML reports
- `reports/performance_report_*.json` - Machine-readable data
- `reports/performance_report_*.md` - Markdown documentation

### Data Exports
- `exports_*/classification_metrics.csv` - Model accuracy metrics
- `exports_*/trading_metrics.csv` - Trading performance data
- `exports_*/model_predictions.csv` - Raw predictions

### Model Checkpoints
- `checkpoints/lstm_model_final.h5` - Trained LSTM model
- `checkpoints/arima_model_final.pkl` - Trained ARIMA model

## Model Performance Metrics

### Classification Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: Reliability of buy signals
- **Recall**: Ability to capture opportunities
- **F1-Score**: Balanced accuracy measure
- **Specificity**: Ability to avoid false signals

### Trading Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of profits to losses

### Statistical Tests
- **McNemar's Test**: Significance of prediction differences
- **T-Test**: Significance of return differences
- **Wilcoxon Test**: Non-parametric return comparison

## Configuration

Key configuration options in `.env`:

```bash
# Data Configuration
SYMBOL=NVDL                    # Stock symbol to predict
LOOKBACK_YEARS=2               # Years of historical data
TEST_SIZE=0.2                  # Fraction for testing

# Trading Configuration
INITIAL_CAPITAL=10000          # Starting portfolio value
RISK_FREE_RATE=0.02           # Risk-free rate for Sharpe ratio

# LSTM Configuration
LSTM_SEQUENCE_LENGTH=60        # Input sequence length
LSTM_UNITS=50                  # Number of LSTM units
LSTM_EPOCHS=100               # Training epochs

# ARIMA Configuration
ARIMA_MAX_P=5                 # Maximum AR order
ARIMA_MAX_D=2                 # Maximum differencing
ARIMA_MAX_Q=5                 # Maximum MA order
```

## Project Structure

```
nvdl-stock-predictor/
├── data/                      # Data collection and preprocessing
│   ├── collector.py          # Market data fetching
│   └── preprocessor.py       # Data cleaning and preparation
├── models/                    # Prediction models
│   ├── lstm_predictor.py     # LSTM neural network
│   ├── arima_predictor.py    # ARIMA time series model
│   ├── model_evaluator.py    # Model evaluation framework
│   └── trading_simulator.py  # Backtesting engine
├── utils/                     # Utility functions
│   ├── error_handler.py      # Error handling framework
│   ├── logger.py             # Structured logging
│   └── results_analyzer.py   # Results analysis and reporting
├── visualization/             # Data visualization
│   └── visualization_engine.py # Interactive charts and dashboards
├── tests/                     # Unit tests
├── results/                   # Generated outputs
├── checkpoints/              # Saved models
├── main.py                   # Main execution pipeline
├── config.py                 # Configuration parameters
└── requirements.txt          # Dependencies
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=.

# Run specific test categories
pytest tests/test_models/
pytest tests/test_data/
pytest tests/test_visualization/
```

## Troubleshooting

### Common Issues

**API Key Error:**
```
Configuration error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set
```
Solution: Set your Alpaca API keys in the `.env` file

**TensorFlow Not Available:**
```
TensorFlow not available, LSTM model will not be available
```
Solution: Install TensorFlow with `pip install tensorflow`

**No Data Retrieved:**
- Check internet connection
- Verify API keys are valid
- Ensure market is open or use historical data

**Memory Issues:**
- Reduce `LOOKBACK_YEARS` in configuration
- Use smaller `LSTM_BATCH_SIZE`
- Close other applications

## Error Handling System

The system includes a comprehensive error handling framework:

- **Robust API Error Handling**: Automatic retry with exponential backoff
- **Data Validation**: Thorough validation with detailed error reporting
- **Model Training Protection**: Error handling for convergence issues
- **Progress Tracking**: Real-time progress monitoring
- **System Monitoring**: Resource usage tracking and alerts
- **Graceful Recovery**: Cleanup procedures and state preservation
- **Detailed Logging**: Structured logging with component-specific files
- **Error Reports**: Comprehensive error reports with diagnostics

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up development environment
- Code style and standards
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Important**: This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Support

For questions, issues, or feature requests:

1. Check the [documentation](results/interpreting_results.md)
2. Search existing issues
3. Create a new issue with detailed information

## Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for providing market data API
- [TensorFlow](https://tensorflow.org/) and [scikit-learn](https://scikit-learn.org/) communities
- [Plotly](https://plotly.com/) for visualization capabilities
- Open source contributors and maintainers

---

**Star ⭐ this repository if you find it useful!**