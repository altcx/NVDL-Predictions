"""
Trading simulation and backtesting module for NVDL Stock Predictor
Implements portfolio tracking and performance metrics calculation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import plotly.graph_objects as go
from utils.logger import get_trading_logger
from config import config


class TradingSimulator:
    """
    Trading simulation class for backtesting prediction models
    Implements buy/sell/hold strategy simulation and performance metrics calculation
    """
    
    def __init__(self, initial_capital: float = None):
        """
        Initialize TradingSimulator with configurable parameters
        
        Args:
            initial_capital: Starting capital for the portfolio (defaults to config)
        """
        self.logger = get_trading_logger()
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.logger.info(f"Initialized TradingSimulator with initial_capital=${self.initial_capital:.2f}")
    
    def simulate_strategy(
        self, 
        prices: pd.Series, 
        signals: np.ndarray,
        commission: float = 0.001,  # 0.1% commission per trade
        slippage: float = 0.001     # 0.1% slippage per trade
    ) -> Dict[str, Union[pd.Series, float, int]]:
        """
        Simulate buy/sell/hold trading strategy based on model predictions
        
        Args:
            prices: Series of asset prices indexed by date
            signals: Array of binary signals (1 for buy, 0 for sell)
            commission: Commission rate per trade as a decimal
            slippage: Slippage rate per trade as a decimal
            
        Returns:
            Dictionary containing equity curve and performance metrics
            
        Raises:
            ValueError: If prices and signals have different lengths
        """
        if len(prices) != len(signals):
            self.logger.error(f"Length mismatch: prices {len(prices)}, signals {len(signals)}")
            raise ValueError("Prices and signals must have the same length")
        
        self.logger.info(f"Simulating trading strategy with {len(signals)} signals")
        
        # Initialize portfolio tracking variables
        equity = pd.Series(index=prices.index, dtype=float)
        equity.iloc[0] = self.initial_capital
        
        position = 0  # Current position (number of shares)
        cash = self.initial_capital  # Available cash
        
        # Transaction history
        transactions = []
        
        # Iterate through prices and signals
        for i in range(1, len(prices)):
            date = prices.index[i]
            current_price = prices.iloc[i]
            signal = signals[i-1]  # Signal from previous day for today's action
            
            # Determine action based on signal
            if signal == 1 and position == 0:  # Buy signal and no position
                # Calculate shares to buy with all available cash
                shares_to_buy = cash / (current_price * (1 + slippage))
                # Account for commission
                transaction_cost = shares_to_buy * current_price * commission
                # Adjust shares to buy after commission
                shares_to_buy = (cash - transaction_cost) / (current_price * (1 + slippage))
                
                # Execute buy
                position = shares_to_buy
                cash = 0
                
                transactions.append({
                    'date': date,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'value': shares_to_buy * current_price,
                    'commission': transaction_cost
                })
                
                self.logger.debug(f"BUY: {shares_to_buy:.2f} shares at ${current_price:.2f}")
                
            elif signal == 0 and position > 0:  # Sell signal and has position
                # Calculate sale value with slippage
                sale_value = position * current_price * (1 - slippage)
                # Account for commission
                transaction_cost = sale_value * commission
                # Adjust cash after commission
                cash = sale_value - transaction_cost
                
                transactions.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': sale_value,
                    'commission': transaction_cost
                })
                
                self.logger.debug(f"SELL: {position:.2f} shares at ${current_price:.2f}")
                position = 0
            
            # Update equity value
            equity.iloc[i] = cash + (position * current_price)
        
        # If still holding at the end, calculate final equity
        if position > 0:
            self.logger.info(f"Still holding {position:.2f} shares at simulation end")
        
        # Create transactions DataFrame
        transactions_df = pd.DataFrame(transactions)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(equity, prices, transactions_df)
        
        # Combine results
        results = {
            'equity_curve': equity,
            'transactions': transactions_df,
            'final_equity': equity.iloc[-1],
            'total_return': metrics['total_return'],
            'annualized_return': metrics['annualized_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'num_trades': metrics['num_trades']
        }
        
        self.logger.info(f"Simulation completed: final equity ${equity.iloc[-1]:.2f}, "
                         f"return {metrics['total_return']:.2%}, "
                         f"trades {metrics['num_trades']}")
        
        return results
    
    def calculate_performance_metrics(
        self, 
        equity_curve: pd.Series,
        prices: pd.Series,
        transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics from equity curve
        
        Args:
            equity_curve: Series of portfolio values over time
            prices: Series of asset prices
            transactions: DataFrame of trading transactions
            
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Calculating performance metrics")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics['total_return'] = total_return
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
            metrics['annualized_return'] = annualized_return
        else:
            metrics['annualized_return'] = 0
        
        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        # Volatility (annualized)
        if len(daily_returns) > 1:
            volatility = daily_returns.std() * np.sqrt(252)
            metrics['volatility'] = volatility
        else:
            metrics['volatility'] = 0
        
        # Sharpe ratio
        if 'volatility' in metrics and metrics['volatility'] > 0:
            excess_return = annualized_return - config.RISK_FREE_RATE
            sharpe_ratio = excess_return / metrics['volatility']
            metrics['sharpe_ratio'] = sharpe_ratio
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum drawdown
        metrics['max_drawdown'] = self.calculate_max_drawdown(equity_curve)
        
        # Trade statistics
        if not transactions.empty:
            # Number of trades
            metrics['num_trades'] = len(transactions[transactions['action'] == 'SELL'])
            
            # Calculate trade returns
            buy_transactions = transactions[transactions['action'] == 'BUY']
            sell_transactions = transactions[transactions['action'] == 'SELL']
            
            if len(buy_transactions) > 0 and len(sell_transactions) > 0:
                # Match buys and sells
                trade_returns = []
                
                for i in range(min(len(buy_transactions), len(sell_transactions))):
                    buy_price = buy_transactions.iloc[i]['price']
                    buy_value = buy_transactions.iloc[i]['value']
                    sell_price = sell_transactions.iloc[i]['price']
                    sell_value = sell_transactions.iloc[i]['value']
                    
                    # Calculate return for this trade
                    trade_return = (sell_value / buy_value) - 1
                    trade_returns.append(trade_return)
                
                # Win rate
                winning_trades = sum(1 for ret in trade_returns if ret > 0)
                metrics['win_rate'] = winning_trades / len(trade_returns) if trade_returns else 0
                
                # Average return per trade
                metrics['avg_trade_return'] = np.mean(trade_returns) if trade_returns else 0
                
                # Profit factor
                gross_profits = sum(ret for ret in trade_returns if ret > 0)
                gross_losses = abs(sum(ret for ret in trade_returns if ret < 0))
                metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            else:
                metrics['win_rate'] = 0
                metrics['avg_trade_return'] = 0
                metrics['profit_factor'] = 0
        else:
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0
            metrics['avg_trade_return'] = 0
            metrics['profit_factor'] = 0
        
        self.logger.info(f"Performance metrics calculated: {metrics}")
        return metrics
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve
        
        Args:
            equity_curve: Series of portfolio values over time
            
        Returns:
            Maximum drawdown as a decimal
        """
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = None,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio for a series of returns
        
        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate (defaults to config)
            periods_per_year: Number of periods in a year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = config.RISK_FREE_RATE
        
        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calculate excess returns
        excess_returns = returns - period_risk_free
        
        # Calculate annualized Sharpe ratio
        if len(excess_returns) > 1 and excess_returns.std() > 0:
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe = 0
        
        return sharpe
    
    def plot_equity_curve(
        self, 
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve"
    ) -> go.Figure:
        """
        Plot equity curve using Plotly
        
        Args:
            equity_curve: Series of portfolio values over time
            benchmark: Optional benchmark series for comparison
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to same starting value
            benchmark_normalized = benchmark * (equity_curve.iloc[0] / benchmark.iloc[0])
            
            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        # Add buy and hold line
        buy_hold = pd.Series(index=equity_curve.index)
        buy_hold.iloc[0] = equity_curve.iloc[0]
        for i in range(1, len(equity_curve)):
            buy_hold.iloc[i] = buy_hold.iloc[0] * (benchmark.iloc[i] / benchmark.iloc[0]) if benchmark is not None else buy_hold.iloc[0]
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            legend=dict(x=0.01, y=0.99),
            template="plotly_white"
        )
        
        return fig
    
    def plot_drawdown_chart(self, equity_curve: pd.Series) -> go.Figure:
        """
        Plot drawdown chart using Plotly
        
        Args:
            equity_curve: Series of portfolio values over time
            
        Returns:
            Plotly figure object
        """
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        fig = go.Figure()
        
        # Add drawdown trace
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ))
        
        # Update layout
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis=dict(tickformat='.1f'),
            template="plotly_white"
        )
        
        return fig
    
    def generate_trade_statistics(self, transactions: pd.DataFrame) -> Dict[str, float]:
        """
        Generate detailed trade statistics
        
        Args:
            transactions: DataFrame of trading transactions
            
        Returns:
            Dictionary of trade statistics
        """
        if transactions.empty:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_holding_period': 0
            }
        
        # Filter buy and sell transactions
        buys = transactions[transactions['action'] == 'BUY']
        sells = transactions[transactions['action'] == 'SELL']
        
        # Match buys and sells to create trades
        trades = []
        
        for i in range(min(len(buys), len(sells))):
            buy = buys.iloc[i]
            sell = sells.iloc[i]
            
            trade = {
                'entry_date': buy['date'],
                'exit_date': sell['date'],
                'entry_price': buy['price'],
                'exit_price': sell['price'],
                'shares': buy['shares'],
                'entry_value': buy['value'],
                'exit_value': sell['value'],
                'return': (sell['value'] / buy['value']) - 1,
                'holding_period': (sell['date'] - buy['date']).days
            }
            
            trades.append(trade)
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate statistics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['return'] > 0]
            losing_trades = trades_df[trades_df['return'] <= 0]
            
            stats = {
                'num_trades': len(trades_df),
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                'avg_win': winning_trades['return'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['return'].mean() if len(losing_trades) > 0 else 0,
                'largest_win': trades_df['return'].max() if len(trades_df) > 0 else 0,
                'largest_loss': trades_df['return'].min() if len(trades_df) > 0 else 0,
                'avg_holding_period': trades_df['holding_period'].mean() if len(trades_df) > 0 else 0
            }
        else:
            stats = {
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_holding_period': 0
            }
        
        return stats