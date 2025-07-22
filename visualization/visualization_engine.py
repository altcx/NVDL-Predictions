"""
Interactive visualization system for NVDL Stock Predictor
Implements Plotly-based charts and dashboards for model results visualization
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.logger import get_visualization_logger


class VisualizationEngine:
    """
    Visualization engine for creating interactive Plotly charts
    Implements price charts, equity curves, and model comparison visualizations
    """
    
    def __init__(self):
        """Initialize VisualizationEngine with logger"""
        self.logger = get_visualization_logger()
        self.logger.info("Initialized VisualizationEngine")
        
        # Default color scheme
        self.colors = {
            'lstm': '#1f77b4',  # blue
            'arima': '#ff7f0e',  # orange
            'buy': '#2ca02c',    # green
            'sell': '#d62728',   # red
            'price': '#7f7f7f',  # gray
            'volume': '#17becf'  # cyan
        }
    
    def plot_price_with_signals(
        self, 
        prices: pd.Series, 
        signals: np.ndarray,
        model_name: str = "Model",
        show_volume: bool = True,
        volume_data: Optional[pd.Series] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create price chart with buy/sell signal overlays
        
        Args:
            prices: Series of asset prices indexed by date
            signals: Array of binary signals (1 for buy, 0 for sell)
            model_name: Name of the model for the title
            show_volume: Whether to include volume subplot
            volume_data: Optional volume data series
            title: Optional custom title
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If prices and signals have different lengths
        """
        if len(prices) != len(signals):
            self.logger.error(f"Length mismatch: prices {len(prices)}, signals {len(signals)}")
            raise ValueError("Prices and signals must have the same length")
        
        self.logger.info(f"Creating price chart with signals for {model_name}")
        
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2 if show_volume and volume_data is not None else 1, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3] if show_volume and volume_data is not None else [1]
        )
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode='lines',
                name='Price',
                line=dict(color=self.colors['price'], width=1.5)
            ),
            row=1, col=1
        )
        
        # Find buy and sell signals
        buy_signals = np.where(signals == 1)[0]
        sell_signals = np.where(signals == 0)[0]
        
        # Add buy signals
        if len(buy_signals) > 0:
            buy_dates = [prices.index[i] for i in buy_signals if i < len(prices)]
            buy_prices = [prices.iloc[i] for i in buy_signals if i < len(prices)]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color=self.colors['buy'],
                        line=dict(width=1, color='darkgreen')
                    )
                ),
                row=1, col=1
            )
        
        # Add sell signals
        if len(sell_signals) > 0:
            sell_dates = [prices.index[i] for i in sell_signals if i < len(prices)]
            sell_prices = [prices.iloc[i] for i in sell_signals if i < len(prices)]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color=self.colors['sell'],
                        line=dict(width=1, color='darkred')
                    )
                ),
                row=1, col=1
            )
        
        # Add volume subplot if requested
        if show_volume and volume_data is not None:
            fig.add_trace(
                go.Bar(
                    x=volume_data.index,
                    y=volume_data.values,
                    name='Volume',
                    marker=dict(color=self.colors['volume'])
                ),
                row=2, col=1
            )
        
        # Set title
        chart_title = title or f"{model_name} Predictions for {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}"
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            hovermode="x unified"
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Add hover tooltips
        fig.update_traces(
            hovertemplate="<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>",
            selector=dict(name='Price')
        )
        
        if show_volume and volume_data is not None:
            fig.update_traces(
                hovertemplate="<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>",
                selector=dict(name='Volume')
            )
        
        self.logger.info(f"Created price chart with {len(buy_signals)} buy and {len(sell_signals)} sell signals")
        return fig
    
    def plot_equity_curves(
        self, 
        lstm_equity: pd.Series, 
        arima_equity: pd.Series,
        benchmark_equity: Optional[pd.Series] = None,
        title: str = "Strategy Comparison"
    ) -> go.Figure:
        """
        Plot equity curves for LSTM and ARIMA strategies
        
        Args:
            lstm_equity: Equity curve for LSTM strategy
            arima_equity: Equity curve for ARIMA strategy
            benchmark_equity: Optional benchmark equity curve (e.g., buy and hold)
            title: Plot title
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If equity curves have different indices
        """
        # Validate inputs
        if not lstm_equity.index.equals(arima_equity.index):
            self.logger.error("Equity curves must have the same index")
            raise ValueError("Equity curves must have the same index")
        
        self.logger.info("Creating equity curve comparison plot")
        
        fig = go.Figure()
        
        # Add LSTM equity curve
        fig.add_trace(go.Scatter(
            x=lstm_equity.index,
            y=lstm_equity.values,
            mode='lines',
            name='LSTM Strategy',
            line=dict(color=self.colors['lstm'], width=2)
        ))
        
        # Add ARIMA equity curve
        fig.add_trace(go.Scatter(
            x=arima_equity.index,
            y=arima_equity.values,
            mode='lines',
            name='ARIMA Strategy',
            line=dict(color=self.colors['arima'], width=2)
        ))
        
        # Add benchmark if provided
        if benchmark_equity is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity.values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        # Calculate and display returns
        lstm_return = (lstm_equity.iloc[-1] / lstm_equity.iloc[0] - 1) * 100
        arima_return = (arima_equity.iloc[-1] / arima_equity.iloc[0] - 1) * 100
        
        annotations = [
            dict(
                x=lstm_equity.index[-1],
                y=lstm_equity.iloc[-1],
                xref="x",
                yref="y",
                text=f"LSTM: {lstm_return:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=50,
                ay=-30,
                font=dict(color=self.colors['lstm'])
            ),
            dict(
                x=arima_equity.index[-1],
                y=arima_equity.iloc[-1],
                xref="x",
                yref="y",
                text=f"ARIMA: {arima_return:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=50,
                ay=30,
                font=dict(color=self.colors['arima'])
            )
        ]
        
        if benchmark_equity is not None:
            benchmark_return = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1) * 100
            annotations.append(
                dict(
                    x=benchmark_equity.index[-1],
                    y=benchmark_equity.iloc[-1],
                    xref="x",
                    yref="y",
                    text=f"B&H: {benchmark_return:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    ax=50,
                    ay=0,
                    font=dict(color='gray')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=annotations,
            template="plotly_white",
            hovermode="x unified"
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        self.logger.info("Created equity curve comparison plot")
        return fig
    
    def plot_model_comparison(
        self, 
        metrics_df: pd.DataFrame,
        highlight_best: bool = True
    ) -> go.Figure:
        """
        Create model comparison visualization with performance metrics
        
        Args:
            metrics_df: DataFrame with models as rows and metrics as columns
            highlight_best: Whether to highlight the best model for each metric
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If metrics_df is empty
        """
        if metrics_df.empty:
            self.logger.error("Metrics DataFrame is empty")
            raise ValueError("metrics_df cannot be empty")
        
        self.logger.info(f"Creating model comparison visualization for {len(metrics_df)} models")
        
        # Select relevant metrics for visualization
        viz_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
        
        # Filter metrics that exist in the DataFrame
        available_metrics = [m for m in viz_metrics if m in metrics_df.columns]
        
        if not available_metrics:
            self.logger.error("No visualization metrics found in DataFrame")
            raise ValueError("No relevant metrics found for visualization")
        
        # Create subplots - one row per metric
        fig = make_subplots(
            rows=len(available_metrics),
            cols=1,
            subplot_titles=[m.replace('_', ' ').title() for m in available_metrics],
            vertical_spacing=0.1
        )
        
        # Colors for different models
        model_colors = {
            'LSTM': self.colors['lstm'],
            'ARIMA': self.colors['arima']
        }
        
        # Default colors for models not in the predefined list
        default_colors = px.colors.qualitative.Plotly
        
        # Add bars for each metric
        for i, metric in enumerate(available_metrics):
            # Determine if higher or lower is better for this metric
            higher_is_better = metric not in ['max_drawdown', 'rmse', 'mae', 'mape']
            
            # Sort values based on whether higher or lower is better
            sorted_df = metrics_df.sort_values(by=metric, ascending=not higher_is_better)
            
            # Get best value for highlighting
            best_value = sorted_df[metric].iloc[0] if len(sorted_df) > 0 else None
            
            # Create bar colors
            bar_colors = []
            for model in sorted_df.index:
                if model in model_colors:
                    color = model_colors[model]
                else:
                    color_idx = list(sorted_df.index).index(model) % len(default_colors)
                    color = default_colors[color_idx]
                
                # Highlight best model if requested
                if highlight_best and best_value is not None and sorted_df.loc[model, metric] == best_value:
                    # Make color more saturated for best model
                    bar_colors.append(color)
                else:
                    # Make color more transparent for non-best models
                    bar_colors.append(color)
            
            # Add bar trace
            fig.add_trace(
                go.Bar(
                    x=sorted_df.index,
                    y=sorted_df[metric],
                    name=metric.replace('_', ' ').title(),
                    marker_color=bar_colors,
                    text=[f"{v:.4f}" for v in sorted_df[metric]],
                    textposition='auto',
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # Format y-axis based on metric
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'win_rate']:
                fig.update_yaxes(
                    tickformat='.0%',
                    range=[0, 1],
                    row=i+1, col=1
                )
            elif metric == 'max_drawdown':
                fig.update_yaxes(
                    tickformat='.0%',
                    row=i+1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Model Performance Comparison",
            height=200 * len(available_metrics),
            template="plotly_white",
            showlegend=False
        )
        
        self.logger.info(f"Created model comparison visualization with {len(available_metrics)} metrics")
        return fig
    
    def create_dashboard(
        self, 
        results: Dict[str, Dict],
        prices: pd.Series,
        volume: Optional[pd.Series] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            results: Dictionary containing results for each model
            prices: Series of asset prices
            volume: Optional volume data
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If results dictionary is empty
        """
        if not results:
            self.logger.error("Results dictionary is empty")
            raise ValueError("results dictionary cannot be empty")
        
        self.logger.info(f"Creating comprehensive dashboard for {len(results)} models")
        
        # Extract model names
        model_names = list(results.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Price Chart with Signals",
                "Equity Curves",
                "Model Performance Metrics",
                "Drawdown Analysis",
                "Trade Win Rate",
                "Return Distribution"
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Add price chart with signals for the first model
        if len(model_names) > 0:
            model = model_names[0]
            if 'predictions' in results[model]:
                signals = results[model]['predictions']
                
                # Add price trace
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices.values,
                        mode='lines',
                        name='Price',
                        line=dict(color=self.colors['price'], width=1.5)
                    ),
                    row=1, col=1
                )
                
                # Find buy and sell signals
                buy_signals = np.where(signals == 1)[0]
                sell_signals = np.where(signals == 0)[0]
                
                # Add buy signals
                if len(buy_signals) > 0:
                    buy_dates = [prices.index[i] for i in buy_signals if i < len(prices)]
                    buy_prices = [prices.iloc[i] for i in buy_signals if i < len(prices)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode='markers',
                            name=f'{model} Buy',
                            marker=dict(
                                symbol='triangle-up',
                                size=8,
                                color=self.colors['buy'],
                                line=dict(width=1, color='darkgreen')
                            )
                        ),
                        row=1, col=1
                    )
                
                # Add sell signals
                if len(sell_signals) > 0:
                    sell_dates = [prices.index[i] for i in sell_signals if i < len(prices)]
                    sell_prices = [prices.iloc[i] for i in sell_signals if i < len(prices)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sell_dates,
                            y=sell_prices,
                            mode='markers',
                            name=f'{model} Sell',
                            marker=dict(
                                symbol='triangle-down',
                                size=8,
                                color=self.colors['sell'],
                                line=dict(width=1, color='darkred')
                            )
                        ),
                        row=1, col=1
                    )
        
        # 2. Add equity curves
        equity_curves = {}
        for model in model_names:
            if 'equity_curve' in results[model]:
                equity_curves[model] = results[model]['equity_curve']
        
        if len(equity_curves) >= 2:
            for i, (model, equity) in enumerate(equity_curves.items()):
                color = self.colors['lstm'] if model == 'LSTM' else self.colors['arima'] if model == 'ARIMA' else px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
                fig.add_trace(
                    go.Scatter(
                        x=equity.index,
                        y=equity.values,
                        mode='lines',
                        name=f'{model} Equity',
                        line=dict(color=color, width=2)
                    ),
                    row=2, col=1
                )
        
        # 3. Add performance metrics
        metrics_data = {}
        for model in model_names:
            model_metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                if metric in results[model]:
                    model_metrics[metric] = results[model][metric]
            
            if model_metrics:
                metrics_data[model] = model_metrics
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data).T
            
            # Select a few key metrics for the dashboard
            key_metrics = ['accuracy', 'sharpe_ratio', 'win_rate']
            available_metrics = [m for m in key_metrics if m in metrics_df.columns]
            
            if available_metrics:
                for i, metric in enumerate(available_metrics):
                    bar_colors = [self.colors['lstm'] if model == 'LSTM' else self.colors['arima'] if model == 'ARIMA' else 
                                 px.colors.qualitative.Plotly[list(metrics_df.index).index(model) % len(px.colors.qualitative.Plotly)] 
                                 for model in metrics_df.index]
                    
                    fig.add_trace(
                        go.Bar(
                            x=metrics_df.index,
                            y=metrics_df[metric],
                            name=metric.replace('_', ' ').title(),
                            marker_color=bar_colors,
                            text=[f"{v:.2f}" for v in metrics_df[metric]],
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
                    break  # Just show one metric in the dashboard
        
        # 4. Add drawdown analysis
        for i, (model, equity) in enumerate(equity_curves.items()):
            if len(equity) > 1:
                # Calculate running maximum
                running_max = equity.cummax()
                
                # Calculate drawdown
                drawdown = (equity - running_max) / running_max
                
                color = self.colors['lstm'] if model == 'LSTM' else self.colors['arima'] if model == 'ARIMA' else px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values * 100,  # Convert to percentage
                        mode='lines',
                        name=f'{model} Drawdown',
                        line=dict(color=color, width=2),
                        fill='tozeroy'
                    ),
                    row=3, col=1
                )
        
        # 5. Add win rate visualization
        win_rates = {}
        for model in model_names:
            if 'win_rate' in results[model]:
                win_rates[model] = results[model]['win_rate']
        
        if win_rates:
            models = list(win_rates.keys())
            values = list(win_rates.values())
            
            bar_colors = [self.colors['lstm'] if model == 'LSTM' else self.colors['arima'] if model == 'ARIMA' else 
                         px.colors.qualitative.Plotly[models.index(model) % len(px.colors.qualitative.Plotly)] 
                         for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name='Win Rate',
                    marker_color=bar_colors,
                    text=[f"{v:.1%}" for v in values],
                    textposition='auto'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="NVDL Stock Prediction Dashboard",
            height=1000,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes titles
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
        
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Model", row=3, col=2)
        fig.update_yaxes(title_text="Win Rate", tickformat='.0%', row=3, col=2)
        
        self.logger.info("Created comprehensive dashboard")
        return fig
    
    def save_figure(
        self, 
        fig: go.Figure, 
        filename: str, 
        formats: List[str] = ['html', 'png']
    ) -> None:
        """
        Save figure to file in multiple formats
        
        Args:
            fig: Plotly figure object
            filename: Base filename without extension
            formats: List of formats to save (html, png, jpg, pdf, svg)
            
        Returns:
            None
        """
        self.logger.info(f"Saving figure to {filename} in formats: {formats}")
        
        for fmt in formats:
            if fmt.lower() == 'html':
                fig.write_html(f"{filename}.html")
            elif fmt.lower() == 'png':
                fig.write_image(f"{filename}.png")
            elif fmt.lower() == 'jpg':
                fig.write_image(f"{filename}.jpg")
            elif fmt.lower() == 'pdf':
                fig.write_image(f"{filename}.pdf")
            elif fmt.lower() == 'svg':
                fig.write_image(f"{filename}.svg")
            else:
                self.logger.warning(f"Unsupported format: {fmt}")
        
        self.logger.info(f"Figure saved successfully")