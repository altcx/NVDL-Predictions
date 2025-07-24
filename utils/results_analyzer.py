"""
Results analysis and reporting system for NVDL Stock Predictor
Implements comprehensive analysis, statistical testing, and report generation
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from utils.logger import get_main_logger
from utils.error_handler import safe_execute
from models.model_evaluator import ModelEvaluator
from visualization.visualization_engine import VisualizationEngine


class ResultsAnalyzer:
    """
    Results analysis and reporting system
    Implements comprehensive analysis, statistical testing, and report generation
    """
    
    def __init__(self):
        """Initialize ResultsAnalyzer with logger"""
        self.logger = get_main_logger()
        self.logger.info("Initialized ResultsAnalyzer")
        self.model_evaluator = ModelEvaluator()
        self.visualization_engine = VisualizationEngine()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/reports', exist_ok=True)
        
    def analyze_model_performance(
        self,
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create comprehensive analysis comparing model performance
        
        Args:
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            
        Returns:
            Dictionary with comprehensive analysis results
            
        Raises:
            ValueError: If input dictionaries are empty or missing required keys
        """
        self.logger.info("Analyzing model performance")
        
        # Validate inputs
        if not model_results or not evaluation_results or not trading_results:
            self.logger.error("Empty input dictionaries")
            raise ValueError("Input dictionaries cannot be empty")
        
        # Initialize analysis results
        analysis = {
            'summary': {},
            'classification_metrics': {},
            'trading_metrics': {},
            'statistical_tests': {},
            'key_insights': []
        }
        
        # Extract model names
        model_names = list(set(model_results.keys()) & set(evaluation_results.keys()) & set(trading_results.keys()))
        
        if not model_names:
            self.logger.error("No common models found in input dictionaries")
            raise ValueError("No common models found in input dictionaries")
        
        self.logger.info(f"Analyzing performance for models: {model_names}")
        
        # 1. Compile classification metrics
        for model in model_names:
            if model in evaluation_results:
                metrics = evaluation_results[model]
                analysis['classification_metrics'][model] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'specificity': metrics.get('specificity', 0)
                }
        
        # 2. Compile trading metrics
        for model in model_names:
            if model in trading_results:
                metrics = trading_results[model]
                analysis['trading_metrics'][model] = {
                    'total_return': metrics.get('total_return', 0),
                    'annualized_return': metrics.get('annualized_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'num_trades': metrics.get('num_trades', 0)
                }
        
        # 3. Perform statistical significance testing
        if len(model_names) >= 2:
            analysis['statistical_tests'] = self._perform_statistical_tests(model_names, model_results, trading_results)
        
        # 4. Generate summary statistics
        analysis['summary'] = self._generate_summary_statistics(model_names, evaluation_results, trading_results)
        
        # 5. Extract key insights
        analysis['key_insights'] = self._extract_key_insights(model_names, analysis)
        
        self.logger.info("Model performance analysis completed")
        return analysis
    
    def _perform_statistical_tests(
        self,
        model_names: List[str],
        model_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical significance testing between models
        
        Args:
            model_names: List of model names
            model_results: Dictionary with model predictions and data
            trading_results: Dictionary with trading simulation results
            
        Returns:
            Dictionary with statistical test results
        """
        self.logger.info("Performing statistical significance testing")
        
        tests = {}
        
        # Only perform tests if we have at least two models
        if len(model_names) < 2:
            return tests
        
        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                comparison_key = f"{model1}_vs_{model2}"
                tests[comparison_key] = {}
                
                # 1. McNemar's test for prediction agreement
                if (model1 in model_results and 'predictions' in model_results[model1] and 
                    model2 in model_results and 'predictions' in model_results[model2] and
                    'y_true' in model_results[model1] and 'y_true' in model_results[model2]):
                    
                    # Ensure predictions are aligned
                    pred1 = model_results[model1]['predictions']
                    pred2 = model_results[model2]['predictions']
                    y_true = model_results[model1]['y_true']
                    
                    # Only use common length
                    min_len = min(len(pred1), len(pred2), len(y_true))
                    pred1 = pred1[:min_len]
                    pred2 = pred2[:min_len]
                    y_true = y_true[:min_len]
                    
                    # Create contingency table
                    # [both correct, model1 correct & model2 wrong]
                    # [model1 wrong & model2 correct, both wrong]
                    contingency_table = [
                        [sum((pred1 == y_true) & (pred2 == y_true)), sum((pred1 == y_true) & (pred2 != y_true))],
                        [sum((pred1 != y_true) & (pred2 == y_true)), sum((pred1 != y_true) & (pred2 != y_true))]
                    ]
                    
                    try:
                        # Perform McNemar's test
                        result = stats.mcnemar(np.array(contingency_table), exact=False, correction=True)
                        tests[comparison_key]['mcnemar_test'] = {
                            'statistic': float(result.statistic),
                            'p_value': float(result.pvalue),
                            'significant': result.pvalue < 0.05,
                            'better_model': model1 if contingency_table[0][1] > contingency_table[1][0] else model2 if contingency_table[1][0] > contingency_table[0][1] else 'tie'
                        }
                    except Exception as e:
                        self.logger.warning(f"McNemar's test failed: {str(e)}")
                
                # 2. Compare daily returns distributions
                if (model1 in trading_results and 'equity_curve' in trading_results[model1] and 
                    model2 in trading_results and 'equity_curve' in trading_results[model2]):
                    
                    equity1 = trading_results[model1]['equity_curve']
                    equity2 = trading_results[model2]['equity_curve']
                    
                    # Calculate daily returns
                    returns1 = equity1.pct_change().dropna()
                    returns2 = equity2.pct_change().dropna()
                    
                    # Align dates
                    common_dates = returns1.index.intersection(returns2.index)
                    if len(common_dates) > 10:  # Need sufficient data points
                        returns1 = returns1.loc[common_dates]
                        returns2 = returns2.loc[common_dates]
                        
                        try:
                            # Perform t-test for returns
                            t_stat, p_value = stats.ttest_rel(returns1, returns2)
                            tests[comparison_key]['returns_ttest'] = {
                                'statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05,
                                'better_model': model1 if returns1.mean() > returns2.mean() else model2
                            }
                        except Exception as e:
                            self.logger.warning(f"T-test failed: {str(e)}")
                        
                        try:
                            # Perform Wilcoxon signed-rank test (non-parametric)
                            w_stat, p_value = stats.wilcoxon(returns1, returns2)
                            tests[comparison_key]['wilcoxon_test'] = {
                                'statistic': float(w_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05,
                                'better_model': model1 if returns1.mean() > returns2.mean() else model2
                            }
                        except Exception as e:
                            self.logger.warning(f"Wilcoxon test failed: {str(e)}")
        
        return tests
    
    def _generate_summary_statistics(
        self,
        model_names: List[str],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for model comparison
        
        Args:
            model_names: List of model names
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            
        Returns:
            Dictionary with summary statistics
        """
        self.logger.info("Generating summary statistics")
        
        summary = {
            'best_model_overall': None,
            'best_classification_model': None,
            'best_trading_model': None,
            'model_rankings': {},
            'performance_summary': {}
        }
        
        # Calculate classification score (weighted average of metrics)
        classification_scores = {}
        for model in model_names:
            if model in evaluation_results:
                metrics = evaluation_results[model]
                # Weight accuracy and F1 score more heavily
                score = (
                    0.3 * metrics.get('accuracy', 0) + 
                    0.2 * metrics.get('precision', 0) + 
                    0.2 * metrics.get('recall', 0) + 
                    0.3 * metrics.get('f1_score', 0)
                )
                classification_scores[model] = score
        
        # Calculate trading score (weighted average of metrics)
        trading_scores = {}
        for model in model_names:
            if model in trading_results:
                metrics = trading_results[model]
                # Weight returns and Sharpe ratio more heavily
                score = (
                    0.3 * min(1.0, max(0.0, metrics.get('total_return', 0))) +  # Cap at 100%
                    0.3 * min(1.0, max(0.0, metrics.get('sharpe_ratio', 0) / 3)) +  # Normalize Sharpe
                    0.2 * metrics.get('win_rate', 0) + 
                    0.2 * (1 - metrics.get('max_drawdown', 0))  # Lower drawdown is better
                )
                trading_scores[model] = score
        
        # Determine best models
        if classification_scores:
            summary['best_classification_model'] = max(classification_scores, key=classification_scores.get)
        
        if trading_scores:
            summary['best_trading_model'] = max(trading_scores, key=trading_scores.get)
        
        # Calculate overall score (average of classification and trading)
        overall_scores = {}
        for model in model_names:
            if model in classification_scores and model in trading_scores:
                overall_scores[model] = 0.5 * classification_scores.get(model, 0) + 0.5 * trading_scores.get(model, 0)
        
        if overall_scores:
            summary['best_model_overall'] = max(overall_scores, key=overall_scores.get)
        
        # Create model rankings
        if classification_scores:
            summary['model_rankings']['classification'] = sorted(
                classification_scores.keys(), 
                key=lambda x: classification_scores[x], 
                reverse=True
            )
        
        if trading_scores:
            summary['model_rankings']['trading'] = sorted(
                trading_scores.keys(), 
                key=lambda x: trading_scores[x], 
                reverse=True
            )
        
        if overall_scores:
            summary['model_rankings']['overall'] = sorted(
                overall_scores.keys(), 
                key=lambda x: overall_scores[x], 
                reverse=True
            )
        
        # Create performance summary for each model
        for model in model_names:
            summary['performance_summary'][model] = {
                'classification_score': classification_scores.get(model, 0),
                'trading_score': trading_scores.get(model, 0),
                'overall_score': overall_scores.get(model, 0),
                'key_strengths': [],
                'key_weaknesses': []
            }
            
            # Identify key strengths and weaknesses
            if model in evaluation_results:
                metrics = evaluation_results[model]
                
                # Check for high accuracy
                if metrics.get('accuracy', 0) > 0.6:
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"High prediction accuracy ({metrics.get('accuracy', 0):.1%})"
                    )
                elif metrics.get('accuracy', 0) < 0.5:
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"Low prediction accuracy ({metrics.get('accuracy', 0):.1%})"
                    )
                
                # Check for imbalanced precision/recall
                if metrics.get('precision', 0) > metrics.get('recall', 0) + 0.2:
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"High precision ({metrics.get('precision', 0):.1%})"
                    )
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"Lower recall ({metrics.get('recall', 0):.1%})"
                    )
                elif metrics.get('recall', 0) > metrics.get('precision', 0) + 0.2:
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"High recall ({metrics.get('recall', 0):.1%})"
                    )
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"Lower precision ({metrics.get('precision', 0):.1%})"
                    )
            
            if model in trading_results:
                metrics = trading_results[model]
                
                # Check for good returns
                if metrics.get('total_return', 0) > 0.2:  # 20% return
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"Strong total return ({metrics.get('total_return', 0):.1%})"
                    )
                elif metrics.get('total_return', 0) < 0:
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"Negative total return ({metrics.get('total_return', 0):.1%})"
                    )
                
                # Check for good Sharpe ratio
                if metrics.get('sharpe_ratio', 0) > 1.0:
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"Good risk-adjusted return (Sharpe: {metrics.get('sharpe_ratio', 0):.2f})"
                    )
                elif metrics.get('sharpe_ratio', 0) < 0.5:
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"Poor risk-adjusted return (Sharpe: {metrics.get('sharpe_ratio', 0):.2f})"
                    )
                
                # Check for high drawdown
                if metrics.get('max_drawdown', 0) > 0.3:  # 30% drawdown
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"High maximum drawdown ({metrics.get('max_drawdown', 0):.1%})"
                    )
                elif metrics.get('max_drawdown', 0) < 0.15:  # 15% drawdown
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"Controlled maximum drawdown ({metrics.get('max_drawdown', 0):.1%})"
                    )
                
                # Check win rate
                if metrics.get('win_rate', 0) > 0.6:
                    summary['performance_summary'][model]['key_strengths'].append(
                        f"High win rate ({metrics.get('win_rate', 0):.1%})"
                    )
                elif metrics.get('win_rate', 0) < 0.4:
                    summary['performance_summary'][model]['key_weaknesses'].append(
                        f"Low win rate ({metrics.get('win_rate', 0):.1%})"
                    )
        
        return summary
    
    def _extract_key_insights(
        self,
        model_names: List[str],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Extract key insights from analysis results
        
        Args:
            model_names: List of model names
            analysis: Dictionary with analysis results
            
        Returns:
            List of key insights as strings
        """
        self.logger.info("Extracting key insights")
        
        insights = []
        
        # Get summary data
        summary = analysis.get('summary', {})
        best_model_overall = summary.get('best_model_overall')
        best_classification_model = summary.get('best_classification_model')
        best_trading_model = summary.get('best_trading_model')
        
        # Get statistical tests
        stat_tests = analysis.get('statistical_tests', {})
        
        # 1. Overall best model
        if best_model_overall:
            insights.append(f"The {best_model_overall} model performs best overall, balancing prediction accuracy and trading performance.")
        
        # 2. Best models by category
        if best_classification_model and best_trading_model:
            if best_classification_model == best_trading_model:
                insights.append(f"The {best_classification_model} model excels in both prediction accuracy and trading performance.")
            else:
                insights.append(f"The {best_classification_model} model has the best prediction accuracy, while the {best_trading_model} model achieves the best trading performance.")
        
        # 3. Statistical significance
        significant_tests = []
        for test_key, test_results in stat_tests.items():
            for test_name, test_data in test_results.items():
                if test_data.get('significant', False):
                    model_comparison = test_key.replace('_vs_', ' vs ')
                    better_model = test_data.get('better_model', '')
                    if test_name == 'mcnemar_test':
                        significant_tests.append(f"The {better_model} model's predictions are significantly different from the other model ({model_comparison}).")
                    elif test_name == 'returns_ttest' or test_name == 'wilcoxon_test':
                        significant_tests.append(f"The {better_model} model's trading returns are significantly better than the other model ({model_comparison}).")
        
        if significant_tests:
            insights.extend(significant_tests[:2])  # Limit to top 2 significant findings
        
        # 4. Model-specific insights
        for model in model_names:
            if model in summary.get('performance_summary', {}):
                model_summary = summary['performance_summary'][model]
                
                # Add strengths
                strengths = model_summary.get('key_strengths', [])
                if strengths:
                    insights.append(f"{model} model strengths: {', '.join(strengths[:2])}.")
                
                # Add weaknesses (limit to one weakness per model)
                weaknesses = model_summary.get('key_weaknesses', [])
                if weaknesses:
                    insights.append(f"{model} model challenges: {weaknesses[0]}.")
        
        # 5. General trading insights
        trading_metrics = analysis.get('trading_metrics', {})
        if trading_metrics:
            # Check if any model is profitable
            profitable_models = [model for model, metrics in trading_metrics.items() 
                               if metrics.get('total_return', 0) > 0]
            
            if profitable_models:
                best_return_model = max(trading_metrics.keys(), 
                                      key=lambda m: trading_metrics[m].get('total_return', 0))
                best_return = trading_metrics[best_return_model].get('total_return', 0)
                
                insights.append(f"The {best_return_model} model achieved the highest return of {best_return:.1%}.")
            else:
                insights.append("None of the models achieved positive returns during the test period.")
        
        # Limit to top 10 insights
        return insights[:10]
    
    def generate_performance_report(
        self,
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        test_prices: pd.Series,
        test_volume: Optional[pd.Series] = None,
        report_format: str = 'html'
    ) -> str:
        """
        Generate detailed performance report with statistical significance testing
        
        Args:
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            test_prices: Test set price data
            test_volume: Optional test set volume data
            report_format: Format of the report ('html', 'json', 'md')
            
        Returns:
            Path to the generated report file
            
        Raises:
            ValueError: If input dictionaries are empty or missing required keys
        """
        self.logger.info(f"Generating performance report in {report_format} format")
        
        # Perform comprehensive analysis
        analysis = self.analyze_model_performance(model_results, evaluation_results, trading_results)
        
        # Generate timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report based on format
        if report_format.lower() == 'html':
            report_path = self._generate_html_report(analysis, model_results, evaluation_results, 
                                                   trading_results, test_prices, test_volume, timestamp)
        elif report_format.lower() == 'json':
            report_path = self._generate_json_report(analysis, model_results, evaluation_results, 
                                                   trading_results, timestamp)
        elif report_format.lower() == 'md':
            report_path = self._generate_markdown_report(analysis, model_results, evaluation_results, 
                                                      trading_results, timestamp)
        else:
            self.logger.error(f"Unsupported report format: {report_format}")
            raise ValueError(f"Unsupported report format: {report_format}")
        
        self.logger.info(f"Performance report generated: {report_path}")
        return report_path
    
    def _generate_html_report(
        self,
        analysis: Dict[str, Any],
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        test_prices: pd.Series,
        test_volume: Optional[pd.Series],
        timestamp: str
    ) -> str:
        """
        Generate HTML performance report
        
        Args:
            analysis: Dictionary with analysis results
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            test_prices: Test set price data
            test_volume: Optional test set volume data
            timestamp: Timestamp string for the report
            
        Returns:
            Path to the generated HTML report
        """
        # Extract model names
        model_names = list(set(model_results.keys()) & set(evaluation_results.keys()) & set(trading_results.keys()))
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NVDL Stock Predictor Performance Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric-good {{ color: #27ae60; }}
                .metric-bad {{ color: #e74c3c; }}
                .insight {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; }}
                .chart-container {{ margin: 30px 0; }}
                .footer {{ margin-top: 50px; text-align: center; font-size: 0.8em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NVDL Stock Predictor Performance Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Executive Summary</h2>
        """
        
        # Add key insights
        html_content += "<div class='insight'><h3>Key Insights</h3><ul>"
        for insight in analysis.get('key_insights', []):
            html_content += f"<li>{insight}</li>"
        html_content += "</ul></div>"
        
        # Add best model summary
        summary = analysis.get('summary', {})
        best_model_overall = summary.get('best_model_overall')
        best_classification_model = summary.get('best_classification_model')
        best_trading_model = summary.get('best_trading_model')
        
        html_content += "<h3>Model Rankings</h3>"
        html_content += "<table><tr><th>Category</th><th>Best Model</th><th>Score</th></tr>"
        
        if best_model_overall:
            overall_score = summary.get('performance_summary', {}).get(best_model_overall, {}).get('overall_score', 0)
            html_content += f"<tr><td>Overall Performance</td><td>{best_model_overall}</td><td>{overall_score:.4f}</td></tr>"
        
        if best_classification_model:
            class_score = summary.get('performance_summary', {}).get(best_classification_model, {}).get('classification_score', 0)
            html_content += f"<tr><td>Classification Accuracy</td><td>{best_classification_model}</td><td>{class_score:.4f}</td></tr>"
        
        if best_trading_model:
            trade_score = summary.get('performance_summary', {}).get(best_trading_model, {}).get('trading_score', 0)
            html_content += f"<tr><td>Trading Performance</td><td>{best_trading_model}</td><td>{trade_score:.4f}</td></tr>"
        
        html_content += "</table>"
        
        # Add classification metrics comparison
        html_content += "<h2>Classification Performance</h2>"
        html_content += "<table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Specificity</th></tr>"
        
        for model in model_names:
            if model in analysis.get('classification_metrics', {}):
                metrics = analysis['classification_metrics'][model]
                html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{metrics.get('accuracy', 0):.4f}</td>
                    <td>{metrics.get('precision', 0):.4f}</td>
                    <td>{metrics.get('recall', 0):.4f}</td>
                    <td>{metrics.get('f1_score', 0):.4f}</td>
                    <td>{metrics.get('specificity', 0):.4f}</td>
                </tr>
                """
        
        html_content += "</table>"
        
        # Add trading performance comparison
        html_content += "<h2>Trading Performance</h2>"
        html_content += "<table><tr><th>Model</th><th>Total Return</th><th>Annualized Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Win Rate</th><th>Profit Factor</th><th>Trades</th></tr>"
        
        for model in model_names:
            if model in analysis.get('trading_metrics', {}):
                metrics = analysis['trading_metrics'][model]
                html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{metrics.get('total_return', 0):.2%}</td>
                    <td>{metrics.get('annualized_return', 0):.2%}</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    <td>{metrics.get('max_drawdown', 0):.2%}</td>
                    <td>{metrics.get('win_rate', 0):.2%}</td>
                    <td>{metrics.get('profit_factor', 0):.2f}</td>
                    <td>{metrics.get('num_trades', 0)}</td>
                </tr>
                """
        
        html_content += "</table>"
        
        # Add statistical tests
        if analysis.get('statistical_tests'):
            html_content += "<h2>Statistical Significance Tests</h2>"
            html_content += "<table><tr><th>Comparison</th><th>Test</th><th>Statistic</th><th>P-Value</th><th>Significant</th><th>Better Model</th></tr>"
            
            for comparison, tests in analysis['statistical_tests'].items():
                for test_name, test_results in tests.items():
                    html_content += f"""
                    <tr>
                        <td>{comparison.replace('_vs_', ' vs ')}</td>
                        <td>{test_name.replace('_', ' ').title()}</td>
                        <td>{test_results.get('statistic', 0):.4f}</td>
                        <td>{test_results.get('p_value', 1):.4f}</td>
                        <td>{'Yes' if test_results.get('significant', False) else 'No'}</td>
                        <td>{test_results.get('better_model', 'N/A')}</td>
                    </tr>
                    """
            
            html_content += "</table>"
        
        # Add model strengths and weaknesses
        html_content += "<h2>Model Strengths and Weaknesses</h2>"
        
        for model in model_names:
            if model in summary.get('performance_summary', {}):
                model_summary = summary['performance_summary'][model]
                
                html_content += f"<h3>{model} Model</h3>"
                html_content += "<h4>Strengths</h4><ul>"
                
                for strength in model_summary.get('key_strengths', []):
                    html_content += f"<li>{strength}</li>"
                
                if not model_summary.get('key_strengths'):
                    html_content += "<li>No significant strengths identified</li>"
                
                html_content += "</ul><h4>Weaknesses</h4><ul>"
                
                for weakness in model_summary.get('key_weaknesses', []):
                    html_content += f"<li>{weakness}</li>"
                
                if not model_summary.get('key_weaknesses'):
                    html_content += "<li>No significant weaknesses identified</li>"
                
                html_content += "</ul>"
        
        # Add footer
        html_content += """
                <div class="footer">
                    <p>NVDL Stock Predictor &copy; 2025</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = f"results/reports/performance_report_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_json_report(
        self,
        analysis: Dict[str, Any],
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        timestamp: str
    ) -> str:
        """
        Generate JSON performance report
        
        Args:
            analysis: Dictionary with analysis results
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            timestamp: Timestamp string for the report
            
        Returns:
            Path to the generated JSON report
        """
        # Helper function to make values JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        # Create report data structure
        report_data = {
            'report_type': 'NVDL Stock Predictor Performance Report',
            'generated_at': datetime.now().isoformat(),
            'analysis': make_serializable(analysis),
            'evaluation_results': {
                model: make_serializable(metrics)
                for model, metrics in evaluation_results.items()
            },
            'trading_results': {
                model: {k: make_serializable(v) 
                       for k, v in metrics.items() if k != 'equity_curve' and k != 'transactions'}
                for model, metrics in trading_results.items()
            }
        }
        
        # Save JSON report
        report_path = f"results/reports/performance_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path
    
    def _generate_markdown_report(
        self,
        analysis: Dict[str, Any],
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        timestamp: str
    ) -> str:
        """
        Generate Markdown performance report
        
        Args:
            analysis: Dictionary with analysis results
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            timestamp: Timestamp string for the report
            
        Returns:
            Path to the generated Markdown report
        """
        # Extract model names
        model_names = list(set(model_results.keys()) & set(evaluation_results.keys()) & set(trading_results.keys()))
        
        # Create Markdown content
        md_content = f"""# NVDL Stock Predictor Performance Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

"""
        
        # Add key insights
        md_content += "### Key Insights\n\n"
        for insight in analysis.get('key_insights', []):
            md_content += f"- {insight}\n"
        
        # Add best model summary
        summary = analysis.get('summary', {})
        best_model_overall = summary.get('best_model_overall')
        best_classification_model = summary.get('best_classification_model')
        best_trading_model = summary.get('best_trading_model')
        
        md_content += "\n### Model Rankings\n\n"
        md_content += "| Category | Best Model | Score |\n"
        md_content += "| --- | --- | --- |\n"
        
        if best_model_overall:
            overall_score = summary.get('performance_summary', {}).get(best_model_overall, {}).get('overall_score', 0)
            md_content += f"| Overall Performance | {best_model_overall} | {overall_score:.4f} |\n"
        
        if best_classification_model:
            class_score = summary.get('performance_summary', {}).get(best_classification_model, {}).get('classification_score', 0)
            md_content += f"| Classification Accuracy | {best_classification_model} | {class_score:.4f} |\n"
        
        if best_trading_model:
            trade_score = summary.get('performance_summary', {}).get(best_trading_model, {}).get('trading_score', 0)
            md_content += f"| Trading Performance | {best_trading_model} | {trade_score:.4f} |\n"
        
        # Add classification metrics comparison
        md_content += "\n## Classification Performance\n\n"
        md_content += "| Model | Accuracy | Precision | Recall | F1 Score | Specificity |\n"
        md_content += "| --- | --- | --- | --- | --- | --- |\n"
        
        for model in model_names:
            if model in analysis.get('classification_metrics', {}):
                metrics = analysis['classification_metrics'][model]
                md_content += f"| {model} | {metrics.get('accuracy', 0):.4f} | {metrics.get('precision', 0):.4f} | "
                md_content += f"{metrics.get('recall', 0):.4f} | {metrics.get('f1_score', 0):.4f} | {metrics.get('specificity', 0):.4f} |\n"
        
        # Add trading performance comparison
        md_content += "\n## Trading Performance\n\n"
        md_content += "| Model | Total Return | Annualized Return | Sharpe Ratio | Max Drawdown | Win Rate | Profit Factor | Trades |\n"
        md_content += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        
        for model in model_names:
            if model in analysis.get('trading_metrics', {}):
                metrics = analysis['trading_metrics'][model]
                md_content += f"| {model} | {metrics.get('total_return', 0):.2%} | {metrics.get('annualized_return', 0):.2%} | "
                md_content += f"{metrics.get('sharpe_ratio', 0):.2f} | {metrics.get('max_drawdown', 0):.2%} | "
                md_content += f"{metrics.get('win_rate', 0):.2%} | {metrics.get('profit_factor', 0):.2f} | {metrics.get('num_trades', 0)} |\n"
        
        # Add statistical tests
        if analysis.get('statistical_tests'):
            md_content += "\n## Statistical Significance Tests\n\n"
            md_content += "| Comparison | Test | Statistic | P-Value | Significant | Better Model |\n"
            md_content += "| --- | --- | --- | --- | --- | --- |\n"
            
            for comparison, tests in analysis['statistical_tests'].items():
                for test_name, test_results in tests.items():
                    md_content += f"| {comparison.replace('_vs_', ' vs ')} | {test_name.replace('_', ' ').title()} | "
                    md_content += f"{test_results.get('statistic', 0):.4f} | {test_results.get('p_value', 1):.4f} | "
                    md_content += f"{'Yes' if test_results.get('significant', False) else 'No'} | {test_results.get('better_model', 'N/A')} |\n"
        
        # Add model strengths and weaknesses
        md_content += "\n## Model Strengths and Weaknesses\n\n"
        
        for model in model_names:
            if model in summary.get('performance_summary', {}):
                model_summary = summary['performance_summary'][model]
                
                md_content += f"### {model} Model\n\n"
                md_content += "#### Strengths\n\n"
                
                for strength in model_summary.get('key_strengths', []):
                    md_content += f"- {strength}\n"
                
                if not model_summary.get('key_strengths'):
                    md_content += "- No significant strengths identified\n"
                
                md_content += "\n#### Weaknesses\n\n"
                
                for weakness in model_summary.get('key_weaknesses', []):
                    md_content += f"- {weakness}\n"
                
                if not model_summary.get('key_weaknesses'):
                    md_content += "- No significant weaknesses identified\n"
                
                md_content += "\n"
        
        # Save Markdown report
        report_path = f"results/reports/performance_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(md_content)
        
        return report_path
    
    def create_model_comparison_dashboard(
        self,
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        test_prices: pd.Series,
        test_volume: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive model comparison dashboard with filtering
        
        Args:
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            test_prices: Test set price data
            test_volume: Optional test set volume data
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If input dictionaries are empty or missing required keys
        """
        self.logger.info("Creating model comparison dashboard")
        
        # Extract model names
        model_names = list(set(model_results.keys()) & set(evaluation_results.keys()) & set(trading_results.keys()))
        
        if not model_names:
            self.logger.error("No common models found in input dictionaries")
            raise ValueError("No common models found in input dictionaries")
        
        # Create dashboard with subplots
        fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=[
                "Model Performance Metrics",
                "Trading Returns Comparison",
                "Prediction Accuracy Over Time",
                "Drawdown Analysis",
                "Win Rate by Model",
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
        
        # 1. Model Performance Metrics (Top row)
        metrics_df = pd.DataFrame({
            model: {
                'Accuracy': evaluation_results[model].get('accuracy', 0),
                'Precision': evaluation_results[model].get('precision', 0),
                'Recall': evaluation_results[model].get('recall', 0),
                'F1 Score': evaluation_results[model].get('f1_score', 0),
                'Sharpe Ratio': trading_results[model].get('sharpe_ratio', 0),
                'Win Rate': trading_results[model].get('win_rate', 0)
            }
            for model in model_names
        }).T
        
        # Create a grouped bar chart for metrics
        for i, metric in enumerate(metrics_df.columns):
            fig.add_trace(
                go.Bar(
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    name=metric,
                    text=[f"{v:.2f}" for v in metrics_df[metric]],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # 2. Trading Returns Comparison (Middle left)
        for model in model_names:
            if 'equity_curve' in trading_results[model]:
                equity = trading_results[model]['equity_curve']
                # Normalize to percentage return
                equity_pct = (equity / equity.iloc[0] - 1) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=equity.index,
                        y=equity_pct,
                        mode='lines',
                        name=f"{model} Return",
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        # 3. Prediction Accuracy Over Time (Middle right)
        for model in model_names:
            if 'predictions' in model_results[model] and 'y_true' in model_results[model] and 'test_dates' in model_results[model]:
                predictions = model_results[model]['predictions']
                y_true = model_results[model]['y_true']
                dates = model_results[model]['test_dates']
                
                # Calculate rolling accuracy (20-day window)
                correct = (predictions == y_true).astype(int)
                rolling_accuracy = pd.Series(correct).rolling(window=20).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=dates[:len(rolling_accuracy)],
                        y=rolling_accuracy,
                        mode='lines',
                        name=f"{model} Accuracy",
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
        
        # 4. Drawdown Analysis (Bottom left)
        for model in model_names:
            if 'equity_curve' in trading_results[model]:
                equity = trading_results[model]['equity_curve']
                
                # Calculate drawdown
                peak = equity.cummax()
                drawdown = (equity - peak) / peak * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        mode='lines',
                        name=f"{model} Drawdown",
                        line=dict(width=2),
                        fill='tozeroy'
                    ),
                    row=3, col=1
                )
        
        # 5. Win Rate by Model (Bottom right)
        win_rates = [trading_results[model].get('win_rate', 0) for model in model_names]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=win_rates,
                name="Win Rate",
                text=[f"{wr:.1%}" for wr in win_rates],
                textposition='auto'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="NVDL Stock Predictor Model Comparison Dashboard",
            height=900,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes titles
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy (20-day MA)", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Model", row=3, col=2)
        fig.update_yaxes(title_text="Win Rate", tickformat='.0%', row=3, col=2)
        
        # Save if path provided
        if save_path:
            self.visualization_engine.save_figure(fig, save_path)
            self.logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def export_results(
        self,
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        export_format: str = 'csv',
        export_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export results and visualizations to various formats
        
        Args:
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            export_format: Format to export ('csv', 'excel', 'json')
            export_path: Optional path to export directory
            
        Returns:
            Dictionary mapping export type to file path
            
        Raises:
            ValueError: If export_format is not supported
        """
        self.logger.info(f"Exporting results in {export_format} format")
        
        # Create export directory if not provided
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"results/exports_{timestamp}"
        
        os.makedirs(export_path, exist_ok=True)
        
        export_files = {}
        
        # 1. Export classification metrics
        classification_data = {}
        for model, metrics in evaluation_results.items():
            classification_data[model] = {k: v for k, v in metrics.items() 
                                        if k in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']}
        
        classification_df = pd.DataFrame(classification_data).T
        
        # 2. Export trading metrics
        trading_data = {}
        for model, metrics in trading_results.items():
            trading_data[model] = {k: v for k, v in metrics.items() 
                                 if k in ['total_return', 'annualized_return', 'sharpe_ratio', 
                                         'max_drawdown', 'win_rate', 'profit_factor', 'num_trades']}
        
        trading_df = pd.DataFrame(trading_data).T
        
        # 3. Export predictions
        predictions_data = {}
        for model, results in model_results.items():
            if 'predictions' in results and 'test_dates' in results:
                predictions = results['predictions']
                dates = results['test_dates']
                
                # Create Series with dates as index
                if len(predictions) <= len(dates):
                    predictions_data[model] = pd.Series(predictions, index=dates[:len(predictions)])
        
        predictions_df = pd.DataFrame(predictions_data)
        
        # Export based on format
        if export_format.lower() == 'csv':
            # Export to CSV
            classification_path = f"{export_path}/classification_metrics.csv"
            trading_path = f"{export_path}/trading_metrics.csv"
            predictions_path = f"{export_path}/model_predictions.csv"
            
            classification_df.to_csv(classification_path)
            trading_df.to_csv(trading_path)
            predictions_df.to_csv(predictions_path)
            
            export_files = {
                'classification_metrics': classification_path,
                'trading_metrics': trading_path,
                'predictions': predictions_path
            }
            
        elif export_format.lower() == 'excel':
            # Export to Excel
            excel_path = f"{export_path}/nvdl_predictor_results.xlsx"
            
            try:
                with pd.ExcelWriter(excel_path) as writer:
                    classification_df.to_excel(writer, sheet_name='Classification Metrics')
                    trading_df.to_excel(writer, sheet_name='Trading Metrics')
                    predictions_df.to_excel(writer, sheet_name='Model Predictions')
                
                export_files = {'excel': excel_path}
            except ImportError:
                self.logger.warning("openpyxl module not available, falling back to CSV export")
                # Fall back to CSV export
                classification_path = f"{export_path}/classification_metrics.csv"
                trading_path = f"{export_path}/trading_metrics.csv"
                predictions_path = f"{export_path}/model_predictions.csv"
                
                classification_df.to_csv(classification_path)
                trading_df.to_csv(trading_path)
                predictions_df.to_csv(predictions_path)
                
                export_files = {
                    'classification_metrics': classification_path,
                    'trading_metrics': trading_path,
                    'predictions': predictions_path
                }
            
        elif export_format.lower() == 'json':
            # Export to JSON
            classification_path = f"{export_path}/classification_metrics.json"
            trading_path = f"{export_path}/trading_metrics.json"
            predictions_path = f"{export_path}/model_predictions.json"
            
            classification_df.to_json(classification_path, orient='index', indent=2)
            trading_df.to_json(trading_path, orient='index', indent=2)
            predictions_df.to_json(predictions_path, orient='columns', indent=2)
            
            export_files = {
                'classification_metrics': classification_path,
                'trading_metrics': trading_path,
                'predictions': predictions_path
            }
            
        else:
            self.logger.error(f"Unsupported export format: {export_format}")
            raise ValueError(f"Unsupported export format: {export_format}")
        
        self.logger.info(f"Results exported to {export_path}")
        return export_files