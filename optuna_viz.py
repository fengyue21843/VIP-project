"""
Optuna Visualization Utility Module

This module provides functions to create visualizations from Optuna study objects.
It generates various plots to analyze hyperparameter optimization results.

Visualizations include:
- Optimization history: Shows how the objective value changes over trials
- Parameter importance: Ranks parameters by their impact on the objective
- Parameter relationships: Shows how pairs of parameters affect the objective
- Parallel coordinate plot: Visualizes the relationship between all parameters and the objective
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
import optuna


def plot_optimization_history(study: optuna.Study, title: Optional[str] = None) -> go.Figure:
    """
    Plot the optimization history showing objective value vs trial number.

    Args:
        study: Optuna study object
        title: Optional custom title for the plot

    Returns:
        Plotly figure object
    """
    df = study.trials_dataframe()

    if title is None:
        title = "Optimization History"

    fig = go.Figure()

    # Plot all trial values
    fig.add_trace(go.Scatter(
        x=df['number'],
        y=df['value'],
        mode='markers',
        name='Trial Value',
        marker=dict(size=8, color='lightblue', opacity=0.6),
        hovertemplate='Trial: %{x}<br>Value: %{y:.6f}<extra></extra>'
    ))

    # Plot best value line
    best_values = df['value'].cummax() if study.direction == optuna.study.StudyDirection.MAXIMIZE else df['value'].cummin()
    fig.add_trace(go.Scatter(
        x=df['number'],
        y=best_values,
        mode='lines',
        name='Best Value',
        line=dict(color='red', width=2),
        hovertemplate='Trial: %{x}<br>Best Value: %{y:.6f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Trial Number",
        yaxis_title="Objective Value",
        hovermode='closest',
        template='plotly_white',
        height=400
    )

    return fig


def plot_param_importances(study: optuna.Study, title: Optional[str] = None) -> go.Figure:
    """
    Plot parameter importances showing which parameters have the most impact.

    Args:
        study: Optuna study object
        title: Optional custom title for the plot

    Returns:
        Plotly figure object
    """
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        # If importance calculation fails, return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not calculate parameter importances: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    if title is None:
        title = "Parameter Importances"

    params = list(importances.keys())
    values = list(importances.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=params,
        orientation='h',
        marker=dict(color='steelblue'),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Parameter",
        template='plotly_white',
        height=max(300, len(params) * 40)
    )

    return fig


def plot_param_slice(study: optuna.Study, param_name: str, title: Optional[str] = None) -> go.Figure:
    """
    Plot slice plot for a single parameter showing its effect on the objective.

    Args:
        study: Optuna study object
        param_name: Name of the parameter to visualize
        title: Optional custom title for the plot

    Returns:
        Plotly figure object
    """
    df = study.trials_dataframe()

    if title is None:
        title = f"Parameter Slice: {param_name}"

    # Get parameter values
    param_col = f"params_{param_name}"
    if param_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Parameter '{param_name}' not found in study",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    x_values = df[param_col]
    y_values = df['value']

    # Check if parameter is categorical or numerical
    if df[param_col].dtype == 'object' or df[param_col].dtype.name == 'category':
        # Categorical parameter - use box plot
        fig = go.Figure()
        for category in x_values.unique():
            mask = x_values == category
            fig.add_trace(go.Box(
                y=y_values[mask],
                name=str(category),
                hovertemplate=f'{param_name}={category}<br>Value: %{{y:.6f}}<extra></extra>'
            ))
        fig.update_layout(
            xaxis_title=param_name,
            yaxis_title="Objective Value"
        )
    else:
        # Numerical parameter - use scatter plot
        fig = go.Figure(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(size=8, color='steelblue', opacity=0.6),
            hovertemplate=f'{param_name}: %{{x}}<br>Value: %{{y:.6f}}<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title=param_name,
            yaxis_title="Objective Value"
        )

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )

    return fig


def plot_param_contour(study: optuna.Study, param1: str, param2: str, title: Optional[str] = None) -> go.Figure:
    """
    Plot contour plot showing the relationship between two parameters.

    Args:
        study: Optuna study object
        param1: Name of the first parameter
        param2: Name of the second parameter
        title: Optional custom title for the plot

    Returns:
        Plotly figure object
    """
    df = study.trials_dataframe()

    if title is None:
        title = f"Parameter Contour: {param1} vs {param2}"

    param1_col = f"params_{param1}"
    param2_col = f"params_{param2}"

    if param1_col not in df.columns or param2_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Parameters '{param1}' or '{param2}' not found in study",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Create scatter plot with color representing objective value
    fig = go.Figure(go.Scatter(
        x=df[param1_col],
        y=df[param2_col],
        mode='markers',
        marker=dict(
            size=10,
            color=df['value'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Objective<br>Value")
        ),
        hovertemplate=f'{param1}: %{{x}}<br>{param2}: %{{y}}<br>Value: %{{marker.color:.6f}}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=param1,
        yaxis_title=param2,
        template='plotly_white',
        height=400
    )

    return fig


def plot_parallel_coordinate(study: optuna.Study, title: Optional[str] = None) -> go.Figure:
    """
    Plot parallel coordinate plot showing all parameters and objective value.

    Args:
        study: Optuna study object
        title: Optional custom title for the plot

    Returns:
        Plotly figure object
    """
    df = study.trials_dataframe()

    if title is None:
        title = "Parallel Coordinate Plot"

    # Get parameter columns
    param_cols = [col for col in df.columns if col.startswith('params_')]

    if not param_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No parameters found in study",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Create dimensions for parallel coordinates
    dimensions = []

    # Add parameter dimensions
    for col in param_cols:
        param_name = col.replace('params_', '')

        # Check if categorical
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Convert categorical to numerical
            categories = df[col].unique()
            cat_to_num = {cat: i for i, cat in enumerate(categories)}
            values = df[col].map(cat_to_num)

            dimensions.append(dict(
                label=param_name,
                values=values,
                tickvals=list(range(len(categories))),
                ticktext=[str(cat) for cat in categories]
            ))
        else:
            dimensions.append(dict(
                label=param_name,
                values=df[col]
            ))

    # Add objective value dimension
    dimensions.append(dict(
        label='Objective Value',
        values=df['value']
    ))

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df['value'],
            colorscale='Viridis',
            showscale=True,
            cmin=df['value'].min(),
            cmax=df['value'].max()
        ),
        dimensions=dimensions
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500
    )

    return fig


def create_optimization_dashboard(study: optuna.Study, model_name: str = "Model") -> dict:
    """
    Create a comprehensive dashboard with all optimization visualizations.

    Args:
        study: Optuna study object
        model_name: Name of the model for titles

    Returns:
        Dictionary of plotly figures with keys:
        - 'history': Optimization history plot
        - 'importance': Parameter importance plot
        - 'parallel': Parallel coordinate plot
        - 'param_slices': Dictionary of parameter slice plots
        - 'best_params': Best parameters as a formatted string
    """
    df = study.trials_dataframe()
    param_cols = [col.replace('params_', '') for col in df.columns if col.startswith('params_')]

    dashboard = {
        'history': plot_optimization_history(study, f"{model_name} - Optimization History"),
        'importance': plot_param_importances(study, f"{model_name} - Parameter Importances"),
        'parallel': plot_parallel_coordinate(study, f"{model_name} - Parallel Coordinate Plot"),
        'param_slices': {},
        'best_params': study.best_params
    }

    # Create slice plots for each parameter
    for param in param_cols:
        dashboard['param_slices'][param] = plot_param_slice(study, param, f"{model_name} - {param}")

    return dashboard


if __name__ == "__main__":
    """
    Test visualization functions with a dummy Optuna study
    """
    print("Testing Optuna Visualization Module")
    print("=" * 80)

    # Create a dummy study for testing
    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_int('y', 0, 10)
        z = trial.suggest_categorical('z', ['a', 'b', 'c'])

        score = x**2 + y**2 + (0 if z == 'a' else 5)
        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    print(f"\nCreated dummy study with {len(study.trials)} trials")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # Test each visualization function
    print("\nTesting visualization functions:")

    print("  - Optimization history... ", end="")
    fig = plot_optimization_history(study)
    print("✓")

    print("  - Parameter importances... ", end="")
    fig = plot_param_importances(study)
    print("✓")

    print("  - Parallel coordinate plot... ", end="")
    fig = plot_parallel_coordinate(study)
    print("✓")

    print("  - Parameter slice plots... ", end="")
    for param in ['x', 'y', 'z']:
        fig = plot_param_slice(study, param)
    print("✓")

    print("  - Parameter contour plot... ", end="")
    fig = plot_param_contour(study, 'x', 'y')
    print("✓")

    print("  - Complete dashboard... ", end="")
    dashboard = create_optimization_dashboard(study, "Test Model")
    print("✓")

    print("\n" + "=" * 80)
    print("✓ All visualization functions working correctly!")
    print("=" * 80)