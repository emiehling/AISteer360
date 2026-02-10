"""Visualization utilities for benchmark profiles."""

from typing import Any, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_plot_style() -> None:
    """Apply specific matplotlib rcParams for scientific style."""
    plt.rcParams.update({
        # fonts: prefer clean sans-serif
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": "medium",
        "axes.labelsize": "medium",

        # spines & ticks: show all four sides
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # grid: subtle and in the background
        "axes.grid": False,
        "grid.color": "#cbcbcb",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,

        # colors
        "axes.prop_cycle": plt.cycler(color=[
            "#348ABD", 
            "#E24A33", 
            "#988ED5", 
            "#777777",
            "#FBC15E", 
            "#8EBA42", 
            "#FFB5B8"
        ]),
    })


def _clean_axes(ax: plt.Axes) -> None:
    """Helper to ensure all four axis spines are visible."""
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def plot_metric_by_config(
    summary: pd.DataFrame,
    metric: str,
    x_col: str = "config_id",
    baseline_value: float | None = None,
    baseline_std: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **errorbar_kwargs: Any,
) -> plt.Axes:
    """Plot a metric with error bars across configurations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    _clean_axes(ax)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # lighter error bars to emphasize the mean point
    defaults = {
        "fmt": "o-",
        "capsize": 0,    # remove caps for cleaner look
        "elinewidth": 1,
        "markersize": 6,
        "markeredgewidth": 1,
        "markeredgecolor": "white", # separates marker from line
        "zorder": 3
    }
    defaults.update(errorbar_kwargs)

    ax.errorbar(
        summary[x_col],
        summary[mean_col],
        yerr=summary[std_col],
        **defaults,
    )

    if baseline_value is not None:
        # use a muted color for baseline reference
        ax.axhline(baseline_value, color="#444444", linestyle="--", linewidth=1, label="Baseline", zorder=1)
        if baseline_std is not None:
            ax.axhspan(
                baseline_value - baseline_std,
                baseline_value + baseline_std,
                color="#999999",
                alpha=0.1,
                edgecolor="none",
                zorder=0
            )

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric)

    # left-aligned title is easier to read
    if title:
        ax.set_title(title, loc="left", fontweight="bold")

    # grid behind data
    ax.grid(True, axis="y", zorder=-1)

    # minimal legend
    ax.legend(frameon=False, loc="best")

    return ax


def plot_tradeoff_scatter(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_col: str | None = None,
    label_col: str | None = None,
    baseline_row: pd.Series | None = None,
    ax: plt.Axes | None = None,
    title: str = "metric tradeoff",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    **scatter_kwargs: Any,
) -> plt.Axes:
    """Plot a scatter of two metrics showing their tradeoff."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    _clean_axes(ax)

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"
    x_std = f"{x_metric}_std"
    y_std = f"{y_metric}_std"

    # clear distinction between data and annotations
    defaults = {
        "s": 100,
        "edgecolors": "white",
        "linewidth": 0.5,
        "alpha": 0.9,
        "zorder": 3
    }
    defaults.update(scatter_kwargs)

    # plot error bars first (behind points)
    # using 'zorder' to ensure points sit on top of error bars
    for _, row in summary.iterrows():
        ax.errorbar(
            row[x_mean],
            row[y_mean],
            xerr=row.get(x_std, 0),
            yerr=row.get(y_std, 0),
            fmt="none",
            color="#bbbbbb", # muted grey for errors
            alpha=0.6,
            capsize=0,
            elinewidth=1,
            zorder=2
        )

    if color_col is not None and color_col in summary.columns:
        defaults["c"] = summary[color_col]
        defaults["cmap"] = cmap
        scatter = ax.scatter(summary[x_mean], summary[y_mean], **defaults)
        # cleaner colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=color_col)
        cbar.outline.set_visible(False) # remove box around colorbar
        cbar.ax.tick_params(size=0)     # clean ticks
    else:
        ax.scatter(summary[x_mean], summary[y_mean], **defaults)

    # labels with slightly less visual weight
    if label_col is not None and label_col in summary.columns:
        for _, row in summary.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row[x_mean], row[y_mean]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="#333333",
                alpha=0.8
            )

    # baseline marker
    if baseline_row is not None:
        ax.scatter(
            baseline_row[x_mean],
            baseline_row[y_mean],
            marker="X",
            s=120,
            c="#E24A33", # specific alert color
            edgecolors="white",
            linewidth=1,
            zorder=4,
            label="Baseline",
        )

    ax.set_xlabel(xlabel or x_metric)
    ax.set_ylabel(ylabel or y_metric)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, zorder=-1)

    if baseline_row is not None:
        ax.legend(frameon=False)

    return ax


def plot_metric_heatmap(
    pivot_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    square: bool = False,
    col_label_decimals: int | None = 2,
) -> plt.Axes:
    """Plot a heatmap from a pivoted DataFrame.

    Args:
        pivot_df: Pivoted DataFrame with values to plot.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        title: Title for the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cbar_label: Label for colorbar.
        square: Whether to enforce square cells.
        col_label_decimals: Number of decimal places to round column labels to.
            Set to None to disable rounding.

    Returns:
        The matplotlib axes with the heatmap.
    """
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError("seaborn is required for heatmap plots") from e

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # round column labels if requested
    plot_df = pivot_df.copy()
    if col_label_decimals is not None:
        plot_df.columns = [round(c, col_label_decimals) if isinstance(c, float) else c for c in plot_df.columns]

    cbar_kws = {"label": cbar_label} if cbar_label else {}

    # seaborn integration
    sns.heatmap(
        plot_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws=cbar_kws,
        square=square,
        linewidths=0.5, # grid for heatmap
        linecolor='white'
    )

    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # clean up the colorbar axes if accessible
    if ax.collections:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(size=0)

    return ax


def plot_comparison_bars(
    comparison_df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_col: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str = "Value",
    colors: Sequence[str] | None = None,
    bar_width: float = 0.35,
) -> plt.Axes:
    """Plot grouped bar chart comparing metrics across groups."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    _clean_axes(ax)

    n_groups = len(comparison_df)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)

    if colors is None:
        # use a qualitative cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    total_width = bar_width * n_metrics
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_metrics)

    for i, (col, offset) in enumerate(zip(metric_cols, offsets)):
        ax.bar(
            x + offset,
            comparison_df[col],
            bar_width,
            label=col,
            color=colors[i % len(colors)],
            edgecolor="none", # no borders on bars
            zorder=3
        )

    # strong zero line
    ax.axhline(0, color="#333333", linewidth=1, zorder=4)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[group_col], rotation=0, ha="center")

    if title:
        ax.set_title(title, loc="left", fontweight="bold")

    ax.grid(True, axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.1), ncol=n_metrics)

    return ax


def plot_sensitivity(
    swept: pd.DataFrame,
    metric: str,
    sweep_col: str,
    baseline: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    metric_label: str | None = None,
    sweep_label: str | None = None,
    title: str | None = None,
    color: str = "#348ABD",
    marker: str = "o",
) -> plt.Axes:
    """Plot a single metric's sensitivity to a swept parameter.

    Args:
        swept: DataFrame of swept configurations with {metric}_mean and {metric}_std columns.
        metric: Name of the metric (used to find {metric}_mean and {metric}_std columns).
        sweep_col: Column name for the swept parameter (x-axis).
        baseline: Optional DataFrame with baseline row(s) for reference line.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        metric_label: Label for the y-axis. Defaults to metric name.
        sweep_label: Label for the x-axis. Defaults to sweep_col.
        title: Plot title. Defaults to "{metric_label} sensitivity".
        color: Color for the line and markers.
        marker: Marker style.

    Returns:
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    _clean_axes(ax)

    metric_label = metric_label or metric
    sweep_label = sweep_label or sweep_col
    title = title or f"{metric_label} sensitivity"

    ax.errorbar(
        swept[sweep_col],
        swept[f"{metric}_mean"],
        yerr=swept[f"{metric}_std"],
        fmt=f"{marker}-",
        capsize=0,
        linewidth=1.5,
        markersize=6,
        color=color,
        markeredgecolor="white",
        zorder=3
    )

    if baseline is not None and not baseline.empty:
        base_val = baseline[f"{metric}_mean"].iloc[0]
        base_std = baseline[f"{metric}_std"].iloc[0]
        ax.axhline(base_val, color="#555555", linestyle="--", label="baseline")
        ax.axhspan(base_val - base_std, base_val + base_std, color="#999999", alpha=0.1)

    ax.set_xlabel(sweep_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", zorder=-1)

    return ax


def plot_tradeoff_with_pareto(
    swept: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sweep_col: str,
    baseline: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    sweep_label: str | None = None,
    title: str = "tradeoff analysis",
    cmap: str = "viridis",
    show_pareto: bool = True,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> plt.Axes:
    """Plot a tradeoff scatter with optional Pareto frontier overlay.

    Args:
        swept: DataFrame of swept configurations with metric columns.
        x_metric: Metric for x-axis (uses {x_metric}_mean and {x_metric}_std).
        y_metric: Metric for y-axis (uses {y_metric}_mean and {y_metric}_std).
        sweep_col: Column for color-coding points.
        baseline: Optional DataFrame with baseline row(s) for reference marker.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        x_label: Label for x-axis. Defaults to x_metric.
        y_label: Label for y-axis. Defaults to y_metric.
        sweep_label: Label for colorbar. Defaults to sweep_col.
        title: Plot title.
        cmap: Colormap for scatter points.
        show_pareto: Whether to overlay the Pareto frontier.
        maximize_x: Whether higher x values are better (for Pareto).
        maximize_y: Whether higher y values are better (for Pareto).

    Returns:
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    _clean_axes(ax)

    x_label = x_label or x_metric
    y_label = y_label or y_metric
    sweep_label = sweep_label or sweep_col

    # scatter plot with color-coded sweep parameter
    scatter = ax.scatter(
        swept[f"{x_metric}_mean"],
        swept[f"{y_metric}_mean"],
        c=swept[sweep_col],
        cmap=cmap,
        s=100,
        edgecolors="white",
        zorder=3,
    )

    # background error bars
    for _, row in swept.iterrows():
        ax.errorbar(
            row[f"{x_metric}_mean"],
            row[f"{y_metric}_mean"],
            xerr=row[f"{x_metric}_std"],
            yerr=row[f"{y_metric}_std"],
            fmt="none",
            color="#cccccc",
            alpha=0.5,
            zorder=2
        )

    # baseline marker
    if baseline is not None and not baseline.empty:
        brow = baseline.iloc[0]
        ax.scatter(
            brow[f"{x_metric}_mean"],
            brow[f"{y_metric}_mean"],
            marker="X",
            s=120,
            c="#444444",
            edgecolors="white",
            zorder=4,
            label="baseline",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, zorder=-1)

    # colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=sweep_label)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)

    # Pareto frontier
    if show_pareto:
        _overlay_pareto_frontier(
            ax,
            swept,
            x_metric,
            y_metric,
            maximize_x=maximize_x,
            maximize_y=maximize_y,
        )

    # legend if baseline was added
    if baseline is not None and not baseline.empty:
        ax.legend(frameon=False, loc="best")

    return ax


def _overlay_pareto_frontier(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> list[tuple[float, float]]:
    """Compute and draw the Pareto frontier on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        summary: DataFrame with {x_metric}_mean and {y_metric}_mean columns.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.

    Returns:
        List of (x, y) tuples representing the Pareto frontier points.
    """
    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"

    # sort by x (descending if maximize, ascending otherwise)
    sorted_df = summary.sort_values(x_mean, ascending=not maximize_x)

    pareto_points = []
    best_y = float("-inf") if maximize_y else float("inf")

    for _, row in sorted_df.iterrows():
        y_val = row[y_mean]
        is_better = (y_val > best_y) if maximize_y else (y_val < best_y)
        if is_better:
            pareto_points.append((row[x_mean], y_val))
            best_y = y_val

    if pareto_points:
        pareto_x, pareto_y = zip(*sorted(pareto_points))
        ax.plot(
            pareto_x,
            pareto_y,
            color="#E24A33",
            linestyle="-",
            linewidth=1.5,
            alpha=0.4,
            zorder=2,
            label="pareto frontier",
        )

    return pareto_points


def plot_pareto_frontier(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    ax: plt.Axes | None = None,
    maximize_x: bool = True,
    maximize_y: bool = True,
    frontier_style: dict[str, Any] | None = None,
) -> tuple[plt.Axes, list[tuple[float, float]]]:
    """Overlay Pareto frontier on an existing or new scatter plot.

    Args:
        summary: DataFrame with {x_metric}_mean and {y_metric}_mean columns.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.
        frontier_style: Dict of style kwargs for the frontier line.

    Returns:
        Tuple of (axes, list of Pareto frontier points as (x, y) tuples).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        _clean_axes(ax)

    if frontier_style is None:
        frontier_style = {
            "color": "#E24A33",
            "linestyle": "-",
            "linewidth": 1.5,
            "alpha": 0.4,
            "zorder": 2,
        }

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"

    sorted_df = summary.sort_values(x_mean, ascending=not maximize_x)

    pareto_points = []
    best_y = float("-inf") if maximize_y else float("inf")

    for _, row in sorted_df.iterrows():
        y_val = row[y_mean]
        is_better = (y_val > best_y) if maximize_y else (y_val < best_y)
        if is_better:
            pareto_points.append((row[x_mean], y_val))
            best_y = y_val

    if pareto_points:
        pareto_x, pareto_y = zip(*sorted(pareto_points))
        ax.plot(pareto_x, pareto_y, label="pareto frontier", **frontier_style)
        ax.legend(frameon=False, loc="best")

    return ax, pareto_points
