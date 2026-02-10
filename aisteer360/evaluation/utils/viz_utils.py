"""Visualization utilities for benchmark profiles."""

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    """Plot a metric with error bars across configurations.

    Args:
        summary: DataFrame from `summarize_by_config` containing `{metric}_mean` and `{metric}_std`.
        metric: Base name of the metric column (will look for `{metric}_mean` and `{metric}_std`).
        x_col: Column to use for x-axis values.
        baseline_value: Optional baseline value to draw as horizontal line.
        baseline_std: Optional baseline std to draw as shaded region around baseline.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label. Defaults to metric name.
        **errorbar_kwargs: Additional kwargs passed to `ax.errorbar()`.

    Returns:
        The matplotlib Axes object.

    Example:
        >>> summary = summarize_by_config(df, ["accuracy"])
        >>> plot_metric_by_config(summary, "accuracy", x_col="alpha")
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    defaults = {"fmt": "o-", "capsize": 4, "capthick": 1.5, "markersize": 8}
    defaults.update(errorbar_kwargs)

    ax.errorbar(
        summary[x_col],
        summary[mean_col],
        yerr=summary[std_col],
        **defaults,
    )

    if baseline_value is not None:
        ax.axhline(baseline_value, color="red", linestyle="--", label="Baseline")
        if baseline_std is not None:
            ax.axhspan(
                baseline_value - baseline_std,
                baseline_value + baseline_std,
                color="red",
                alpha=0.1,
            )

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_tradeoff_scatter(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_col: str | None = None,
    label_col: str | None = None,
    baseline_row: pd.Series | None = None,
    ax: plt.Axes | None = None,
    title: str = "Metric Tradeoff",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    **scatter_kwargs: Any,
) -> plt.Axes:
    """Plot a scatter of two metrics showing their tradeoff.

    Args:
        summary: DataFrame from `summarize_by_config`.
        x_metric: Base name for x-axis metric (looks for `{x_metric}_mean`, `{x_metric}_std`).
        y_metric: Base name for y-axis metric.
        color_col: Column to use for point coloring. If None, uses uniform color.
        label_col: Column to use for point labels (annotations).
        baseline_row: Optional Series representing the baseline for special marking.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        xlabel: X-axis label. Defaults to x_metric.
        ylabel: Y-axis label. Defaults to y_metric.
        cmap: Colormap name if `color_col` is provided.
        **scatter_kwargs: Additional kwargs passed to `ax.scatter()`.

    Returns:
        The matplotlib Axes object.

    Example:
        >>> plot_tradeoff_scatter(
        ...     summary, x_metric="accuracy", y_metric="reward",
        ...     color_col="alpha", label_col="config"
        ... )
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"
    x_std = f"{x_metric}_std"
    y_std = f"{y_metric}_std"

    defaults = {"s": 120, "edgecolors": "black", "zorder": 3}
    defaults.update(scatter_kwargs)

    if color_col is not None and color_col in summary.columns:
        defaults["c"] = summary[color_col]
        defaults["cmap"] = cmap
        scatter = ax.scatter(summary[x_mean], summary[y_mean], **defaults)
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(summary[x_mean], summary[y_mean], **defaults)

    # Error crosses
    for _, row in summary.iterrows():
        ax.errorbar(
            row[x_mean],
            row[y_mean],
            xerr=row.get(x_std, 0),
            yerr=row.get(y_std, 0),
            fmt="none",
            color="gray",
            alpha=0.5,
            capsize=2,
        )

    # Labels
    if label_col is not None and label_col in summary.columns:
        for _, row in summary.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row[x_mean], row[y_mean]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    # Baseline marker
    if baseline_row is not None:
        ax.scatter(
            baseline_row[x_mean],
            baseline_row[y_mean],
            marker="X",
            s=180,
            c="red",
            edgecolors="black",
            zorder=4,
            label="Baseline",
        )
        ax.errorbar(
            baseline_row[x_mean],
            baseline_row[y_mean],
            xerr=baseline_row.get(x_std, 0),
            yerr=baseline_row.get(y_std, 0),
            fmt="none",
            color="red",
            alpha=0.5,
            capsize=2,
        )

    ax.set_xlabel(xlabel or x_metric)
    ax.set_ylabel(ylabel or y_metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

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
) -> plt.Axes:
    """Plot a heatmap from a pivoted DataFrame.

    This is a thin wrapper around seaborn's heatmap for consistent styling.

    Args:
        pivot_df: Pivoted DataFrame (index=rows, columns=cols, values=metric).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
        cmap: Colormap name.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        cbar_label: Label for the colorbar.

    Returns:
        The matplotlib Axes object.

    Example:
        >>> pivot = df.pivot(index="instruction_type", columns="alpha", values="follow_rate")
        >>> plot_metric_heatmap(pivot, title="Follow Rate by Type", vmin=0, vmax=1)
    """
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError("seaborn is required for heatmap plots: pip install seaborn") from e

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    cbar_kws = {"label": cbar_label} if cbar_label else {}

    sns.heatmap(
        pivot_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws=cbar_kws,
    )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

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
    """Plot grouped bar chart comparing metrics across groups.

    Args:
        comparison_df: DataFrame with one row per group.
        metric_cols: Columns to plot as bars.
        group_col: Column containing group labels (used for x-axis).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        ylabel: Y-axis label.
        colors: Colors for each metric. If None, uses default cycle.
        bar_width: Width of each bar.

    Returns:
        The matplotlib Axes object.

    Example:
        >>> comparison = pd.DataFrame({
        ...     "instruction_type": ["A", "B"],
        ...     "baseline": [0.5, 0.6],
        ...     "steered": [0.7, 0.8],
        ... })
        >>> plot_comparison_bars(comparison, ["baseline", "steered"], "instruction_type")
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    n_groups = len(comparison_df)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_metrics]

    total_width = bar_width * n_metrics
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_metrics)

    for i, (col, offset) in enumerate(zip(metric_cols, offsets)):
        ax.bar(
            x + offset,
            comparison_df[col],
            bar_width,
            label=col,
            color=colors[i % len(colors)],
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[group_col], rotation=45, ha="right")
    if title:
        ax.set_title(title)
    ax.legend()

    return ax


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
        summary: DataFrame with `{metric}_mean` columns.
        x_metric: Base name for x-axis metric.
        y_metric: Base name for y-axis metric.
        ax: Matplotlib axes. If None, creates new figure.
        maximize_x: If True, larger x is better; if False, smaller x is better.
        maximize_y: If True, larger y is better; if False, smaller y is better.
        frontier_style: Style kwargs for the frontier line.
            Defaults to `{"color": "green", "linestyle": "--", "linewidth": 2, "alpha": 0.5}`.

    Returns:
        Tuple of (axes, pareto_points) where pareto_points is a list of (x, y) tuples.

    Example:
        >>> ax = plot_tradeoff_scatter(summary, "accuracy", "reward")
        >>> ax, frontier = plot_pareto_frontier(summary, "accuracy", "reward", ax=ax)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if frontier_style is None:
        frontier_style = {"color": "green", "linestyle": "--", "linewidth": 2, "alpha": 0.5}

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"

    # Sort by x
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
        ax.plot(pareto_x, pareto_y, label="Pareto Frontier", **frontier_style)
        ax.legend()

    return ax, pareto_points


def create_tradeoff_figure(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sweep_col: str,
    baseline_pipeline: str = "baseline",
    x_label: str | None = None,
    y_label: str | None = None,
    sweep_label: str | None = None,
    figsize: tuple[float, float] = (15, 4.5),
    save_path: str | None = None,
) -> plt.Figure:
    """Create a 3-panel figure showing metric tradeoffs.

    Panel 1: x_metric vs sweep_col
    Panel 2: y_metric vs sweep_col
    Panel 3: y_metric vs x_metric (scatter with baseline highlighted)

    Args:
        summary: DataFrame from `summarize_by_config` with both metrics.
        x_metric: First metric (e.g., "accuracy").
        y_metric: Second metric (e.g., "reward").
        sweep_col: Column representing the swept parameter (e.g., "alpha").
        baseline_pipeline: Pipeline name to identify baseline rows.
        x_label: Label for x_metric. Defaults to x_metric.
        y_label: Label for y_metric. Defaults to y_metric.
        sweep_label: Label for sweep_col. Defaults to sweep_col.
        figsize: Figure size.
        save_path: If provided, saves the figure to this path.

    Returns:
        The matplotlib Figure object.

    Example:
        >>> fig = create_tradeoff_figure(
        ...     summary, x_metric="accuracy", y_metric="reward",
        ...     sweep_col="alpha", save_path="tradeoff.png"
        ... )
    """
    x_label = x_label or x_metric
    y_label = y_label or y_metric
    sweep_label = sweep_label or sweep_col

    baseline = summary[summary["pipeline"] == baseline_pipeline]
    swept = summary[summary["pipeline"] != baseline_pipeline].sort_values(sweep_col)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: x_metric vs sweep
    ax1 = axes[0]
    ax1.errorbar(
        swept[sweep_col],
        swept[f"{x_metric}_mean"],
        yerr=swept[f"{x_metric}_std"],
        fmt="o-",
        capsize=4,
        capthick=1.5,
        markersize=8,
    )
    if not baseline.empty:
        base_val = baseline[f"{x_metric}_mean"].iloc[0]
        base_std = baseline[f"{x_metric}_std"].iloc[0]
        ax1.axhline(base_val, color="red", linestyle="--", label="Baseline")
        ax1.axhspan(base_val - base_std, base_val + base_std, color="red", alpha=0.1)
    ax1.set_xlabel(sweep_label)
    ax1.set_ylabel(x_label)
    ax1.set_title(f"{x_label} vs {sweep_label}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: y_metric vs sweep
    ax2 = axes[1]
    ax2.errorbar(
        swept[sweep_col],
        swept[f"{y_metric}_mean"],
        yerr=swept[f"{y_metric}_std"],
        fmt="s-",
        capsize=4,
        capthick=1.5,
        markersize=8,
        color="orange",
    )
    if not baseline.empty:
        base_val = baseline[f"{y_metric}_mean"].iloc[0]
        base_std = baseline[f"{y_metric}_std"].iloc[0]
        ax2.axhline(base_val, color="red", linestyle="--", label="Baseline")
        ax2.axhspan(base_val - base_std, base_val + base_std, color="red", alpha=0.1)
    ax2.set_xlabel(sweep_label)
    ax2.set_ylabel(y_label)
    ax2.set_title(f"{y_label} vs {sweep_label}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: scatter
    ax3 = axes[2]
    scatter = ax3.scatter(
        swept[f"{x_metric}_mean"],
        swept[f"{y_metric}_mean"],
        c=swept[sweep_col],
        cmap="viridis",
        s=120,
        edgecolors="black",
        zorder=3,
    )
    for _, row in swept.iterrows():
        ax3.errorbar(
            row[f"{x_metric}_mean"],
            row[f"{y_metric}_mean"],
            xerr=row[f"{x_metric}_std"],
            yerr=row[f"{y_metric}_std"],
            fmt="none",
            color="gray",
            alpha=0.5,
            capsize=2,
        )
    if not baseline.empty:
        brow = baseline.iloc[0]
        ax3.scatter(
            brow[f"{x_metric}_mean"],
            brow[f"{y_metric}_mean"],
            marker="X",
            s=180,
            c="red",
            edgecolors="black",
            zorder=4,
            label="Baseline",
        )
        ax3.errorbar(
            brow[f"{x_metric}_mean"],
            brow[f"{y_metric}_mean"],
            xerr=brow[f"{x_metric}_std"],
            yerr=brow[f"{y_metric}_std"],
            fmt="none",
            color="red",
            alpha=0.5,
            capsize=2,
        )
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_title(f"{y_label} vs {x_label} Tradeoff")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label=sweep_label)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
