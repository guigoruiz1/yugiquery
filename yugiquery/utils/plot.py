# yugiquery/utils/plot.py

# -*- coding: utf-8 -*-

# --- Imports: Standard Library --- #
import colorsys
from typing import List, Tuple, Callable, Any

# --- Imports: Third-Party --- #
import numpy as np
import pandas as pd
from matplotlib.colors import cnames, to_rgb, rgb_to_hsv, hex2color, LogNorm
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib_venn import venn2
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

# --- Imports: Local Application --- #
from .helpers import *
from .dirs import dirs


# --- Matplotlib Settings & Overrides --- #
if dirs.is_notebook:
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    plt.style.use("default")  # TODO: Make this configurable
    set_matplotlib_formats("svg")  # Needed for dynanmic theme


# --- Plot Dictionaries --- #
colors_dict = load_json(dirs.get_asset("json", "colors.json"))
"""
Dictionary mapping card or type names to color hex codes, loaded from the colors.json asset.
Used for consistent color assignment in plots.
"""

DEFAULT_FONTSIZES: dict[str, int] = {"label": 14, "title": 20, "tick": 12, "legend": 12, "suptitle": 20}
"""
Default font sizes for plot elements (labels, titles, ticks, legends, suptitles).
Keys are element names, values are font sizes in points.
"""

_angle_map = {
    "→": 0,
    "↗": np.pi / 4,
    "↑": np.pi / 2,
    "↖": 3 * np.pi / 4,
    "←": np.pi,
    "↙": 5 * np.pi / 4,
    "↓": 3 * np.pi / 2,
    "↘": 7 * np.pi / 4,
}
"""Mapping of arrow symbols to angles in radians for polar plotting of link arrows.
Keys are arrow symbols (e.g., "→"), values are corresponding angles in radians (e.g., 0 for "→", π/4 for "↗", etc.).
Used in the arrows() plotting function to convert arrow symbols to angles for polar bar plots.
"""


# --- Custom Legend Handler Class --- #


class MulticolorPatchHandler:
    """
    Custom legend handler to display a multicolored rectangle with a single uniform hatch across the entire box.
    """

    # --- Helper Functions --- #
    def __init__(self, colors: List[str], hatch: str | None = None, edgecolor: str = "black", **kwargs):
        self.colors = colors  # List of colors for different segments
        self.hatch = hatch  # Single hatch applied across the entire box
        self.edgecolor = edgecolor  # Single edge color for the whole box
        self.kwargs = kwargs

    def legend_artist(self, legend, orig_handle, fontsize: int, handlebox) -> mpatches.Rectangle:
        width = handlebox.width
        height = handlebox.height

        # Define the width for each color segment
        color_width = width / len(self.colors)

        # Create multicolored patches
        for i, color in enumerate(self.colors):
            patch = mpatches.Rectangle(
                (handlebox.xdescent + i * color_width, -handlebox.ydescent),
                color_width,
                height,
                facecolor=color,
                transform=handlebox.get_transform(),
                **self.kwargs,
            )
            handlebox.add_artist(patch)

        # Apply a transparent hatch over the entire box
        hatch_patch = mpatches.Rectangle(
            (handlebox.xdescent, -handlebox.ydescent),
            width,
            height,
            facecolor="none",  # No facecolor to avoid covering the colors underneath
            hatch=self.hatch,  # Apply the uniform hatch across the full box
            edgecolor=self.edgecolor,  # No edge color for the hatch
            transform=handlebox.get_transform(),
            **self.kwargs,
        )
        handlebox.add_artist(hatch_patch)
        return hatch_patch


# --- Helper Functions --- #


def _is_light_color(color: str | tuple, threshold: float = 0.6) -> bool:
    """
    Check if a given color is light or dark based on a specified threshold.

    Args:
        color (str | tuple): The color to be checked. Can be a hex string or RGB tuple.
        threshold (float): The threshold to determine if a color is light or dark. Default is 0.6.

    Returns:
        bool: True if the color is light, False otherwise.

    Raises:
        ValueError: If the color value is invalid.
    """
    try:
        rgb = hex2color(color) if isinstance(color, str) else color
        hsv = rgb_to_hsv(rgb)
        return hsv[2] > threshold
    except Exception as e:
        raise ValueError(f"Invalid color value for _is_light_color: {color}") from e


def _adjust_lightness(color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """
    Adjust the lightness of a given color by a specified amount.

    Args:
        color (str): The color to be adjusted, in string format (hex or named color).
        amount (float, optional): The amount by which to adjust the lightness of the color. Default value is 0.5.

    Returns:
        tuple: The adjusted color in RGB format.

    Raises:
        ValueError: If the color value is invalid.
    """
    try:
        rgb = to_rgb(cnames.get(color, color))
        # Lighten by blending with white in RGB space
        if amount >= 1:
            blend = min(1, amount - 1)  # 0 = original, 1 = white
            r = rgb[0] + (1 - rgb[0]) * blend
            g = rgb[1] + (1 - rgb[1]) * blend
            b = rgb[2] + (1 - rgb[2]) * blend
            return (r, g, b)
        else:
            # Darken by blending with black in RGB space
            blend = max(0, 1 - amount)  # 0 = original, 1 = black
            r = rgb[0] * (1 - blend)
            g = rgb[1] * (1 - blend)
            b = rgb[2] * (1 - blend)
            return (r, g, b)
    except Exception as e:
        raise ValueError(f"Invalid color value for _adjust_lightness: {color}") from e


def _align_yaxis(ax1: Axes, v1: float, ax2: Axes, v2: float) -> None:
    """
    Adjust the y-axis of two subplots so that the specified values in each subplot are aligned.

    Args:
        ax1 (Axes): The first subplot.
        v1 (float): The value in ax1 to align.
        ax2 (Axes): The second subplot.
        v2 (float): The value in ax2 to align.

    Raises:
        RuntimeError: If alignment fails.
    """
    try:
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        _adjust_yaxis(ax=ax2, ydif=(y1 - y2) / 2, v=v2)
        _adjust_yaxis(ax=ax1, ydif=(y2 - y1) / 2, v=v1)
    except Exception as e:
        raise RuntimeError("Failed to align y-axes.") from e


# --- Rate Utilities --- #
def _adjust_yaxis(ax: Axes, ydif: float, v: float) -> None:
    """
    Shift the y-axis of a subplot by a specified amount, while maintaining the location of a specified point.

    Args:
        ax (Axes): The subplot whose y-axis is to be adjusted.
        ydif (float): The amount by which to adjust the y-axis.
        v (float): The location of the point whose position should remain unchanged.

    Raises:
        RuntimeError: If adjustment fails.
    """
    try:
        inv = ax.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
        miny, maxy = ax.get_ylim()
        miny, maxy = miny - v, maxy - v
        if -miny > maxy or (-miny == maxy and dy > 0):
            nminy = miny
            nmaxy = miny * (maxy + dy) / (miny + dy)
        else:
            nmaxy = maxy
            nminy = maxy * (miny + dy) / (maxy + dy)
        ax.set_ylim(bottom=nminy + v, top=nmaxy + v)
    except Exception as e:
        raise RuntimeError("Failed to adjust y-axis.") from e


def _generate_rate_grid(
    df: pd.DataFrame,
    ax: Axes,
    xlabel: str | None = "Date",
    size_pct: str | float = "150%",
    pad: int = 0,
    colors: List[str] | None = None,
    cumsum: bool = True,
    fill: bool = False,
    limit_year: bool = False,
) -> list[Axes]:
    """
    Generate a grid of subplots displaying cumulative and yearly rates from a DataFrame indexed by date.

    Args:
        df (pd.DataFrame): DataFrame with datetime-like index and one or more columns of numeric data.
        ax (matplotlib.axes.Axes): The axis to plot the main (cumulative) plot on.
        xlabel (str | None, optional): Label for the x-axis. Defaults to 'Date'.
        size_pct (str | float, optional): Size of the bottom subplot as a percentage of the top subplot. Defaults to '150%'.
        pad (int, optional): Padding between subplots in pixels. Defaults to 0.
        colors (List[str] | None, optional): Colors for the plot lines or areas. Defaults to matplotlib default cycle.
        cumsum (bool, optional): If True, plot cumulative sum; if False, only yearly/monthly rates. Defaults to True.
        fill (bool, optional): If True, fill area under cumulative curve. Defaults to False.
        limit_year (bool, optional): If True, limit x-axis to next full year. Defaults to False.

    Returns:
        list[matplotlib.axes.Axes]: List containing the main and secondary axes (cumulative and yearly).

    Raises:
        ValueError: If DataFrame index is not datetime-like or is empty.
    """
    if colors is None:
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    index_name = f" {str(df.index.name).lower()}" if df.index.name else ""
    if cumsum:
        cumsum_ax = ax
        divider = make_axes_locatable(axes=cumsum_ax)
        yearly_ax = divider.append_axes(position="bottom", size=size_pct, pad=pad)
        cumsum_ax.figure.add_axes(yearly_ax)
        cumsum_ax.set_xticklabels([])
        axes = [cumsum_ax, yearly_ax]

        y = df.fillna(0).cumsum()

        if len(df.columns) == 1:
            cumsum_ax.plot(y, label="Cumulative", c=colors[0], antialiased=True)
            if fill:
                cumsum_ax.fill_between(x=y.index, y1=y.values.T[0], color=colors[0], alpha=0.1, hatch="x")
            cumsum_ax.set_ylabel(f"Cumulative {y.columns[0]}")  # Wrap text
            cumsum_ax.legend(loc="upper left", ncols=int(len(df.columns) / 5 + 1))  # Test
        else:
            cumsum_ax.stackplot(y.index, y.values.T, labels=y.columns, colors=colors, antialiased=True)
            cumsum_ax.set_ylabel(f"Cumulative{index_name}")
            cumsum_ax.figure.legend(
                loc="upper center", bbox_to_anchor=(0.5, 0), ncols=len(df.columns), frameon=False
            )  # Test

        yearly_ax.set_ylabel(f"Yearly{index_name} rate")

        def func(x, pos):
            return "" if np.isclose(x, 0) else f"{round(x):.0f}"

        cumsum_ax.yaxis.set_major_formatter(FuncFormatter(func))

    else:
        yearly_ax = ax
        axes = [yearly_ax]

        if len(df.columns) == 1:
            yearly_ax.set_ylabel(f"{df.columns[0]}\nYearly{index_name} rate")
        else:
            yearly_ax.set_ylabel(f"Yearly{index_name} rate")

    # Remove the last year if it is incomplete
    yearly_rate = df.resample("YE").sum()
    if limit_year and yearly_rate.index[-1].timestamp() > arrow.utcnow().shift(years=1).timestamp():
        yearly_rate = yearly_rate[:-1]

    if len(df.columns) == 1:
        monthly_ax = yearly_ax.twinx()
        monthly_rate = df.resample("ME").sum()
        monthly_ax.bar(
            x=monthly_rate.index,
            height=monthly_rate.T.values[0],
            width=monthly_rate.index.to_series().diff(),
            label="Monthly rate",
            color=colors[2],
            antialiased=True,
        )
        monthly_ax.set_ylabel(f"Monthly{index_name} rate")
        monthly_ax.legend(loc="upper right")

        yearly_ax.plot(
            yearly_rate,
            label="Yearly rate",
            ls="--",
            c=colors[1],
            antialiased=True,
        )
        yearly_ax.legend(loc="upper left", ncols=int(len(df.columns) / 8 + 1))

    else:
        yearly_ax.stackplot(
            yearly_rate.index, yearly_rate.values.T, labels=yearly_rate.columns, colors=colors, antialiased=True
        )
        if not cumsum:
            yearly_ax.legend(loc="upper left", ncols=int(len(df.columns) / 8 + 1))

    if xlabel is not None:
        yearly_ax.set_xlabel(xlabel)
    else:
        yearly_ax.set_xticklabels([])

    for temp_ax in axes:
        temp_ax.set_xlim(
            (
                df.index.min() - pd.Timedelta(weeks=13),
                df.index.max() + pd.Timedelta(weeks=52),
            )
        )
        temp_ax.xaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.yaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.xaxis.set_major_locator(mdates.YearLocator())
        temp_ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        temp_ax.set_axisbelow(True)
        temp_ax.grid(ls=":")

    yearly_ax.tick_params(axis="x", rotation=45)

    if len(df.columns) == 1:
        _align_yaxis(ax1=yearly_ax, v1=0, ax2=monthly_ax, v2=0)
        l = yearly_ax.get_ylim()
        l2 = monthly_ax.get_ylim()

        def f(x):
            return l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])

        ticks = f(yearly_ax.get_yticks())
        monthly_ax.yaxis.set_major_locator(FixedLocator(ticks.tolist()))
        monthly_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{round(x):.0f}"))
        monthly_ax.yaxis.set_minor_locator(AutoMinorLocator())
        axes.append(monthly_ax)

    return axes


def rate(
    df: pd.DataFrame,
    figsize: Tuple[int, int] | None = None,
    title: str = "",
    colors: List[str] | None = None,
    cumsum: bool = True,
    bg: pd.DataFrame | None = None,
    vlines: pd.Series | None = None,
    fill: bool = False,
    limit_year: bool = False,
    subplots: bool = False,
    hspace: float = 0.05,
    **kwargs,
) -> Figure:
    """
    Create a visualization of rate changes over time for multiple variables in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot. Index must be datetime-like.
        figsize (Tuple[int, int] | None, optional): Size of the figure. Defaults to None.
        title (str, optional): Title of the plot. Defaults to an empty string.
        colors (List[str] | None, optional): List of colors for the plot lines. Defaults to None.
        cumsum (bool, optional): Whether to plot cumulative sum of data. Defaults to True.
        bg (pd.DataFrame | None, optional): Data for background shading. Defaults to None.
        vlines (pd.Series | None, optional): Series for vertical lines. Defaults to None.
        fill (bool, optional): Whether to fill the area under the cumulative sum curve. Defaults to False.
        limit_year (bool, optional): Whether to limit the x-axis to the next full year. Defaults to False.
        subplots (bool, optional): Whether to create a grid of subplots for each column in the DataFrame. Defaults to False.
        hspace (float, optional): Height space between subplots. Defaults to 0.05.
        **kwargs: Additional keyword arguments to pass to the plotting functions. Not implemented.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If input DataFrame is empty or index is not datetime-like.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime-like.")
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Order columns by the first date where they have a value > 0 (ascending).
    mask = df.gt(0)
    first_dates = mask.idxmax().where(mask.any(), pd.Timestamp.max)
    new_order = first_dates.sort_values().index
    df = df[new_order]

    # Reorder any user-provided colors to match column order
    if colors is not None:
        try:
            seq = list(colors)
            if len(seq) == len(first_dates.index):
                mapping = dict(zip(list(first_dates.index), seq))
                colors = [mapping[col] for col in new_order]
        except Exception:
            pass  # Fall back to original colors if reordering fails

    num_cols = len(df.columns)
    top_space = 0.5

    # Setup figure and gridspec
    figsize = figsize or ((14, num_cols * 3 * (1 + cumsum)) if subplots else (14, 6))
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(num_cols, 1, height_ratios=[3] * num_cols, hspace=hspace) if subplots else GridSpec(1, 1, hspace=hspace)

    fig.suptitle(
        f'{title or df.index.name}{f" by {str(df.columns.name).lower()}" if df.columns.name else ""}',
        y=1,
    )

    # Initialize colors if not provided
    if colors is None:
        colors = (
            list(plt.cm.get_cmap("tab20").colors)  # pyright: ignore[reportAttributeAccessIssue]
            if subplots
            else list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        )

    # Create subplots and apply shading, vertical lines
    axes = []
    for i, col in enumerate(df.columns):
        ax = fig.add_subplot(gs[i] if subplots else gs[0])

        # Select colors for this subplot
        if subplots:
            subplot_colors = [
                colors[2 * i % len(colors)],
                colors[2 * i % len(colors)],
                colors[(2 * i + 1) % len(colors)],
            ]
        else:
            subplot_colors = colors

        sub_axes = _generate_rate_grid(
            df=df[col].to_frame() if subplots else df,
            ax=ax,
            colors=subplot_colors,
            cumsum=cumsum,
            fill=fill,
            limit_year=limit_year,
            size_pct="100%" if subplots else "150%",
            xlabel="Date" if (i + 1) == len(df.columns) or not subplots else None,
        )
        axes.extend(sub_axes[:2])
        if not subplots:
            break

    # Add background shading and vertical lines
    if bg is not None and "end" in bg:
        bg = bg.copy()
        bg["end"] = bg["end"].fillna(df.index.max())
        _add_background_shading(axes=axes, bg=bg)
    if vlines is not None:
        _add_vertical_lines(axes=axes, vlines=vlines, cumsum=cumsum)

    fig.subplots_adjust(top=1 - top_space / fig.get_figheight())
    return fig


def _add_background_shading(axes: List[Axes], bg: pd.DataFrame, colors: List | None = None) -> None:
    """
    Add background shading to the subplots.

    Args:
        axes (List[plt.Axes]): List of axes to apply background shading.
        bg (pd.DataFrame): DataFrame for background shading. Must contain 'begin' and 'end' columns.
        colors (list | None, optional): Colormap for the shading. If None, colors_dict is used. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If required columns are missing in bg.
    """
    if not all(col in bg.columns for col in ["begin", "end"]):
        raise ValueError("Background DataFrame must contain 'begin' and 'end' columns.")
    sec_ax = axes[0].secondary_xaxis("top")
    sec_ax.set_xticks(bg.mean(axis=1))
    sec_ax.set_xticklabels(bg.index)
    for ax in axes:
        c = 0
        for idx, row in bg.iterrows():
            if row["end"] > pd.to_datetime(ax.get_xlim()[0], unit="D"):
                ax.axvspan(
                    row["begin"],
                    row["end"],
                    alpha=0.1,
                    color=(colors[c] if colors is not None else colors_dict.get(idx, f"C{c}")),
                    zorder=-1,
                )
                c += 1


def _add_vertical_lines(axes: List[Axes], vlines: pd.Series, color="maroon", cumsum: bool = False) -> None:
    """
    Add vertical lines to the subplots.

    Args:
        axes (List[plt.Axes]): List of axes to apply vertical lines.
        vlines (pd.Series): Series for vertical lines.
        color (str): Color for the vertical lines.
        cumsum (bool, optional): Whether cumulative sum is being plotted. Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If vlines is empty or not a Series.
    """
    if not isinstance(vlines, pd.Series) or vlines.empty:
        raise ValueError("vlines must be a non-empty pandas Series.")
    for ix, ax in enumerate(axes):
        xlim = pd.to_datetime(ax.get_xlim()[0], unit="D")
        for idx, row in vlines.items():
            if row > xlim:
                line = ax.axvline(row, ls="-.", c=color, lw=1)
                if ix == 0:
                    (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                    ax.text(
                        (x0 + x1) / 2 + 25,
                        (0.05 if cumsum else 0.95),
                        str(idx),
                        c=color,
                        ha="left",
                        va=("bottom" if cumsum else "top"),
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                    )


# --- Dedicated Plot Types --- #
def arrows(arrows: pd.Series, figsize: Tuple[int, int] = (6, 6), **kwargs) -> Figure:
    """
    Create a polar plot to visualize the frequency of each arrow direction in a pandas Series.

    Args:
        arrows (pandas.Series): A pandas Series containing arrow symbols as string data type.
        figsize (Tuple[int, int], optional): The width and height of the figure. Defaults to (6, 6).
        **kwargs: Additional keyword arguments to be passed to the bar() method.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If input Series is empty or contains invalid arrow symbols.
    """
    if arrows.empty:
        raise ValueError("Input Series for arrows is empty.")
    # Count the frequency of each arrow direction
    counts = arrows.value_counts().sort_index()

    if not all(a in _angle_map for a in counts.index):
        raise ValueError("Input Series contains invalid arrow symbols.")
    angles = counts.index.map(_angle_map)

    # Create a polar plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(polar=True)
    ax.bar(x=angles, height=counts, width=0.5, color=colors_dict.get("Link Monster", "blue"), **kwargs)

    # Set the label for each arrow
    ax.set_xticks(list(_angle_map.values()))
    ax.set_xticklabels(["▶", "◥", "▲", "◤", "◀", "◣", "▼", "◢"], fontsize=18)

    # Set radius grid location
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(list(ticks[1:])))
    ax.set_rorigin(-5)  # pyright: ignore[reportAttributeAccessIssue]

    # Set the title of the plot
    ax.set_title("Link Arrows")
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def box(
    df: pd.DataFrame, mean: bool = True, group_string: str = "%Y", x: str | None = None, y: str | None = None, **kwargs
) -> Figure:
    """
    Plots a box plot of a given DataFrame using seaborn, with the year of the timestamp column on the x-axis and the remaining column on the y-axis.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the timestamps and another numeric column.
        group_string (str, optional): The string format to group the timestamps by. Defaults to "%Y", representing year.
        mean (bool, optional): If True, plots a line representing the mean of each box. Defaults to True.
        x (str | None, optional): The column name to use for the x-axis. Defaults to None.
        y (str | None, optional): The column name to use for the y-axis. Defaults to None.
        **kwargs: Additional keyword arguments to pass to seaborn.boxplot().

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If the DataFrame is empty or has no suitable x/y columns.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    df = df.dropna().copy()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    if x is None:
        x_candidates = df.columns[df.columns.str.contains("release|debut|time|date", case=False)].to_list()
        if x_candidates:
            x = x_candidates[0]
        else:
            raise ValueError("'Release' or 'Debut' column not found in df. Pass the column name as x.")
    if y is None:
        y_candidates = df.columns.difference([x])
        if y_candidates.empty:
            raise ValueError("No suitable y column found in DataFrame.")
        y = y_candidates[0]
        df[y] = df[y].apply(pd.to_numeric, errors="coerce")

    df[x] = df[x].dt.strftime(group_string)

    sns.boxplot(ax=ax, data=df, y=y, x=x, width=0.5, **kwargs)
    if mean:
        df.groupby(x).mean(numeric_only=True).plot(ax=ax, c="r", ls="--", alpha=0.75, grid=True, legend=False)

    if df[y].max() < 15:  # Level/Rank/Link/Pendulum
        ax.set_yticks(np.arange(0, df[y].max() + 1, 1))
    else:  # ATK/DEF
        ax.set_yticks(np.arange(0, 5500, 500))
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_axisbelow(True)
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def pyramid(
    series: pd.Series,
    use_area: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    grid: bool = False,
    colors: List[str] | None = None,
    alpha: float = 1,
    **kwargs,
) -> Figure:
    """
    Create a pyramid (horizontal stacked bar or area) plot from a pandas Series.

    Args:
        series (pd.Series): Data to plot, with index as labels and values as bar/area sizes.
        use_area (bool, optional): If True, use area to represent values (triangular segments); if False, use bar width. Defaults to False.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 8).
        grid (bool, optional): Whether to show a grid on the x-axis. Defaults to False.
        colors (List[str] | None, optional): List of colors for each segment. Defaults to matplotlib default cycle.
        alpha (float, optional): Transparency of the bars/areas. Defaults to 1.
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        matplotlib.figure.Figure: The generated pyramid plot figure.

    """
    series = series.sort_values(ascending=False)
    n = len(series)
    yticks = []

    if colors is None:
        colors = [f"C{i}" for i in range(n)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    if use_area:
        total_area = series.sum()
        bottom = 2 * np.sqrt(total_area / np.sqrt(3))
        xlim = bottom / 2
        heights = []
        total_height = 0
        for i, (j, area) in enumerate(series.items()):
            height = (bottom - np.sqrt(bottom**2 - 4 * area / np.sqrt(3))) / (2 / np.sqrt(3))
            top = bottom - 2 * height / np.sqrt(3)
            y = [total_height, total_height + height]
            x1 = [-bottom / 2, -top / 2]
            x2 = [bottom / 2, top / 2]
            ax.fill_betweenx(y, x1, x2, alpha=alpha, color=colors[i])
            yticks.append(total_height + height / 2)
            bottom = top
            total_height += height
            heights.append(total_height)
        ax.set_yticks(heights[:-1], minor=True)

    else:
        total_height = n
        xlim = series.max() / 2
        for i, (j, k) in enumerate(series.items()):
            y = [i, i + 1]
            x1 = [-k / 2, -series.iloc[i + 1] / 2 if (i + 1) < n else 0]
            x2 = [k / 2, series.iloc[i + 1] / 2 if (i + 1) < n else 0]
            ax.fill_betweenx(y, x1, x2, alpha=alpha, color=colors[i])
            yticks.append(i + 0.5)
        yticks = np.array(yticks)
        ax.set_yticks(yticks[1:] - 0.5, minor=True)

    ax.xaxis.set_major_locator(MaxNLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(-xlim, xlim)

    ax.set_yticks(yticks)
    ax.set_ylim(0, total_height)
    ax.tick_params(axis="both", which="major", direction="inout", length=10)
    ax.tick_params(axis="both", which="minor", direction="inout", length=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position("center")
    ax.set_yticklabels(series.index)

    ax2 = ax.twinx()
    ax2.set_yticks(yticks)
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(axis="y", which="major", length=10, direction="inout")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_position("center")
    ax2.set_yticklabels(series)

    if grid:
        ax.grid(axis="x", ls=":")

    fig.suptitle(str(series.name) if series.name is not None else "")
    fig.tight_layout()

    return fig


# --- Deck Specific Plots --- #


def deck_composition(
    deck_df: pd.DataFrame,
    grid_cols: int = 3,
    ring_radius: float = 0.3,
    pctdistances: List[float] = [0.85, 0.75],
    scale: float = 1,
    pie_kwargs: dict = {},
    bar_kwargs: dict = {},
    **kwargs,
) -> Figure:
    """
    Create a grid of pie charts displaying the composition of each deck in a DataFrame.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        grid_cols (int, optional): The number of columns in the grid. Defaults to 3.
        ring_radius (float, optional): The radius of the ring in the pie chart. Defaults to 0.3.
        pctdistances (List[float], optional): The distances of the percentage labels from the center of the chart rings. Defaults to [0.85, 0.75].
        scale (float, optional): The scale factor for the plot size. Defaults to 1.
        pie_kwargs (dict, optional): Additional keyword arguments to pass to the pie chart plotting function. Defaults to None.
        bar_kwargs (dict, optional): Additional keyword arguments to pass to the side bar plotting function. Defaults to None.
        **kwargs: Additional keyword arguments such as font sizes.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Deck data preparation
    decks = deck_df["Deck"].unique()
    temp = deck_df.copy()
    temp["Primary type"] = deck_df["Primary type"].fillna(deck_df["Card type"])
    main_df = temp[temp["Section"] == "Main"].groupby(["Deck", "Primary type"])["Count"].sum().unstack(0)
    extra_df = temp[temp["Section"] == "Extra"].groupby(["Deck", "Primary type"])["Count"].sum().unstack(0)
    side_df = temp[temp["Section"] == "Side"].groupby(["Deck", "Primary type"])["Count"].sum().unstack(0)

    # Colors
    MAIN_TYPES = ["Effect Monster", "Normal Monster", "Ritual Monster", "Trap Card", "Spell Card"]
    EXTRA_TYPES = ["Fusion Monster", "Synchro Monster", "Xyz Monster", "Link Monster"]
    colors_all = {type: colors_dict.get(type, f"C{i}") for i, type in enumerate(temp["Primary type"].unique())}
    colors_main = {type: colors_all[type] for type in MAIN_TYPES if type in main_df.index.union(side_df.index)}
    colors_extra = {type: colors_all[type] for type in EXTRA_TYPES if type in extra_df.index.union(side_df.index)}

    # Font sizes
    label_fontsize = kwargs.get("label_fontsize", DEFAULT_FONTSIZES["label"])
    title_fontsize = kwargs.get("title_fontsize", DEFAULT_FONTSIZES["title"])
    suptitle_fontsize = kwargs.get("suptitle_fontsize", DEFAULT_FONTSIZES["suptitle"])
    legend_fontsize = kwargs.get("legend_fontsize", DEFAULT_FONTSIZES["legend"])

    # Plot layout calculations
    plot_width = 5 * scale
    plot_height = 5 * scale
    horizontal_space = max(2 * scale, 0.5)
    vertical_space = max(1 * scale, 0.5)
    header_space = max(scale * legend_fontsize / 5, 2)
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    fig_height = plot_height * rows + (rows - 1) * vertical_space + header_space

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=rows,
        ncols=cols,
        wspace=horizontal_space / plot_width,
        hspace=vertical_space / plot_height,
    )

    for i, deck in enumerate(sorted(decks)):
        sub_gs = gs[(i // cols), i % cols].subgridspec(2, 1, height_ratios=[9, 1], hspace=0.2)
        ax_pie = fig.add_subplot(sub_gs[0, 0])
        colors = list(colors_main[type] for type in main_df[deck].dropna().index)
        _plot_pie(
            ax_pie,
            main_df[deck].dropna(),
            colors,
            _make_autopct(main_df[deck].dropna()),
            ring_radius,
            pctdistances[0],
        )
        if deck in extra_df.columns:
            colors = list(colors_extra[type] for type in extra_df[deck].dropna().index)
            _plot_pie(
                ax=ax_pie,
                data=extra_df[deck].dropna(),
                colors=colors,
                autopct=_make_autopct(extra_df[deck].dropna()),
                ring_radius=ring_radius,
                pctdistance=pctdistances[1],
                radius=1 - ring_radius,
                **pie_kwargs,
            )
        ax_pie.text(
            0,
            0,
            f"Main: {main_df[deck].sum().astype(int)}\nExtra: {extra_df[deck].sum().astype(int)}",
            ha="center",
            va="center",
            fontsize=label_fontsize,
        )
        ax_pie.set_title(deck, fontsize=title_fontsize)
        ax_pie.set_xlim(-1, 1)
        ax_pie.set_ylim(-1, 1)
        ax_pie.set_aspect("equal", adjustable="box")

        # Side bar setup
        ax_bar = fig.add_subplot(sub_gs[1, 0])
        ax_bar.axis("off")
        if deck in side_df and side_df[deck] is not None:
            sorted_side = side_df[deck].sort_values(ascending=True).dropna()
            side_total = sorted_side.sum().astype(int)
            colors = list(colors_all[type] for type in sorted_side.index)
            _plot_side_bar(ax_bar, sorted_side, side_total, colors=colors, **bar_kwargs)
            ax_bar.set_title(f"Side: {side_total}", fontsize=label_fontsize)
            ax_bar.set_xlim(-side_total, 0)
            ax_bar.set_ylim(-0.05, 0.05)
            ax_bar.set_aspect(side_total, adjustable="box")
        else:
            ax_bar.set_title(f"Side: 0", fontsize=label_fontsize)

    # Legend setup
    handles1 = [mpatches.Patch(color=colors_main[type], label=type) for type in colors_main.keys()]
    handles2 = [mpatches.Patch(color=colors_extra[type], label=type) for type in colors_extra.keys()]

    top = 1 - header_space / fig_height
    legend_y = top + 3 * (1 - top) / 5

    fig.subplots_adjust(top=top, bottom=0)

    fig.legend(
        handles=handles1,
        title="Main deck",
        loc="lower center",
        fontsize=legend_fontsize,
        ncol=len(handles1),
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        borderaxespad=0,
        title_fontsize=legend_fontsize + 2,
    )
    fig.legend(
        handles=handles2,
        title="Extra deck",
        loc="upper center",
        fontsize=legend_fontsize,
        ncol=len(handles2),
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        borderaxespad=0,
        title_fontsize=legend_fontsize + 2,
    )

    fig.suptitle("Deck composition", fontsize=suptitle_fontsize, y=1)
    return fig


def _make_autopct(values: List[float] | pd.Series) -> Callable[[float], str]:
    """
    Return a function to format pie chart percentage labels with value counts.

    Args:
        values (iterable): The values represented in the pie chart.

    Returns:
        Callable[..., str]: Formatter function for autopct in matplotlib pie charts.
    """

    def my_autopct(pct) -> str:
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f"{pct:.0f}%\n({val})"

    return my_autopct


def _plot_pie(
    ax: Axes,
    data: List[float] | pd.Series,
    colors: List[str],
    autopct: Callable[[float], str],
    ring_radius: float,
    pctdistance: float,
    startangle: float = 90,
    radius: float = 1,
    counterclock: bool = False,
    **kwargs,
) -> None:
    """
    Plot a ring (donut) pie chart on the given axis with custom label formatting and colors.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        data (iterable): The data values for the pie chart.
        colors (list): Colors for each wedge.
        autopct (callable): Formatter for percentage labels.
        ring_radius (float): Width of the ring.
        pctdistance (float): Distance of percentage labels from center.
        startangle (float, optional): Starting angle for the pie chart. Defaults to 90.
        radius (float, optional): Radius of the pie chart. Defaults to 1.
        counterclock (bool, optional): Plot wedges counterclockwise. Defaults to False.
        **kwargs: Additional keyword arguments to pass to ax.pie().

    Returns:
        None
    """
    pie_result = ax.pie(
        data,
        autopct=autopct,
        startangle=startangle,
        radius=radius,
        wedgeprops=dict(width=ring_radius, edgecolor="w"),
        pctdistance=pctdistance,
        colors=colors,
        counterclock=counterclock,
        **kwargs,
    )
    # ax.pie can return 2 or 3 values depending on autopct
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
    else:
        wedges, texts = pie_result
        autotexts = []
    for wedge, text in zip(wedges, autotexts):
        color = wedge.get_facecolor()[:3]
        text.set_color("black" if _is_light_color(color) else "white")


def _plot_side_bar(
    ax: Axes,
    sorted_side: pd.Series | List[float],
    side_total: int,
    colors: List[str],
    **kwargs,
) -> None:
    """
    Plot a horizontal bar representing the side deck composition with percentage and count labels.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sorted_side (pd.Series): Series mapping of card type to count, sorted.
        side_total (int): Total number of cards in the side deck.
        colors (list): List of colors for each bar segment.
        **kwargs: Additional keyword arguments to pass to ax.barh().
    """
    left = 0
    height = 0.1
    for idx, count in enumerate(sorted_side):
        left -= count
        bc = ax.barh(
            0,
            width=count,
            height=height,
            left=left,
            color=colors[idx],
            edgecolor="white",
            **kwargs,
        )
        color = bc.patches[0].get_facecolor()[:3]
        ax.bar_label(
            bc,
            labels=[f"{count/side_total*100:.0f}%\n({count:.0f})"],
            label_type="center",
            color="black" if _is_light_color(color) else "white",
        )


def deck_distribution(
    deck_df: pd.DataFrame,
    column: str,
    scale: float = 1.0,
    grid_cols: int = 2,
    colors: Dict[str, str] | List[str] | None = None,
    hatches: Dict[str, str] | List[str] | str = "",
    edgecolors: Dict[str, str] | List[str] | str = "white",
    align=False,
    **kwargs,
) -> Figure:
    """
    Create a grid of horizontal bar charts displaying the distribution of a specified column in each deck.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        column (str): The column to be plotted.
        scale (float, optional): The scaling factor for the plot. Defaults to 1.0.
        grid_cols (int, optional): The number of columns in the grid. Defaults to 2.
        colors (Dict[str, str] | List[str] | None, optional): A dictionary of colors for each section, or a list of colors to be used in the plot. If not provided, colors_dict is used. Defaults to None.
        hatches (Dict[str, str] | List[str] | str, optional): A dictionary of hatches for each section, or a list of hatches to be used in the plot. If passed, must be the same length as the number of sections in deck_df or a single string for the entire plot. Defaults to "".
        edgecolors (Dict[str, str] | List[str] | str, optional): The colors of the edges of the bars and hatches.  If passed, must be the same length as the number of sections in deck_df or a single string for the entire plot. Defaults to "white".
        **kwargs: Extra keyword arguments such as font sizes. Passed to the bar plotting function.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    title_fontsize = kwargs.pop("title_fontsize", DEFAULT_FONTSIZES["title"])
    label_fontsize = kwargs.pop("label_fontsize", DEFAULT_FONTSIZES["label"])
    legend_fontsize = kwargs.pop("legend_fontsize", DEFAULT_FONTSIZES["legend"])

    section_counts = deck_df[deck_df[column].notna()].groupby("Section")["Count"].sum()
    sorted_sections = _sort_sections(section_counts)

    section_colors = _make_section_colors(
        colors=colors,
        sections=sorted_sections,
        index=sorted(deck_df[column].dropna().unique()),
    )

    hatches_series = pd.Series(
        (
            [hatches.get(section, "") for section in sorted_sections]
            if isinstance(hatches, dict)
            else hatches[: len(sorted_sections)] if isinstance(hatches, list) else hatches
        ),
        index=sorted_sections,
    )
    edgecolors_series = pd.Series(
        (
            [edgecolors.get(section, "white") for section in sorted_sections]
            if isinstance(edgecolors, dict)
            else edgecolors[: len(sorted_sections)] if isinstance(edgecolors, list) else edgecolors
        ),
        index=sorted_sections,
    )

    # Base values
    plot_width = 6 * scale
    bar_height = 0.45 * scale
    horizontal_space = 3 * scale
    left_margin = 0.07
    right_margin = 0.97
    subplot_margin = 0.22 * scale
    row_gap = max(1.2 * scale, 0.8)
    top_margin_in = max(1.2 * scale, 0.8)

    # Dynamically increase horizontal_space for long labels
    label_texts = [str(x) for x in deck_df[column].dropna().unique()]
    max_label_len = max([len(x) for x in label_texts])
    mean_labels = deck_df.groupby("Deck")[column].nunique()
    mean_labels = mean_labels[mean_labels > 0].mean()
    extra_hspace = 0
    if max_label_len > 10:
        # 0.45 is a base value, scale it linearly
        extra_hspace = 0.45 * scale * (max_label_len - 10)

    horizontal_space = max(horizontal_space + extra_hspace, 1.6)

    # Calculate number of columns and rows
    decks = sorted(deck_df[deck_df[column].notna()]["Deck"].unique())
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    # Manual axes placement for per-subplot heights
    axes = []
    subplot_heights = []
    temp_dfs = []
    for deck in decks:
        temp_df = deck_df[deck_df["Deck"] == deck].groupby(["Section", column])["Count"].sum().unstack(0)
        temp_dfs.append(temp_df)
        num_bars = len(temp_df) if temp_df is not None else 0
        subplot_heights.append(bar_height * max(num_bars, 1) + subplot_margin)  # margin for title/xlabel

    # Compute total height in inches for each column
    col_heights = [
        sum(subplot_heights[i] for i in range(j, len(decks), cols))
        + row_gap * (len([i for i in range(j, len(decks), cols)]) - 1)
        for j in range(cols)
    ]
    max_col_height = max(col_heights)
    fig_height = max_col_height + top_margin_in
    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    col_width = plot_width / fig_width
    col_gap = (horizontal_space * 0.25) / fig_width  # Keep horizontal gap as before

    # Precompute row heights if align is enabled
    if align:
        row_heights = []
        for row in range(rows):
            row_idxs = [i for i in range(len(decks)) if i // cols == row]
            row_heights.append(max([subplot_heights[i] for i in row_idxs]))
        row_ypos = []
        ypos = top_margin_in
        for h in row_heights:
            row_ypos.append(ypos)
            ypos += h + row_gap
    else:
        col_ypos = [top_margin_in for _ in range(cols)]

    fig = plt.figure(figsize=(fig_width, fig_height))
    for idx, (deck, temp_df) in enumerate(zip(decks, temp_dfs)):
        row = idx // cols
        col = idx % cols
        left = col * (col_width + col_gap) + left_margin
        width = col_width * (right_margin - left_margin)
        height = subplot_heights[idx] / fig_height
        if align:
            bottom = 1 - (row_ypos[row] + subplot_heights[idx]) / fig_height
        else:
            bottom = 1 - (col_ypos[col] + subplot_heights[idx]) / fig_height
        ax = fig.add_axes((float(left), float(bottom), float(width), float(height)))
        if not temp_df.empty:
            temp_df = temp_df[_sort_sections(temp_df.sum())]
            plot_colors = {
                section: [section_colors[section][idx] for idx in temp_df[section].index] for section in temp_df.columns
            }
            _plot_distribution_bar(
                ax,
                temp_df,
                plot_colors,
                hatches_series,
                edgecolors_series,
                **kwargs,
            )
        ax.set_title(deck, fontsize=label_fontsize)
        axes.append(ax)
        if not align:
            col_ypos[col] += subplot_heights[idx] + row_gap

    # Dynamically center the legend above the axes area, just below the title
    fig.canvas.draw()  # Ensure positions are up to date
    axes_pos = [ax.get_position() for ax in fig.axes if hasattr(ax, "get_position")]
    if axes_pos:
        lefts = [pos.x0 for pos in axes_pos]
        rights = [pos.x1 for pos in axes_pos]
        axes_left = min(lefts)
        axes_right = max(rights)
        axes_center = (axes_left + axes_right) / 2
        legend_x = axes_center
    else:
        legend_x = 0.5
    legend_y = 1 - (top_margin_in / fig_height) / 2

    handler = {}
    for section in sorted_sections:
        # Build color list for legend using all unique indices
        color_dict = section_colors[section]
        color_list = list({color_dict[idx] for idx in color_dict})  # unique colors only
        handler[mpatches.Patch(label=section)] = MulticolorPatchHandler(
            color_list, hatches_series[section], edgecolor=edgecolors_series[section]
        )

    # Add legend with a fixed position
    fig.legend(
        handles=list(handler.keys()),
        handler_map=handler,
        loc="center",
        fontsize=legend_fontsize,
        ncol=3,
        bbox_to_anchor=(legend_x, legend_y),
        borderaxespad=0.5,
        frameon=False,
        handlelength=3,
        handleheight=1.2,
    )
    # Dynamically center the suptitle above the axes area, not just the figure
    fig.suptitle(f"{column} distribution", fontsize=title_fontsize, y=1, x=legend_x, ha="center")

    return fig


def _plot_distribution_bar(
    ax: Axes,
    deck_df: pd.DataFrame,
    section_colors: Dict[str, Any],
    hatches_series: pd.Series,
    edgecolors_series: pd.Series,
    **kwargs,
) -> None:
    """
    Plot a horizontal stacked bar chart for a single deck's distribution.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        deck_df (pd.DataFrame): DataFrame with counts for each section/label.
        section_colors (Dict[str, Any]): Colors for each section/bar.
        hatches_series (pd.Series): Hatches for each section/bar.
        edgecolors_series (pd.Series): Edge colors for each section/bar.
        **kwargs: Additional keyword arguments for font sizes, etc.

    Returns:
        None
    """

    num_bars = len(deck_df)
    label_fontsize = kwargs.pop("label_fontsize", DEFAULT_FONTSIZES["label"])
    tick_fontsize = kwargs.pop("tick_fontsize", DEFAULT_FONTSIZES["tick"])

    bar_ax = deck_df.plot.barh(
        ax=ax,
        stacked=True,
        legend=False,
        fontsize=label_fontsize,
        color=section_colors,
        **kwargs,
    )
    for j, bar in enumerate(bar_ax.patches):
        hatch_index = j // (len(bar_ax.patches) // len(deck_df.columns))
        bar.set_hatch(hatches_series.iloc[hatch_index])
        bar.set_edgecolor(edgecolors_series.iloc[hatch_index])

    ax.set_ylabel("")
    ax.set_xlabel("Count", fontsize=label_fontsize)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", which="major", linestyle=":")
    ax.set_ylim(-0.5, num_bars - 0.5)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(MultipleLocator(1))


def deck_stem(
    deck_df: pd.DataFrame,
    columns: str | List[str],
    scale: float = 1.0,
    grid_cols: int = 2,
    colors: Dict[str, str] | List[str] | None = None,
    markers: Dict[str, str] | List[str] = ["s", "o", "+"],
    hollow: bool = False,
    marker_size: int = 10,
    align_zero: bool = False,
    **kwargs,
) -> Figure:
    """
    Create a grid of stem plots displaying the distribution of a specified column in each deck.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        columns (str | List[str]): The column(s) to be plotted. If a single string is passed, it will be converted to a list with one element. Must be one or two columns.
        scale (float, optional): The scaling factor for the plot size. Defaults to 1.0.
        grid_cols (int, optional): The number of columns in the grid. Defaults to 2.
        colors (Dict[str, str] | List[str] | None, optional): A dictionary of colors for each section, or a list of colors to be used in the plot. If not provided, colors_dict is used. Defaults to None.
        markers (Dict[str, str], optional): A dictionary mapping section names to marker styles. Defaults to {"Section1": "s", "Section2": "o", "Section3": "+"}.
        hollow (bool, optional): Whether to make the markers hollow. Defaults to False.
        marker_size (int, optional): The initial size of the markers. Defaults to 10.
        align_zero (bool, optional): Whether to align the zero point of the y-axis across all subplots. Defaults to False.
        **kwargs: Additional keyword arguments such as font sizes. Passed to the stem plotting function.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    columns = [columns] if isinstance(columns, str) else columns
    if len(columns) > 2:
        raise ValueError("deck_stem only supports up to two columns (got %d)" % len(columns))
    decks = deck_df["Deck"].unique()
    section_counts = deck_df[deck_df[columns].notna().any(axis=1)].groupby("Section")["Count"].sum()
    sorted_sections = _sort_sections(section_counts)

    title_fontsize = kwargs.pop("title_fontsize", DEFAULT_FONTSIZES["title"])
    legend_fontsize = kwargs.pop("legend_fontsize", DEFAULT_FONTSIZES["legend"])
    label_fontsize = kwargs.pop("label_fontsize", DEFAULT_FONTSIZES["label"])

    # Set constants for plot sizes and spacing
    plot_width = 8 * scale
    plot_height = 2 * len(columns) * scale
    horizontal_space = 2 * scale
    vertical_space = max(1 * scale, 0.8)
    header_space = max(legend_fontsize / 10 * scale, 1)

    # Calculate number of columns and rows
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    # Dynamically calculate the figure size based on the number of rows and columns
    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    fig_height = plot_height * rows + (rows - 1) * vertical_space + header_space

    section_colors = _make_section_colors(
        colors=colors,
        sections=sorted_sections,
        index=columns,
        fallback_override={"Side": "#000000"},
        index_fallback=False,
    )
    if not isinstance(markers, dict):
        markers = {section: markers[i % len(markers)] for i, section in enumerate(sorted_sections)}

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=rows,
        ncols=cols,
        wspace=horizontal_space / plot_width,  # Adjusted for figure width
        hspace=vertical_space / plot_height,  # Adjusted for plot height
    )

    for i, deck in enumerate(decks):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        sub_df = deck_df[deck_df["Deck"] == deck]
        if sub_df.empty:
            continue
        _plot_stem_subplot(
            ax=ax,
            deck_df=sub_df,
            columns=columns,
            colors=section_colors,
            markers=markers,
            hollow=hollow,
            marker_size=marker_size,
            align_zero=align_zero,
            **kwargs,
        )
        ax.set_title(deck, fontsize=label_fontsize)

    top = 1 - header_space / fig_height
    legend_y = top + (1 - top) / 2
    fig.subplots_adjust(
        top=top,
        bottom=0,
    )

    handles = None
    ncols = None
    stat_colors = [
        [section_colors[s][columns[0]] for s in sorted_sections],
    ]
    if len(columns) == 2:
        stat_colors.append([section_colors[s][columns[1]] for s in sorted_sections])
        stat_labels = [str(c) for c in columns]
        if not all(a == b for a, b in zip(stat_colors[0], stat_colors[1])):
            # Stat label (no marker), then section markers in correct order, all in a single row
            stat1_handle = Line2D([], [], color="none", marker="", linestyle="None", label=f"{stat_labels[0]}:")
            stat2_handle = Line2D([], [], color="none", marker="", linestyle="None", label=f"{stat_labels[1]}:")
            handles = [stat1_handle, stat2_handle]
            for i, s in enumerate(sorted_sections):
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=markers[s] if isinstance(markers, dict) and s in markers else "o",
                        color=stat_colors[0][i],
                        linestyle="None",
                        markersize=marker_size,
                        label=s,
                    )
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=markers[s] if isinstance(markers, dict) and s in markers else "o",
                        color=stat_colors[1][i],
                        linestyle="None",
                        markersize=marker_size,
                        label=s,
                    )
                )
                ncols = len(handles) // 2

    if not handles:
        # Only one line needed
        handles = [
            Line2D(
                [0],
                [0],
                marker=markers[s] if isinstance(markers, dict) and s in markers else "o",
                color=stat_colors[0][i],
                label=s,
                linestyle="None",
                markersize=marker_size,
            )
            for i, s in enumerate(sorted_sections)
        ]

    fig.legend(
        handles=handles,
        ncol=len(handles),
        loc="center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize=legend_fontsize,
        frameon=False,
        ncols=ncols if ncols else len(handles),
        handletextpad=1.0,
        columnspacing=1.5,
    )

    fig.suptitle(f"{' & '.join(columns)} distribution", fontsize=title_fontsize, y=1)
    return fig


def _plot_stem_subplot(
    ax: Axes,
    deck_df: pd.DataFrame,
    columns: List[str],
    colors: Dict[str, Dict[str, Any]],
    markers: Dict[str, str],
    hollow: bool,
    marker_size: int,
    align_zero: bool = False,
    **kwargs,
) -> None:
    """
    Plot a stem plot for a single deck on the provided axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        deck_df (pd.DataFrame): DataFrame for the current deck.
        columns (List[str]): List of columns to plot (e.g., ["ATK", "DEF"]).
        colors (Dict[str, Dict[str, Any]]): Mapping from section name and column name to color.
        markers (Dict[str, str]): Mapping from section name to marker style.
        hollow (bool): Whether to use hollow markers.
        marker_size (int): Initial marker size.
        align_zero (bool): Whether to align the y-axis limits to center at 0.
        **kwargs: Additional keyword arguments for font sizes, etc.

    Returns:
        None
    """
    label_fontsize = kwargs.pop("label_fontsize", DEFAULT_FONTSIZES["label"])
    tick_fontsize = kwargs.pop("tick_fontsize", DEFAULT_FONTSIZES["tick"])
    msize = marker_size
    max_idx = 0
    min_idx = np.inf
    hasna = False

    section_counts = deck_df.groupby("Section")["Count"].sum()
    sorted_sections = _sort_sections(section_counts)
    steps = (100, 500) if deck_df[columns].map(pd.to_numeric, errors="coerce").diff().max().max() > 12 else (1, 1)
    for k, col in enumerate(columns):
        msize = marker_size
        for j, s in enumerate(sorted_sections):
            sub_sub_df = deck_df[deck_df["Section"] == s]
            if sub_sub_df.empty:
                continue
            series = sub_sub_df.groupby(col)["Count"].sum().mul(np.power(-1, k))
            if series.empty:
                continue
            index = pd.to_numeric(series.index.to_series(), errors="coerce")
            if not index.isna().all():
                max_idx = max(int(index.max()), max_idx)
                min_idx = min(int(index.min()), min_idx)
            if index.isna().any():
                hasna = True
            series.index = index.fillna(max_idx + steps[1])
            color = colors.get(s, {}).get(col, "C0")
            marker = markers.get(s, "o")
            stem = ax.stem(
                series.index,
                series,
                linefmt=":",
                markerfmt=marker,
                basefmt=":",
                **kwargs,
            )
            if hollow:
                stem.markerline.set_markeredgecolor(color)
                stem.markerline.set_markerfacecolor("none")
            else:
                stem.markerline.set_color(color)
            stem.stemlines.set_color(color)
            stem.baseline.set_color(color)
            stem.markerline.set_markersize(msize)
            msize = max(msize - 2, 2)
    if steps[1] < 10:
        xticks = np.arange(0, 14, 1)
    else:
        xticks = np.arange(int(min_idx / steps[1]) * steps[1], max_idx + steps[1], steps[1])
    xticks = xticks.tolist()
    minor_xticks = np.arange(0, (len(xticks) - 1) * steps[1], steps[0]).tolist()
    if hasna:
        xticks = xticks + [max_idx + steps[1]]
    xticks_labels = [str(x) for x in (xticks[:-1] + ["?"] if hasna else xticks)]
    ax.set_xticks(ticks=xticks, labels=xticks_labels, rotation=45 * (xticks[-1] > 100))
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xlim(-min(steps[0], steps[1] / 2) + min(xticks), max(xticks) + min(steps[0], steps[1] / 2))
    plim = int(ax.get_ylim()[1] + 1)
    nlim = int(ax.get_ylim()[0] - 1) if ax.get_ylim()[0] < -1 else 0
    if align_zero:
        plim = max(plim, abs(nlim))
        nlim = -plim
    ax.set_ylim(nlim, plim)
    if nlim < 0:
        ax.set_ylabel("← " + " | ".join(reversed(columns)) + " →", fontsize=label_fontsize)
    else:
        ax.set_ylabel(columns[0], fontsize=label_fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(abs(x)))))
    ax.xaxis.set_minor_locator(MultipleLocator(steps[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(ls=":", axis="y")
    ax.set_axisbelow(True)


def _make_section_colors(
    colors: dict | list | str | None = None,
    index: list | pd.Index = [0],
    sections: list = ["Main", "Extra", "Side"],
    fallback_override: Dict[str, str] = {},
    index_fallback: bool = True,
) -> dict:
    """
    Generate a mapping from section names to color(s) for plotting.
    Always returns {section: {index: color}}.

    Args:
        colors (dict | list | str | None): User color input. Accepts dict, list, str, or None.
        index (list | pd.Index): List of row values or columns.
        sections (list): List of section names.
        fallback_override (dict, optional): Dictionary mapping section names to override color keys in internal fallback colors.
        index_fallback (bool): Whether to prefer index-based fallback colors over section-based ones.
    Returns:
        dict: {section: {index: color}}
    """
    fallback_colors = {"Main": "#FF8B53", "Extra": "#A086B7", "Side": "#C0C0C0"}
    fallback_colors.update(fallback_override)
    section_colors = {}
    color_repeat_count = {idx: {} for idx in index}

    for i, section in enumerate(sections):
        section_colors[section] = {}
        for j, idx in enumerate(index):
            color = None
            # {section: {index: color}}
            if (
                isinstance(colors, dict)
                and section in colors
                and isinstance(colors[section], dict)
                and idx in colors[section]
            ):
                color = colors[section][idx]
            # {index: {section: color}}
            elif isinstance(colors, dict) and idx in colors and isinstance(colors[idx], dict) and section in colors[idx]:
                color = colors[idx][section]
            # {section: color}
            elif isinstance(colors, dict) and section in colors and not isinstance(colors[section], dict):
                color = colors[section]
            # {index: color}
            elif isinstance(colors, dict) and idx in colors and not isinstance(colors[idx], dict):
                color = colors[idx]
            # {section: list}
            elif isinstance(colors, dict) and section in colors and isinstance(colors[section], list):
                if len(colors[section]) == len(index):
                    color = colors[section][j]
                else:
                    color = colors[section][j % len(colors[section])]
            # {index: list}
            elif isinstance(colors, dict) and idx in colors and isinstance(colors[idx], list):
                if len(colors[idx]) == len(sections):
                    color = colors[idx][i]
                else:
                    color = colors[idx][i % len(colors[idx])]
            # list: if matches sections, treat as per-section; if matches index, treat as per-index; else cycle
            elif isinstance(colors, list):
                if len(colors) == len(sections):
                    color = colors[i]
                elif len(colors) == len(index):
                    color = colors[j]
                else:
                    color = colors[j % len(colors)]
            # string: single color
            elif isinstance(colors, str):
                color = colors
            # fallback: prefer colors_dict[section] if available, else default color
            if color is None:
                if index_fallback:
                    color = colors_dict.get(idx)
                else:
                    color = colors_dict.get(section)

                color = color or fallback_colors.get(section, f"C{i}")

            # Cumulative lightness adjustment for repeated colors in the same row
            count = color_repeat_count[idx].get(color, 0)
            adjusted_color = color
            if count > 0:
                adjusted_color = _adjust_lightness(adjusted_color, amount=1 + 0.25 * count)
            color_repeat_count[idx][color] = count + 1
            section_colors[section][idx] = adjusted_color

    return section_colors


def _sort_sections(section_counts, section_order=["Main", "Extra", "Side"]):
    """
    Sort sections by fixed priority order first, then by descending size for extras.
    Args:
        section_counts (pd.Series): Series mapping section name to total count.
        section_order (list): List of section names to prioritize.
    Returns:
        list: Sorted section names.
    """
    available_sections = section_counts.index.tolist()
    sorted_sections = [s for s in section_order if s in available_sections]
    extra_sections = [s for s in available_sections if s not in section_order]
    extra_sections_sorted = sorted(extra_sections, key=lambda s: section_counts[s], reverse=True)
    return sorted_sections + extra_sections_sorted
