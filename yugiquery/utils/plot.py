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
from matplotlib_venn import venn2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# --- Imports: Local Application --- #
from .helpers import *
from .dirs import dirs


# --- Matplotlib Settings & Overrides --- #
plt.style.use("default")  # TODO: Make this configurable
if dirs.is_notebook:
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")  # Needed for dynanmic theme


# --- Plot Dictionaries --- #
colors_dict = load_json(dirs.get_asset("json", "colors.json"))
# TODO: Adapt colors to style
DEFAULT_FONT_SIZES = {"label": 14, "title": 20, "tick": 12, "legend": 12, "suptitle": 20}


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

    # Map the arrows to angles
    angle_map = {
        "→": 0,
        "↗": np.pi / 4,
        "↑": np.pi / 2,
        "↖": 3 * np.pi / 4,
        "←": np.pi,
        "↙": 5 * np.pi / 4,
        "↓": 3 * np.pi / 2,
        "↘": 7 * np.pi / 4,
    }
    if not all(a in angle_map for a in counts.index):
        raise ValueError("Input Series contains invalid arrow symbols.")
    angles = counts.index.map(angle_map)

    # Create a polar plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(polar=True)
    ax.bar(x=angles, height=counts, width=0.5, color=colors_dict["Link Monster"], **kwargs)

    # Set the label for each arrow
    ax.set_xticks(list(angle_map.values()))
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
    grid_spacing: Tuple[int, int] = (2, 1),
    grid_cols: int = 3,
    plot_size: Tuple[int, int] = (5, 5),
    ring_radius: float = 0.3,
    pctdistances: List[float] = [0.85, 0.75],
    **kwargs,
) -> Figure:
    """
    Create a grid of pie charts displaying the composition of each deck in a DataFrame.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        grid_spacing (Tuple[int, int], optional): The horizontal and vertical spacing between plots. Defaults to (2, 1).
        grid_cols (int, optional): The number of columns in the grid. Defaults to 3.
        plot_size (Tuple[int, int], optional): The width and height of each plot. Defaults to (5, 5).
        ring_radius (float, optional): The radius of the ring in the pie chart. Defaults to 0.3.
        pctdistances (List[float], optional): The distances of the percentage labels from the center of the chart rings. Defaults to [0.85, 0.75].
        **kwargs: Not implemented.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    decks = deck_df["Deck"].unique()
    temp = deck_df.copy()
    temp["Primary type"] = deck_df["Primary type"].fillna(deck_df["Card type"])
    main_df = temp[temp["Section"] == "Main"].groupby(["Deck", "Primary type"])["Count"].sum().unstack(0)
    extra_df = temp[temp["Section"] == "Extra"].groupby(["Deck", "Primary type"])["Count"].sum().unstack(0)
    side_df = temp[temp["Section"] == "Side"].groupby(["Deck", "Primary type"])["Count"].sum().unstack(0)

    label_font_size = kwargs.get("label_font_size", DEFAULT_FONT_SIZES["label"])
    title_font_size = kwargs.get("title_font_size", DEFAULT_FONT_SIZES["title"])
    suptitle_font_size = kwargs.get("suptitle_font_size", DEFAULT_FONT_SIZES["suptitle"])
    legend_font_size = kwargs.get("legend_font_size", DEFAULT_FONT_SIZES["legend"])

    plot_width = plot_size[0]
    plot_height = plot_size[1]
    horizontal_space = grid_spacing[0]
    vertical_space = grid_spacing[1]
    header_space = (2 * legend_font_size + 2) / 10
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    colors_main = [colors_dict[type] for type in main_df.index]
    colors_extra = [colors_dict[type] for type in extra_df.index]
    colors_remaining = side_df.index.difference(main_df.index.union(extra_df.index))

    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    fig_height = plot_height * rows + (rows - 1) * vertical_space + header_space

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=rows,
        ncols=cols,
        wspace=horizontal_space / plot_width,
        hspace=vertical_space / plot_height,
    )

    for i, deck in enumerate(decks):
        sub_gs = gs[(i // cols), i % cols].subgridspec(2, 1, height_ratios=[9, 1], hspace=0.2)
        ax_pie = fig.add_subplot(sub_gs[0, 0])
        wedges1, texts1, autotexts1 = _plot_pie(
            ax_pie,
            main_df[deck].dropna(),
            np.array(colors_main)[main_df[deck].notna()].tolist(),
            _make_autopct(main_df[deck].dropna()),
            ring_radius,
            pctdistances[0],
        )
        if deck in extra_df.columns:
            wedges2, texts2, autotexts2 = _plot_pie(
                ax=ax_pie,
                data=extra_df[deck].dropna(),
                colors=np.array(colors_extra)[extra_df[deck].notna()].tolist(),
                autopct=_make_autopct(extra_df[deck].dropna()),
                ring_radius=ring_radius,
                pctdistance=pctdistances[1],
                radius=1 - ring_radius,
            )
        ax_pie.text(
            0,
            0,
            f"Main: {main_df[deck].sum().astype(int)}\nExtra: {extra_df[deck].sum().astype(int)}",
            ha="center",
            va="center",
            fontsize=label_font_size,
        )
        ax_pie.set_title(deck, fontsize=title_font_size)
        ax_pie.set_xlim(-1, 1)
        ax_pie.set_ylim(-1, 1)
        ax_pie.set_aspect("equal", adjustable="box")

        ax_bar = fig.add_subplot(sub_gs[1, 0])
        ax_bar.axis("off")
        if deck in side_df and side_df[deck] is not None:
            sorted_side = side_df[deck].sort_values(ascending=True).dropna()
            side_total = sorted_side.sum().astype(int)
            _plot_side_bar(ax_bar, sorted_side, side_total, **kwargs)
            ax_bar.set_title(f"Side: {side_total}", fontsize=label_font_size)
            ax_bar.set_xlim(-side_total, 0)
            ax_bar.set_ylim(-0.05, 0.05)
            ax_bar.set_aspect(side_total, adjustable="box")
        else:
            ax_bar.set_title(f"Side: 0", fontsize=label_font_size)

    colors_main += [
        colors_dict[type]
        for type in colors_remaining
        if type not in ["Fusion Monster", "Synchro Monster", "Xyz Monster", "Link Monster"]
    ]
    colors_main += [
        colors_dict[type]
        for type in colors_remaining
        if type in ["Fusion Monster", "Synchro Monster", "Xyz Monster", "Link Monster"]
    ]
    handles1 = [mpatches.Patch(color=colors_dict[type], label=type) for type in main_df.index]
    handles2 = [mpatches.Patch(color=colors_dict[type], label=type) for type in extra_df.index]

    top = 1 - header_space / fig_height
    legend_y = top + 3 * (1 - top) / 5

    fig.subplots_adjust(top=top, bottom=0)

    fig.legend(
        handles=handles1,
        title="Main deck",
        loc="lower center",
        fontsize=legend_font_size,
        ncol=len(handles1),
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        borderaxespad=0,
        title_fontsize=legend_font_size + 2,
    )
    fig.legend(
        handles=handles2,
        title="Extra deck",
        loc="upper center",
        fontsize=legend_font_size,
        ncol=len(handles2),
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
        borderaxespad=0,
        title_fontsize=legend_font_size + 2,
    )

    fig.suptitle("Deck composition", fontsize=suptitle_font_size, y=1)
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
) -> tuple:
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

    Returns:
        tuple: (wedges, texts, autotexts) from matplotlib pie chart.
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
    return wedges, texts, autotexts


def _plot_side_bar(
    ax: Axes,
    sorted_side: pd.Series | dict,
    side_total: int,
    **kwargs,
) -> None:
    """
    Plot a horizontal bar representing the side deck composition with percentage and count labels.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sorted_side (dict): Mapping of card type to count, sorted.
        side_total (int): Total number of cards in the side deck.
        **kwargs: Additional keyword arguments (unused).
    """
    left = 0
    height = 0.1
    for name, count in sorted_side.items():
        left -= count
        color = colors_dict[name]
        bc = ax.barh(
            0,
            width=count,
            height=height,
            left=left,
            color=color,
            edgecolor="white",
        )
        ax.bar_label(
            bc,
            labels=[f"{count/side_total*100:.0f}%\n({count:.0f})"],
            label_type="center",
            color="black" if _is_light_color(color) else "white",
        )


def deck_distribution(
    deck_df: pd.DataFrame,
    column: str,
    grid_spacing: Tuple[int, int] = (3, 1),
    grid_cols: int = 2,
    plot_size: Tuple[int, int] | None = None,
    colors: Dict[str, str] | List[str] | None = None,
    hatches: List[str] | str = "",
    edgecolors: List[str] | str = "white",
    **kwargs,
) -> Figure:
    """
    Create a grid of horizontal bar charts displaying the distribution of a specified column in each deck.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        column (str): The column to be plotted.
        grid_spacing (Tuple[int, int], optional): The horizontal and vertical spacing between plots. Defaults to (3, 1).
        grid_cols (int, optional): The number of columns in the grid. Defaults to 2.
        plot_size (Tuple[int, int], optional): The width and height of each plot. If None, is calculated to fit all labels. Defaults to None.
        colors (Dict[str, str] | List[str] | None, optional): A dictionary of colors for each section, or a list of colors to be used in the plot. If not provided, colors_dict is used. Defaults to None.
        hatches (List[str] | str, optional): A list of hatches to be used in the plot. If passed, must be the same length as the number of sections in deck_df or a single string for the entire plot. Defaults to "".
        edgecolors (List[str] | str, optional): The colors of the edges of the bars and hatches.  If passed, must be the same length as the number of sections in deck_df or a single string for the entire plot. Defaults to "white".
        font_size (Dict[str,int], optional): The dictionary of font sizes to override defaults for the labels, title, suptitle, and legend. Defaults to {"label": 14, "title": 16, "suptitle": 20, "legend": 12}.
        **kwargs: Not implemented.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    decks = deck_df[deck_df[column].notna()]["Deck"].unique()
    max_label_len = max([len(x) for x in deck_df[column].dropna().unique()])
    mean_labels = deck_df.groupby("Deck")[column].nunique()
    mean_labels = mean_labels[mean_labels > 0].mean()

    title_font_size = kwargs.pop("title_font_size", DEFAULT_FONT_SIZES["title"])
    label_font_size = kwargs.pop("label_font_size", DEFAULT_FONT_SIZES["label"])
    legend_font_size = kwargs.pop("legend_font_size", DEFAULT_FONT_SIZES["legend"])

    # Set constants for plot sizes and spacing
    plot_width = 6 if plot_size is None else plot_size[0]  # Width of each plot
    # Fixed height for each plot
    plot_height = max(mean_labels / 2, 0.5) if plot_size is None else plot_size[1]
    # Fixed horizontal space between plots
    horizontal_space = grid_spacing[0] + max(2 * int(max_label_len / 10) - 3, 0)
    vertical_space = grid_spacing[1]  # Fixed vertical space between plots
    # Fixed space between figure top and subplots
    header_space = legend_font_size / 10

    # Calculate number of columns and rows
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    # Dynamically calculate the figure size based on the number of rows and columns
    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    fig_height = plot_height * rows + (rows - 1) * vertical_space + header_space

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=rows,
        ncols=cols,
        wspace=horizontal_space / plot_width,
        hspace=vertical_space / plot_height,
    )

    section_colors = _make_section_colors(colors=colors, index=sorted(deck_df[column].dropna().unique()))

    sorted_sections = (
        deck_df[deck_df[column].notna()].groupby("Section")["Count"].sum().sort_values(ascending=False).index.tolist()
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

    # Plotting each deck's data
    for i, deck in enumerate(decks):
        temp_df = deck_df[deck_df["Deck"] == deck].groupby(["Section", column])["Count"].sum().unstack(0)
        temp_df = temp_df[temp_df.sum().sort_values(ascending=False).index]
        if not temp_df.empty:
            ax = fig.add_subplot(gs[i // cols, i % cols])
            plot_colors = (
                {
                    section: (
                        section_colors[section].loc[temp_df[section].index]
                        if len(section_colors[section]) > 1
                        else section_colors[section]
                    )
                    for section in temp_df.columns
                }
                if colors is not None
                else section_colors
            )
            _plot_distribution_bar(
                ax,
                temp_df,
                plot_colors,
                hatches_series,
                edgecolors_series,
                **kwargs,
            )
            ax.set_title(deck, fontsize=label_font_size)

    # Adjust margins and add suptitle
    top = 1 - header_space / fig_height
    legend_y = top + (1 - top) / 2

    fig.subplots_adjust(
        top=top,
        bottom=0,
    )

    handler = {}
    for section in sorted_sections:
        color = section_colors[section]
        if isinstance(color, str):
            color = [color]
        color = list(color)
        handler[mpatches.Patch(label=section)] = MulticolorPatchHandler(
            color, hatches_series[section], edgecolor=edgecolors_series[section]
        )

    # Add legend with a fixed position
    fig.legend(
        handles=list(handler.keys()),
        handler_map=handler,
        loc="center",
        fontsize=legend_font_size,
        ncol=3,
        bbox_to_anchor=(
            0.5,
            legend_y,
        ),
        borderaxespad=0.5,
        frameon=False,
        handlelength=3,
        handleheight=1.2,
    )
    fig.suptitle(f"{column} distribution", fontsize=title_font_size, y=1)

    return fig


def _make_section_colors(
    colors: Dict[str, str] | List[str] | None = None, index: List[Any] | pd.Index[Any] = [0]
) -> Dict[str, Any] | Dict[Any, pd.Series[Any]]:
    """
    Generate a mapping from section names to color(s) for plotting.

    Args:
        colors (Dict[str, str] | List[str] | None, optional):
            - If None, uses default color mapping from colors_dict and DEFAULT_COLORS.
            - If dict, maps section names to color(s).
            - If list, uses the list for all sections.
        index (List[Any] | pd.Index[Any], optional):
            Index to use for the pd.Series if colors is not None. Defaults to [0].

    Returns:
        dict: Mapping from section name to color string or pd.Series of colors.
    """
    sections = ["Main", "Extra", "Side"]
    DEFAULT_COLORS = ["Effect Monster", "Fusion Monster", "Counter"]
    section_colors = {}

    # For each row, track how many times a color has been used so far
    color_repeat_count = {idx: {} for idx in index}
    color_last_adjusted = {idx: {} for idx in index}

    for i, section in enumerate(sections):
        color_series = []
        for j, idx in enumerate(index):
            color = None
            # 1. {section: {row: color}}
            if (
                isinstance(colors, dict)
                and section in colors
                and isinstance(colors[section], dict)
                and idx in colors[section]
            ):
                color = colors[section][idx]
            # 2. {row: {section: color}}
            elif isinstance(colors, dict) and idx in colors and isinstance(colors[idx], dict) and section in colors[idx]:
                color = colors[idx][section]  # pyright: ignore[reportArgumentType]
            # 3. {section: color}
            elif isinstance(colors, dict) and section in colors and not isinstance(colors[section], dict):
                color = colors[section]
            # 4. {row: color}
            elif isinstance(colors, dict) and idx in colors and not isinstance(colors[idx], dict):
                color = colors[idx]
            # 5. list (cycled)
            elif isinstance(colors, list):
                color = colors[j % len(colors)]
            # 6. string (single color)
            elif isinstance(colors, str):
                color = colors
            # 7. fallback (default color)
            if color is None:
                color = colors_dict.get(DEFAULT_COLORS[i], f"C{i}")

            # Cumulative lightness adjustment for repeated colors in the same row
            count = color_repeat_count[idx].get(color, 0)
            adjusted_color = color
            if count > 0:
                adjusted_color = _adjust_lightness(adjusted_color, amount=1 + 0.25 * count)
            print(section, idx, adjusted_color)
            color_repeat_count[idx][color] = count + 1
            color_series.append(adjusted_color)
        section_colors[section] = pd.Series(color_series, index=index)

    return section_colors


def _plot_distribution_bar(
    ax: Axes,
    temp_df: pd.DataFrame,
    section_colors: Dict[str, Any],
    hatches_series: pd.Series,
    edgecolors_series: pd.Series,
    **kwargs,
) -> None:
    """
    Plot a horizontal stacked bar chart for a single deck's distribution.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        temp_df (pd.DataFrame): DataFrame with counts for each section/label.
        section_colors (Dict[str, Any]): Colors for each section/bar.
        hatches_series (pd.Series): Hatches for each section/bar.
        edgecolors_series (pd.Series): Edge colors for each section/bar.
        **kwargs: Additional keyword arguments for font sizes, etc.

    Returns:
        None
    """

    num_bars = len(temp_df)
    max_labels = temp_df.shape[0] if temp_df.shape[0] > 0 else 1
    bar_height_scale = (num_bars) / (2 * max_labels)
    label_font_size = kwargs.get("label_font_size", DEFAULT_FONT_SIZES["label"])
    tick_font_size = kwargs.get("tick_font_size", DEFAULT_FONT_SIZES["tick"])

    bar_ax = temp_df.plot.barh(
        ax=ax,
        stacked=True,
        legend=False,
        fontsize=label_font_size,
        color=section_colors,
        width=bar_height_scale,
    )
    for j, bar in enumerate(bar_ax.patches):
        hatch_index = j // (len(bar_ax.patches) // len(temp_df.columns))
        bar.set_hatch(hatches_series.iloc[hatch_index])
        bar.set_edgecolor(edgecolors_series.iloc[hatch_index])
    ax.set_ylabel("")
    ax.set_xlabel("Count", fontsize=label_font_size)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", which="major", linestyle=":")
    ax.set_ylim(-0.5, num_bars - 0.5)
    ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(MultipleLocator(1))


def deck_stem(
    deck_df: pd.DataFrame,
    y1: str,
    y2: str | None = None,
    plot_size: Tuple[int, int] | None = None,
    grid_spacing: Tuple[int, int] = (2, 1),
    grid_cols: int = 2,
    colors: Dict[str, str] | List[str] | None = None,
    markers: List[str] = ["s", "o", "+"],
    hollow: bool = False,
    marker_size: int = 10,
    **kwargs,
) -> Figure:
    """
    Create a grid of stem plots displaying the distribution of a specified column in each deck.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        y1 (str): The first column to be plotted.
        y2 (str, optional): The second column to be plotted. Defaults to None.
        plot_size (Tuple[int, int], optional): The width and height of each plot. If None, is calculated to fit all labels. Defaults to None.
        grid_spacing (Tuple[int, int], optional): The horizontal and vertical spacing between plots. Defaults to (2, 1).
        grid_cols (int, optional): The number of columns in the grid. Defaults to 2.
        colors (Dict[str, str] | List[str] | None, optional): A dictionary of colors for each section, or a list of colors to be used in the plot. If not provided, colors_dict is used. Defaults to None.
        label_font_size, title_font_size, legend_font_size, tick_font_size (int, optional): Font sizes for labels, titles, legend, and ticks. Pass as keyword arguments if you want to override defaults.
        markers (List[str], optional): The list of markers to be used in the plot for each deck section. Defaults to ["s", "o", "+"].
        hollow (bool, optional): Whether to make the markers hollow. Defaults to False.
        marker_size (int, optional): The initial size of the markers. Defaults to 10.
        **kwargs: Not implemented.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    columns = [y1] if y2 is None else [y1, y2]
    decks = deck_df["Deck"].unique()
    sorted_sections = (
        deck_df[deck_df[columns].notna().any(axis=1)]
        .groupby("Section")["Count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    title_font_size = kwargs.pop("title_font_size", DEFAULT_FONT_SIZES["title"])
    legend_font_size = kwargs.pop("legend_font_size", DEFAULT_FONT_SIZES["legend"])

    # Set constants for plot sizes and spacing
    plot_width = 9 if plot_size is None else plot_size[0]  # Width of each plot
    # Fixed height for each plot
    plot_height = 4 if plot_size is None else plot_size[1]
    horizontal_space = grid_spacing[0]  # Fixed horizontal space between plots
    vertical_space = grid_spacing[1]  # Fixed vertical space between plots
    # Fixed space between figure top and subplots
    header_space = legend_font_size / 10

    # Calculate number of columns and rows
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    # Dynamically calculate the figure size based on the number of rows and columns
    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    fig_height = plot_height * rows + (rows - 1) * vertical_space + header_space

    if colors is None:
        colors = {
            section: colors_dict.get(c, f"C{i}")
            for i, (section, c) in enumerate(
                zip(["Main", "Extra", "Side"], ["Effect Monster", "Fusion Monster", "Xyz Monster"])
            )
        }
    else:
        colors = {
            section: (colors.get(section, f"C{i}") if isinstance(colors, dict) else colors[i])
            for i, section in enumerate(sorted_sections)
        }

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
            ax,
            sub_df,
            columns,
            colors,
            markers,
            hollow,
            marker_size,
            **kwargs,
        )

    top = 1 - header_space / fig_height
    legend_y = top + (1 - top) / 2

    fig.subplots_adjust(
        top=top,
        bottom=0,
    )
    fig.legend(
        sorted_sections,
        ncols=3,
        loc="center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize=legend_font_size,
        frameon=False,
    )
    fig.suptitle(f"{', '.join(columns)} distribution", fontsize=title_font_size, y=1)
    return fig


def _plot_stem_subplot(
    ax: Axes,
    sub_df: pd.DataFrame,
    columns: List[str],
    colors: Dict[str, str],
    markers: List[str],
    hollow: bool,
    marker_size: int,
    **kwargs,
) -> None:
    """
    Plot a stem plot for a single deck on the provided axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        sub_df (pd.DataFrame): DataFrame for the current deck.
        columns (List[str]): List of columns to plot (e.g., ["ATK", "DEF"]).
        colors (Dict[str, str]): Mapping from section name to color string.
        markers (List[str]): List of marker styles for each section.
        hollow (bool): Whether to use hollow markers.
        marker_size (int): Initial marker size.
        **kwargs: Additional keyword arguments for font sizes, etc.

    Returns:
        None
    """
    label_font_size = kwargs.get("label_font_size", DEFAULT_FONT_SIZES["label"])
    tick_font_size = kwargs.get("tick_font_size", DEFAULT_FONT_SIZES["tick"])
    msize = marker_size
    max_idx = 0
    min_idx = np.inf
    hasna = False
    it_columns = sub_df[columns].dropna(axis=1, how="all").columns
    sorted_sections = sub_df.groupby("Section")["Count"].sum().sort_values(ascending=False).index.tolist()
    steps = (100, 500) if sub_df[columns].map(pd.to_numeric, errors="coerce").diff().max().max() > 12 else (1, 1)
    for k, col in enumerate(it_columns):
        msize = marker_size
        for j, s in enumerate(sorted_sections):
            sub_sub_df = sub_df[sub_df["Section"] == s]
            if sub_sub_df.empty:
                continue
            series = sub_sub_df.groupby(col)["Count"].sum().mul(np.power(-1, k))
            if series.empty:
                continue
            index = pd.to_numeric(series.index, errors="coerce")
            if not index.isna().all():
                max_idx = max(int(index.max()), max_idx)
                min_idx = min(int(index.min()), min_idx)
            if index.isna().any():
                hasna = True
            series.index = index.fillna(max_idx + steps[1])
            stem = ax.stem(
                series.index,
                series,
                linefmt=":",
                markerfmt=markers[(j)],
                basefmt=":",
            )
            if hollow:
                stem.markerline.set_markeredgecolor(colors.get(s, f"C{j}"))
                stem.markerline.set_markerfacecolor("none")
            else:
                stem.markerline.set_color(colors.get(s, f"C{j}"))
            stem.stemlines.set_color(colors.get(s, f"C{j}"))
            stem.baseline.set_color(colors.get(s, f"C{j}"))
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
    ax.set_ylim(nlim, plim)
    if nlim < 0:
        ax.set_ylabel("← " + " | ".join(reversed(it_columns)) + " →", fontsize=label_font_size)
    else:
        ax.set_ylabel(it_columns[0], fontsize=label_font_size)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(abs(x)))))
    ax.xaxis.set_minor_locator(MultipleLocator(steps[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
    ax.grid(ls=":", axis="y")
    ax.set_axisbelow(True)
