# yugiquery/utils/plot.py

# -*- coding: utf-8 -*-

# =============== #
# Plotting module #
# =============== #

# ======= #
# Imports #
# ======= #

# Standard library imports
import warnings
import colorsys
from typing import List, Tuple, Callable

# Third-party imports
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LogNorm, Normalize, ListedColormap, cnames, to_rgb, rgb_to_hsv, hex2color
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Local application imports
from .helpers import *
from .dirs import dirs

# Matplotlib default settings overrides
plt.style.use("default")  # TODO: Make this configurable
if dirs.is_notebook:
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")  # Needed for dynanmic theme

# ========= #
# Variables #
# ========= #

#: Dictionary containing the colors used in the plots.
colors_dict = load_json(dirs.get_asset("json", "colors.json"))
# TODO: Adapt colors to style


# ======= #
# Classes #
# ======= #


class MulticolorPatchHandler:
    """
    Custom legend handler to display a multicolored rectangle with a single uniform hatch across the entire box.
    """

    def __init__(self, colors, hatch=None, edgecolor="black", **kwargs):
        self.colors = colors  # List of colors for different segments
        self.hatch = hatch  # Single hatch applied across the entire box
        self.edgecolor = edgecolor  # Single edge color for the whole box
        self.kwargs = kwargs

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width = handlebox.width
        height = handlebox.height

        # Define the width for each color segment
        color_width = width / len(self.colors)

        # Create multicolored patches
        for i, color in enumerate(self.colors):
            patch = mpatches.Rectangle(
                [handlebox.xdescent + i * color_width, -handlebox.ydescent],
                color_width,
                height,
                facecolor=color,
                transform=handlebox.get_transform(),
                **self.kwargs,
            )
            handlebox.add_artist(patch)

        # Apply a transparent hatch over the entire box
        hatch_patch = mpatches.Rectangle(
            [handlebox.xdescent, -handlebox.ydescent],
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


# ========= #
# Functions #
# ========= #


# Helpers
def is_light_color(color, threshold=0.6) -> bool:
    """
    Check if a given color is light or dark based on a specified threshold.

    Args:
        color: The color to be checked.
        threshold: The threshold to determine if a color is light or dark.

    Returns:
        bool: True if the color is light, False otherwise.
    """
    # Convert color to RGB if it's in hex format
    if isinstance(color, str):
        color = hex2color(color)
    # Convert RGB to HSV
    hsv = rgb_to_hsv(color)
    # Check the brightness (value in HSV)
    return hsv[2] > threshold


def adjust_lightness(color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """
    Adjust the lightness of a given color by a specified amount.

    Args:
        color (str): The color to be adjusted, in string format.
        amount (float, optional): The amount by which to adjust the lightness of the color. Default value is 0.5.

    Returns:
        tuple: The adjusted color in RGB format.
    """

    try:
        c = cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*to_rgb(c))
    return colorsys.hls_to_rgb(h=c[0], l=max(0, min(1, amount * c[1])), s=c[2])


def align_yaxis(ax1: plt.Axes, v1: float, ax2: plt.Axes, v2: float) -> None:
    """
    Adjust the y-axis of two subplots so that the specified values in each subplot are aligned.

    Args:
        ax1 (AxesSubplot): The first subplot.
        v1 (float): The value in ax1 that should be aligned with v2 in ax2.
        ax2 (AxesSubplot): The second subplot.
        v2 (float): The value in ax2 that should be aligned with v1 in ax1.

    Returns:
        None
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax=ax2, ydif=(y1 - y2) / 2, v=v2)
    adjust_yaxis(ax=ax1, ydif=(y2 - y1) / 2, v=v1)


# Rates
def adjust_yaxis(ax: plt.Axes, ydif: float, v: float) -> None:
    """
    Shift the y-axis of a subplot by a specified amount, while maintaining the location of a specified point.

    Args:
        ax (AxesSubplot): The subplot whose y-axis is to be adjusted.
        ydif (float): The amount by which to adjust the y-axis.
        v (float): The location of the point whose position should remain unchanged.

    Returns:
        None
    """
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


def generate_rate_grid(
    df: pd.DataFrame,
    ax: plt.Axes,
    xlabel: str = "Date",
    size_pct: str | float = "150%",
    pad: int = 0,
    colors: List[str] | None = None,
    cumsum: bool = True,
    fill: bool = False,
    limit_year: bool = False,
) -> plt.axes:
    """
    Generate a grid of subplots displaying yearly and monthly rates from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the data to be plotted.
        ax (AxesSubplot): The subplot onto which to plot the grid.
        xlabel (str, optional): The label to be used for the x-axis. Default value is 'Date'.
        size_pct (float, optional): The size of the bottom subplot as a percentage of the top subplot. Default value is '150%'.
        pad (int, optional): The amount of padding between the two subplots in pixels. Default value is 0.
        colors (List[str] | None, optional): A list of colors to be used in the plot. If not provided, the default Matplotlib color cycle is used. Default value is None.
        cumsum (bool, optional): If True, plot the cumulative sum of the data. If False, plot only the yearly and monthly rates. Default value is True.
        fill (bool, optional): If True, fill the area under the cumulative sum curve. Default value is False.
        limit_year (bool, optional): If True, limit the x-axis to the next full year. Default value is False.

    Returns:
        matplotlib.axes.Axes: The generated subplot axes.
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if cumsum:
        cumsum_ax = ax
        divider = make_axes_locatable(axes=cumsum_ax)
        yearly_ax = divider.append_axes(position="bottom", size=size_pct, pad=pad)
        cumsum_ax.figure.add_axes(yearly_ax)
        cumsum_ax.set_xticklabels([])
        axes = [cumsum_ax, yearly_ax]

        y = df.fillna(0).cumsum()

        if len(df.columns) == 1:
            cumsum_ax.plot(y, label="Cummulative", c=colors[0], antialiased=True)
            if fill:
                cumsum_ax.fill_between(x=y.index, y1=y.values.T[0], color=colors[0], alpha=0.1, hatch="x")
            cumsum_ax.set_ylabel(f"{y.columns[0]}")  # Wrap text
        else:
            cumsum_ax.stackplot(y.index, y.values.T, labels=y.columns, colors=colors, antialiased=True)
            cumsum_ax.set_ylabel(f"Cumulative {y.index.name.lower()}")

        yearly_ax.set_ylabel(f"Yearly {df.index.name.lower()} rate")
        cumsum_ax.legend(loc="upper left", ncols=int(len(df.columns) / 5 + 1))  # Test
        func = lambda x, pos: "" if np.isclose(x, 0) else f"{round(x):.0f}"
        cumsum_ax.yaxis.set_major_formatter(FuncFormatter(func))

    else:
        yearly_ax = ax
        axes = [yearly_ax]

        if len(df.columns) == 1:
            yearly_ax.set_ylabel(f"{df.columns[0]}\nYearly {df.index.name.lower()} rate")
        else:
            yearly_ax.set_ylabel(f"Yearly {df.index.name.lower()} rate")

    if len(df.columns) == 1:
        monthly_ax = yearly_ax.twinx()
        monthly_rate = df.resample("ME").sum()
        monthly_ax.bar(
            x=monthly_rate.index,
            height=monthly_rate.T.values[0],
            width=monthly_rate.index.diff(),
            label="Monthly rate",
            color=colors[2],
            antialiased=True,
        )
        monthly_ax.set_ylabel(f"Monthly {df.index.name.lower()} rate")
        monthly_ax.legend(loc="upper right")
        yearly_rate = df.resample("YE").sum()

        # Remove the last year if it is incomplete
        if limit_year and yearly_rate.index[-1].timestamp() > arrow.utcnow().shift(years=1).timestamp():
            yearly_rate = yearly_rate[:-1]

        yearly_ax.plot(
            yearly_rate,
            label="Yearly rate",
            ls="--",
            c=colors[1],
            antialiased=True,
        )
        yearly_ax.legend(loc="upper left", ncols=int(len(df.columns) / 8 + 1))

    else:
        dy2 = df.resample("YE").sum()
        yearly_ax.stackplot(dy2.index, dy2.values.T, labels=dy2.columns, colors=colors, antialiased=True)
        if not cumsum:
            yearly_ax.legend(loc="upper left", ncols=int(len(df.columns) / 8 + 1))

    if xlabel is not None:
        yearly_ax.set_xlabel(xlabel)
    else:
        yearly_ax.set_xticklabels([])

    for temp_ax in axes:
        temp_ax.set_xlim(
            [
                df.index.min() - pd.Timedelta(weeks=13),
                df.index.max() + pd.Timedelta(weeks=52),
            ]
        )
        temp_ax.xaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.yaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.xaxis.set_major_locator(mdates.YearLocator())
        temp_ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        temp_ax.set_axisbelow(True)
        temp_ax.grid(ls=":")

    yearly_ax.tick_params(axis="x", rotation=45)

    if len(df.columns) == 1:
        align_yaxis(ax1=yearly_ax, v1=0, ax2=monthly_ax, v2=0)
        l = yearly_ax.get_ylim()
        l2 = monthly_ax.get_ylim()
        f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
        ticks = f(yearly_ax.get_yticks())
        monthly_ax.yaxis.set_major_locator(FixedLocator(ticks))
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
    vlines: pd.DataFrame | None = None,
    fill: bool = False,
    limit_year: bool = False,
    subplots: bool = False,
    hspace: float = 0.05,
) -> plt.figure:
    """
    Creates a visualization of rate changes over time for multiple variables in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        figsize (Tuple[int, int] | None, optional): Size of the figure. Defaults to None.
        title (str, optional): Title of the plot. Defaults to an empty string.
        colors (List[str] | None, optional): List of colors for the plot lines. Defaults to None.
        cumsum (bool, optional): Whether to plot cumulative sum of data. Defaults to True.
        bg (pd.DataFrame | None, optional): Data for background shading. Defaults to None.
        vlines (pd.DataFrame | None, optional): Data for vertical lines. Defaults to None.
        fill (bool, optional): Whether to fill the area under the cumulative sum curve. Defaults to False.
        limit_year (bool, optional): Whether to limit the x-axis to the next full year. Defaults to False.
        subplots (bool, optional): Whether to create a grid of subplots for each column in the DataFrame. Defaults to False.
        hspace (float, optional): Height space between subplots. Defaults to 0.5.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    num_cols = len(df.columns)
    top_space = 0.5

    if figsize is None:
        figsize = (14, num_cols * 3 * (1 + cumsum)) if subplots else (14, 6)

    # Setup figure and gridspec
    fig = plt.figure(figsize=figsize)
    if subplots:
        gs = GridSpec(num_cols, 1, height_ratios=[3] * num_cols, hspace=hspace)
    else:
        gs = GridSpec(1, 1, hspace=hspace)

    fig.suptitle(
        f'{title if title else df.index.name}{f" by {df.columns.name.lower()}" if df.columns.name else ""}',
        y=1,
    )

    if colors is None and subplots:
        colors = plt.cm.get_cmap("tab20").colors

    # Create subplots and apply shading, vertical lines
    axes = []
    for i, col in enumerate(df.columns):
        ax = fig.add_subplot(gs[i] if subplots else gs[0])
        sub_axes = generate_rate_grid(
            df=df[col].to_frame() if subplots else df,
            ax=ax,
            colors=(
                [colors[2 * i % len(colors)], colors[2 * i % len(colors)], colors[(2 * i + 1) % len(colors)]]
                if subplots
                else colors
            ),
            cumsum=cumsum,
            fill=fill,
            limit_year=limit_year,
            size_pct="100%" if subplots else "150%",
            xlabel="Date" if (i + 1) == len(df.columns) else None,
        )
        axes.extend(sub_axes[:2])
        if not subplots:
            break

    # Add background shading and vertical lines separately
    if bg is not None and "end" in bg:
        bg["end"] = bg["end"].fillna(df.index.max())
        add_background_shading(axes=axes, bg=bg)
    if vlines is not None:
        add_vertical_lines(axes=axes, vlines=vlines, cumsum=cumsum)

    fig.subplots_adjust(top=1 - top_space / fig.get_figheight())
    return fig


def add_background_shading(axes: List[plt.Axes], bg: pd.DataFrame, colors: List | None = None) -> None:
    """
    Add background shading to the subplots.

    Args:
        axes (List[plt.Axes]): List of axes to apply background shading.
        bg (pd.DataFrame): DataFrame for background shading. Must contain 'begin' and 'end' columns.
        colors (list | None, optional): Colormap for the shading. If None, colors_dict is used. Defaults to None.

    Returns:
        None
    """
    if all(col in bg.columns for col in ["begin", "end"]):
        sec_ax = axes[0].secondary_xaxis("top")
        sec_ax.set_xticks(bg.mean(axis=1))
        sec_ax.set_xticklabels(bg.index)
        for ix, ax in enumerate(axes):
            c = 0
            for idx, row in bg.iterrows():
                if row["end"] > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                    ax.axvspan(
                        row["begin"],
                        row["end"],
                        alpha=0.1,
                        color=(colors[c] if colors is not None else colors_dict.get(idx, f"C{c}")),
                        zorder=-1,
                    )
                    c += 1


def add_vertical_lines(axes: List[plt.Axes], vlines: pd.DataFrame, color="maroon", cumsum: bool = False) -> None:
    """
    Add vertical lines to the subplots.

    Args:
        axes (List[plt.Axes]): List of axes to apply vertical lines.
        vlines (pd.DataFrame): DataFrame for vertical lines.
        color (str): Color for the vertical lines.
        cumsum (bool, optional): Whether cumulative sum is being plotted. Defaults to False.

    Returns:
        None
    """
    for ix, ax in enumerate(axes):
        for idx, row in vlines.items():
            if row > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                line = ax.axvline(row, ls="-.", c=color, lw=1)
                if ix == 0:
                    (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                    ax.text(
                        (x0 + x1) / 2 + 25,
                        (0.05 if cumsum else 0.95),
                        idx,
                        c=color,
                        ha="left",
                        va=("bottom" if cumsum else "top"),
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                    )


# Dedicated plots
def arrows(arrows: pd.Series, figsize: Tuple[int, int] = (6, 6), **kwargs) -> plt.figure:
    """
    Create a polar plot to visualize the frequency of each arrow direction in a pandas Series.

    Args:
        arrows (pandas.Series): A pandas Series containing arrow symbols as string data type.
        figsize (Tuple[int, int], optional): The width and height of the figure. Defaults to (6, 6).
        **kwargs: Additional keyword arguments to be passed to the bar() method.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
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
    ax.yaxis.set_major_locator(FixedLocator(ticks[1:]))
    ax.set_rorigin(-5)

    # Set the title of the plot
    ax.set_title("Link Arrows")
    ax.set_axisbelow(True)

    # Display the plot
    fig.tight_layout()
    return fig


def box(df, mean: bool = True, **kwargs) -> plt.figure:
    """
    Plots a box plot of a given DataFrame using seaborn, with the year of the Release column on the x-axis and the remaining column on the y-axis.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the Release dates and another numeric column.
        mean (bool, optional): If True, plots a line representing the mean of each box. Defaults to True.
        **kwargs: Additional keyword arguments to pass to seaborn.boxplot().

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If the DataFrame has no Release column.
    """
    df = df.dropna().copy()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    col = df.columns.difference(["Release"])[0]
    df["year"] = df["Release"].dt.strftime("%Y")
    df[col] = df[col].apply(pd.to_numeric, errors="coerce")

    sns.boxplot(ax=ax, data=df, y=col, x="year", width=0.5, **kwargs)
    if mean:
        df.groupby("year").mean(numeric_only=True).plot(ax=ax, c="r", ls="--", alpha=0.75, grid=True, legend=False)

    if df[col].max() < 5000:
        ax.set_yticks(np.arange(0, df[col].max() + 1, 1))
    elif df[col].max() == 5000:
        ax.set_yticks(np.arange(start=0, stop=5500, step=500))
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_axisbelow(True)
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def pyramid(
    series: pd.Series,
    use_area: bool = False,
    size: Tuple[int, int] = (8, 8),
    grid: bool = False,
    colors: List[str] | None = None,
    alpha: int = 1,
    **kwargs,
) -> plt.Figure:
    """
    Creates a pyramid plot from a pandas Series.

    Args:
        series (pd.Series): The data to plot.
        use_area (bool, optional): Whether to use area to represent the values. Defaults to False.
        size (Tuple[int, int], optional): The size of the plot. Defaults to (8, 8).
        grid (bool, optional): Whether to show a grid. Defaults to False.
        colors (List[str], optional): The colors of the bars. If None, use default color cycler. Defaults to None.
        alpha (int, optional): The transparency of the bars. Defaults to 1.
        kwargs: Additional keyword arguments to pass to the plot. Not implemented.

    Returns:
        plt.Figure: The plot.
    """
    series = series.sort_values(ascending=False)
    n = len(series)
    yticks = []

    if colors is None:
        colors = [f"C{i}" for i in range(n)]

    fig = plt.figure(figsize=size)
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

    fig.suptitle(series.name)
    fig.tight_layout()

    return fig


# =================== #
# Deck specific plots #
# =================== #


def deck_composition(
    deck_df: pd.DataFrame,
    grid_spacing: Tuple[int, int] = (2, 1),
    grid_cols: int = 3,
    plot_size: Tuple[int, int] = (5, 5),
    ring_radius: float = 0.3,
    pctdistances: List[float] = [0.85, 0.75],
    font_size: Dict[str, int] = {"label": 14, "title": 16, "suptitle": 20, "legend": 12},
    **kwargs,
) -> plt.Figure:
    """
    Create a grid of pie charts displaying the composition of each deck in a DataFrame.

    Args:
        deck_df (pd.DataFrame): The DataFrame containing the deck data.
        grid_spacing (Tuple[int, int], optional): The horizontal and vertical spacing between plots. Defaults to (2, 1).
        grid_cols (int, optional): The number of columns in the grid. Defaults to 3.
        plot_size (Tuple[int, int], optional): The width and height of each plot. Defaults to (5, 5).
        ring_radius (float, optional): The radius of the ring in the pie chart. Defaults to 0.3.
        pctdistances (List[float], optional): The distances of the percentage labels from the center of the chart rings. Defaults to [0.85, 0.75].
        font_size (Dict[str,int], optional): The dictionary of font sizes to override defaults for the labels, title, suptitle, and legend. Defaults to {"label": 14, "title": 16, "suptitle": 20, "legend": 12}.
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

    # Font sizes
    label_font_size = font_size.get("label", 14)
    title_font_size = font_size.get("title", 16)
    suptitle_font_size = font_size.get("suptitle", 20)
    legend_font_size = font_size.get("legend", 12)

    plot_width = plot_size[0]  # Width of each plot
    plot_height = plot_size[1]  # Fixed height for each plot
    horizontal_space = grid_spacing[0]  # Fixed horizontal space between plots
    vertical_space = grid_spacing[1]  # Fixed vertical space between plots
    header_space = (2 * legend_font_size + 2) / 10  # Fixed space between top and first row of plots
    cols = min(grid_cols, len(decks))
    rows = int(np.ceil(len(decks) / cols))

    colors_main = [colors_dict[type] for type in main_df.index]
    colors_extra = [colors_dict[type] for type in extra_df.index]
    colors_remaining = side_df.index.difference(main_df.index.union(extra_df.index))

    # Dynamically calculate the figure size based on the number of rows and columns
    fig_width = plot_width * cols + (cols - 1) * horizontal_space
    fig_height = plot_height * rows + (rows - 1) * vertical_space + header_space

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=rows,
        ncols=cols,
        wspace=horizontal_space / plot_width,  # Adjusted for figure width
        hspace=vertical_space / plot_height,  # Adjusted for plot height
    )

    def make_autopct(values) -> Callable[..., str]:
        def my_autopct(pct) -> str:
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.0f}%\n({val})"

        return my_autopct

    for i, deck in enumerate(decks):
        # Create sub-grid for pie and bar plots
        sub_gs = gs[(i // cols), i % cols].subgridspec(2, 1, height_ratios=[9, 1], hspace=0.2)

        # Main plot in the upper sub-grid
        ax_pie = fig.add_subplot(sub_gs[0, 0])
        wedges1, texts1, autotexts1 = ax_pie.pie(
            main_df[deck].dropna(),
            autopct=make_autopct(main_df[deck].dropna()),
            startangle=90,
            radius=1,
            wedgeprops=dict(width=ring_radius, edgecolor="w"),
            pctdistance=pctdistances[0],
            colors=np.array(colors_main)[main_df[deck].notna()],
            counterclock=False,
        )

        if deck in extra_df.columns:
            wedges2, texts2, autotexts2 = ax_pie.pie(
                extra_df[deck].dropna(),
                autopct=make_autopct(extra_df[deck].dropna()),
                startangle=90,
                radius=1 - ring_radius,
                wedgeprops=dict(width=ring_radius, edgecolor="w"),
                pctdistance=pctdistances[1],
                colors=np.array(colors_extra)[extra_df[deck].notna()],
                counterclock=False,
            )

        for wedge, text in zip(wedges1, autotexts1):
            color = wedge.get_facecolor()[:3]
            text.set_color("black" if is_light_color(color) else "white")
        for wedge, text in zip(wedges2, autotexts2):
            color = wedge.get_facecolor()[:3]
            text.set_color("black" if is_light_color(color) else "white")

        ax_pie.text(
            0,
            0,
            f"Main: {main_df[deck].sum()}\nExtra: {extra_df[deck].sum()}",
            ha="center",
            va="center",
            fontsize=label_font_size,
        )
        ax_pie.set_title(deck, fontsize=title_font_size)
        ax_pie.set_xlim(-1, 1)
        ax_pie.set_ylim(-1, 1)
        ax_pie.set_aspect("equal", adjustable="box")

        ax_bar = fig.add_subplot(sub_gs[1, 0])  # Bar plot in the odd row
        ax_bar.axis("off")
        # Create bar plot in the lower sub-grid
        if deck in side_df and side_df[deck] is not None:
            sorted_side = side_df[deck].sort_values(ascending=True).dropna()
            side_total = sorted_side.sum()
            left = 0
            height = 0.1
            for j, (name, count) in enumerate(sorted_side.items()):
                left -= count
                color = colors_dict[name]
                bc = ax_bar.barh(
                    0,
                    width=count,
                    height=height,
                    left=left,
                    color=color,
                    edgecolor="white",
                )
                ax_bar.bar_label(
                    bc,
                    labels=[f"{count/side_total*100:.0f}%\n({count})"],
                    label_type="center",
                    color="black" if is_light_color(color) else "white",
                )
            ax_bar.set_title(f"Side: {side_total}", fontsize=label_font_size)
            ax_bar.set_xlim(-side_total, 0)
            ax_bar.set_ylim(-0.05, 0.05)
            ax_bar.set_aspect(side_total, adjustable="box")

        else:
            ax_bar.set_title(f"Side: 0", fontsize=label_font_size)

    # Create custom legend handles for main_df and extra_df
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

    # Adjust the legend position
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


def deck_distribution(
    deck_df: pd.DataFrame,
    column: str,
    grid_spacing: Tuple[int, int] = (3, 1),
    grid_cols: int = 2,
    plot_size: Tuple[int, int] | None = None,
    colors: Dict[str, str] | List[str] | None = None,
    hatches: List[str] | str = "",
    edgecolors: List[str] | str = "white",
    font_size: Dict[str, int] = {"label": 14, "title": 20, "tick": 12, "legend": 12},
    **kwargs,
) -> plt.Figure:
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
    max_labels = deck_df.groupby("Deck")[column].nunique().max()
    sorted_sections = (
        deck_df[deck_df[column].notna()].groupby("Section")["Count"].sum().sort_values(ascending=False).index.tolist()
    )

    # Font sizes
    label_font_size = font_size.get("label", 14)
    title_font_size = font_size.get("title", 20)
    legend_font_size = font_size.get("legend", 12)
    tick_font_size = font_size.get("tick", 12)

    # Set constants for plot sizes and spacing
    plot_width = 6 if plot_size is None else plot_size[0]  # Width of each plot
    plot_height = max(mean_labels / 2, 0.5) if plot_size is None else plot_size[1]  # Fixed height for each plot
    horizontal_space = grid_spacing[0] + max(2 * int(max_label_len / 10) - 3, 0)  # Fixed horizontal space between plots
    vertical_space = grid_spacing[1]  # Fixed vertical space between plots
    header_space = legend_font_size / 10  # Fixed space between figure top and subplots

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
        wspace=horizontal_space / plot_width,  # Adjusted for figure width
        hspace=vertical_space / plot_height,  # Adjusted for plot height
    )

    if colors is None:
        section_colors = {
            section: colors_dict.get(c, f"C{i}")
            for i, (section, c) in enumerate(zip(["Main", "Extra", "Side"], ["Effect Monster", "Fusion Monster", "Counter"]))
        }
    else:
        section_colors = {
            section: (
                pd.Series(
                    colors.get(section, f"C{i}") if isinstance(colors, dict) else colors,
                    index=(
                        sorted(deck_df[column].dropna().unique())
                        if ((isinstance(colors, dict) and isinstance(colors[section], list)) or isinstance(colors, list))
                        else [0]
                    ),
                )
            )
            for i, section in enumerate(sorted_sections)
        }

    hatches = pd.Series(
        (
            [hatches.get(section, "") for section in sorted_sections]
            if isinstance(hatches, dict)
            else hatches[: len(sorted_sections)] if isinstance(hatches, list) else hatches
        ),
        index=sorted_sections,
    )
    edgecolors = pd.Series(
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
            num_bars = len(temp_df)
            # Scale factor to adjust bar height based on the maximum number of bars
            bar_height_scale = (num_bars) / (2 * max_labels)

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

            bar_ax = temp_df.plot.barh(
                ax=ax,
                stacked=True,
                legend=False,
                fontsize=label_font_size,
                color=plot_colors,
                width=bar_height_scale,
            )
            for j, bar in enumerate(bar_ax.patches):
                hatch_index = j // (len(bar_ax.patches) // len(temp_df.columns))
                bar.set_hatch(hatches.iloc[hatch_index])
                bar.set_edgecolor(edgecolors.iloc[hatch_index])

            ax.set_ylabel("")
            ax.set_xlabel("Count", fontsize=label_font_size)
            ax.set_title(deck, fontsize=label_font_size)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="x", which="major", linestyle=":")
            ax.set_ylim(-0.5, num_bars - 0.5)
            ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
            ax.set_axisbelow(True)

            ax.xaxis.set_minor_locator(MultipleLocator(1))

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
        handler[mpatches.Patch(label=section)] = MulticolorPatchHandler(
            color, hatches[section], edgecolor=edgecolors[section]
        )

    # Add legend with a fixed position
    leg = fig.legend(
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


def deck_stem(
    deck_df: pd.DataFrame,
    y1: str,
    y2: str | None = None,
    plot_size: Tuple[int, int] | None = None,
    grid_spacing: Tuple[int, int] = (2, 1),
    grid_cols: int = 2,
    colors: Dict[str, str] | List[str] | None = None,
    font_size: Dict[str, int] = {"label": 14, "title": 20, "tick": 12, "legend": 12},
    markers: List[str] = ["s", "o", "+"],
    hollow: bool = False,
    marker_size: int = 10,
    **kwargs,
) -> plt.Figure:
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
        font_size (Dict[str,int], optional): The dictionary of font sizes to override defaults for the labels, title, suptitle, and legend. Defaults to {"label": 14, "title": 16, "suptitle": 20, "legend": 12}.
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
    steps = (100, 500) if deck_df[columns].map(pd.to_numeric, errors="coerce").diff().max().max() > 12 else (1, 1)

    # Font sizes
    label_font_size = font_size.get("label_font_size", 14)
    tick_font_size = font_size.get("tick_font_size", 12)
    title_font_size = font_size.get("title_font_size", 20)
    legend_font_size = font_size.get("legend_font_size", 12)

    # Set constants for plot sizes and spacing
    plot_width = 9 if plot_size is None else plot_size[0]  # Width of each plot
    plot_height = 4 if plot_size is None else plot_size[1]  # Fixed height for each plot
    horizontal_space = grid_spacing[0]  # Fixed horizontal space between plots
    vertical_space = grid_spacing[1]  # Fixed vertical space between plots
    header_space = legend_font_size / 10  # Fixed space between figure top and subplots

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
        max_idx = 0
        min_idx = np.inf
        hasna = False
        sub_df = deck_df[deck_df["Deck"] == deck]
        if sub_df.empty:
            continue
        it_columns = sub_df[columns].dropna(axis=1, how="all").columns
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
                    stem.markerline.set_markeredgecolor(colors.get(s, f"C{j}"))  # Marker edge color
                    stem.markerline.set_markerfacecolor("none")  # Hollow marker (no fill)
                else:
                    stem.markerline.set_color(colors.get(s, f"C{j}"))
                stem.stemlines.set_color(colors.get(s, f"C{j}"))  # Stem line color
                stem.baseline.set_color(colors.get(s, f"C{j}"))  # Baseline color
                stem.markerline.set_markersize(msize)
                msize = max(msize - 2, 2)

        if steps[1] < 10:
            xticks = np.arange(0, 14, 1)
        else:
            xticks = np.arange(int(min_idx / steps[1]) * steps[1], max_idx + steps[1], steps[1])
        minor_xticks = np.arange(0, (len(xticks) - 1) * steps[1], steps[0])
        if hasna:
            xticks = list(xticks) + [max_idx + steps[1]]

        xticks_labels = xticks[:-1] + ["?"] if hasna else xticks
        ax.set_xticks(xticks, xticks_labels, rotation=45 * (xticks[-1] > 100))
        ax.set_xticks(minor_xticks, minor=True)

        ax.set_title(deck)
        ax.set_xlim(-min(steps[0], steps[1] / 2) + min(xticks), max(xticks) + min(steps[0], steps[1] / 2))
        plim = int(ax.get_ylim()[1] + 1)
        nlim = int(ax.get_ylim()[0] - 1) if ax.get_ylim()[0] < -1 else 0
        ax.set_ylim(nlim, plim)
        if nlim < 0:
            ax.set_ylabel("← " + " | ".join(reversed(it_columns)) + " →", fontsize=label_font_size)
        else:
            ax.set_ylabel(it_columns[0], fontsize=label_font_size)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(abs(x))))
        ax.xaxis.set_minor_locator(MultipleLocator(steps[0]))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
        ax.grid(ls=":", axis="y")
        ax.set_axisbelow(True)

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
    fig.suptitle("/".join(columns) + " Distribution", y=1, fontsize=title_font_size)

    return fig
