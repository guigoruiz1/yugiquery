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
from typing import (
    List,
    Tuple,
)

# Third-party imports
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, ListedColormap, cnames, to_rgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    MaxNLocator,
)
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
colors_dict = load_json(dirs.get_asset("json", "colors.json"))  # Colors dictionary to associate to series and cards
# TODO: Adapt colors to style

# ========= #
# Functions #
# ========= #


def adjust_lightness(color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """Adjust the lightness of a given color by a specified amount.

    Args:
        color (str): The color to be adjusted, in string format.
        amount (float): The amount by which to adjust the lightness of the color. Default value is 0.5.

    Returns:
        tuple: The adjusted color in RGB format.

    Raises:
        KeyError: If the specified color is not a valid Matplotlib color name.
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
    dy: pd.DataFrame,
    ax: plt.Axes,
    xlabel: str = "Date",
    size: str = "150%",
    pad: int = 0,
    colors: List[str] = None,
    cumsum: bool = True,
) -> plt.axes:
    """
    Generate a grid of subplots displaying yearly and monthly rates from a Pandas DataFrame.

    Args:
        dy (pd.DataFrame): A Pandas DataFrame containing the data to be plotted.
        ax (AxesSubplot): The subplot onto which to plot the grid.
        xlabel (str): The label to be used for the x-axis. Default value is 'Date'.
        size (str): The size of the bottom subplot as a percentage of the top subplot. Default value is '150%'.
        pad (int): The amount of padding between the two subplots in pixels. Default value is 0.
        colors (List[str]): A list of colors to be used in the plot. If not provided, the default Matplotlib color cycle is used. Default value is None.
        cumsum (bool): If True, plot the cumulative sum of the data. If False, plot only the yearly and monthly rates. Default value is True.

    Returns:
        matplotlib.axes.Axes: The generated subplot axes.
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if cumsum:
        cumsum_ax = ax
        divider = make_axes_locatable(axes=cumsum_ax)
        yearly_ax = divider.append_axes(position="bottom", size=size, pad=pad)
        cumsum_ax.figure.add_axes(yearly_ax)
        cumsum_ax.set_xticklabels([])
        axes = [cumsum_ax, yearly_ax]

        y = dy.fillna(0).cumsum()

        if len(dy.columns) == 1:
            cumsum_ax.plot(y, label="Cummulative", c=colors[0], antialiased=True)
            cumsum_ax.fill_between(x=y.index, y1=y.values.T[0], color=colors[0], alpha=0.1, hatch="x")
            cumsum_ax.set_ylabel(f"{y.columns[0]}")  # Wrap text
        else:
            cumsum_ax.stackplot(y.index, y.values.T, labels=y.columns, colors=colors, antialiased=True)
            cumsum_ax.set_ylabel(f"Cumulative {y.index.name.lower()}")

        yearly_ax.set_ylabel(f"Yearly {dy.index.name.lower()} rate")
        cumsum_ax.legend(loc="upper left", ncols=int(len(dy.columns) / 5 + 1))  # Test

    else:
        yearly_ax = ax
        axes = [yearly_ax]

        if len(dy.columns) == 1:
            yearly_ax.set_ylabel(f"{dy.columns[0]}\nYearly {dy.index.name.lower()} rate")
        else:
            yearly_ax.set_ylabel(f"Yearly {dy.index.name.lower()} rate")

    if len(dy.columns) == 1:
        monthly_ax = yearly_ax.twinx()
        monthly_rate = dy.resample("ME").sum()
        monthly_ax.bar(
            x=monthly_rate.index,
            height=monthly_rate.T.values[0],
            width=monthly_rate.index.diff(),
            label="Monthly rate",
            color=colors[2],
            antialiased=True,
        )
        monthly_ax.set_ylabel(f"Monthly {dy.index.name.lower()} rate")
        monthly_ax.legend(loc="upper right")

        yearly_ax.plot(
            dy.resample("YE").sum(),
            label="Yearly rate",
            ls="--",
            c=colors[1],
            antialiased=True,
        )
        yearly_ax.legend(loc="upper left", ncols=int(len(dy.columns) / 8 + 1))

    else:
        dy2 = dy.resample("YE").sum()
        yearly_ax.stackplot(dy2.index, dy2.values.T, labels=dy2.columns, colors=colors, antialiased=True)
        if not cumsum:
            yearly_ax.legend(loc="upper left", ncols=int(len(dy.columns) / 8 + 1))

    if xlabel is not None:
        yearly_ax.set_xlabel(xlabel)
    else:
        yearly_ax.set_xticklabels([])

    for temp_ax in axes:
        temp_ax.set_xlim(
            [
                dy.index.min() - pd.Timedelta(weeks=13),
                dy.index.max() + pd.Timedelta(weeks=52),
            ]
        )
        temp_ax.xaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.yaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.xaxis.set_major_locator(mdates.YearLocator())
        temp_ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        temp_ax.set_axisbelow(True)
        temp_ax.grid()

    if len(dy.columns) == 1:
        align_yaxis(ax1=yearly_ax, v1=0, ax2=monthly_ax, v2=0)
        l = yearly_ax.get_ylim()
        l2 = monthly_ax.get_ylim()
        f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
        ticks = f(yearly_ax.get_yticks())
        monthly_ax.yaxis.set_major_locator(FixedLocator(ticks))
        monthly_ax.yaxis.set_minor_locator(AutoMinorLocator())
        axes.append(monthly_ax)

    return axes


def rate_subplots(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = None,
    title: str = "",
    xlabel: str = "Date",
    colors: List[str] = None,
    cumsum: bool = True,
    bg: pd.DataFrame = None,
    vlines: pd.DataFrame = None,
) -> None:
    """
    Creates a grid of subplots to visualize rates of change over time of multiple variables in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data to plot.
        figsize (Tuple[int, int] or None): The size of the figure to create. If None, default size is (16, len(df.columns)*2*(1+cumsum)).
        title (str): The title of the figure. Default is an empty string.
        xlabel (str): The label of the x-axis. Default is 'Date'.
        colors (List[str]): The list of colors to use for the lines. If None, default colors are used.
        cumsum (bool): Whether to plot the cumulative sum of the data. Default is True.
        bg (pd.DataFrame): A DataFrame containing the background shading data. Default is None.
        vlines (pd.DataFrame): A DataFrame containing the vertical line data. Default is None.

    Returns:
        None: Displays the generated plot.
    """
    if figsize is None:
        figsize = (12, len(df.columns) * 2 * (1 + cumsum))

    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=figsize, sharex=True)
    fig.suptitle(
        f'{title if title is not None else df.index.name.capitalize()}{f" by {df.columns.name.lower()}" if df.columns.name is not None else ""}',
        y=1,
    )

    if colors is None:
        cmap = plt.cm.tab20
    else:
        if len(colors) == len(df.columns):
            cmap = ListedColormap([adjust_lightness(color=c, amount=i * 0.5 + 0.75) for c in colors for i in (0, 1)])
        else:
            cmap = ListedColormap(colors)

    if bg is not None and all(col in bg.columns for col in ["begin", "end"]):
        bg = bg.copy()
        bg["end"] = bg["end"].fillna(df.index.max())
        sec_ax = axes[0].secondary_xaxis("top")
        sec_ax.set_xticks(bg.mean(axis=1))
        sec_ax.set_xticklabels(bg.index)

    c = 0
    for i, col in enumerate(df.columns):
        sub_axes = generate_rate_grid(
            dy=df[col].to_frame(),
            ax=axes[i],
            colors=[cmap(2 * c), cmap(2 * c), cmap(2 * c + 1)],
            size="100%",
            xlabel="Date" if (i + 1) == len(df.columns) else None,
            cumsum=cumsum,
        )

        for ix, ax in enumerate(sub_axes[:2]):
            if bg is not None and all(col in bg.columns for col in ["begin", "end"]):
                for idx, row in bg.iterrows():
                    if row["end"] > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                        filled_poly = ax.axvspan(
                            row["begin"],
                            row["end"],
                            alpha=0.1,
                            color=colors_dict[idx],
                            zorder=-1,
                        )

            if vlines is not None:
                for idx, row in vlines.items():
                    if row > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                        line = ax.axvline(row, ls="-.", c="maroon", lw=1)
                        if i == 0 and ix == 0:
                            (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                            ax.text(
                                (x0 + x1) / 2 + 25,
                                (0.02 if cumsum else 0.98),
                                idx,
                                c="maroon",
                                ha="left",
                                va=("bottom" if cumsum else "top"),
                                rotation=90,
                                transform=ax.get_xaxis_transform(),
                            )

        c += 1
        if 2 * c + 1 >= cmap.N:
            c = 0

    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )

    fig.tight_layout()
    plt.show()

    warnings.filterwarnings(
        action="default",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )


def rate(
    dy: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    title: str = None,
    xlabel: str = "Date",
    colors: List[str] = None,
    cumsum: bool = True,
    bg: pd.DataFrame = None,
    vlines: pd.DataFrame = None,
) -> None:
    """
    Creates a single plot to visualize the rate of change over time of a single variable in a pandas DataFrame.

    Args:
        dy (pd.DataFrame): The pandas DataFrame containing the data to plot.
        figsize (Tuple[int, int]): The size of the figure to create. Default is (16, 6).
        title (str): The title of the figure. Default is None.
        xlabel (str): The label of the x-axis. Default is 'Date'.
        colors (List[str]): The list of colors to use for the lines. If None, default colors are used.
        cumsum (bool): Whether to plot the cumulative sum of the data. Default is True.
        bg (pd.DataFrame): A DataFrame containing the background shading data. Default is None.
        vlines (pd.DataFrame): A DataFrame containing the vertical line data. Default is None.

    Returns:
        None: Displays the generated plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    fig.suptitle(
        f'{title if title is not None else dy.index.name.capitalize()}{f" by {dy.columns.name.lower()}" if dy.columns.name is not None else ""}'
    )

    axes = generate_rate_grid(dy=dy, ax=ax, size="100%", colors=colors, cumsum=cumsum)
    if bg is not None and all(col in bg.columns for col in ["begin", "end"]):
        bg = bg.copy()
        bg["end"] = bg["end"].fillna(dy.index.max())
        sec_ax = axes[0].secondary_xaxis("top")
        sec_ax.set_xticks(bg.mean(axis=1))
        sec_ax.set_xticklabels(bg.index)

    for i, ax in enumerate(axes[:2]):
        if bg is not None and all(col in bg.columns for col in ["begin", "end"]):
            for idx, row in bg.iterrows():
                if row["end"] > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                    filled_poly = ax.axvspan(
                        row["begin"],
                        row["end"],
                        alpha=0.1,
                        color=colors_dict[idx],
                        zorder=-1,
                    )

        if vlines is not None:
            for idx, row in vlines.items():
                if row > pd.to_datetime(ax.get_xlim()[0], unit="d"):
                    line = ax.axvline(row, ls="-.", c="maroon", lw=1)
                    if i == 0:
                        (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                        ax.text(
                            (x0 + x1) / 2 + 25,
                            (0.02 if cumsum or len(dy.columns) > 1 else 0.98),
                            idx,
                            c="maroon",
                            ha="left",
                            va=("bottom" if cumsum or len(dy.columns) > 1 else "top"),
                            rotation=90,
                            transform=ax.get_xaxis_transform(),
                        )

    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )

    fig.tight_layout()
    plt.show()

    warnings.filterwarnings(
        action="default",
        category=UserWarning,
        message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    )


def arrows(arrows: pd.Series, figsize: Tuple[int, int] = (6, 6), **kwargs) -> None:
    """
    Create a polar plot to visualize the frequency of each arrow direction in a pandas Series.

    Args:
        arrows (pandas.Series): A pandas Series containing arrow symbols as string data type.
        figsize (Tuple[int, int], optional): The width and height of the figure. Defaults to (6, 6).
        **kwargs: Additional keyword arguments to be passed to the bar() method.

    Returns:
        None: Displays the generated plot.
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

    # Display the plot
    fig.tight_layout()
    plt.show()


def box(df, mean=True, **kwargs) -> None:
    """
    Plots a box plot of a given DataFrame using seaborn, with the year of the Release column on the x-axis and the remaining column on the y-axis.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the Release dates and another numeric column.
        mean (bool, optional): If True, plots a line representing the mean of each box. Defaults to True.
        **kwargs: Additional keyword arguments to pass to seaborn.boxplot().

    Returns:
        None

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
    plt.show()
