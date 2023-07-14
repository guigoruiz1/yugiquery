# -*- coding: utf-8 -*-

# ======= #
# Imports #
# ======= #

import yugiquery as yq

__author__ = yq.__author__
__copyright__ = yq.__copyright__
__license__ = yq.__license__
__version__ = yq.__version__
__maintainer__ = yq.__maintainer__
__email__ = yq.__email__
__status__ = yq.__status__

# from yugiquery import os, glob, subprocess, io, re, json, Enum, datetime, timezone, git, pd, dotenv_values

# Native python packages
import argparse
import os
import glob
import random
import subprocess
import platform
import asyncio
import io
import re
import json
from enum import Enum
from datetime import datetime, timezone

# PIP packages - installed by yugiquery
import git
import discord
from discord.ext import commands
import pandas as pd
import multiprocessing as mp
from dotenv import dotenv_values
from tqdm.contrib.discord import tqdm as discord_pbar
from tqdm.auto import tqdm, trange


# ======= #
# Helpers #
# ======= #


# Data loaders
def load_secrets_with_args(args):
    """
    Load secrets from command-line arguments, and update them with values from
    environment variables or a .env file if necessary.

    Args:
        args:

    Returns:
        dict: A dictionary containing the loaded secrets.

    Raises:
        KeyError: If a required secret is not found in the loaded secrets.
    """
    secrets = {key: value for key, value in args.items() if value is not None}
    missing = [key for key, value in args.items() if value is None]
    if len(missing) > 0:
        try:
            loaded_secrets = yq.load_secrets(
                requested_secrets=missing,
                secrets_file=os.path.join(yq.PARENT_DIR, "assets/secrets.env"),
                required=True,
            )
        except:
            print("Secrets not found. Exiting...")
            exit()

        secrets = secrets | loaded_secrets

    return secrets


def load_repo_vars():
    # Open the repository
    repo = git.Repo(yq.PARENT_DIR, search_parent_directories=True)

    # Get the remote repository
    remote = repo.remote()
    remote_url = remote.url

    # Extract the GitHub page URL from the remote URL
    # by removing the ".git" suffix and splitting the URL
    # by the "/" character
    remote_url_parts = remote_url[:-4].split("/")

    return remote_url_parts[-2:]


def init_reports_enum():
    """
    Initializes and returns an Enum object containing the available reports.
    The reports are read from the yugiquery.SCRIPT_DIR directory, where they are expected to be Jupyter notebooks.
    The Enum object is created using the reports' file names, with the .ipynb extension removed and the first letter capitalized.

    Returns:
        Enum: An Enum object containing the available reports.
    """
    reports_dict = {"All": "all"}
    reports = sorted(glob.glob(os.path.join(yq.SCRIPT_DIR, "*.ipynb")))
    for report in reports:
        reports_dict[os.path.basename(report)[:-6].capitalize()] = report

    return Enum("Reports", reports_dict)


# ========= #
# Variables #
# ========= #

# Repository
(author, repo) = load_repo_vars()
## URLs
repository_api_url = f"https://api.github.com/repos/{author}/{repo}"
repository_url = f"https://github.com/{author}/{repo}"
webpage_url = f"https://{author}.github.io/{repo}"

# Discord API
intents = discord.Intents(messages=True, guilds=True, members=True)
bot = commands.Bot(command_prefix="/", intents=intents)

# Other
Reports = init_reports_enum()
process = None

# ======== #
# Commands #
# ======== #


@bot.hybrid_command(name="shutdown", description="Shutdown bot", with_app_command=True)
@commands.is_owner()
async def shutdown(ctx):
    """
    Shuts down the bot gracefully by sending a message and closing the connection.

    Args:
        ctx (commands.Context): The context of the command.
    """
    await ctx.send(content="Shutting down...")
    await bot.close()


@bot.hybrid_command(
    name="run", description="Run full YugiQuery workflow", with_app_command=True
)
@commands.is_owner()
@commands.cooldown(1, 12 * 60 * 60, commands.BucketType.user)
async def run(ctx, report: Reports = Reports.All):
    """
    Runs a YugiQuery workflow by launching a separate process and monitoring its progress.
    The progress is reported back to the Discord channel where the command was issued.
    The command has a cooldown period of 12 hours per user.

    Args:
        ctx (commands.Context): The context of the command.
        report (Reports): An Enum value indicating which YugiQuery report to run.

    Raises:
        discord.ext.commands.CommandOnCooldown: If the command is on cooldown for the user.
    """
    global process
    if process is not None:
        await ctx.send(
            content="Query already running. Try again after it has finished.",
            ephemeral=True,
            delete_after=60,
        )
        return

    original_response = await ctx.send(
        content="Initializing...", ephemeral=True, delete_after=60
    )

    queue = mp.Queue()

    def progress_handler(iterable=None, API_status: bool = True, **kwargs):
        queue.put(API_status)
        if iterable and ctx.channel.id != int(secrets["DISCORD_CHANNEL_ID"]):
            return discord_pbar(
                iterable,
                token=secrets["DISCORD_TOKEN"],
                channel_id=ctx.channel.id,
                file=io.StringIO(),
                **kwargs,
            )

    try:
        process = mp.Process(target=yq.run, args=[report.value, progress_handler])
        process.start()
        await original_response.edit(content="Running...")
    except:
        await original_response.edit(content="Initialization failed!")

    async def await_result():
        while process.is_alive():
            await asyncio.sleep(1)
        API_error = False
        while not queue.empty():
            API_error = not queue.get()
        return process.exitcode, API_error

    exitcode, API_error = await await_result()
    process.close()
    process = None

    if API_error:
        await ctx.channel.send(
            content="Unable to comunicate to the API. Try again later."
        )
    else:
        if exitcode is None:
            await ctx.channel.send(content="Query execution failed!")
        elif exitcode == 0:
            await ctx.channel.send(content="Query execution completed!")
        elif exitcode == -15:
            await ctx.channel.send(content="Query execution aborted!")
        else:
            await ctx.channel.send(
                content=f"Query execution exited with exit code: {exitcode}"
            )

    # Reset cooldown in case query did not complete
    if API_error or exitcode != 0:
        ctx.command.reset_cooldown(ctx)


@bot.hybrid_command(
    name="abort", description="Abort running YugiQuery workflow", with_app_command=True
)
@commands.is_owner()
async def abort(ctx):
    """
    Aborts a running YugiQuery workflow by terminating the process.

    Args:
        ctx (commands.Context): The context of the command.
    """
    original_response = await ctx.send(
        content="Aborting...", ephemeral=True, delete_after=60
    )

    try:
        process.terminate()
        await original_response.edit(content="Aborted")
    except:
        await original_response.edit(content="Abort failed")


@bot.hybrid_command(
    name="benchmark",
    description="Show average time each report takes to complete",
    with_app_command=True,
)
async def benchmark(ctx):  # Improve function
    """
    Returns the average time each report takes to complete and the latest time for each report.

    Args:
        ctx (discord_slash.context.SlashContext): The context of the slash command.

    Returns:
        None: Sends an embed message with the benchmark data.
    """
    await ctx.defer()
    try:
        with open(os.path.join(yq.PARENT_DIR, "data/benchmark.json"), "r") as file:
            data = json.load(file)
    except:
        await ctx.send(
            "Unable to find benchmark records at this time. Try again later."
        )
        return

    embed = discord.Embed(
        title="Benchmark",
        description="The average time each report takes to complete",
        color=discord.Colour.gold(),
    )
    # Get benchmark
    value = ""
    for key, values in data.items():
        weighted_sum = 0
        total_weight = 0
        for entry in values:
            weighted_sum += entry["average"] * entry["weight"]
            total_weight += entry["weight"]

        avg_time = pd.Timestamp(weighted_sum / total_weight, unit="s")
        latest_time = pd.Timestamp(entry["average"], unit="s")

        avg_time_str = (
            f"{avg_time.strftime('%-M')} minutes and {avg_time.strftime('%-S.%f')} seconds"
            if avg_time.minute > 0
            else f"{avg_time.strftime('%-S.%f')} seconds"
        )
        latest_time_str = (
            f"{latest_time.strftime('%-M')} minutes and {latest_time.strftime('%-S.%f')} seconds"
            if latest_time.minute > 0
            else f"{latest_time.strftime('%-S.%f')} seconds"
        )

        value = f"• Average: {avg_time_str}\n• Latest: {latest_time_str}"
        embed.add_field(name=key.capitalize(), value=value, inline=False)

    await ctx.send(embed=embed)


@bot.hybrid_command(
    name="latest",
    description="Show latest time each report was generated",
    with_app_command=True,
)
async def latest(ctx):
    """
    Displays the timestamp of the latest local and live reports generated. Reads the report files from `yq.PARENT_DIR` and
    queries the GitHub API for the latest commit timestamp for each file. Returns the result as an embedded message in
    the channel.

    Args:
        ctx (discord.ext.commands.Context): The context of the command.

    Raises:
        `discord.ext.commands.CommandInvokeError`: If unable to find local or live reports.

    Returns:
        None
    """
    await ctx.defer()
    reports = sorted(glob.glob(os.path.join(yq.PARENT_DIR, "*.html")))
    embed = discord.Embed(
        title="Latest reports generated",
        description="The live reports may not always be up to date with the local reports",
        color=discord.Colour.orange(),
    )

    # Get local files timestamps
    local_value = ""
    for report in reports:
        local_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(os.path.getmtime(report),unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

    embed.add_field(name="Local", value=local_value, inline=False)

    # Get live files timestamps
    try:
        live_value = ""
        for report in reports:
            result = pd.read_json(
                f"{repository_api_url}/commits?path={os.path.basename(report)}"
            )
            timestamp = pd.DataFrame(result.loc[0, "commit"]).loc["date", "author"]
            live_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(timestamp, utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

        embed.add_field(name="Live", value=live_value, inline=False)
    except:
        pass

    await ctx.send(embed=embed)


@bot.hybrid_command(
    name="links", description="Show YugiQuery links", with_app_command=True
)
async def links(ctx):
    """
    Displays the links to the YugiQuery webpage, repository, and data. Returns the links as an embedded message in
    the channel.

    Args:
        ctx (discord.ext.commands.Context): The context of the command.

    Returns:
        None
    """
    embed = discord.Embed(
        title="YugiQuery links",
        description=f"[Webpage]({webpage_url}) • [Repository]({repository_url}) • [Data]({repository_url}/tree/main/data)",
        color=discord.Colour.green(),
    )

    await ctx.send(embed=embed)


@bot.hybrid_command(
    name="data", description="Send latest data files", with_app_command=True
)
async def data(ctx):
    """
    This command sends the latest data files available in the repository as direct download links.

    Parameters:
        ctx (discord_slash.context.SlashContext): The context of the slash command.

    Returns:
        None
    """
    await ctx.defer()

    try:
        embed = discord.Embed(
            title="Latest data files",
            description="Direct links to download files from GitHub.",
            color=discord.Colour.magenta(),
        )

        files = pd.read_json(f"{repository_api_url}/contents/data")
        files = files[
            files["name"].str.endswith(".bz2")
        ]  # Remove .json files from lists
        files["Group"] = files["name"].apply(
            lambda x: re.search(
                r"(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).bz2", x
            ).group(1)
        )
        files["Timestamp"] = files["name"].apply(
            lambda x: re.search(
                r"(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).bz2", x
            ).group(3)
        )
        files["Timestamp"] = pd.to_datetime(files["Timestamp"], utc=True)
        index = files.groupby("Group")["Timestamp"].idxmax()
        latest_files = files.loc[index, ["name", "download_url"]]

        data_value = ""
        changelog_value = ""
        for idx, file in latest_files.iterrows():
            if "changelog" in file["name"]:
                changelog_value += f'• [{file["name"]}]({file["download_url"]})\n'
            else:
                data_value += f'• [{file["name"]}]({file["download_url"]})\n'

        embed.add_field(name="Data", value=data_value, inline=False)
        embed.add_field(name="Changelog", value=changelog_value, inline=False)
        await ctx.send(embed=embed)

    except:
        await ctx.send("Unable to obtain latest files at this time. Try again later.")


@bot.hybrid_command(
    name="ping", description="Test the bot connection latency", with_app_command=True
)
async def ping(ctx):
    """
    This command tests the bot's connection latency and sends the result back to the user.

    Parameters:
        ctx (discord_slash.context.SlashContext): The context of the slash command.

    Returns:
        None
    """
    await ctx.send(
        content="🏓 Pong! {0}ms".format(round(bot.latency * 1000, 1)),
        ephemeral=True,
        delete_after=60,
    )


@bot.hybrid_command(
    name="battle",
    description="Simulate a battle of all monster cards",
    with_app_command=True,
)
@commands.is_owner()
async def battle(ctx, atk_weight: int = 4, def_weight: int = 1):
    """
    This function loads the list of all Monster Cards and simulates a battle between them. Each card is represented by its name, attack (ATK), and defense (DEF) stats. At the beginning of the battle, a random card is chosen as the initial contestant. Then, for each subsequent card, a random stat (ATK or DEF) is chosen to compare with the corresponding stat of the current winner. If the challenger's stat is higher, the challenger becomes the new winner. If the challenger's stat is lower, the current winner retains its position. If the stats are tied, the comparison is repeated with the other stat. The battle continues until there is only one card left standing.

    Args:
        ctx (commands.Context): The context of the command that triggered the function.
        atk_weight (int, optional): The weight to use for the ATK stat when randomly choosing the monster's stat to compare. This affects the probability that ATK will be chosen over DEF. The default value is 4.
        def_weight (int, optional): The weight to use for the DEF stat when randomly choosing the monster's stat to compare. This affects the probability that DEF will be chosen over ATK. The default value is 1.

    Returns:
        None
    """
    await ctx.defer()

    MONSTER_STATS = ["Name", "ATK", "DEF"]
    weights = [atk_weight, def_weight]
    cards_files = sorted(
        glob.glob(os.path.join(yq.PARENT_DIR, "data/all_cards_*.bz2")),
        key=os.path.getmtime,
    )
    if not cards_files:
        await ctx.send(
            content="Cards data not found... Try again later.",
            ephemeral=True,
            delete_after=60,
        )
        return

    cards = pd.read_csv(cards_files[0])

    monsters = cards[
        (cards["Card type"] == "Monster Card")
        & (cards["Primary type"] != "Monster Token")
    ][MONSTER_STATS].set_index("Name")
    monsters = monsters.applymap(
        lambda x: x if x != "?" else random.randrange(0, 51) * 100
    )
    monsters = monsters.applymap(pd.to_numeric, errors="coerce").fillna(0).reset_index()

    # Shuffle the monsters and select the first one as the initial winner
    monsters = monsters.sample(frac=1).reset_index(drop=True)
    winner = (monsters.iloc[0], 0)
    longest = (monsters.iloc[0], 0)

    embed = discord.Embed(
        title="Battle",
        description="Simulate a battle of all monster cards",
        color=discord.Colour.purple(),
    )

    embed.add_field(name="First contestant", value=winner[0]["Name"], inline=False)
    embed.set_footer(text="Still battling... ⏳")
    original_response = await ctx.send(embed=embed)

    iterator = trange(1, len(monsters), desc="Battle")
    for i in iterator:
        current_winner = (winner[0].copy(), winner[1])
        next_monster = monsters.iloc[i].copy()
        chosen_stat = random.choices(MONSTER_STATS[1:], weights=weights)[0]
        iterator.set_postfix(winner=current_winner[0]["Name"])
        not_chosen_stat = [stat for stat in MONSTER_STATS[1:] if stat != chosen_stat][0]
        if next_monster[chosen_stat] > current_winner[0][chosen_stat]:
            next_monster[chosen_stat] -= current_winner[0][chosen_stat]
            if current_winner[1] > longest[1]:
                longest = current_winner
            current_winner = (next_monster, 0)
        elif next_monster[chosen_stat] < current_winner[0][chosen_stat]:
            current_winner[0][chosen_stat] -= next_monster[chosen_stat]
        elif next_monster[chosen_stat] == current_winner[0][chosen_stat]:
            if next_monster[not_chosen_stat] > current_winner[0][not_chosen_stat]:
                next_monster[not_chosen_stat] -= current_winner[0][not_chosen_stat]
                if current_winner[1] > longest[1]:
                    longest = current_winner
                current_winner = (next_monster, 0)
            elif next_monster[not_chosen_stat] < current_winner[0][not_chosen_stat]:
                current_winner[0][not_chosen_stat] -= next_monster[not_chosen_stat]

        winner = (current_winner[0], current_winner[1] + 1)

    embed.add_field(name="Winner", value=winner[0]["Name"], inline=True)
    embed.add_field(name="Wins", value=winner[1], inline=True)
    embed.add_field(
        name="Stats remaining",
        value=f'ATK={winner[0]["ATK"]}, DEF={winner[0]["DEF"]}',
        inline=True,
    )

    embed.add_field(name="Longest streak", value=longest[0]["Name"], inline=True)
    embed.add_field(name="Wins", value=longest[1], inline=True)
    embed.add_field(
        name="Stats when defeated",
        value=f'ATK={longest[0]["ATK"]}, DEF={longest[0]["DEF"]}',
        inline=True,
    )
    embed.remove_footer()
    await original_response.edit(embed=embed)


@bot.hybrid_command(
    name="status",
    description="Display bot status and system information",
    with_app_command=True,
)
async def status(ctx):
    """
    Displays information about the bot, including uptime, guilds, users, channels, available commands,
    bot version, discord.py version, python version, and operating system.

    Args:
        ctx (commands.Context): The context of the command that triggered the function.

    Returns:
        None
    """
    uptime = datetime.now() - bot.start_time

    appInfo = await bot.application_info()
    admin = appInfo.owner
    users = 0
    channels = 0
    guilds = len(bot.guilds)
    for guild in bot.guilds:
        users += len(guild.members)
        channels += len(guild.channels)

    if len(bot.commands):
        commandsInfo = "\n".join(sorted([i.name for i in bot.commands]))

    embed = discord.Embed(color=ctx.me.colour)
    embed.set_footer(text="Time to duel!")
    embed.set_thumbnail(url=ctx.me.avatar)
    embed.add_field(name="Admin", value=admin, inline=False)
    embed.add_field(name="Uptime", value=uptime, inline=False)
    embed.add_field(name="Guilds", value=guilds, inline=True)
    embed.add_field(name="Users", value=users, inline=True)
    embed.add_field(name="Channels", value=channels, inline=True)
    embed.add_field(name="Available Commands", value=commandsInfo, inline=True)
    embed.add_field(name="Bot Version", value=__version__, inline=True)
    embed.add_field(name="Discord.py Version", value=discord.__version__, inline=True)
    embed.add_field(name="Python Version", value=platform.python_version(), inline=True)
    embed.add_field(
        name="Operating System",
        value=f"System: {platform.system()}\nRelease: {platform.release()}\nMachine: {platform.machine()}\nVersion: {platform.version()}",
        inline=False,
    )
    await ctx.send("**:information_source:** Information about this bot:", embed=embed)


# ====== #
# Events #
# ====== #


@bot.event
async def on_ready():
    """
    Event that runs when the bot is ready to start receiving events and commands.
    Prints out the bot's username and the guilds it's connected to.
    """
    bot.start_time = datetime.now()
    print("You are logged as {0.user}".format(bot))
    await bot.tree.sync()

    print(f"{bot.user} is connected to the following guilds:")
    for guild in bot.guilds:
        print(f"{guild.name}(id: {guild.id})")
        members = "\n - ".join([member.name for member in guild.members])
        print(f"Guild Members:\n - {members}")


@bot.event
async def on_message(message):
    """
    Event that runs whenever a message is sent in a server where the bot is present.
    Responds with a greeting to any message starting with 'hi'.
    """
    if message.author == bot.user:
        return

    await bot.process_commands(message)
    if message.content.lower().startswith("hi"):
        await message.channel.send(content=f"Hello, {message.author.name}!")


@bot.event
async def on_command_error(ctx, error):
    """
    Event that runs whenever a command invoked by the user results in an error.
    Sends a message to the channel indicating the type of error that occurred.
    """
    print(error)
    if isinstance(error, commands.errors.CommandOnCooldown):
        await ctx.send(content=error)
    elif isinstance(error, commands.errors.NotOwner):
        await ctx.send(content=error)
    elif isinstance(error, commands.errors.CheckFailure):
        await ctx.send(content=error)


# ========= #
# Execution #
# ========= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--token", dest="DISCORD_TOKEN", type=str, help="Discord API token"
    )
    parser.add_argument(
        "-c",
        "--channel",
        dest="DISCORD_CHANNEL_ID",
        type=int,
        help="Discord channel id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enable debug flag",
    )
    args = vars(parser.parse_args())
    debug = args.pop("debug", False)

    # Load secrets
    secrets = load_secrets_with_args(args)
    # Run
    bot.run(secrets["DISCORD_TOKEN"])
