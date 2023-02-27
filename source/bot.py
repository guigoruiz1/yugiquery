# -*- coding: utf-8 -*-

__author__ = "Guilherme Ruiz"
__copyright__ = "2023, Guilherme Ruiz"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Guilherme Ruiz"
__email__ = "57478888+guigoruiz1@users.noreply.github.com"
__status__ = "Development"

# ======= #
# Imports #
# ======= #

import yugiquery as yq

# native python packages
import os
import glob
import random
import subprocess
import discord
import asyncio
import io
import re
import json

# PIP packages
import pandas as pd
from enum import Enum
import multiprocessing as mp
from discord.ext import commands
from dotenv import dotenv_values
from tqdm.contrib.discord import tqdm as discord_pbar

# ======= #
# Helpers #
# ======= #

# Data loaders
def load_secrets(secrets_file):
    """
    Loads secrets from the specified file using dotenv_values.
    If the file is not found, or if any of the required secrets are missing or empty, the function exits the program.
    
    Args:
        secrets_file (str): Path to the secrets file.
        
    Returns:
        dict: A dictionary containing the loaded secrets.
        
    Raises:
        SystemExit: If the secrets file is not found or if any of the required secrets are missing or empty.
    """
    required_secrets = ['DISCORD_TOKEN','DISCORD_CHANNEL_ID']
    if os.path.isfile(secrets_file):
        secrets=dotenv_values(secrets_file)
        if all(key in secrets.keys() for key in required_secrets) and all(secrets[key] for key in required_secrets):
            return secrets
    print(secrets_file)
    print(os.getcwd)
    print('Secrets not found. Exiting...')
    exit()

def init_reports_enum():
    """
    Initializes and returns an Enum object containing the available reports.
    The reports are read from the yugiquery.SCRIPT_DIR directory, where they are expected to be Jupyter notebooks.
    The Enum object is created using the reports' file names, with the .ipynb extension removed and the first letter capitalized.
    
    Returns:
        Enum: An Enum object containing the available reports.
    """
    reports_dict = {'All': 'all'}
    reports = sorted(glob.glob(os.path.join(yq.SCRIPT_DIR,'*.ipynb')))
    for report in reports:
        report = os.path.basename(report)
        reports_dict[report[:-6].capitalize()] = report

    return  Enum('Reports', reports_dict)

# ========= #
# Variables #
# ========= #

# URLs
repository_api_url = "https://api.github.com/repos/guigoruiz1/yugiquery"
repository_url = 'https://github.com/guigoruiz1/yugiquery'
webpage_url = 'https://guigoruiz1.github.io/yugiquery'

# Secrets
secrets = load_secrets(os.path.join(yq.PARENT_DIR,'assets/secrets.env'))

# Discord API
intents = discord.Intents(messages=True, guilds=True, members=True)
bot = commands.Bot(command_prefix='/', intents=intents)

# Other
Reports = init_reports_enum()
process = None

# ======== #
# Commands #
# ======== #

@bot.hybrid_command(name='shutdown', description='Shutdown bot', with_app_command = True)
@commands.is_owner()
async def shutdown(ctx):
    """
    Shuts down the bot gracefully by sending a message and closing the connection.
    
    Args:
        ctx (commands.Context): The context of the command.
    """
    await ctx.send(content='Shutting down...')
    await bot.close()

@bot.hybrid_command(name='run', description='Run full Yugiquery workflow', with_app_command = True)
@commands.is_owner()
@commands.cooldown(1, 12*60*60, commands.BucketType.user)
async def run(ctx, report: Reports):
    """
    Runs a Yugiquery workflow by launching a separate process and monitoring its progress.
    The progress is reported back to the Discord channel where the command was issued.
    The command has a cooldown period of 12 hours per user.
    
    Args:
        ctx (commands.Context): The context of the command.
        report (Reports): An Enum value indicating which Yugiquery report to run.
        
    Raises:
        discord.ext.commands.CommandOnCooldown: If the command is on cooldown for the user.
    """
    global process
    if process is not None:
        await ctx.send(
            content='Query already running. Try again after it has finished.', 
            ephemeral=True,
            delete_after=60
        )
        return
    
    original_response = await ctx.send(
        content='Initializing...', 
        ephemeral=True, 
        delete_after=60
    )
    
    API_error = False
    def progress_handler(iterable=None, API_status=None, **kwargs):
        if iterable and ctx.channel.id != int(secrets['DISCORD_CHANNEL_ID']):
                return discord_pbar(
                    iterable, 
                    token = secrets['DISCORD_TOKEN'], 
                    channel_id=ctx.channel.id, 
                    file=io.StringIO(),
                    **kwargs
                )
        elif API_status is not None:
            nonlocal API_error
            API_error = not API_status
            return
        
    try:
        process = mp.Process(
            target=yq.run, 
            args=[report.value, progress_handler]
        )
        process.start()
        await original_response.edit(content='Running...')
    except:
        await original_response.edit(content='Initialization failed!')
    
    async def await_result():
        while process.is_alive():
            await asyncio.sleep(1)
        return process.exitcode
    
    exitcode = await await_result()
    process.close()
    process = None
    
    if API_error:
        await ctx.channel.send(content='Unable to comunicate to the API. Try again later.') 
    else:
        if exitcode is None:
            await ctx.channel.send(content='Query execution failed!') 
        elif exitcode==0:
            await ctx.channel.send(content='Query execution completed!')
        elif exitcode==-15:
            await ctx.channel.send(content='Query execution aborted!')
        else:
            await ctx.channel.send(content=f'Query execution exited with exit code: {exitcode}')
        
    # Reset cooldown in case query did not complete
    if API_error or exitcode!=0:
        ctx.command.reset_cooldown(ctx)
    
        
@bot.hybrid_command(name='abort', description='Abort running Yugiquery workflow', with_app_command=True)
@commands.is_owner()
async def abort(ctx):
    """
    Aborts a running Yugiquery workflow by terminating the process.
    
    Args:
        ctx (commands.Context): The context of the command.
    """
    original_response = await ctx.send(
        content='Aborting...', 
        ephemeral=True, 
        delete_after=60
    )
    
    try:
        process.terminate()
        await original_response.edit(content='Aborted')
    except:
        await original_response.edit(content='Abort failed')

@bot.hybrid_command(name='benchmark', description='Show average time each report takes to complete', with_app_command=True)
async def benchmark(ctx): # Improve function
    """
    Returns the average time each report takes to complete and the latest time for each report.

    Args:
        ctx (discord_slash.context.SlashContext): The context of the slash command.

    Returns:
        None: Sends an embed message with the benchmark data.
    """
    await ctx.defer()
    try:
        with open(os.path.join(yq.PARENT_DIR,'data/benchmark.json'), 'r') as file:
            data = json.load(file)
    except:
        await ctx.send('Unable to find benchmark records at this time. Try again later.')
        return

    embed = discord.Embed(
        title='Benchmark', 
        description='The average time each report takes to complete', 
        color=discord.Colour.blue()
    )
    # Get benchmark
    value=''
    for key, values in data.items():
        weighted_sum = 0
        total_weight = 0
        for entry in values:
            weighted_sum+= entry['average']*entry['weight']
            total_weight+= entry['weight']
            
        avg_time = pd.Timestamp(weighted_sum/total_weight, unit='s')
        latest_time = pd.Timestamp(entry['average'], unit='s')
        
        avg_time_str = f"{avg_time.strftime('%-M')} minutes and {avg_time.strftime('%-S.%f')} seconds" if avg_time.minute > 0 else f"{avg_time.strftime('%-S.%f')} seconds"
        latest_time_str = f"{latest_time.strftime('%-M')} minutes and {latest_time.strftime('%-S.%f')} seconds" if latest_time.minute > 0 else f"{latest_time.strftime('%-S.%f')} seconds"
        
        value = f"• Average: {avg_time_str}\n• Latest: {latest_time_str}"
        embed.add_field(
            name=key.capitalize(), 
            value=value, 
            inline=False
        )
        
    await ctx.send(embed=embed)
        
@bot.hybrid_command(name='latest', description='Show latest time each report was generated', with_app_command=True)
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
    reports = sorted(glob.glob(yq.PARENT_DIR))
    embed = discord.Embed(
        title='Latest reports generated', 
        description='The live reports may not always be up to date with the local reports', 
        color=discord.Colour.blue()
    )
    
    # Get local files timestamps
    local_value=''
    for report in reports:
        local_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(os.path.getmtime(report),unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'
    
    embed.add_field(
        name='Local:', 
        value=local_value, 
        inline=False
    )
     
    # Get live files timestamps
    try:
        live_value=''
        for report in reports:
            result = pd.read_json(f'{repository_api_url}/commits?path={os.path.basename(report)}')
            timestamp = pd.DataFrame(result.loc[0,'commit']).loc['date','author']
            live_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(timestamp, utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

        embed.add_field(
            name='Live:', 
            value=live_value, 
            inline=False
        )
    except:
        pass
        
    await ctx.send(embed=embed)
    
@bot.hybrid_command(name='links', description='Show Yugiquery links', with_app_command=True)
async def links(ctx):
    """
    Displays the links to the Yugiquery webpage, repository, and data. Returns the links as an embedded message in
    the channel.

    Args:
        ctx (discord.ext.commands.Context): The context of the command.

    Returns:
        None
    """
    embed = discord.Embed(
        title="Yugiquery links",
        description=f'[Webpage]({webpage_url}) • [Repository]({repository_url}) • [Data]({repository_url}/tree/main/data)',
        color=discord.Colour.blue()
    )
    
    await ctx.send(embed=embed)
    
@bot.hybrid_command(name='data', description='Send latest data files', with_app_command=True)
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
            description='Direct links to download files from GitHub.',
            color=discord.Colour.blue()
        )
        
        files = pd.read_json(f'{repository_api_url}/contents/data')
        files = files[files['name'].str.endswith('.csv')] # Remove .json files from lists
        files['Group'] = files['name'].apply(lambda x: re.search(r'(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).csv', x).group(1))
        files['Timestamp'] = files['name'].apply(lambda x: re.search(r'(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).csv', x).group(3))
        files['Timestamp'] = pd.to_datetime(files['Timestamp'], utc=True)
        index = files.groupby('Group')['Timestamp'].idxmax()
        latest_files = files.loc[index,['name','download_url']]

        data_value=''
        changelog_value=''
        for idx, file in latest_files.iterrows():
            if 'changelog' in file["name"]:
                changelog_value+=f'• [{file["name"]}]({file["download_url"]})\n'
            else:
                data_value+=f'• [{file["name"]}]({file["download_url"]})\n'
        
        embed.add_field(
            name='Data:', 
            value=data_value, 
            inline=False
        )
        embed.add_field(
            name='Changelog:', 
            value=changelog_value, 
            inline=False
        )
        await ctx.send(embed=embed)
        
    except:
        await ctx.send('Unable to obtain latest files at this time. Try again later.')

@bot.hybrid_command(name='ping', description='Test the bot connection latency', with_app_command=True)
async def ping(ctx):
    """
    This command tests the bot's connection latency and sends the result back to the user.

    Parameters:
        ctx (discord_slash.context.SlashContext): The context of the slash command.

    Returns:
        None
    """
    await ctx.send(
        content='🏓 Pong! {0}ms'.format(round(bot.latency*1000, 1)), 
        ephemeral=True, 
        delete_after=60
    )

# ====== #
# Events #
# ====== #

@bot.event
async def on_ready():
    """
    Event that runs when the bot is ready to start receiving events and commands. 
    Prints out the bot's username and the guilds it's connected to.
    """
    print('You are logged as {0.user}'.format(bot))
    await bot.tree.sync()
    
    print(f'{bot.user} is connected to the following guilds:')
    for guild in bot.guilds: 
        print(f'{guild.name}(id: {guild.id})')
        members = '\n - '.join([member.name for member in guild.members])
        print(f'Guild Members:\n - {members}')
        
@bot.event
async def on_message(message):
    """
    Event that runs whenever a message is sent in a server where the bot is present. 
    Responds with a greeting to any message starting with 'hi'.
    """
    if message.author == bot.user:
        return
     
    await bot.process_commands(message)
    if message.content.lower().startswith('hi'):
        await message.channel.send(content=f'Hello, {message.author.name}!')

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
    bot.run(secrets['DISCORD_TOKEN'])