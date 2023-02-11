import os
import glob
import random
import subprocess
import discord
import asyncio
import io
import re
import pandas as pd
from enum import Enum
import multiprocessing as mp
import yugiquery as yq
from discord.ext import commands
from dotenv import dotenv_values
from tqdm.contrib.discord import tqdm as discord_pbar

def load_secrets(secrets_file):
    required_secrets = ['DISCORD_TOKEN','DISCORD_CHANNEL_ID']
    if os.path.isfile(secrets_file):
        secrets=dotenv_values(secrets_file)
        if all(key in secrets.keys() for key in required_secrets) and all(secrets[key] for key in required_secrets):
            return secrets
    
    print('Secrets not found. Exiting...')
    exit()

def init_reports_enum():
    reports = sorted(glob.glob('*.ipynb'))
    if reports:
        reports = {report[:-6].capitalize(): report for report in sorted(reports)}
    else:
        reports = {}

    reports['All'] = 'all'
    return  Enum('DynamicEnum', reports)

repository_api_url = "https://api.github.com/repos/guigoruiz1/yugiquery"
repository_url = 'https://github.com/guigoruiz1/yugiquery'
webpage_url = 'https://guigoruiz1.github.io/yugiquery'
secrets = load_secrets('../assets/secrets.env')
intents = discord.Intents(messages=True, guilds=True, members=True)
bot = commands.Bot(command_prefix='/', intents=intents)
Reports = init_reports_enum()
process = None

@bot.hybrid_command(name='shutdown', description='Shutdown bot', with_app_command = True)
@commands.is_owner()
async def shutdown(ctx):
    await ctx.send(content='Shutting down...')
    await bot.close()

@bot.hybrid_command(name='run', description='Run full Yugiquery workflow', with_app_command = True)
@commands.is_owner()
@commands.cooldown(1, 12*60*60, commands.BucketType.user)
async def run(ctx, report: Reports):
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
    if API_error or exitcode!=1:
        ctx.command.reset_cooldown(ctx)
    
        
@bot.hybrid_command(name='abort', description='Abort running Yugiquery workflow', with_app_command=True)
@commands.is_owner()
async def abort(ctx):
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

@bot.hybrid_command(name='latest', description='Show latest time each report was generated', with_app_command=True)
async def latest(ctx):
    await ctx.defer()
    reports = sorted(glob.glob('../*.html'))
    embed = discord.Embed(
        title='Latest reports generated', 
        description='The live reports may not always be up to date with the local reports', 
        color=discord.Colour.blue()
    )
    
    # Get local files timestamps
    local_value=''
    for report in reports:
        local_value += f'• {os.path.basename(report).rstrip(".html")}: {pd.to_datetime(os.path.getmtime(report),unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'
    
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
            live_value += f'• {os.path.basename(report).rstrip(".html")}: {pd.to_datetime(timestamp, utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'

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
    
    embed = discord.Embed(
        title="Yugiquery links",
        description=f'[Webpage]({webpage_url}) • [Repository]({repository_url}) • [Data]({repository_url}/tree/main/data)',
        color=discord.Colour.blue()
    )
    
    await ctx.send(embed=embed)
    
@bot.hybrid_command(name='data', description='Send latest data files', with_app_command=True)
async def data(ctx):
    await ctx.defer()
    
    try:
        embed = discord.Embed(
            title="Latest data files",
            description='Direct links to download files from GitHub.',
            color=discord.Colour.blue()
        )
        
        files = pd.read_json(f'{repository_api_url}/contents/data')
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
    await ctx.send(
        content='🏓 Pong! {0}ms'.format(round(bot.latency*1000, 1)), 
        ephemeral=True, 
        delete_after=60
    )

@bot.hybrid_command(name='test', description='test', with_app_command=True)
@commands.cooldown(1, 60, commands.BucketType.user)
@commands.is_owner()
async def test(ctx):
    await ctx.send(
        content=f'Teste efetuado, {ctx.author.name}', 
        ephemeral=True, 
        delete_after=60
    )
    
@bot.event
async def on_ready():
    print('You are logged as {0.user}'.format(bot))
    await bot.tree.sync()
    
    print(f'{bot.user} is connected to the following guilds:')
    for guild in bot.guilds: 
        print(f'{guild.name}(id: {guild.id})')
        members = '\n - '.join([member.name for member in guild.members])
        print(f'Guild Members:\n - {members}')
        
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
     
    await bot.process_commands(message)
    if message.content.lower().startswith('hi'):
        await message.channel.send(content=f'Hello, {message.author.name}!')

@bot.event
async def on_command_error(ctx, error):
    print(error)
    if isinstance(error, commands.errors.CommandOnCooldown):
        await ctx.send(content=error)
    elif isinstance(error, commands.errors.NotOwner):
        await ctx.send(content=error)
    elif isinstance(error, commands.errors.CheckFailure):
        await ctx.send(content=error)

bot.run(secrets['DISCORD_TOKEN'])