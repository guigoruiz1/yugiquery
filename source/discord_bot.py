import os
import glob
import random
import subprocess
import discord
import asyncio
import io
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

secrets = load_secrets('../assets/secrets.env')
intents = discord.Intents(messages=True, guilds=True, members=True)
bot = commands.Bot(command_prefix='/', intents=intents)
checks = discord.app_commands.checks
Reports = init_reports_enum()
process = None

def is_owner():
    async def predicate(ctx):
        return await bot.is_owner(ctx.user)
    return checks.check(predicate)

@bot.tree.command(name='shutdown', description='Shutdown bot')
@is_owner()
async def shutdown(ctx):
    await ctx.response.send_message(content='Shutting down...')
    await bot.close()

@bot.tree.command(name='run', description='Run full Yugiquery workflow')
@is_owner()
@checks.cooldown(1, 12*60*60, key=lambda i: (i.guild_id, i.user.id))
async def run(ctx, report: Reports):
    global process
    if process is not None:
        await ctx.response.send_message(content='Query already running. Try again after it has finished.', ephemeral=True, delete_after=60)
        return
    
    await ctx.response.send_message(content='Initializing...', ephemeral=True, delete_after=60)
    
    API_error = False
    def progress_handler(iterable=None, API_status=None, **kwargs):
        if iterable and ctx.channel.id != int(secrets['DISCORD_CHANNEL_ID']):
                return discord_pbar(iterable, token = secrets['DISCORD_TOKEN'], channel_id=ctx.channel.id, file=io.StringIO(), **kwargs)
        elif API_status is not None:
            nonlocal API_error
            API_error = not API_status
            return
        
    try:
        process = mp.Process(target=yq.run, args=[report.value, progress_handler])
        process.start()
        await ctx.edit_original_response(content='Running...')
    except:
        await ctx.edit_original_response(content='Initialization failed!')
    
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
    
        
@bot.tree.command(name='abort', description='Abort running Yugiquery workflow')
@is_owner()
async def abort(ctx):

    await ctx.response.send_message('Aborting...', ephemeral=True, delete_after=60)
    
    try:
        process.terminate()
        await ctx.edit_original_response(content='Aborted')
    except:
        await ctx.edit_original_response(content='Abort failed')
        
@bot.tree.command(name='latest', description='Show latest time each report was generated')
async def latest(ctx):
    await ctx.response.defer()
    response='Latest reports generated:'
    reports = sorted(glob.glob('../*.html'))
    for report in reports:
        response += f'\n- {os.path.basename(report)[:-5]}: {yq.pd.to_datetime(os.path.getmtime(report),unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}'
    
    await ctx.followup.send(response)
    
@bot.tree.command(name='links', description='Show Yugiquery links')
async def links(ctx):
    response = '\n'.join(
        ['Webpage: https://guigoruiz1.github.io/yugiquery/',
         'Repository: https://github.com/guigoruiz1/yugiquery/',
         'Data: https://github.com/guigoruiz1/yugiquery/tree/main/data']
    )
    await ctx.response.send_message(response)
    
@bot.tree.command(name='data', description='Send latest data files')
async def data(ctx):
    response = 'Under construction'
    await ctx.response.send_message(response)

@bot.tree.command(name='ping', description='Test the bot connection latency')
async def ping(ctx):
    await ctx.response.send_message('Pong! {0}ms'.format(round(bot.latency*1000, 1)), ephemeral=True, delete_after=60)

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
        await message.channel.send(f'Hello, {message.author.name}!')
    
@bot.tree.error
async def on_error(ctx, error):
    if isinstance(error, discord.app_commands.errors.CommandOnCooldown):
        await ctx.response.send_message(error)
    elif isinstance(error, discord.app_commands.errors.CheckFailure):
        await ctx.response.send_message(error)

bot.run(secrets['DISCORD_TOKEN'])