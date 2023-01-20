import os
import random
import subprocess
import discord
import asyncio
import yugiquery as yq
import multiprocessing as mp
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

secrets = load_secrets('../Assets/secrets.env')
intents = discord.Intents(messages=True, guilds=True, members=True)
bot = commands.Bot(command_prefix='/', intents=intents)
loop = None
task = None
process = None
    
async def check_owner(ctx):
    return await bot.is_owner(ctx.message.author)

@bot.tree.command(name='shutdown', description='Shutdown bot')
@commands.is_owner()
async def shutdown(ctx):
    await ctx.response.send_message(content='Shutting down...')
    await bot.close()

@bot.tree.command(name='run', description='Run full Yugiquery workflow')
@commands.cooldown(1, 12*60*60)
@commands.is_owner()
async def run(ctx):
    await ctx.response.send_message(content='Initializing...')
    
    if ctx.channel.id != int(secrets['DISCORD_CHANNEL_ID']):
        def progress_handler(iterator, **kwargs):
            return discord_pbar(iterator, token = secrets['DISCORD_TOKEN'], channel_id=ctx.channel.id, **kwargs)
    else:
        progress_handler = None
    
    try: 
        global process
        process = mp.Process(target=yq.run_all, args=[progress_handler])
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
    
    if exitcode is None:
        await ctx.edit_original_response(content='Failed!') 
    elif exitcode==0:
        await ctx.edit_original_response(content='Completed!')
    elif exitcode==-15:
        await ctx.edit_original_response(content='Aborted!')
    else:
        await ctx.edit_original_response(content=f'Unknown exit code: {exitcode}')
    
        
@bot.tree.command(name='abort', description='Abort running Yugiquery workflow')
@commands.is_owner()
async def abort(ctx):

    await ctx.response.send_message('Aborting...')
    
    try:
        process.terminate()
        await ctx.edit_original_response(content='Aborted')
        await asyncio.sleep(5)
        await ctx.delete_original_response()
    except:
        await ctx.edit_original_response(content='Abort failed')
    
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

bot.run(secrets['DISCORD_TOKEN'])