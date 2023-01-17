import discord
import requests
import os
import random
from discord.ext import commands
 
secrets_file = '../Assets/secrets.txt'
if not os.path.isfile(secrets_file):
    exit()
    
secrets=dotenv_values("../Assets/secrets.env")
if not all(key in secrets.keys() for key in ['DISCORD_TOKEN','DISCORD_CHANNEL_ID']):
    exit()

if not (secrets['DISCORD_CHANNEL_ID'] and secrets['DISCORD_TOKEN']):
    exit()
    
    
intents = discord.Intents(messages=True)
# client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix='/', intents=intents)

@bot.command(name='run', help='Run full Yugiquery workflow')
async def nine_nine(ctx):
    response = 'Under construction. Try again later'
    await ctx.send(response)

@bot.command(name='roll_dice', help='Simulates rolling dice.')
async def roll(ctx, number_of_dice: int, number_of_sides: int):
    dice = [
        str(random.choice(range(1, number_of_sides + 1)))
        for _ in range(number_of_dice)
    ]
    await ctx.send(', '.join(dice))
    
@bot.event
async def on_ready():
    print('We have logged in as {0.user}'.format(bot))
    await bot.tree.sync()
    
    for guild in bot.guilds: 
        print(
            f'{bot.user} is connected to the following guilds:\n'
            f'{guild.name}(id: {guild.id})'
        )
    
    members = '\n - '.join([member.name for member in guild.members])
    print(f'Guild Members:\n - {members}')
 
@bot.event
async def on_message(message):
    print("message-->", message)
    print("message content-->", message.content)
    print("message attachments-->", message.attachments)
    print("message id", message.author.id)
    a_id = message.author.id
    # if a_id != secrets['DISCORD_TOKEN']:
   
        # for x in message.attachments:
        #     print("attachment-->",x.url)
        #     d_url = requests.get(x.url)
        #     file_name = x.url.split('/')[-1]
        #     with open(file_name, "wb") as f:
        #         f.write(d_url.content)
 
    if message.author == bot.user:
        return
     
    await bot.process_commands(message)
    if message.content.lower().startswith('hi'):
        await message.channel.send(f'Hello, {message.author.name}!')
 
#     if message.content.startswith('image'):
#         await message.channel.send(file=discord.File('download.jpg'))
 
#     if message.content.startswith('video'):
#         await message.channel.send(file=discord.File('sample-mp4-file-small.mp4'))
 
#     if message.content.startswith('audio'):
#         await message.channel.send(file=discord.File('file_example_MP3_700KB.mp3'))
 
#     if message.content.startswith('file'):
#         await message.channel.send(file=discord.File('sample.pdf'))
 
bot.run(secrets['DISCORD_TOKEN'])