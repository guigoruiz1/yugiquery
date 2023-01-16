import discord
import requests
import os
 
secrets_file = '../Assets/secrets.txt'
if os.path.isfile(secrets_file):
    secrets={}
    with open(secrets_file) as f:
        for line in f:
            line=line.strip()
            if not line.startswith('#'):
                name, value = line.split("=")
                secrets[name.strip()] = value.strip()
    
intents = discord.Intents(messages=True)
client = discord.Client(command_prefix='!', intents=intents)

bot = commands.Bot(command_prefix='!')
@bot.command(name='run', help='Run full Yugiquery workflow')
async def nine_nine(ctx):
    response = 'Under construction. Try again later'
    await ctx.send(response)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

    for guild in client.guilds:
        if guild.name == GUILD:
            break

    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )
    
    members = '\n - '.join([member.name for member in guild.members])
    print(f'Guild Members:\n - {members}')
 
@client.event
async def on_message(message):
    print("message-->", message)
    print("message content-->", message.content)
    print("message attachments-->", message.attachments)
    print("message id", message.author.id)
    a_id = message.author.id
    if a_id != secrets['DISCORD_TOKEN']:
   
        # for x in message.attachments:
        #     print("attachment-->",x.url)
        #     d_url = requests.get(x.url)
        #     file_name = x.url.split('/')[-1]
        #     with open(file_name, "wb") as f:
        #         f.write(d_url.content)
 
    if message.author == client.user:
        return
 
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
 
client.run(secrets['DISCORD_TOKEN'])