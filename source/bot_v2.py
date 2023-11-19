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

# Native python packages
import argparse
import asyncio
import glob
import io
import json
import multiprocessing as mp
import os
import platform
import random
import re
import subprocess
from datetime import datetime, timezone
from enum import Enum

# PIP packages - installed by yugiquery
import git
import pandas as pd
from dotenv import dotenv_values
from tqdm.auto import tqdm, trange

# Telegram
import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
# from tqdm.contrib.telegram import tqdm as telegram_pbar

# Discord
import discord
from discord.ext import commands
# from tqdm.contrib.discord import tqdm as discord_pbar


# ============== #
# Helper methods #
# ============== #

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
    subclass = args.pop("subclass").upper()
    secrets = {f"{subclass}_{key}": value for key, value in args.items() if value is not None}
    missing = [f"{subclass}_{key}" for key, value in args.items() if value is None]
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

def escape_chars(string):
    for char in ['_', '.', '-', '+', '#', '@', '=']:
        string = string.replace(char, '\\'+char)
    return string

# ============== #
# Bot Superclass #
# ============== #

class Bot:

    def __init__(self, token, channel, **kwargs):
        self.start_time = datetime.now()
        self.token = token
        self.channel = channel
        self.init_reports_enum()
        self.load_repo_vars()
        
    # Other
    process = None
    Reports = Enum("Reports", {"All": "all"})

    # ======================== #
    # Bot Superclass Functions #
    # ======================== #

    def load_repo_vars(self):
        # Open the repository
        repo = git.Repo(yq.PARENT_DIR, search_parent_directories=True)
    
        # Get the remote repository
        remote = repo.remote()
        remote_url = remote.url
    
        # Extract the GitHub page URL from the remote URL
        # by removing the ".git" suffix and splitting the URL
        # by the "/" character
        remote_url_parts = remote_url[:-4].split("/")
    
        # Repository
        (author, repo) = remote_url_parts[-2:]
        # URLs
        self.repository_api_url = f"https://api.github.com/repos/{author}/{repo}"
        self.repository_url = f"https://github.com/{author}/{repo}"
        self.webpage_url = f"https://{author}.github.io/{repo}"

    def init_reports_enum(self):
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
    
        self.Reports = Enum("Reports", reports_dict)
    
    def abort(self):
        """
        Aborts a running YugiQuery workflow by terminating the process.
        """
        try:
            self.process.terminate()
            return "Aborted"
        except:
            return "Abort failed"

    def benchmark(self):
        """
        Returns the average time each report takes to complete and the latest time for each report.
    
        Args:
            update (telegram.Update): The update object.
            context (telegram.ext.CallbackContext): The callback context.
        """
        try:
            with open(os.path.join(yq.PARENT_DIR, "data/benchmark.json"), "r") as file:
                data = json.load(file)
        except:
            return {"error": "Unable to find benchmark records at this time. Try again later."}
    
        response = {
            "title":"Benchmark",
            "description":"The average time each report takes to complete",
        }
         
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
            response[key.capitalize()] = value

        return response

    def latest(self):
        """
        Displays the timestamp of the latest local and live reports generated.
        Reads the report files from `yq.PARENT_DIR` and queries the GitHub API
        for the latest commit timestamp for each file. Returns the result as an 
        message in the channel.
        """
        
        reports = sorted(glob.glob(os.path.join(yq.PARENT_DIR, "*.html")))
        response = {"title": "Latest reports generated",
                    "description": "The live reports may not always be up to date with the local reports"
                   }
    
        # Get local files timestamps
        local_value = ""
        for report in reports:
            local_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(os.path.getmtime(report),unit="s", utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'
    
        response["local"] = local_value
    
        # Get live files timestamps
        try:
            live_value = ""
            for report in reports:
                result = pd.read_json(f"{repository_api_url}/commits?path={os.path.basename(report)}")
                timestamp = pd.DataFrame(result.loc[0, "commit"]).loc["date", "author"]
                live_value += f'• {os.path.basename(report).split(".html")[0]}: {pd.to_datetime(timestamp, utc=True).strftime("%d/%m/%Y %H:%M %Z")}\n'
    
            response["live"] = live_value
        except:
            pass
    
        return response
    
    def links(self):
        """
        Displays the links to the YugiQuery webpage, repository, and data.
        Returns the links as an message in the channel.
        """
        response = {
            "title": "YugiQuery links", 
            "description": f"[Webpage]({self.webpage_url}) • [Repository]({self.repository_url}) • [Data]({self.repository_url}/tree/main/data)"
        }
    
        return response
    
    def data(self):
        """
        This command sends the latest data files available in the repository as direct download links.
        """
        try:
            files = pd.read_json(f"{repository_api_url}/contents/data")
            files = files[files["name"].str.endswith(".bz2")]  # Remove .json files from lists
            files["Group"] = files["name"].apply(
                lambda x: re.search(r"(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).bz2", x).group(1)
            )
            files["Timestamp"] = files["name"].apply(
                lambda x: re.search(r"(\w+_\w+)_(.*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).bz2", x).group(3)
            )
            files["Timestamp"] = pd.to_datetime(files["Timestamp"], utc=True)
            index = files.groupby("Group")["Timestamp"].idxmax()
            latest_files = files.loc[index, ["name", "download_url"]]
    
            data_value = ""
            changelog_value = ""
            for _, file in latest_files.iterrows():
                if "changelog" in file["name"]:
                    changelog_value += f'• [{file["name"]}]({file["download_url"]})\n'
                else:
                    data_value += f'• [{file["name"]}]({file["download_url"]})\n'
    
            response = {
                "title": "Latest data files",
                "description": "Direct links to download files from GitHub.", 
                "data": data_value, 
                "changelog": changelog_value
            }
            return response
    
        except:
            return {"error": "Unable to obtain the latest files at this time. Try again later."}


    async def battle(self, callback, atk_weight: int = 4, def_weight: int = 1):
        """
        This function loads the list of all Monster Cards and simulates a battle between them.
        """
        MONSTER_STATS = ["Name", "ATK", "DEF"]
        cards_files = sorted(
            glob.glob(os.path.join(yq.PARENT_DIR, "data/all_cards_*.bz2")),
            key=os.path.getmtime,
        )
        if not cards_files:
            return {"error": "Cards data not found... Try again later."}

        cards = pd.read_csv(cards_files[0])
        weights = [atk_weight, def_weight]
        monsters = cards[
            (cards["Card type"] == "Monster Card")
            & (cards["Primary type"] != "Monster Token")
        ][MONSTER_STATS].set_index("Name")
        monsters = monsters.map(
            lambda x: x if x != "?" else random.randrange(0, 51) * 100
        )
        monsters = monsters.map(pd.to_numeric, errors="coerce").fillna(0).reset_index()
    
        # Shuffle the monsters and select the first one as the initial winner
        monsters = monsters.sample(frac=1).reset_index(drop=True)
        winner = (monsters.iloc[0], 0)
        longest = (monsters.iloc[0], 0)

        await callback(winner[0]["Name"])
    
        for i in range(1, len(monsters)):
            current_winner = (winner[0].copy(), winner[1])
            next_monster = monsters.iloc[i].copy()
            chosen_stat = random.choices(MONSTER_STATS[1:], weights=weights)[0]
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

        return {"winner": winner, "longest": longest}


    # PENDING
    async def run(self, callback, channel_id, report: Reports = Reports.All, progress_bar = None):
        """
        Runs a YugiQuery workflow by launching a separate thread and monitoring its progress.
        The progress is reported back to the Telegram chat where the command was issued.
        """
        if self.process is not None:
            return {"error": "Query already running. Try again after it has finished."}
            
        queue = mp.Queue()
    
        def progress_handler(iterable=None, API_status: bool = True, **kwargs):
            queue.put(API_status)
            if iterable and (channel_id != self.channel) and (progress_bar is not None):
                 return progress_bar(
                    iterable,
                    token=self.token,
                    channel_id=channel_id,
                    file=io.StringIO(),
                    **kwargs,
                )
    
        try:
            self.process = mp.Process(target=yq.run, args=[report.value, progress_handler, True])
            self.process.start()
            callback("Running...")
        except:
            callback("Initialization failed!")
    
        async def await_result():
            while self.process.is_alive():
                await asyncio.sleep(1)
            API_error = False
            while not queue.empty():
                API_error = not queue.get()
            return process.exitcode, API_error
    
        exitcode, API_error = await await_result()
        self.process.close()
        self.process = None
    
        if API_error:
            return {"error": "Unable to communicate with the API. Try again later."}
        else:
            if exitcode is None:
                return {"error": "Query execution failed!"}
            elif exitcode == 0:
                return {"content", "Query execution completed!"}
            elif exitcode == -15:
                return {"error", "Query execution aborted!"}
            else:
                return {"error": f"Query execution exited with exit code: {exitcode}"}

# ===================== #
# Telegram Bot Subclass #
# ===================== #

class Telegram(Bot):
    def __init__(self, token, channel):
        super().__init__(token, channel)
        # Initialize the Telegram bot
        self.application = ApplicationBuilder().token(token).build()
        self.register_commands()
        self.register_events()
    
    def run(self):
        self.application.run_polling()

        
    # ======== #
    # Commands #
    # ======== #
           
    def register_commands(self):

        async def shutdown(update: Update, context: CallbackContext):
            """
            Shuts down the bot gracefully by sending a message and stopping the polling.
        
            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Shutting down...")
            self.application.stop_running()
    
        async def run(update: Update, context: CallbackContext, report: self.Reports = self.Reports.All):
            """
            Runs a YugiQuery workflow by launching a separate thread and monitoring its progress.
            The progress is reported back to the Telegram chat where the command was issued.
        
            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text="Initializing...")
            def callback(content):
                nonlocal original_response
                original_response.edit_text(content)
               
            response = await self.run(callback=callback, report=report, channel_id = update.effective_chat.id, progress_bar = telegram_pbar)
            if error in response:
                await context.bot.send_message(chat_id=update.effective_chat.id, text = response["error"])
                # Reset cooldown in case query did not complete
                ctx.command.reset_cooldown(ctx)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text = response["content"])
            
        
        async def abort(update: Update, context: CallbackContext):
            """
            Aborts a running YugiQuery workflow by terminating the thread.
        
            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            original_response = await context.bot.send_message(chat_id=update.effective_chat.id, text="Aborting...")
            response = self.abort()
            await original_response.edit_text(response)
        
        
        async def benchmark(update: Update, context: CallbackContext):
            """
            Returns the average time each report takes to complete and the latest time for each report.
        
            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.benchmark()
            if "error" in response: 
                await context.bot.send_message(chat_id=update.effective_chat.id, text=response["error"])
                return
        
            message = response.pop("title")
            message += "\n"
            message += response.pop("description")
            message += "\n\n"
            for key, value in response.items():
                message += f"<b>{key}</b>\n{value}\n\n"
        
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="HTML")
        
        
        async def latest(update: Update, context: CallbackContext):
            """
            Displays the timestamp of the latest local and live reports generated.
            Reads the report files from `yq.PARENT_DIR` and queries the GitHub API
            for the latest commit timestamp for each file. Returns the result as an
            embedded message in the channel.
        
            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.latest()
        
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode="HTML")
        
        async def links(update: Update, context: CallbackContext):
            """
            Displays the links to the YugiQuery webpage, repository, and data.
            Returns the links as an embedded message in the channel.
        
            Args:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            response = self.links()
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response["title"]+"\n"+response["description"], parse_mode="MarkdownV2")
        
        async def data(update: Update, context: CallbackContext):
            """
            This command sends the latest data files available in the repository as direct download links.
        
            Parameters:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            
            response=self.data()
            if "error" in response:
                message = response["error"]
            else:
                message = response["title"]+"\n"+response["description"]+"\n\n"
                message += response["data"]+"\n"+response["changelog"]
                
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode="MarkdownV2")
    
        # PENDING
        async def ping(update: Update, context: CallbackContext):
            """
            This command tests the bot's connection latency and sends the result back to the user.
        
            Parameters:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            start_time = datetime.now()
            original_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Calculating latency...")
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds()*1e3
            response = f"🏓 Pong! {round(latency_ms,1)}ms"
            await original_message.edit_text(response)
        
        
        async def battle(update: Update, context: CallbackContext, atk_weight: int = 4, def_weight: int = 1):
            """
            This function loads the list of all Monster Cards and simulates a battle between them.
            ...
        
            Parameters:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
                atk_weight (int, optional): The weight to use for the ATK stat when randomly choosing the monster's stat to compare. This affects the probability that ATK will be chosen over DEF. The default value is 4.
                def_weight (int, optional): The weight to use for the DEF stat when randomly choosing the monster's stat to compare. This affects the probability that DEF will be chosen over ATK. The default value is 1.
            """
            original_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Simulating a battle... ⚔️")
            async def callback(first):
                first = escape_chars(first)
                await original_message.edit_text(
                    f"**First contestant**: {first}\n\nStill battling\.\.\. ⏳", parse_mode="MarkdownV2"
                )
            response = await self.battle(callback=callback, atk_weight=atk_weight, def_weight=def_weight)
            if "error" in response:
                message = response["error"]
            else:
                winner = response["winner"]
                longest = response["longest"]
                message = (
                    f"**Winner**: {winner[0]['Name']}\n"
                    f"**Wins**: {winner[1]}\n"
                    f"**Stats remaining**: ATK={winner[0]['ATK']}, DEF={winner[0]['DEF']}\n"
                    f"\n"
                    f"**Longest streak**: {longest[0]['Name']}\n"
                    f"**Wins**: {longest[1]}\n"
                    f"**Stats when defeated**: ATK={longest[0]['ATK']}, DEF={longest[0]['DEF']}"
                )
            
            message = escape_chars(message)
            await original_message.edit_text(
                message, parse_mode="MarkdownV2"
            )
            
        async def status(update: Update, context: CallbackContext):
            """
            Displays information about the bot, including uptime, guilds, users, channels, available commands,
            bot version, discord.py version, python version, and operating system.
            
            Parameters:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            uptime = datetime.now() - self.start_time
        
            app_info = await context.bot.get_me()
            bot_name = app_info.username

            message = (
                f"**Bot name**: {bot_name}\n"
                f"**Uptime**: {uptime}\n"
                f"**Bot Version**: {__version__}\n"
                f"**Telegram Bot API Version**: {telegram.__version__}\n"
                f"**Python Version**: {platform.python_version()}\n"
                f"**Operating System:**\n"
                f" - Name: {platform.system()}\n"
                f" - Release: {platform.release()}\n"
                f" - Machine: {platform.machine()}\n"
                f" - Version: {platform.version()}"
            ).replace('_','\_').replace('.','\.').replace('-','\-').replace('+','\+').replace('#','\#')
            for char in ['_', '.', '-', '+', '#']:
                message.replace(char, '\\'+char)
        
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode="MarkdownV2")

        # Register the command handlers
        self.application.add_handler(CommandHandler("shutdown", shutdown, filters=filters.Chat(chat_id=int(self.channel))))
        self.application.add_handler(CommandHandler("run", run, block=False))
        self.application.add_handler(CommandHandler("abort", abort))
        self.application.add_handler(CommandHandler("benchmark", benchmark))
        self.application.add_handler(CommandHandler("latest", latest))
        self.application.add_handler(CommandHandler("links", links))
        self.application.add_handler(CommandHandler("data", data))
        self.application.add_handler(CommandHandler("ping", ping))
        self.application.add_handler(CommandHandler("battle", battle))
        self.application.add_handler(CommandHandler("status", status))
    
    
    # ====== #
    # Events #
    # ====== #

    # PENDING
    def register_events(self):
        # async def start(update: Update, context: CallbackContext):
        #     """Send a message when the command /start is issued."""
        #     user = update.effective_user
        #     await update.message.reply_html(
        #         rf"Hi {user.mention_html()}!",
        #         reply_markup=ForceReply(selective=True),
        #     )
    
    
        async def on_command_error(update: Update, context: CallbackContext):
            """
            Event that runs whenever a command invoked by the user results in an error.
            Sends a message to the chat indicating the type of error that occurred.
            
            Parameters:
                update (telegram.Update): The update object.
                context (telegram.ext.CallbackContext): The callback context.
            """
            error = context.error
            print(error)
            # if isinstance(error, CommandError):
            #     # Assuming the exceptions are adapted for Telegram, you may need to customize these messages
            #     if isinstance(error, CommandOnCooldown):
            #         await update.message.reply_text(f"Cooldown error: {error}", quote=True)
            #     elif isinstance(error, NotOwner):
            #         await update.message.reply_text(f"Not owner error: {error}", quote=True)
            #     elif isinstance(error, CheckFailure):
            #         await update.message.reply_text(f"Check failure error: {error}", quote=True)

        # self.application.add_handler(CommandHandler("start", start))
        self.application.add_error_handler(on_command_error)

# ==================== #
# Discord Bot Subclass #
# ==================== #

class Discord(Bot, commands.Bot):
    def __init__(self, token, channel):
        Bot.__init__(self, token, channel)
        intents = discord.Intents(messages=True, guilds=True, members=True)
        # Initialize the Discord bot
        commands.Bot.__init__(self, command_prefix="/", intents=intents)
        self.register_commands()

    def run(self):
        commands.Bot.run(self, self.token)


    
    # ====== #
    # Events #
    # ====== #

    async def on_ready(self):
        """
        Event that runs when the bot is ready to start receiving events and commands.
        Prints out the bot's username and the guilds it's connected to.
        """
        print("You are logged as {}".format(self.user))
        await self.tree.sync()
    
        print(f"{self.user} is connected to the following guilds:")
        for guild in self.guilds:
            print(f"{guild.name}(id: {guild.id})")
            members = "\n - ".join([member.name for member in guild.members])
            print(f"Guild Members:\n - {members}")
    
    async def on_message(self, message):
        """
        Event that runs whenever a message is sent in a server where the bot is present.
        Responds with a greeting to any message starting with 'hi'.
        """
        if message.author == self.user:
            return
    
        await self.process_commands(message)
        # response = super().on_message(message.content)
        # await message.channel.send(content=response)
        if message.content.lower().startswith("hi"):
            await message.channel.send(content=f"Hello, {message.author.name}!")
    
    async def on_command_error(self, ctx, error):
        """
        Event that runs whenever a command invoked by the user results in an error.
        Sends a message to the channel indicating the type of error that occurred.
        """
        print(error)
        if isinstance(error, commands.errors.CommandOnCooldown):
            await ctx.send(content=error, ephemeral=True, delete_after=60)
        elif isinstance(error, commands.errors.NotOwner):
            await ctx.send(content=error, ephemeral=True, delete_after=60)
        elif isinstance(error, commands.errors.CheckFailure):
            await ctx.send(content=error, ephemeral=True, delete_after=60)
    
    
    # ======== #
    # Commands #
    # ======== #


    def register_commands(self):
        @self.hybrid_command(name="shutdown", description="Shutdown bot", with_app_command=True)
        @commands.is_owner()
        async def shutdown(ctx):
            """
            Shuts down the bot gracefully by sending a message and closing the connection.
        
            Args:
                ctx (commands.Context): The context of the command.
            """
            await ctx.send(content="Shutting down...")
            await self.close()
        
        @self.hybrid_command(
            name="run", description="Run full YugiQuery workflow", with_app_command=True
        )
        @commands.is_owner()
        @commands.cooldown(1, 12 * 60 * 60, commands.BucketType.user)
        async def run(ctx, report: self.Reports = self.Reports.All):
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
            original_response = await ctx.send(
                content="Initializing...", ephemeral=True, delete_after=60
            )
            def callback(content):
                nonlocal original_response
                original_response.edit(content=content)
                
            response = await self.run(callback = callback, report = report, channel_id = str(ctx.channel.id))
            if error in response:
                await original_response.send(
                    content=response["error"]
                )
                # Reset cooldown in case query did not complete
                ctx.command.reset_cooldown(ctx)
            else:
                await original_response.send(
                    content=response["content"]
                )
        
        
        @self.hybrid_command(
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
            response = self.abort()
            await original_response.edit(content=response)
    
        
        @self.hybrid_command(
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
            response = self.benchmark()
            if error in response:
                await ctx.send(
                    response["error"]
                )
                return
            title = response.pop("title")
            description = response.pop("description")
            embed = discord.Embed(
                title=title,
                description=description,
                color=discord.Colour.gold(),
            )
    
            for key, value in response.items():
                embed.add_field(name=key, value=value, inline=False)
        
            await ctx.send(embed=embed)
        
        
        @self.hybrid_command(
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
            response = self.latest()
            embed = discord.Embed(
                title = response["title"],
                description=response["description"],
                color=discord.Colour.orange(),
            )
            embed.add_field(name="Local", value=response["local"], inline=False)
            if "live" in response:
                embed.add_field(name="Live", value=response["live"], inline=False)
            
            await ctx.send(embed=embed)
        
        
        @self.hybrid_command(
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
            response = self.links()
            embed = discord.Embed(
                title=response["title"],
                description=response["description"],
                color=discord.Colour.green(),
            )
        
            await ctx.send(embed=embed)
        
        
        @self.hybrid_command(
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
            response = self.data()
            if "error" in response:
                await ctx.send(response["error"])
        
            else:
                embed = discord.Embed(
                    title=response["title"],
                    description=response["description"],
                    color=discord.Colour.magenta(),
                )
                embed.add_field(name="Data", value=response["data"], inline=False)
                embed.add_field(name="Changelog", value=response["changelog"], inline=False)
                await ctx.send(embed=embed)
                
        
        
        @self.hybrid_command(
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
                content="🏓 Pong! {0}ms".format(round(self.latency * 1000, 1)),
                ephemeral=True,
                delete_after=60,
            )
    
    
        @self.hybrid_command(
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
            embed = discord.Embed(
                title="Battle",
                description="Simulate a battle of all monster cards",
                color=discord.Colour.purple(),
            )

            original_response = None
            async def callback(first):
                embed.add_field(name="First contestant", value=first, inline=False)
                embed.set_footer(text="Still battling... ⏳")
                nonlocal original_response
                original_response = await ctx.send(embed=embed)
            
            response = await self.battle(atk_weight=atk_weight, def_weight=def_weight, callback=callback)
        
            if "error" in response:
                await ctx.send(
                    content="Cards data not found... Try again later.",
                    ephemeral=True,
                    delete_after=60,
                )
                return
        
            winner = response["winner"]
            longest = response["longest"]
        
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
        
    
        # PENDING
        @self.hybrid_command(
            name="status",
            description="Display bot status and system information",
            with_app_command=True
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
            uptime = datetime.now() - self.start_time
        
            appInfo = await self.application_info()
            admin = appInfo.owner
            users = 0
            channels = 0
            guilds = len(self.guilds)
            for guild in self.guilds:
                users += len(guild.members)
                channels += len(guild.channels)
        
            if len(self.commands):
                commandsInfo = "\n".join(sorted([i.name for i in self.commands]))
        
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
                value=f"Name: {platform.system()}\nRelease: {platform.release()}\nMachine: {platform.machine()}\nVersion: {platform.version()}",
                inline=False,
            )
            await ctx.send("**:information_source:** Information about this bot:", embed=embed)
    



# ========= #
# Execution #
# ========= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subclass",
        choices = ["discord", "telegram"],
        default = "discord",
        help = "Select between a Discord or a Telegram bot"
    )
    parser.add_argument(
        "-t", "--token", dest="TOKEN", type=str, help="Bot API token"
    )
    parser.add_argument(
        "-c",
        "--channel",
        dest="CHANNEL_ID",
        type=int,
        help="Bot responses channel id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enable debug flag",
    )
    
    args = vars(parser.parse_args())
    subclass = args["subclass"]
    debug = args.pop("debug")

    # Load secrets
    secrets = load_secrets_with_args(args)
    
    if subclass == "discord":
        # Initialize the Discord bot
        bot = Discord(secrets["DISCORD_TOKEN"], secrets["DISCORD_CHANNEL_ID"])
        bot.run()

    elif subclass == "telegram":
        # Initialize the Telegram bot
        bot = Telegram(secrets["TELEGRAM_TOKEN"], secrets["TELEGRAM_CHANNEL_ID"])
        bot.run()