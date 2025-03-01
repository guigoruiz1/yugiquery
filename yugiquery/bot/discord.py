#!/usr/bin/env python3

# yugiquery/bot/discord.py

# -*- coding: utf-8 -*-

# ======= #
# Imports #
# ======= #

# Standard library packages
import platform
import io

# Local application imports
from ..metadata import __version__
from .base import Bot, GitCommands

# Discord
try:
    import discord
    from discord.ext import commands

except ImportError:
    raise RuntimeError(
        'Missing Discord bot specif packages. Please install the required packages with "pip install yugiquery[discord]".'
    )

# Silence discord.py pynacl optional dependency warning.
discord.VoiceClient.warn_nacl = False

# ==================== #
# Discord Bot Subclass #
# ==================== #


class Discord(Bot, commands.Bot):
    """
    Discord bot subclass. Inherits from Bot class and discord.ext.commands.Bot.

    Args:
        token (str): The token for the Discord bot.
        channel_id (str | int): The channel ID for the bot.

    Attributes:
        discord.ext.commands.Bot attributes
        Bot attributes
    """

    DISCORD_MESSAGE_LIMIT = 2000  # Discord's character limit for messages

    def __init__(self, token: str, channel_id: str | int):
        """
        Initializes the Discord bot subclass.

        Args:
            token (str): The token for the Discord bot.
            channel_id (str | int): The channel ID for the Discord bot.
        """
        from tqdm.contrib.discord import tqdm as discord_tqdm

        self.discord_pbar = discord_tqdm
        Bot.__init__(self)
        self.token = token
        self.channel_id = int(channel_id)
        # Initialize the Discord bot
        intents = discord.Intents(
            messages=True,
            guilds=True,
            members=True,
        )
        help_command = commands.DefaultHelpCommand(no_category="Commands")
        description = "Bot to manage YugiQuery data and execution."
        activity = discord.Activity(type=discord.ActivityType.watching, name="for /status")
        commands.Bot.__init__(
            self,
            command_prefix="/",
            intents=intents,
            activity=activity,
            description=description,
            help_command=help_command,
        )
        self.register_commands()

    def run(self) -> None:
        """
        Starts running the discord Bot.
        """
        commands.Bot.run(self, token=self.token)

    async def send_long_message(self, ctx, content: str, filename: str = "message.txt", **kwargs):
        """
        Sends a message, ensuring it does not exceed Discord's message limit.
        If the message is too long, sends it as a text file attachment.

        Args:
            ctx (commands.Context): The context of the message.
            content (str): The message content.
            filename (str): The name of the file if the message is sent as an attachment.
            **kwargs: Additional keyword arguments to pass to the send method.
        """
        if len(content) <= self.DISCORD_MESSAGE_LIMIT:
            await ctx.send(content=content, **kwargs)
        else:
            file = discord.File(io.StringIO(content), filename=filename)
            await ctx.send(content="Response too long, sending as an attachment:", file=file, **kwargs)

    # ====== #
    # Events #
    # ====== #

    async def on_ready(self) -> None:
        """
        Event callback that runs when the bot is ready to start receiving events and commands.
        Prints out the bot's username and the guilds it's connected to.
        """
        print("You are logged as {}".format(self.user))
        await self.tree.sync()

        print(f"{self.user} is connected to the following guilds:")
        for guild in self.guilds:
            print(f"{guild.name}(id: {guild.id})")
            members = "\n - ".join([member.name for member in guild.members])
            print(f"Guild Members:\n - {members}")

        info = await self.application_info()
        await info.owner.send(f"Hello {info.owner.global_name}!\n{info.name} bot is online.")

    async def on_message(self, message: discord.Message) -> None:
        """
        Event callback that runs whenever a message is sent in a server where the bot is present.
        Responds with a greeting to any message starting with 'hi'.

        Args:
            ctx (commands.Context): The context of the message.
            message (discord.Message): The message received.
        """
        if message.author == self.user:
            return

        await self.process_commands(message)
        if message.content.lower().startswith("hi"):
            await message.channel.send(content=f"Hello, {message.author.name}!")

    async def on_command_error(self, ctx, error: commands.CommandError) -> None:
        """
        Event callback that runs whenever a command invoked by the user results in an error.
        Sends a message to the channel indicating the type of error that occurred.

        Args:
            ctx (commands.Context): The context of the error.
            error (commands.CommandError): The error received.
        """
        print(error)
        # TODO: handle errors separatelly
        if isinstance(error, commands.CommandOnCooldown):
            await self.send_long_message(ctx, content=error, ephemeral=True, delete_after=60)
        elif isinstance(error, commands.NotOwner):
            await self.send_long_message(ctx, content=error, ephemeral=True, delete_after=60)
        elif isinstance(error, commands.CheckFailure):
            await self.send_long_message(ctx, content=error, ephemeral=True, delete_after=60)
        else:
            await self.send_long_message(ctx, content=error, ephemeral=True, delete_after=60)

    # ======== #
    # Commands #
    # ======== #

    def register_commands(self) -> None:
        """
        Register command handlers for the Discord bot.

        Command descriptions:
            abort - Aborts a running YugiQuery flow by terminating the process.
            battle - Simulate a battle of all monster cards.
            benchmark - Show average time each report takes to complete.
            data - Send latest data files.
            git - Run a Git command.
            latest - Show latest time each report was generated.
            links - Show YugiQuery links.
            ping - Test the bot connection latency.
            run - Run full YugiQuery flow.
            status - Display bot status and system information.
            shutdown - Shutdown bot.
        """

        @self.hybrid_command(
            name="abort",
            description="Abort running YugiQuery flow.",
            with_app_command=True,
        )
        @commands.is_owner()
        async def abort(ctx) -> None:
            """
            Aborts a running YugiQuery flow by terminating the process.

            Args:
                ctx (commands.Context): The context of the command.
            """
            original_response = await ctx.send(content="Aborting...", ephemeral=True, delete_after=60)
            response = self.abort()
            await original_response.edit(content=response)

        @self.hybrid_command(
            name="battle",
            description="Simulate a battle of all monster cards.",
            with_app_command=True,
        )
        @commands.is_owner()
        async def battle(ctx, atk_weight: int = 4, def_weight: int = 1) -> None:
            """
            Loads the list of all Monster Cards and simulates a battle between them.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
                atk_weight (int): Weight for ATK stat.
                def_weight (int): Weight for DEF stat.
            """
            await ctx.defer()
            embed = discord.Embed(
                title="Battle",
                description="Simulate a battle of all monster cards",
                color=discord.Colour.purple(),
            )

            original_response = None

            async def callback(first) -> None:
                embed.add_field(name="First contestant", value=first, inline=False)
                embed.set_footer(text="Still battling... ⏳")
                nonlocal original_response
                original_response = await ctx.send(embed=embed)

            response = await self.battle(atk_weight=atk_weight, def_weight=def_weight, callback=callback)

            if "error" in response.keys():
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

        @self.hybrid_command(
            name="benchmark",
            description="Show average time each report takes to complete.",
            with_app_command=True,
        )
        async def benchmark(ctx) -> None:  # Improve function
            """
            Returns the average time each report takes to complete and the latest time for each report.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.defer()
            response = self.benchmark()
            if "error" in response.keys():
                await ctx.send(response["error"])
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

        @self.hybrid_command(name="data", description="Send latest data files.", with_app_command=True)
        async def data(ctx) -> None:
            """
            This command sends the latest data files available in the repository as direct download links.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.defer()
            response = self.data()
            if "error" in response.keys():
                await ctx.send(response["error"])

            else:
                embed = discord.Embed(
                    title=response["title"],
                    description=response["description"],
                    color=discord.Colour.magenta(),
                )
                for field, content in response["fields"].items():
                    embed.add_field(name=field, value=content, inline=False)

                await ctx.send(embed=embed)

        @self.hybrid_command(name="git", description="Execute a Git command in the repository.", with_app_command=True)
        @commands.is_owner()
        async def git_cmd(ctx, command: GitCommands, passphrase: str = "") -> None:
            """
            Executes a Git command in the repository.
            The passphrase is used to decrypt the Git keychain if it is encrypted.
            """

            await ctx.defer()
            response = self.git_cmd(command=command, passphrase=passphrase)
            await self.send_message(ctx, content=response)

        @self.hybrid_command(
            name="latest",
            description="Show latest time each report was generated.",
            with_app_command=True,
        )
        async def latest(ctx) -> None:
            """
            Displays the timestamp of the latest local and live reports generated. Reads the report files from `dirs.REPORTS` and
            queries the GitHub API for the latest commit timestamp for each file. Returns the result as an embedded message in
            the channel.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.defer()
            response = self.latest()
            embed = discord.Embed(
                title=response["title"],
                description=response["description"],
                color=discord.Colour.orange(),
            )
            embed.add_field(name="Local", value=response["local"], inline=False)
            if "live" in response:
                embed.add_field(name="Live", value=response["live"], inline=False)

            await ctx.send(embed=embed)

        @self.hybrid_command(name="links", description="Show YugiQuery links.", with_app_command=True)
        async def links(ctx) -> None:
            """
            Displays the links to the YugiQuery webpage, repository, and data. Returns the links as an embedded message in
            the channel.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            response = self.links()
            embed = discord.Embed(
                title=response["title"],
                description=response["description"],
                color=discord.Colour.green(),
            )

            await ctx.send(embed=embed)

        @self.hybrid_command(
            name="ping",
            description="Test the bot connection latency.",
            with_app_command=True,
        )
        async def ping(ctx) -> None:
            """
            This command tests the bot's connection latency and sends the result back to the user.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            await ctx.send(
                content="🏓 Pong! {0}ms".format(round(self.latency * 1000, 1)),
                ephemeral=True,
                delete_after=60,
            )

        @self.hybrid_command(name="run", description="Run full YugiQuery flow.", with_app_command=True)
        @commands.is_owner()
        @commands.cooldown(rate=1, per=self.cooldown_limit, type=commands.BucketType.user)
        # Typehinting for report needs to be this way to handle dynamic loading of reports
        async def run_query(ctx, report: self.Reports = self.Reports.All) -> None:
            """
            Runs a YugiQuery flow by launching a separate process and monitoring its progress.
            The progress is reported back to the Discord channel where the command was issued.
            The command has a cooldown period of 12 hours per user.

            Args:
                ctx (commands.Context): The context of the command.
                report (Bot.Reports): An Enum value indicating which YugiQuery report to run.

            Raises:
                discord.ext.commands.CommandOnCooldown: If the command is on cooldown for the user.
            """
            original_response = await ctx.send(content="Initializing...", ephemeral=True, delete_after=60)

            async def callback(content: str) -> None:
                await original_response.edit(content=content)

            response = await self.run_query(
                callback=callback,
                report=report,
                progress_bar=self.discord_pbar,
                channel_id=ctx.channel.id,
                token=self.token,
            )
            if "error" in response.keys():
                await self.send_message(ctx.channel, content=response["error"])
                # Reset cooldown in case query did not complete
                ctx.command.reset_cooldown(ctx)
            else:
                await self.send_message(ctx.channel, content=response["content"])

        @self.hybrid_command(
            name="status",
            description="Displays bot status and system information.",
            with_app_command=True,
        )
        async def status(ctx) -> None:
            """
            Displays information about the bot, including uptime, guilds, users, channels, available commands,
            bot version, discord.py version, python version, and operating system.

            Args:
                ctx (discord.ext.commands.Context): The context of the command.
            """
            appInfo = await self.application_info()
            admin = appInfo.owner
            users = 0
            channels = 0
            guilds = len(self.guilds)
            for guild in self.guilds:
                users += len(guild.members)
                channels += len(guild.channels)

            if len(self.commands):
                commandsInfo = " • `\\" + "\n • `\\".join(
                    sorted([f"{i.name}`: {i.description}" for i in self.commands if not i.name == "help"])
                )

            embed = discord.Embed(color=ctx.me.colour)
            embed.set_footer(text="Time to duel!")
            embed.set_thumbnail(url=ctx.me.avatar)
            embed.add_field(name="Admin", value=admin, inline=False)
            embed.add_field(name="Uptime", value=self.uptime(), inline=False)
            embed.add_field(name="Guilds", value=guilds, inline=True)
            embed.add_field(name="Users", value=users, inline=True)
            embed.add_field(name="Channels", value=channels, inline=True)
            embed.add_field(name="Available Commands", value=commandsInfo, inline=False)
            embed.add_field(name="Bot Version", value=__version__, inline=True)
            embed.add_field(name="Discord.py Version", value=discord.__version__, inline=True)
            embed.add_field(name="Python Version", value=platform.python_version(), inline=True)
            embed.add_field(
                name="Operating System",
                value=f" • **Name**: {platform.system()}\n • **Release**: {platform.release()}\n • **Machine**: {platform.machine()}\n • **Version**: {platform.version()}",
                inline=False,
            )
            await ctx.send("**:information_source:** Information about this bot:", embed=embed)

        @self.hybrid_command(name="shutdown", description="Shutdown bot.", with_app_command=True)
        @commands.is_owner()
        async def shutdown(ctx) -> None:
            """
            Shuts down the bot gracefully by sending a message and closing the connection.

            Args:
                ctx (commands.Context): The context of the command.
            """
            await ctx.send(content="Shutting down...")
            await self.close()
