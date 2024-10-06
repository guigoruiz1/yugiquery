<div align='center'>
    <pre>
        <br>
    ██    ██ ██    ██  ██████  ██  ██████  ██    ██ ███████ ██████  ██    ██ 
     ██  ██  ██    ██ ██       ██ ██    ██ ██    ██ ██      ██   ██  ██  ██  
      ████   ██    ██ ██   ███ ██ ██    ██ ██    ██ █████   ██████    ████   
       ██    ██    ██ ██    ██ ██ ██ ▄▄ ██ ██    ██ ██      ██   ██    ██    
       ██     ██████   ██████  ██  ██████   ██████  ███████ ██   ██    ██    
                                      ▀▀                                     
    </pre>
</div>

[![License](https://img.shields.io/github/license/guigoruiz1/yugiquery)](https://github.com/guigoruiz1/yugiquery/blob/main/LICENSE.md)
![Repo size](https://img.shields.io/github/repo-size/guigoruiz1/yugiquery)
![Code size](https://img.shields.io/github/languages/code-size/guigoruiz1/yugiquery)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Read the Docs](https://img.shields.io/readthedocs/yugiquery/latest)](https://yugiquery.readthedocs.io/en/latest/)
[![Pages-build-deployment](https://github.com/guigoruiz1/yugiquery/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/guigoruiz1/yugiquery/actions/workflows/pages/pages-build-deployment)
[![Codespaces Prebuilds](https://github.com/guigoruiz1/yugiquery/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/guigoruiz1/yugiquery/actions/workflows/codespaces/create_codespaces_prebuilds)
[![CodeQL](https://github.com/guigoruiz1/yugiquery/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/guigoruiz1/yugiquery/actions/workflows/github-code-scanning/codeql)
<!-- [![hackmd-github-sync-badge](https://hackmd.io/VkEfdO3nRyuIZedC4FRPZA/badge)](https://hackmd.io/VkEfdO3nRyuIZedC4FRPZA) -->

# What is it?

YugiQuery is a Python package to query and display Yu-Gi-Oh! data extracted from the [yugipedia](http://yugipedia.com) database. It is entirely built on Jupyter notebooks and Git. The notebooks are rendered as HTML reports and can be displayed as an "always up to date" static web page by laveraging on GitHub pages. The raw data is kept as CSV files with timestamps and changelogs for a thorough record of the game's history. Every operation is recorded on git with a descriptive commit message. 

# Reports

Below are listed all the available reports and their execution timestamps. 

|                    Report | Last execution       |
| -------------------------:|:-------------------- |
| [Bandai](reports/Bandai.html) | 04/10/2024 19:18 UTC |
| [Cards](reports/Cards.html) | 04/10/2024 19:20 UTC |
| [Rush](reports/Rush.html) | 04/10/2024 19:20 UTC |
| [Sets](reports/Sets.html) | 04/10/2024 19:24 UTC |
| [Speed](reports/Speed.html) | 04/10/2024 19:25 UTC |
| [Timeline](reports/Timeline.html) | 04/10/2024 19:26 UTC |


The full YugiQuery flow was last executed at `06/10/2024 23:00 UTC`

# Usage

The full YugiQuery workflow can be run directly with 

```
>> yugiquery
```

All commands and options can be displayed with the command
```
>> yugiquery -h
```

Any Jupyter notebook in the ***notebooks*** directory will be assumed to be a report and will be executed and saved as HTML in the ***reports*** directory. The index.md and README.md files will be updated, using their respective template files in the ***assets*** directory, to include a table with all the reports available and their timestamps. The source notebooks will then be cleared of their outputs and all changes will be commited to Git.

Template notebooks are included in the `notebooks/templates` folder.

Further user input can be made through the command
```
>> yugiquery run
```

To use the optional Discord bot, run
```
>> yugiquery bot SUBCLASS
```
Where `SUBCLASS` can be either `telegram` or `discord`.

Both the `yugiquery.py` and `bot.py` modules within the `yugiquery` package accept command line arguments. Using `-h` or `--help` will print an useful help message listing the parameters that can be passed and their usage.

## Installation

YugiQuery is meant to be user friendly to users without much coding experience. It can be used "as is" from the repository, or installed via pip.

The `post_install.py` script in the assets directory has options to install: 
1 - A nbconvert template which adds dynamic light and dark modes to the exported html report. This is the default template used by YugiQuery.
2 - The TQDM fork needed to run the discord bot subclass.
3 - A jupyter kernel for the current envyronment.

It can be run from the main yugiquery CLI with the command
```
>> yugiquery install
```

Further details can be found the [documentation](#documentation).

## Repository hierarchy

The repository is structured such that its root contains the web page index files, while the package files are kept in the ***yugiquery*** directory. Any template files (markdown, nbconvert, notebook, etc) and files used for reference such as dictionaries are kept in the ***assets*** directory. The raw data used by the reports is saved in the ***data*** directory. The *Read The Docs* source files are kept in the ***docs*** directory. The HTML reports are generated from the notebooks in the **notebooks** directory and saved in the **reports** directory. Below is an skeleton of the directory structure.

```
yugiquery/
├─ assets/
│  ├─ json/
│  │  ├─ colors.json
│  │  ├─ dates.json
│  │  ├─ headers.json
│  │  ├─ rarities.json
│  │  └─ regions.json
│  ├─ markdown/
│  │  ├─ footer.md
│  │  ├─ header.md
│  │  ├─ index.md
│  │  └─ REAMDME.md
│  ├─ scripts/
│  │  ├─ git_filters.sh
│  │  ├─ post_install.py
│  │  └─ unlock_git.sh
│  ├─ nbconvert/
|  |  └─ labdynamic/
|  │      ├─ conf.json
|  │      ├─ dynamic.css
|  │      └─ index.html.j2
│  ├─ templates/
│  │  ├─ Collection.ipynb
│  │  └─ Template.ipynb
│  ├─ Gateway.html
│  └─ secrets.env
├─ data/
│  ├─ benchmark.json
│  ├─ report_data.bz2
│  └─ report_changelog.bz2
├─ docs/
│  ├─ Makefile
│  ├─ make.bat
│  ├─ conf.py
│  ├─ utils.rst
│  ├─ index.rst
│  ├─ bot.rst
│  └─ yugiquery.rst
├─ notebooks/
│  ├─ Bandai.ipynb
│  ├─ Cards.ipynb
│  ├─ Rush.ipynb
│  ├─ Sets.ipynb
│  ├─ Speed.ipynb
│  └─ Timeline.ipynb
├─ reports/
│  └─ report.html
├─ yugiquery/
│  ├─ utils
│  |  ├─ __init__.py
│  |  ├─ api.py
│  |  ├─ dirs.py
│  |  ├─ git.py
│  |  ├─ helpers.py
│  |  ├─ plot.py
│  |  └─ progress_handler.py
│  ├─ bot
│  |  ├─ __init__.py
│  |  ├─ __main__.py
│  |  ├─ base.py
│  |  ├─ discord.py
│  |  └─ telegram.py
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ metadata.py
│  └─ yugiquery.py
├─ _config.yml
├─ .devcontainer.json
├─ .readthedocs.yaml
├─ index.md
├─ LICENSE.md
├─ pyproject.toml
├─ README.md
└─ requirements.txt
```

Ideally, files in the ***assets*** directory should not be edited unless you know what you are doing. Files in the ***data*** directory are read and write files for the generation of the reports. The root of the repository should only contain files intended for the web page generation by GitHub pages or files that cannot be in another location.

## Documentation

The documentation can be found at [ReadTheDocs](https://yugiquery.readthedocs.io/en/latest/)

## Known limitations

At present, `TQDM` relies on the deprecated `disco-py` package which won't build. To circunvent this problem until the official`TQDM` release drops the `disco-py` dependency, we install `TQDM` from [this fork](https://github.com/guigoruiz1/tqdm), which uses pure REST API and/or `discord.py`.

Recent updates to `IPython` broke `HALO` in Jupyter notebooks. Until `HALO` conforms to the new IPython API, we install it from [this fork](https://github.com/guigoruiz1/halo).

---

###### tags: `Personal` `Public` `yugioh` `python`
