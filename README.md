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

YugiQuery is a Python script to query and display Yu-Gi-Oh! data extracted from the [yugipedia](http://yugipedia.com) database. It is entirely built on Jupyter notebooks and Git. The notebooks are rendered as HTML reports and can be displayed as an "always up to date" static web page by laveraging on GitHub pages. The raw data is kept as CSV files with timestamps and changelogs for a thorough record of the game's history. Every operation is recorded on git with a descriptive commit message. 

# Reports

Below are listed all the available reports and their execution timestamps. 

|                    Report | Last execution       |
| -------------------------:|:-------------------- |
| [Bandai](Bandai.html) | 26/02/2024 10:38 UTC |
| [Cards](Cards.html) | 26/02/2024 10:41 UTC |
| [Rush](Rush.html) | 26/02/2024 10:43 UTC |
| [Sets](Sets.html) | 26/02/2024 10:56 UTC |
| [Speed](Speed.html) | 26/02/2024 10:57 UTC |
| [Timeline](Timeline.html) | 26/02/2024 11:01 UTC |


The full YugiQuery flow was last executed at `26/02/2024 11:01 UTC`

# Usage

The full YugiQuery workflow can be run with 

```
python yugiquery.py
```

Any Jupyter notebook in the ***source*** directory will be assumed to be a report and will be executed and exported to HTML. The index.md and README.md files will be updated, using their respective template files in the ***assets*** directory, to include a table with all the reports available and their timestamps. The source notebooks will then be cleared of their outputs and all changes will be commited to Git.

Report templates are included in the `assets/notebook` folder. Moving them to the source folder will enable them for execution.

To use the optional Discord bot, run

```
python bot.py discord
```

Alternatively, to use the optional Telegram bot, run

```
python bot.py telegram
```

Both `yugiquery.py` and `bot.py` accept command line arguments. Using `-h` or `--help` will print an useful help message listing the parameters that can be passed and their usage. It is also possible to call the script directly as an executable using `./`, although that may be OS dependant.

Further use cases can be found in the [documentation](#documentation).

## Installation

YugiQuery is meant to be user friendly to users without much coding experience. Provided you have Python and Git installed, upon first execution YugiQuery will try to install all its dependencies. If the operation is not succesfull, the user may try to install the dependencies manually, relying on the `install.sh` script and the pip `requirements.txt` file provided. The `install.sh`` script also install a nbconvert template which adds dynamic light and dark modes to the exported html report. This is the default template used by YugiQuery. In case it cannot be installed, the user should change the selected template on each report notebook.

Further details can be found the [documentation](#documentation).

## Repository hierarchy

The repository is structured such that its root contains the web page source files, while the actual executable files are kept in the ***source*** directory. Any template files (markdown, nbconvert, notebook, etc) and files used for reference such as dictionaries are kept in the ***assets*** directory. The raw data used by the reports is saved in the ***data*** directory. The *Read The Docs* source files are kept in the ***docs*** directory. Below is an skeleton of the directory structure.

```
yugiquery/
├─ assets/
│  ├─ json/
│  │  ├─ colors.json
│  │  ├─ dates.json
│  │  ├─ rarities.json
│  │  └─ regions.json
│  ├─ markdown/
│  │  ├─ footer.md
│  │  ├─ header.md
│  │  ├─ index.md
│  │  └─ REAMDME.md
│  ├─ nbconvert/
│  │  ├─ conf.json
│  │  ├─ dynamic.css
│  │  └─ index.html.j2
│  ├─ notebook/
│  │  └─ Template.ipynb
│  ├─ Gateway.html
│  └─ secrets.env
├─ data/
│  ├─ benchmark.json
│  ├─ report_data.csv
│  └─ report_changelog.csv
├─ docs/
│  ├─ Makefile
│  ├─ make.bat
│  ├─ conf.py
│  ├─ index.rst
│  ├─ bot.rst
│  └─ yugiquery.rst
├─ source/
│  ├─ install.sh
│  ├─ requirements.txt
│  ├─ Report.ipynb
│  ├─ bot.py
│  └─ yugiquery.py
├─ _config.yml
├─ .devcontainer.json
├─ .readthedocs.yaml
├─ index.md
├─ LICENSE.md
├─ README.md
└─ Report.html
```

Ideally, files in the ***assets*** directory should be read-only files exclusively for reference. Files in the ***data*** directory are read and write files for the generation of the reports. The root of the repository should only contain files intended for the web page generation by GitHub pages or files that cannot be in another location.

## Documentation

The documentation can be found at [ReadTheDocs](https://yugiquery.readthedocs.io/en/latest/)

## Known limitations

At present, `TQDM` relies on the deprecated `disco-py` package which won't build. To circunvent this problem until the official`TQDM` release drops the `disco-py` dependency, we install `TQDM` from [this fork](https://github.com/guigoruiz1/tqdm), which uses pure REST API and/or `discord.py`.

Recent updates to `IPython` broke `HALO` in Jupyter notebooks. Until `HALO` conforms to the new IPython API, we install it from [this fork](https://github.com/guigoruiz1/halo).

---

###### tags: `Personal` `Public` `yugioh` `python`
