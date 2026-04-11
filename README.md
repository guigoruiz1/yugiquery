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
[![CodeQL](https://github.com/guigoruiz1/yugiquery/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/guigoruiz1/yugiquery/actions/workflows/github-code-scanning/codeql)

# What is it?

YugiQuery is a Python package to query and display Yu-Gi-Oh! data extracted from the [yugipedia](http://yugipedia.com) database. It is entirely built on Jupyter notebooks and Git. The notebooks are rendered as HTML reports and can be displayed as an "always up to date" static web page by leveraging on GitHub pages. The raw data is kept as CSV files with timestamps and changelogs for a thorough record of the game's history. Every operation is recorded on git with a descriptive commit message. 

# Reports

Below are listed all the available reports and their execution timestamps. 

<!-- REPORT_TABLE_START -->

|                    Report | Last execution       |
| -------------------------:|:-------------------- |
| [Bandai](reports/Bandai.html) | 11/04/2026 14:20 UTC |
| [Cards](reports/Cards.html) | 11/04/2026 14:20 UTC |
| [Rush](reports/Rush.html) | 11/04/2026 14:20 UTC |
| [Sets](reports/Sets.html) | 11/04/2026 14:20 UTC |
| [Speed](reports/Speed.html) | 11/04/2026 14:20 UTC |
| [Timeline](reports/Timeline.html) | 11/04/2026 14:20 UTC |

<!-- REPORT_TABLE_END -->

YugiQuery flow was last executed at `11/04/2026 17:11 UTC`.

# Usage

The full YugiQuery workflow can be run directly with 

```bash
yugiquery
```

All commands and options can be displayed with the command

```bash
yugiquery -h
```

Any Jupyter notebook in the ***notebooks*** directory will be assumed to be a report and will be executed and saved as HTML in the ***reports*** directory. The index.md and README.md files will be updated, using their respective template files in the ***assets*** directory, to include a table with all the reports available and their timestamps. The source notebooks will then be cleared of their outputs and all changes will be committed to Git.

Template notebooks are included in the `assets/templates` folder.

Further user input can be made through the command

```bash
yugiquery run
```

To use the optional Discord bot, run

```bash
yugiquery bot SUBCLASS
```
Where `SUBCLASS` can be either `telegram` or `discord`.

Both the main CLI (`yugiquery` / `python -m yugiquery`) and bot commands (`yugiquery bot ...`) accept command line arguments. Using `-h` or `--help` prints a help message with the available parameters and usage.


## Installation

YugiQuery is meant to be user friendly to users without much coding experience. It can be used "as is" from the repository, or installed via pip.

### Requirements

- Python >= 3.11.
- Full dependency list: [pyproject.toml](pyproject.toml) / [requirements.txt](requirements.txt).

### Install

Install from source:

```bash
pip install -e .
# optional: pip install -r requirements.txt
```

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
yugiquery run
```

Reports are written to `reports/`.

### Optional installer

`yugiquery install` (or `python assets/scripts/post_install.py`) supports: `--templates`, `--kernel`, `--nbconvert`, `--filters`, `--venv`.

- `--templates`: copy notebook/spreadsheet templates from `assets/templates`.
- `--kernel`: create/register a `yugiquery` IPython kernel.
- `--nbconvert`: install the custom nbconvert template.
- `--filters`: install Git filters to help clean notebooks (updates repo git config).
- `--venv`: create a venv and register its Python as the `yugiquery` kernel.

Most actions need write access to the repo or Jupyter config; `sudo` is usually not required.

Further details can be found in the [documentation](#documentation).


## Repository hierarchy

The repository is organized with package code in `yugiquery/`, templates and reference files in `assets/`, data and exports in `data/`, generated reports in `reports/`, and source notebooks in `notebooks/`. Jupyter notebooks are executed as reports and saved as HTML. The ReadTheDocs documentation source lives in `docs/`.

```
yugiquery/
├─ assets/
│  ├─ html/                   # Reusable HTML snippets for rendered pages
│  ├─ json/                   # Reference dictionaries (colors, dates, headers, rarities, regions)
│  ├─ scripts/                # Utility scripts (e.g., git filters)
│  └─ templates/              # Notebook templates used by installer/workflow
├─ data/                      # Persistent datasets and exports
│  ├─ *.ydk / *.txt           # Deck files (Yu-Gi-Oh card decks)
│  ├─ collection.csv / .xlsx  # User card collection
│  ├─ ygoprodeck.json         # API cache from YGOProDeck
│  ├─ benchmark.json          # Performance/test data snapshots
│  └─ *_data_*.bz2 / *_changelog_*.bz2  # Historical compressed data/changelog snapshots
├─ docs/                      # ReadTheDocs source (Sphinx)
├─ notebooks/                 # Source report notebooks
├─ reports/                   # Generated HTML reports
├─ tests/                     # Automated tests
├─ yugiquery/
│  ├─ __init__.py             # Package exports
│  ├─ __main__.py             # Module entry point (python -m yugiquery)
│  ├─ cli.py                  # CLI command orchestration
│  ├─ metadata.py             # Package metadata/constants
│  ├─ api/                    # Yugipedia/YGOProDeck access layer
│  ├─ bot/                    # Telegram/Discord bot implementations
│  ├─ core/                   # Data/deck/pipeline/maintenance workflows
│  ├─ scripts/                # Internal maintenance/automation scripts
│  └─ utils/                  # Dirs, git, notebook, image, plot, progress helpers
├─ pyproject.toml             # Package/project configuration
├─ requirements.txt           # Pinned dependencies
├─ index.md                   # Project site landing page content
└─ README.md                  # Main project guide
```

## Documentation

The documentation can be found at [ReadTheDocs](https://yugiquery.readthedocs.io/en/latest/).

## Known limitations

Recent updates to `IPython` broke `HALO` in Jupyter notebooks. Until `HALO` conforms to the new IPython API, we install it from [this fork](https://github.com/guigoruiz1/halo).

---

###### tags: `Personal` `Public` `yugioh` `python`

