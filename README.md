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
| [Bandai](Bandai.html) | 19/06/2023 08:20 UTC |
| [Cards](Cards.html) | 19/06/2023 08:23 UTC |
| [Rush](Rush.html) | 19/06/2023 08:24 UTC |
| [Sets](Sets.html) | 19/06/2023 08:35 UTC |
| [Speed](Speed.html) | 19/06/2023 08:36 UTC |
| [Timeline](Timeline.html) | 19/06/2023 08:40 UTC |


The full YugiQuery flow was last executed at `19/06/2023 08:40 UTC`

# Usage

The full YugiQuery workflow can be run with 

```
python yugiquery.py
```

Any Jupyter notebook in the *source* directory will be assumed to be a report and will be executed and exported to HTML. The index.md and README.md files will be updated, using their template files in the *assets* directory, to include a table with all the reports available and their timestamps. The source notebooks will then be cleared of their outputs and all changes will be commited to Git.

## Installation

YugiQuery is meant to be user friendly to users without much coding experience. Provided you have Python and Git installed, upon first execution YugiQuery will try to install all its dependencies. If the operation is not succesfull, the user may try to install the dependencies manually relying on the install.sh script. A pip requirements.txt file is also provided, but it does not install every dependency installed by the install.sh script.

## Repository hierarchy

The repository is structured such that its root contains the web page source files while the actual executable files are kept in the *source* directory. Any template files and files used for reference such as dictionaries are kept in the *assets* directory. The raw data is saved in the *data* directory and the ReadTheDocs source files are kept the *docs* directory. Below is an example of the basic structure of the directory.

```
yugiquery/
├─ assets/
│  ├─ colors/
│  ├─ dates.json
│  ├─ footer.md
│  ├─ Gateway.html
│  ├─ header.md
│  ├─ index.md
│  ├─ rarities.json
│  ├─ README.md
│  ├─ regions.json
│  ├─ secrets.env
│  └─ Template.ipynb
├─ data/
│  ├─ benchmark.json
│  ├─ data.csv
│  └─ data_changelog.csv
├─ docs/
│  ├─ bot.rst
│  ├─ conf.py
│  ├─ index.rst
│  └─ yugiquery.rst
├─ source/
│  ├─ bot.py
│  ├─ install.sh
│  ├─ Report.ipynb
│  ├─ requirements.txt
│  └─ yugiquery.py
├─ .readthedocs.yaml
├─ index.md
├─ LICENSE.md
├─ README.md
├─ Report.html
└─ _config.yml
```

Ideally, files in the *assets* directory should be read-only files for reference only. Files in the *data* directory are read and write files for the generation of the reports. The root of the repository should only contain files intended for the web page generation by GitHub pages or files that cannot be in another location.

## Documentation

The documentation can be found at [ReadTheDocs](https://yugiquery.readthedocs.io/en/latest/)

---

###### tags: `Personal` `Public` `yugioh` `python`
