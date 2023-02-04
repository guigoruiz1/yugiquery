__author__ = "Guilherme Ruiz"
__copyright__ = "Copyright 2023, Yugiquery"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Guilherme Ruiz"
__email__ = "57478888+guigoruiz1@users.noreply.github.com"
__status__ = "Development"


###########
# Imports #
###########

loop = 0
while True:
    try:
        import git
        import subprocess
        import ipynbname
        import glob
        import os
        import string
        import calendar
        import warnings
        import colorsys
        import logging
        import io
        import nbformat
        import hashlib
        import asyncio
        import aiohttp
        import requests
        import socket
        import json
        import re
        from enum import Enum
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import urllib.parse as up
        import wikitextparser as wtp
        import papermill as pm
        import matplotlib.pyplot as plt
        import matplotlib.colors as mc # LogNorm, Normalize, ListedColormap, cnames, to_rgb
        import matplotlib.dates as mdates
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FixedLocator
        from matplotlib_venn import venn2
        from datetime import datetime, timezone
        from ast import literal_eval
        from IPython.display import Markdown
        from textwrap import wrap
        from tqdm.auto import tqdm, trange
        from ipylab import JupyterFrontEnd
        from dotenv import dotenv_values
        
        # Defaults overrides
        pd.set_option('display.max_columns', 40)

        break

    except ImportError:
        if loop>1:
            print("Failed to install required packages twice. Aborting...")
            quit()
        
        loop+=1   
        import subprocess
        print("Missing required packages. Trying to install now...")
        subprocess.call(['sh', './install.sh'])


###########      
# Helpers #
###########

# Secrets
def load_secrets(secrets_file, requested_secrets=[], required=False):
    if os.path.isfile(secrets_file):
        secrets = dotenv_values(secrets_file)
        if not requested_secrets:
            return secrets
        
        found_secrets = {key: secrets[key] for key in requested_secrets if key in secrets.keys() and secrets[key]}
        if required:
            for i, key in enumerate(requested_secrets):
                check = required if isinstance(required, bool) else required[i]
                if check and key not in found_secrets.keys():
                    raise KeyError(f'Secret \"{requested_secrets[i]}\" not found')
                
        return found_secrets   
    
    else:
        raise FileNotFoundError(f'No such file or directory: {secrets_file}')
        
def load_json(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            return data
    except:
        print(f'Error loading {json_file}. Ignoring...')
        return {}
    
# Validators
def md5(file_name):
    hash_md5 = hashlib.md5()
    hash_md5.update(file_name.encode())
    return hash_md5.hexdigest()

class CG(Enum):
    CG = 'CG'
    ALL = CG
    BOTH = CG
    TCG = 'TCG'
    OCG = 'OCG'
    
# Attributes list to split monsters query
attributes = [
    'DIVINE', 
    'LIGHT', 
    'DARK', 
    'WATER', 
    'EARTH', 
    'FIRE', 
    'WIND'
]

# Region abreviations dictionary
regions_dict = {
    'EN': 'English', 
    'NA': 'North American English',
    'EU': 'European English', 
    'AU': 'Oceanic English', 
    'PT': 'Portuguese', 
    'DE': 'German', 
    'FC': 'French-Canadian', 
    'FR': 'French', 
    'IT': 'Italian', 
    'SP': 'Spanish', 
    'JP': 'Japanese', 
    'JA': 'Japanese-Asian', 
    'AE': 'Asian-English', 
    'KR': 'Korean', 
    'TC': 'Traditional Chinese', 
    'SC': 'Simplified Chinese'
}

# Arrow unicode simbols dictionary
arrows_dict = {
    'Middle-Left': '\u2190', 
    'Middle-Right': '\u2192', 
    'Top-Left': '\u2196', 
    'Top-Center': '\u2191', 
    'Top-Right': '\u2197', 
    'Bottom-Left': '\u2199', 
    'Bottom-Center': '\u2193', 
    'Bottom-Right': '\u2198'
}

# Images
async def download_images(file_name: pd.DataFrame, save_folder: str = "../images/", max_tasks: int = 10):
    # Prepare URL from file names
    file_name_md5 = file_name.apply(md5)
    urls = file_name_md5.apply(lambda x: f'/{x[0]}/{x[0]}{x[1]}/')+file_name
    
    # Download image from URL
    async def download_image(session, url, save_folder, semaphore, pbar):
        async with semaphore:
            async with session.get(url) as response:
                save_name = url.split("/")[-1]
                if response.status != 200:
                    raise ValueError(f"URL {url} returned status code {response.status}")
                total_size = int(response.headers.get("Content-Length", 0))
                progress = tqdm(unit="B", total=total_size, unit_scale=True, unit_divisor=1024, desc=save_name, leave=False)
                if os.path.isfile(f'{save_folder}/{save_name}'):
                    os.remove(f'{save_folder}/{save_name}')
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    progress.update(len(chunk))
                    with open(f'{save_folder}/{save_name}', 'ab') as f:
                        f.write(chunk)
                progress.close()
                return save_name
            
    # Parallelize image downloads
    semaphore = asyncio.Semaphore(max_tasks)
    async with aiohttp.ClientSession(base_url='https://ms.yugipedia.com/', headers=http_headers) as session:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with tqdm(total=len(urls), unit_scale=True, unit="file") as pbar:
            tasks = [download_image(session, url, save_folder, semaphore, pbar) for url in urls]
            for task in asyncio.as_completed(tasks):
                pbar.update()
                await task

# Data management
def cleanup_data(dry_run: bool = False):
    file_list = glob.glob('../data/*')
    df = pd.DataFrame(file_list, columns=['file'])
    df['timestamp'] = pd.to_datetime(df['file'].apply(os.path.getctime), unit='s')
    df['group'] = df['file'].apply(lambda x: x.split('/')[-1]).apply(lambda x: x[:x.rindex('_')])
    df = df.sort_values(['group', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    # Monthly
    keep_monthly = df.copy()
    keep_monthly['Y+m'] = keep_monthly['timestamp'].dt.strftime('%Y%m')
    keep_monthly.drop_duplicates(['Y+m','group'], keep="first", inplace = True)
    # Weekly
    keep_weekly = keep_monthly.where(keep_monthly['Y+m']==keep_monthly['Y+m'].min())
    keep_weekly['W'] = keep_monthly['timestamp'].dt.strftime('%W')
    keep_weekly.drop_duplicates(['W','group'], keep="first", inplace = True)

    drop_index = keep_monthly.index.join(keep_weekly.index)
    for file in df.loc[~df.index.isin(drop_index),'file']:
        if dry_run:
            print(file)
        else:
            os.remove(file)
            
# Data formating functions
def extract_fulltext(x):
    if len(x)>0:
        if isinstance(x[0], int):
            return str(x[0])
        elif 'fulltext' in x[0]:
            return x[0]['fulltext'].strip('\u200e')
        else:
            return x[0].strip('\u200e')
    else:
        return np.nan
    
def format_df(input_df: pd.DataFrame, include_all: bool = False):
    df = pd.DataFrame(index=input_df.index)
    
    # Cards
    if 'Name' in input_df.columns:
        df['Name'] = input_df['Name'].dropna().apply(extract_fulltext)
    if 'Password' in input_df.columns:
        df['Password'] = input_df['Password'].dropna().apply(extract_fulltext)
    if 'Card type' in input_df.columns:
        df['Card type'] = input_df['Card type'].dropna().apply(extract_fulltext)
    if 'Property' in input_df.columns:
        df['Property'] = input_df['Property'].dropna().apply(extract_fulltext)
    if 'Primary type' in input_df.columns:
        df['Primary type'] = input_df['Primary type'].dropna().apply(lambda x: [i['fulltext'] for i in x] if len(x)>0 else []).apply(lambda y: list(filter(lambda z: z != 'Pendulum Monster', y)) if len(y)>0 else []).apply(lambda y: list(filter(lambda z: z != 'Effect Monster', y))[0] if len(y)>1 else (y[0] if len(y)>0 else np.nan))
    if 'Secondary type' in input_df.columns:
        df['Secondary type'] = input_df['Secondary type'].dropna().apply(extract_fulltext)
    if 'Attribute' in input_df.columns:
        df['Attribute'] = input_df['Attribute'].dropna().apply(extract_fulltext)
    if 'Monster type' in input_df.columns:
        df['Monster type'] = input_df['Monster type'].dropna().apply(extract_fulltext)
    if 'Level/Rank' in input_df.columns:
        df['Level/Rank'] = input_df['Level/Rank'].dropna().apply(extract_fulltext)
    if 'ATK' in input_df.columns:
        df['ATK'] = input_df['ATK'].dropna().apply(extract_fulltext)
    if 'DEF' in input_df.columns:
        df['DEF'] = input_df['DEF'].dropna().apply(extract_fulltext)
    if 'Pendulum Scale' in input_df.columns:
        df['Pendulum Scale'] = input_df['Pendulum Scale'].dropna().apply(extract_fulltext)
    if 'Link' in input_df.columns:
        df['Link'] = input_df['Link'].dropna().apply(extract_fulltext)
    if 'Link Arrows' in input_df.columns:
        df['Link Arrows'] = input_df['Link Arrows'].dropna().apply(lambda x: tuple([arrows_dict[i] for i in sorted(x)]) if len(x)>0 else np.nan)
    if 'Effect type' in input_df.columns:
        df['Effect type'] = input_df['Effect type'].dropna().apply(lambda x: tuple(sorted([i['fulltext'] for i in x])) if len(x)>0 else np.nan)
    if 'Archseries' in input_df.columns:
        df['Archseries'] = input_df['Archseries'].dropna().apply(lambda x: tuple(sorted(x)) if len(x)>0 else np.nan)
    if 'TCG status' in input_df.columns:
        df['TCG status'] = input_df['TCG status'].dropna().apply(extract_fulltext)
    if 'OCG status' in input_df.columns:
        df['OCG status'] = input_df['OCG status'].dropna().apply(extract_fulltext)
    if 'Card image' in input_df.columns:
        df['Card image'] = input_df['Card image'].apply(extract_fulltext)
    
    # Sets
    if 'Series' in input_df.columns:
        df['Series'] = input_df['Series'].apply(extract_fulltext)
    if 'Set type' in input_df.columns:
        df['Set type'] = input_df['Set type'].apply(extract_fulltext)
    if 'Cover card' in input_df.columns:
        df['Cover card'] = input_df['Cover card'].apply(lambda x: tuple(sorted([y['fulltext'] for y in x])) if len(x)>0 else np.nan)
        
    # Category
    if 'Category'in input_df.columns:
        df['Category'] = input_df['Category'].dropna().apply(lambda x: tuple(sorted([y['fulltext'] for y in x])) if len(x)>0 else np.nan)

    # Artworks columns
    if len(input_df.filter(like=' artwork').columns)>0:
        df['Artwork'] = input_df.filter(like=' artworks').applymap(extract_category_bool).apply(format_artwork, axis=1)
    
    # Page columns
    if len(input_df.filter(like='Page ').columns)>0:
        df = df.join(input_df.filter(like='Page '))
    
    # Date columns    
    if len(input_df.filter(like=' date').columns)>0:
        df = df.join(input_df.filter(like=' date').applymap(lambda x:pd.to_datetime(x[0]['timestamp'], unit = 's', errors = 'coerce') if len(x)>0 else np.nan))
    
    # Include other unspecified columns
    if include_all:
        df.join(input_df[~df.columns])
        
    return df

## Cards
def extract_category_bool(x):
    if len(x)>0:
        if x[0]=='f':
            return False
        elif x[0]=='t':
            return True
    
    return np.nan

def format_artwork(row: pd.Series):
    result = tuple()
    if 'OCG/TCG cards with alternate artworks' in row: 
        if row['OCG/TCG cards with alternate artworks']:
            result += ('Alternate',)
    if 'OCG/TCG cards with edited artworks' in row: 
        if row['OCG/TCG cards with edited artworks']:
            result += ('Edited',)
    if result == tuple():
        return np.nan
    else:
        return result

def format_errata(row: pd.Series):
    result = tuple()
    if 'Name errata' in row: 
        if row['Name errata']:
            result += ('Name',)
    if 'Type errata' in row:  
        if row['Type errata']:
            result += ('Type',)
    if result == tuple():
        return np.nan
    else:
        return result 
    
def merge_errata(input_df: pd.DataFrame, input_errata_df: pd.DataFrame, drop: bool = False):
    if 'Page name' in input_df.columns:
        input_errata_df = input_errata_df.apply(format_errata,axis=1).rename('Errata')
        input_df = input_df.merge(input_errata_df, left_on = 'Page name', right_index = True, how='left')
        if drop:
            input_df.drop('Page name', axis=1, inplace=True)
    else:
        print('Error! No \"page\" name column to join errata')
    
    return input_df

## Sets
def merge_set_info(input_df: pd.DataFrame, input_info_df: pd.DataFrame):
    if all([col in input_df.columns for col in ['Set', 'Region']]):
        input_df['Release'] = input_df[['Set','Region']].apply(lambda x: input_info_df[regions_dict[x['Region']]+' release date'][x['Set']] if (x['Region'] in regions_dict.keys() and x['Set'] in input_info_df.index) else np.nan, axis = 1)
        input_df['Release'] = pd.to_datetime(input_df['Release'].astype(str), errors='coerce') # Bug fix
        input_df = input_df.merge(input_info_df.loc[:,:'Modification date'], left_on = 'Set', right_index = True, how = 'outer', indicator = True).reset_index(drop=True) 
        print('Set properties merged')
    else:
        print('Error! No \"Set\" and/or \"Region\" column(s) to join set info')
        
    return input_df


#############
# Changelog #
#############

def generate_changelog(previous_df: pd.DataFrame, current_df: pd.DataFrame, col: str):
    changelog = previous_df.merge(current_df, indicator = True, how='outer').loc[lambda x : x['_merge']!='both'].sort_values(col, ignore_index=True)
    changelog['_merge'].replace(['left_only','right_only'],['Old', 'New'], inplace = True)
    changelog.rename(columns={"_merge": "Version"}, inplace = True)
    nunique = changelog.groupby(col).nunique(dropna=False)
    cols_to_drop = nunique[nunique < 2].dropna(axis=1).columns
    changelog.drop(cols_to_drop, axis=1, inplace = True)
    changelog = changelog.set_index(col)
    
    if all(col in changelog.columns for col in ['Modification date', 'Version']):
        true_changes = changelog.drop(['Modification date', 'Version'], axis = 1)[nunique>1].dropna(axis=0, how='all').index
        new_entries = nunique[nunique['Version'] == 1].dropna(axis=0, how='all').index
        rows_to_keep = true_changes.union(new_entries).unique()
        changelog = changelog.loc[rows_to_keep].sort_values([col,'Version'])
    
    if changelog.empty:
        print('No changes')
        
    return changelog


###########
# Styling #
###########

def style_df(df: pd.DataFrame):
    return df.style.format(hyperlinks='html')

#######################
# Notebook management #
#######################

# Frontend shortcuts
## Force saving the current notebook to disk
def save_notebook():
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    print("Notebook saved to disk")

## Remove output from all notebooks in the source directory
def clear_notebooks(which = 'all'):
    if which=='all':
        # Get reports
        reports = sorted(glob.glob('*.ipynb'))
    else:
        reports = [str(which)] if not isinstance(which,list) else which
    if len(reports)>0:
        subprocess.call(['nbstripout']+reports)

## Run all notebooks in the source directory
def run_notebooks(which='all', progress_handler=None):  
    if which=='all':
        # Get reports
        reports = sorted(glob.glob('*.ipynb'))
    else:
        reports = [str(which)] if not isinstance(which,list) else which

    if progress_handler:
        external_pbar = progress_handler(iterable=reports, desc="Completion", unit='report', unit_scale=True)
    else:
        external_pbar = None

    # Initialize iterators
    try:
        required_secrets = ['DISCORD_TOKEN','DISCORD_CHANNEL_ID']
        secrets_file = '../assets/secrets.env'
        secrets = load_secrets(secrets_file, required_secrets, required=True)
        from tqdm.contrib.discord import tqdm as discord_tqdm
        iterator = discord_tqdm(reports, desc="Completion", unit='report', unit_scale=True, token=secrets['DISCORD_TOKEN'], channel_id=secrets['DISCORD_CHANNEL_ID'])
    except:
        iterator = tqdm(reports, desc="Completion", unit='report')

    # Get papermill logger
    logger = logging.getLogger("papermill")
    logger.setLevel(logging.INFO)

    # Create a StreamHandler and attach it to the logger
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.addFilter(lambda record: record.getMessage().startswith("Ending Cell"))
    logger.addHandler(stream_handler)

    # Define a function to update the output variable
    def update_pbar():
        iterator.update((1/cells))
        if external_pbar:
            external_pbar.update((1/cells))

    for i, report in enumerate(iterator):
        iterator.n = i
        iterator.last_print_n = i
        iterator.refresh()

        with open(report) as f:
            nb = nbformat.read(f,nbformat.NO_CONVERT)
            cells = len(nb.cells)
            # print(f'Number of Cells: {cells}')

        # Attach the update_pbar function to the stream_handler
        stream_handler.flush = update_pbar

        # Update postfix
        tqdm.write(f'Generating {report[:-6]} report')
        iterator.set_postfix(report=report)
        if external_pbar:
            external_pbar.set_postfix(report=report)

        # execute the notebook with papermill
        pm.execute_notebook(
            report,
            report,
            log_output=True,
            progress_bar=True,
        );

    # Close the iterator
    iterator.close()
    if external_pbar:
        external_pbar.close()

    # Close the stream_handler
    stream_handler.close()
    # Clear custom handler
    logger.handlers.clear()


####################
# Markdown editing #
####################

# Update webpage index with timestamps
def update_index(): # Handle paths properly
    index_file_name='README.md'
    timestamp = datetime.now().astimezone(timezone.utc)
    try:
        with open(f'../assets/index.md') as f:
            readme = f.read()
            reports = sorted(glob.glob('../*.html'))
            for report in reports:
                readme = readme.replace(f'@{os.path.basename(report)[:-5].upper()}_TIMESTAMP@', pd.to_datetime(os.path.getmtime(report),unit='s', utc=True).strftime("%d/%m/%Y %H:%M %Z"))

            readme = readme.replace(f'@TIMESTAMP@', timestamp.strftime("%d/%m/%Y %H:%M %Z"))
            with open(f'../{index_file_name}', 'w') as o:
                print(readme, file=o)

        try:
            repo = git.Repo(f'../')
            repo.git.commit('-m', f'index timestamp update-{timestamp.strftime("%d%m%Y")}', f'{index_file_name}')
        except:
            print('Failed to commit to git')

    except:
        print('No "index.md" file in "assets". Aborting...')

# Generate Markdown header
def header(name: str = None):
    if name is None:
        try: 
            name = ipynbname.name()
        except:
            name = ''

    with open('../assets/header.md') as f:
        header = f.read()
        header = header.replace('@TIMESTAMP@', datetime.now().astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M %Z"))
        header = header.replace('@NOTEBOOK@', name)
        return Markdown(header)

# Generate Markdown footer
def footer():
    with open('../assets/footer.md') as f:
        footer = f.read()
        footer = footer.replace('@TIMESTAMP@', datetime.now().astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M %Z"))
        return Markdown(footer)

    
######################
# API call functions #
######################

# Variables
http_headers = {'User-Agent': 'Yugiquery/1.0 - https://guigoruiz1.github.io/yugiquery/'}
base_url = 'https://yugipedia.com/api.php'
media_url='https://ws.yugipedia.com/'
revisions_query_action = '?action=query&format=json&prop=revisions&rvprop=content&titles='
ask_query_action='?action=ask&format=json&query='
askargs_query_action = '?action=askargs&format=json&conditions='
categorymembers_query_action = '?action=query&format=json&list=categorymembers&cmdir=desc&cmsort=timestamp&cmtitle=Category:'
redirects_query_action = '?action=query&format=json&redirects=True&titles='

# Extract results from query response
def extract_results(response: requests.Response):
    json = response.json()
    df = pd.DataFrame(json['query']['results']).transpose()
    df = pd.DataFrame(df['printouts'].values.tolist(), index = df['printouts'].keys())
    page_url=pd.DataFrame(json['query']['results']).transpose()['fullurl'].rename('Page URL')
    page_name=pd.DataFrame(json['query']['results']).transpose()['fulltext'].rename('Page name') # Not necessarily same as card name (Used to merge errata)
    df = pd.concat([df,page_name,page_url],axis=1)
    return df

# Cards Query arguments shortcut
def card_query(default: str = None, _password: bool = True, _card_type: bool = True, _property: bool = True, _primary: bool = True, _secondary: bool = True, _attribute: bool = True, _monster_type: bool = True, _stars: bool = True, _atk: bool = True, _def: bool = True, _scale: bool = True, _link: bool = True, _arrows: bool = True, _effect_type: bool = True, _archseries: bool = True, _name_errata: bool = True, _type_errata: bool = True, _alternate_artwork: bool = True, _edited_artwork: bool = True, _tcg: bool = True, _ocg: bool = True, _date: bool = True, _page_name: bool = True, _category: bool = False, _image_URL: bool = False):
    if default is not None:
        default = default.lower() 
    valid_default = {'spell', 'trap', 'st', 'monster', None}
    if default not in valid_default:
        raise ValueError("results: default must be one of %r." % valid_default)
    elif default=='monster':
        return card_query(
            _property = False
        )
    elif default=='st' or default=='trap' or default=='spell':
        return card_query(
            _primary = False,
            _secondary = False,
            _attribute = False, 
            _monster_type = False, 
            _stars = False, 
            _atk = False, 
            _def = False, 
            _scale = False, 
            _link = False, 
            _arrows = False
        )
    else:
        search_string = '|?English%20name=Name'
        if _password:
            search_string += '|?Password'
        if _card_type:
            search_string += '|?Card%20type'
        if _property:    
            search_string += '|?Property'
        if _primary:
            search_string += '|?Primary%20type'
        if _secondary:
            search_string += '|?Secondary%20type'
        if _attribute:
            search_string += '|?Attribute'
        if _monster_type:
            search_string += '|?Type=Monster%20type'
        if _stars:
            search_string += '|?Stars%20string=Level%2FRank%20'
        if _atk:
            search_string += '|?ATK%20string=ATK'
        if _def:
            search_string += '|?DEF%20string=DEF'
        if _scale:
            search_string += '|?Pendulum%20Scale'
        if _link:
            search_string += '|?Link%20Rating=Link'
        if _arrows:
            search_string += '|?Link%20Arrows'
        if _effect_type:
            search_string += '|?Effect%20type'
        if _archseries:
            search_string += '|?Archseries'
        if _alternate_artwork:
            search_string += '|?Category:OCG/TCG%20cards%20with%20alternate%20artworks'
        if _edited_artwork:
            search_string += '|?Category:OCG/TCG%20cards%20with%20edited%20artworks'
        if _name_errata:
            search_string += '|?Category:Cards%20with%20name%20errata'
        if _type_errata:
            search_string += '|?Category:Cards%20with%20card%20type%20errata'
        if _tcg:
            search_string += '|?TCG%20status'
        if _ocg:
            search_string += '|?OCG%20status'
        if _date:
            search_string += '|?Modification%20date'
        if _image_URL:
            search_string += '|?Card%20image' 
        if _category: # Deprecated - Use for debuging
            search_string += '|?category' 

        return search_string

# Check if API is live and responsive    
def check_API_status():
    params = {'action': 'query', 'meta': 'siteinfo', 'siprop': 'general', 'format': 'json'}

    try:
        response = requests.get(base_url, params=params, headers=http_headers)
        response.raise_for_status()
        print(f"{base_url} is up and running {response.json()['query']['general']['generator']}")
        return True
    except requests.exceptions.RequestException as err:
        print(f"{base_url} is not alive: {err}")  
        domain = up.urlparse(base_url).netloc
        port = 443

        try:
            socket.create_connection((domain, port), timeout=2)
            print(f"{domain} is reachable")
        except socket.timeout:
            print(f"{domain} is not reachable")

        return False

# Fetch category members - still not used
def fetch_categorymembers(category: str, namespace: int = 0, step: int = 500):
    params = {
        'cmtitle': category, 
        'cmlimit': step, 
        'cmnamespace': namespace 
    }

    lastContinue = {}
    all_results = []
    while True:
        params = params.copy()
        params.update(lastContinue)
        result = requests.get(f'{base_url}{categorymembers_query_action}', params=params, headers=http_headers).json()
        if 'error' in result:
            raise Error(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            all_results+=result['query']['categorymembers']
        if 'continue' not in result:
            break
        lastContinue = result['continue']
    
    return all_results
    
# Fetch properties from query and condition - should be called from parent functions
def fetch_properties(condition: str, query: str, step: int = 5000, limit: int = 5000, iterator=None, debug: bool = False):
    df=pd.DataFrame()
    i = 0
    complete = False
    while not complete:
        if iterator is not None:
            iterator.set_postfix(it=i+1)

        response = requests.get(f'{base_url}{ask_query_action}{condition}{query}|limit%3D{step}|offset={i*step}|order%3Dasc', headers=http_headers)
        result = extract_results(response)
        formatted_df = format_df(result)
        df = pd.concat([df, formatted_df], ignore_index=True, axis=0)

        if debug:
            tqdm.write(f'Iteration {i+1}: {len(formatted_df.index)} results')

        if len(formatted_df.index)<step or (i+1)*step>=limit:
            complete = True
        else:
            i+=1

    return df

# Fetch spell or trap cards
def fetch_st(st_query: str, st: str = 'both', cg: CG = CG.ALL, step: int = 1000, limit: int = 5000, debug: bool = False):
    st = st.capitalize()
    valid_st = {'Spell', 'Trap', 'Both', 'All'}
    if st not in valid_st:
        raise ValueError("results: st must be one of %r." % valid_st)
    elif st=='Both' or st=='All':
        concept=f'[[Concept:{cg.value}%20Spell%20Cards]]OR[[Concept:{cg.value}%20Trap%20Cards]]'
        st='Spells and Trap'
    else:
        concept=f'[[Concept:{cg.value}%20{st}%20Cards]]'

    print(f'Downloading {st}s')
    st_df = fetch_properties(
        concept, 
        st_query, 
        step=step, 
        limit=limit, 
        debug=debug
    )

    if debug:
        print('- Total')

    print(f'{len(st_df.index)} results\n')

    return st_df

# Fetch monster cards by splitting into attributes
def fetch_monster(monster_query: str, cg: CG = CG.ALL, step: int = 1000, limit: int = 5000, debug: bool = False):
    valid_cg = cg.value

    print('Downloading monsters')
    monster_df = pd.DataFrame()
    iterator = tqdm(attributes, leave = False, unit='attribute')
    for att in iterator:
        iterator.set_description(att)
        if debug:
            tqdm.write(f"- {att}")

        concept = f'[[Concept:{valid_cg}%20monsters]][[Attribute::{att}]]'

        temp_df = fetch_properties(
            concept, 
            monster_query, 
            step=step, 
            limit=limit, 
            iterator=iterator, 
            debug=debug
        )

        monster_df = pd.concat([monster_df, temp_df], ignore_index=True, axis=0) 

    if debug:
        print('- Total')

    print(f'{len(monster_df.index)} results\n')

    return monster_df

def fetch_errata(errata: str = 'all', limit: int = 1000):
    errata = errata.lower()
    valid = {'name', 'type', 'all', 'both'}
    if errata not in valid:
        raise ValueError("results: errata must be one of %r." % valid)
    elif errata == 'both' or errata=='all':
        errata_list = ['name','type']
    else:
        errata_list = [errata]

    errata_df = pd.DataFrame()
    for errata in errata_list:
        if errata == 'type':
            condition = '[[Category:Cards%20with%20card%20type%20errata]]'
        if errata == 'name':
            condition = '[[Category:Cards%20with%20name%20errata]]'

        print(f'Downloading {errata} errata')  
        errata_query_df = fetch_properties(
            condition,
            query='',
            step=limit,
            limit=limit
        )

        errata_index = [i.split('Card Errata:')[-1].strip() for i in errata_query_df['Page name'] if 'Card Errata:' in i]
        errata_column = f'{errata.capitalize()} errata'
        errata_series = pd.Series(True, index = errata_index, name=errata_column)
        print(f'{len(errata_series)} results\n')

        errata_df = pd.concat([errata_df, errata_series],axis=1).fillna(False)

    return errata_df

# Sets
## Get title of set list pages
def fetch_set_titles(cg: CG = CG.ALL, limit: int = 5000):
    valid_cg = cg.value
    if valid_cg=='CG':
        condition='[[Category:TCG%20Set%20Card%20Lists||OCG%20Set%20Card%20Lists]]'
    else:
        category=f'[[Category:{valid_cg}%20Set%20Card%20Lists]]'

    df = fetch_properties(
        condition,
        query='',
        step=limit,
        limit=limit
    )

    return df['Page name']

## Fetch set lists from page titles
def fetch_set_lists(titles, debug: bool = False):  # Separate formating function
    if debug:
        print(f'{len(titles)} set lists requested')

    titles = up.quote('|'.join(titles))
    rarity_dict = load_json('../assets/rarities.json')
    set_lists_df = pd.DataFrame(columns = ['Set','Card number','Name','Rarity','Print','Quantity','Region'])   
    success = 0
    error = 0

    response = requests.get(f'{base_url}{revisions_query_action}{titles}', headers=http_headers)
    json = response.json()
    contents = json['query']['pages'].values()
    
    for content in contents:
        if 'revisions' in  content.keys():
            temp = content['revisions'][0]['*']
            parsed = wtp.parse(temp)
            for template in parsed.templates:
                if template.name == 'Set list':
                    title = content['title'].split('Lists:')[1]
                    set_df = pd.DataFrame(columns = set_lists_df.columns)

                    region = None
                    rarity = None
                    card_print = None
                    qty = None
                    desc = None
                    opt = None
                    list_df = None

                    for argument in template.arguments:
                        if 'region=' in argument:
                            region = argument.string.split('=')[-1]
                            
                            if region.upper() == 'ES': # Remove duplicated entry for Spain
                                region = 'SP'
                                
                        elif 'rarities=' in argument:
                            rarity = tuple(
                                rarity_dict.get( i.strip(), i.strip()) for i in (argument.string.split('=')[-1]).split(',')
                            )
                            
                        elif 'print=' in argument:
                            card_print = argument.string.split('=')[-1]
                            
                        elif 'qty=' in argument:
                            qty = argument.string.split('=')[-1]
                            
                        elif 'description=' in argument:
                            desc = argument.string.split('=')[-1]
                            
                        elif 'options=' in argument:
                            opt = argument.string.split('=')[-1]
                            
                        else:
                            set_list = argument.string[2:-1]
                            lines = set_list.split('\n')

                            list_df = pd.DataFrame([x.split(';') for x in lines])
                            list_df = list_df[~list_df[0].str.contains('!:')]
                            list_df = list_df.applymap(lambda x: x.split('//')[0] if x is not None else x)
                            list_df = list_df.applymap(lambda x: x.strip() if x is not None else x)
                            list_df.replace(r'^\s*$|^@.*$', None, regex = True, inplace = True)

                    noabbr = (opt == 'noabbr')
                    
                    set_df['Name'] = list_df[1-noabbr].apply(lambda x: x.strip('\u200e').split(' (')[0] if x is not None else x)
                    
                    if not noabbr and len(list_df.columns>1):
                        set_df['Card number'] = list_df[0]
                        
                    if len(list_df.columns)>(2-noabbr): # and rare in str
                        set_df['Rarity'] = list_df[2-noabbr].apply(lambda x: tuple([rarity_dict.get(y.strip(), y.strip()) for y in x.split(',')]) if x is not None else rarity)
                    
                    else:
                        set_df['Rarity'] = [rarity for _ in set_df.index]

                    if len(list_df.columns)>(3-noabbr):
                        if card_print is not None: # and new/reprint in str
                            set_df['Print'] = list_df[3-noabbr].apply(lambda x: card_print if (card_print and x is None) else x)
                            
                            if len(list_df.columns)>(4-noabbr) and qty:
                                set_df['Quantity'] = list_df[4-noabbr].apply(lambda x: x if x is not None else qty)
                        
                        elif qty:
                            set_df['Quantity'] = list_df[3-noabbr].apply(lambda x: x if x is not None else qty)

                    set_df['Set'] = title.split("(")[0].strip()
                    set_df['Region'] = region.upper() 
                    set_lists_df = pd.concat([set_lists_df, set_df], ignore_index=True)
                    success+=1

        else:
            error+=1
            if debug:
                print(f"Error! No content for \"{content['title']}\"")
            
    if debug:
        print(f'{success} set lists received - {error} errors')
        print('-------------------------------------------------')

    return set_lists_df, success, error

## Fecth all set lists
def fetch_all_set_lists(cg: CG = CG.ALL, step: int = 50, debug: bool = False):
    keys = fetch_set_titles(cg) # Get list of sets

    all_set_lists_df = pd.DataFrame(columns = ['Set','Card number','Name','Rarity','Print','Quantity','Region'])
    total_success = 0
    total_error = 0

    for i in trange(np.ceil(len(keys)/step).astype(int), leave=False):
        success = 0
        error = 0
        if debug:
            tqdm.write(f'Iteration {i}:')

        first = i*step
        last = (i+1)*step

        set_lists_df, success, error = fetch_set_lists(keys[first:last])
        all_set_lists_df = pd.concat([all_set_lists_df, set_lists_df], ignore_index=True)
        total_success+=success
        total_error+=error

    all_set_lists_df = all_set_lists_df.convert_dtypes()
    all_set_lists_df.sort_values(by=['Set','Region','Card number']).reset_index(inplace = True)
    print(f'{"Total:" if debug else ""}{total_success} set lists received - {total_error} errors')

    return all_set_lists_df

## Fetch set info for list of sets
def fetch_set_info(sets, step: int = 15, debug: bool = False):
    # Info to ask
    info = ['Series','Set type','Cover card','Modification date']
    # Release to ask
    release = [i+' release date' for i in set(regions_dict.values())]
    # Ask list
    ask = up.quote('|'.join(np.append(info,release)))

    # Get set info
    set_info_df = pd.DataFrame()
    for i in trange(np.ceil(len(sets)/step).astype(int),leave=False):
        first = i*step
        last = (i+1)*step
        titles = up.quote(']]OR[['.join(sets[first:last]))
        response = requests.get(f'{base_url}{askargs_query_action}{titles}&printouts={ask}', headers=http_headers)
        formatted_response = extract_results(response)
        formatted_response.drop('Page name', axis=1, inplace = True) # Page name not needed - no set errata, set name same as page name
        formatted_df = format_df(formatted_response)
        if debug:
            tqdm.write(f'Iteration {i}\n{len(formatted_df)} set properties downloaded - {step-len(formatted_df)} errors')
            tqdm.write('-------------------------------------------------')

        set_info_df = pd.concat([set_info_df, formatted_df])

    set_info_df = set_info_df.convert_dtypes()
    set_info_df.sort_index(inplace = True)

    print(f'{"Total:" if debug else ""}{len(set_info_df)} set properties received - {len(sets)-len(set_info_df)} errors')

    return set_info_df


######################
# Plotting functions #
######################

# Colors dictionary to associate to series and cards
colors_dict = {
    'Effect Monster': '#FF8B53', 
    'Normal Monster': '#FDE68A', 
    'Ritual Monster': '#9DB5CC', 
    'Fusion Monster': '#A086B7', 
    'Synchro Monster': '#CCCCCC', 
    'Xyz Monster': '#000000', 
    'Link Monster': '#00008B', 
    'Pendulum Monster': 'r', 
    'Monster Card': '#FF8B53', 
    'Spell Card': '#1D9E74', 
    'Trap Card': '#BC5A84', 
    'Monster Token': '#C0C0C0', 
    'FIRE': '#fd1b1b', 
    'WATER': '#03a9e6', 
    'EARTH': '#060d0a', 
    'WIND': '#77bb58', 
    'DARK': '#745ea5', 
    'LIGHT': '#9d8047', 
    'DIVINE': '#7e6537', 
    'Level': '#f1a41f',
    'First Series': '#FDE68A',
    'Duel Monsters': '#FF8B53',
    'GX': '#A086B7',
    '5D\'s': '#CCCCCC',
    'ZEXAL': '#000000',
    'ARC-V': 'r',
    'VRAINS': '#00008B',
    'SEVENS': '#1D9E74',
    'GO RUSH!!': '#BC5A84'
}

def adjust_lightness(color: str, amount: float = 0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def align_yaxis(ax1, v1: float, ax2, v2: float):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2,(y1-y2)/2,v2)
    adjust_yaxis(ax1,(y2-y1)/2,v1)

def adjust_yaxis(ax, ydif: float, v: float):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny>maxy or (-miny==maxy and dy > 0):
        nminy = miny
        nmaxy = miny*(maxy+dy)/(miny+dy)
    else:
        nmaxy = maxy
        nminy = maxy*(miny+dy)/(maxy+dy)
    ax.set_ylim(nminy+v, nmaxy+v)

def generate_rate_grid(dy: pd.DataFrame, ax, xlabel: str = 'Date', size: str = "150%", pad: int = 0, colors: list = None, cumsum: bool = True): 
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if cumsum: 
        cumsum_ax = ax
        divider = make_axes_locatable(cumsum_ax)
        yearly_ax = divider.append_axes("bottom", size=size, pad=pad)
        cumsum_ax.figure.add_axes(yearly_ax)   
        cumsum_ax.set_xticklabels([])
        axes = [cumsum_ax, yearly_ax]   

        y = dy.fillna(0).cumsum()

        if len(dy.columns)==1:
            cumsum_ax.plot(y, label = "Cummulative", c=colors[0], antialiased=True)
            cumsum_ax.fill_between(y.index, y.values.T[0], color=colors[0], alpha=0.1, hatch='x')
            cumsum_ax.set_ylabel(f'{y.columns[0]}') # Wrap text
        else:
            cumsum_ax.stackplot(y.index, y.values.T, labels = y.columns, colors=colors, antialiased=True)
            cumsum_ax.set_ylabel(f'Cumulative {y.index.name.lower()}')

        yearly_ax.set_ylabel(f'Yearly {dy.index.name.lower()} rate')
        cumsum_ax.legend(loc='upper left', ncols=int(len(dy.columns)/5+1)) # Test

    else:
        yearly_ax = ax
        axes = [yearly_ax] 

        if len(dy.columns)==1:
            yearly_ax.set_ylabel(f'{dy.columns[0]}\nYearly {dy.index.name.lower()} rate')
        else:
            yearly_ax.set_ylabel(f'Yearly {dy.index.name.lower()} rate')

    if len(dy.columns)==1:
        monthly_ax = yearly_ax.twinx()

        yearly_ax.plot(dy.resample('Y').sum(), label = "Yearly rate", ls='--', c=colors[1], antialiased=True)
        yearly_ax.legend(loc='upper left', ncols=int(len(dy.columns)/8+1))
        monthly_ax.plot(dy.resample('M').sum(), label = "Monthly rate", c=colors[2], antialiased=True)
        monthly_ax.set_ylabel(f'Monthly {dy.index.name.lower()} rate')
        monthly_ax.legend(loc='upper right')

    else:
        dy2=dy.resample('Y').sum()
        yearly_ax.stackplot(dy2.index, dy2.values.T, labels = dy2.columns, colors=colors, antialiased=True)
        if not cumsum:
            yearly_ax.legend(loc='upper left', ncols=int(len(dy.columns)/8+1))

    if xlabel is not None:
        yearly_ax.set_xlabel(xlabel)
    else:
        yearly_ax.set_xticklabels([])

    for temp_ax in axes:
        temp_ax.set_xlim([dy.index.min()-pd.Timedelta(weeks=13),dy.index.max()+pd.Timedelta(weeks=52)])
        temp_ax.xaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.yaxis.set_minor_locator(AutoMinorLocator())
        temp_ax.xaxis.set_major_locator(mdates.YearLocator())
        temp_ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        temp_ax.grid()

    if len(dy.columns)==1:
        align_yaxis(yearly_ax, 0, monthly_ax, 0)
        l = yearly_ax.get_ylim()
        l2 = monthly_ax.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(yearly_ax.get_yticks())
        monthly_ax.yaxis.set_major_locator(FixedLocator(ticks))
        monthly_ax.yaxis.set_minor_locator(AutoMinorLocator())
        axes.append(monthly_ax)

    return axes

def rate_subplots(df: pd.DataFrame, figsize = None, title: str = '', xlabel: str = 'Date', colors: list = None, cumsum: bool = True, bg: pd.DataFrame = None, vlines: pd.DataFrame = None):
    if figsize is None:
        figsize = (16, len(df.columns)*2*(1+cumsum))

    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=figsize, sharex=True)
    fig.suptitle(f'{title if title is not None else df.index.name.capitalize()}{f" by {df.columns.name.lower()}" if df.columns.name is not None else ""}', y=1)

    if colors is None:
        cmap = plt.cm.tab20
    else:
        if len(colors) == len(df.columns):
            cmap = mc.ListedColormap([adjust_lightness(c, i*0.5+0.75) for c in colors for i in (0, 1)])
        else:
            cmap = mc.ListedColormap(colors)

    c=0
    for i, col in enumerate(df.columns):
        sub_axes = generate_rate_grid(df[col].to_frame(), ax=axes[i], colors = [cmap(2*c),cmap(2*c),cmap(2*c+1)], size='100%', xlabel='Date' if (i+1)==len(df.columns) else None, cumsum=cumsum)

        for ix, ax in enumerate(sub_axes[:2]):
            if bg is not None and all(col in bg.columns for col in ['begin','end']):
                bg = bg.copy()
                bg['end'].fillna( df.index.max(), inplace=True)
                for idx, row in bg.iterrows():
                    if row['end']>pd.to_datetime(ax.get_xlim()[0], unit='d'):
                        filled_poly = ax.axvspan(row['begin'], row['end'], alpha=.1, color=colors_dict[idx], zorder = -1)
                        if i==0 and ix==0:
                            (x0, y0), (x1, y1) = filled_poly.get_path().get_extents().get_points()
                            ax.text((x0+x1)/2, y1, idx, ha='center', va='bottom', transform=ax.get_xaxis_transform())

            if vlines is not None:
                for idx, row in vlines.items():
                    if row>pd.to_datetime(ax.get_xlim()[0], unit='d'):
                        line = ax.axvline(row, ls='-.', c='maroon', lw=1)
                        if i==0 and ix==0:
                            (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                            ax.text((x0+x1)/2 + 25, (0.02 if cumsum else 0.98), idx, c='maroon', ha='left', va=('bottom' if cumsum else 'top'), rotation = 90, transform=ax.get_xaxis_transform())

        c+=1
        if 2*c+1>=cmap.N:
            c=0

    warnings.filterwarnings( "ignore", category = UserWarning, message = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")

    fig.tight_layout()
    plt.show()

    warnings.filterwarnings( "default", category = UserWarning, message = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")

def rate_plot(dy: pd.DataFrame, figsize = (16,6), title: str = None, xlabel: str = 'Date', colors: list = None, cumsum=True, bg = None, vlines = None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharey=True, sharex=True)
    fig.suptitle(f'{title if title is not None else dy.index.name.capitalize()}{f" by {dy.columns.name.lower()}" if dy.columns.name is not None else ""}')

    axes = generate_rate_grid(dy, ax, size='100%', colors = colors, cumsum=cumsum)
    for i, ax in enumerate(axes[:2]):
        if bg is not None and all(col in bg.columns for col in ['begin','end']):
            bg = bg.copy()
            bg['end'].fillna(dy.index.max(), inplace=True)
            for idx, row in bg.iterrows():
                if row['end']>pd.to_datetime(ax.get_xlim()[0], unit='d'):
                    filled_poly = ax.axvspan(row['begin'], row['end'], alpha=.1, color=colors_dict[idx], zorder = -1)
                    if i==0:
                        (x0, y0), (x1, y1) = filled_poly.get_path().get_extents().get_points()
                        ax.text((x0+x1)/2, y1, idx, ha='center', va='bottom', transform=ax.get_xaxis_transform())

        if vlines is not None:
            for idx, row in vlines.items():
                if row>pd.to_datetime(ax.get_xlim()[0], unit='d'):
                    line = ax.axvline(row, ls='-.', c='maroon', lw=1)
                    if i==0:
                        (x0, y0), (x1, y1) = line.get_path().get_extents().get_points()
                        ax.text((x0+x1)/2 + 25, (0.02 if cumsum or len(dy.columns)>1 else 0.98), idx, c='maroon', ha='left', va=('bottom' if cumsum or len(dy.columns)>1 else 'top'), rotation = 90, transform=ax.get_xaxis_transform())

    warnings.filterwarnings( "ignore", category = UserWarning, message = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")

    fig.tight_layout()
    plt.show()

    warnings.filterwarnings( "default", category = UserWarning, message = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")

###########################
# Complete execution flow #
###########################

def run(report = 'all', progress_handler = None):
    # Check API status
    if not check_API_status():
        if progress_handler:
            progress_handler(API_status=False)
        return
    # Execute all notebooks in the source directory
    run_notebooks(which = report, progress_handler = progress_handler)
    # Update page index to reflect last execution timestamp
    update_index()
    # Clear notebooks after HTML reports have been created
    clear_notebooks(which = report)
    # Cleanup redundant data files
    # cleanup_data()


#############
# CLI usage #
#############

if __name__ == "__main__":
    # Change working directory to script location
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    # Execute the complete workflow
    run()
    # Exit python
    quit()