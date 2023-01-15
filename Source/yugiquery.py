# Imports
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

        break

    except ImportError:
        if loop>1:
            print("Failed to install required packages twice. Aborting...")
            quit()
        
        loop+=1   
        import subprocess
        print("Missing required packages. Trying to install now...")
        subprocess.call(['sh', './install.sh'])

# CLI usage
def run_all():    
    reports = sorted(glob.glob('*.ipynb'))
    iterator = tqdm(reports, desc="Completion", unit='report')
    
    secrets_file = '../Assets/secrets.txt'
    if os.path.isfile(secrets_file):
        secrets={}
        with open(secrets_file) as f:
            for line in f:
                line=line.strip()
                if not line.startswith('#'):
                    name, value = line.split("=")
                    secrets[name.strip()] = value.strip()
            
        if all(key in secrets.keys() for key in ['DISCORD_TOKEN','DISCORD_CHANNEL_ID']):
            try:
                from tqdm.contrib.discord import tqdm as discord_tqdm
                iterator = discord_tqdm(reports, desc="Completion", unit='report', token=secrets['DISCORD_TOKEN'], channel_id=secrets['DISCORD_CHANNEL_ID'])
            except:
                pass
    
    for report in iterator:
        iterator.set_postfix(report=report)
        tqdm.write(f'Generating {report[:-6]} report')
        pm.execute_notebook(report,report);
        
## Helpers
def clear_notebooks():
    reports = sorted(glob.glob('*.ipynb'))
    if len(reports)>0:
        subprocess.call(['nbstripout']+reports)
    
def update_index(): # Paths
    index_file_name='README.md'
    timestamp = datetime.now().astimezone(timezone.utc)
    try:
        with open(f'../Assets/index.md') as f:
            readme = f.read()

            reports = sorted(glob.glob('../*.html'))
            for report in reports:
                readme = readme.replace(f'@{report[:-6].upper()}_TIMESTAMP@', pd.to_datetime(os.path.getmtime(report),unit='s', utc=True).strftime("%d/%m/%Y %H:%M %Z"))

            readme = readme.replace(f'@TIMESTAMP@', timestamp.strftime("%d/%m/%Y %H:%M %Z"))
            with open(f'../{index_file_name}', 'w') as o:
                print(readme, file=o)

        repo = git.Repo(f'../')
        repo.git.commit('-m', f'index timestamp update-{timestamp.strftime("%d%m%Y")}', f'{index_file_name}')
        
    except:
        print('No "index.md" file in "Assets". Aborting...')
        
def cleanup_data(dry_run=False):
    file_list = glob.glob('../Data/*')
    df = pd.DataFrame(file_list, columns=['file'])
    df['timestamp'] = df['file'].apply(os.path.getctime).apply(pd.to_datetime, unit='s')
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

## If executed from the CLI
if __name__ == "__main__":
    # Change working directory to script location
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    # Execute all notebooks in the source directory
    run_all()
    # Update page index to reflect last execution timestamp
    update_index()
    # Clear notebooks after HTML reports have been created
    clear_notebooks()
    # Exit python
    quit()

# API variables
api_url = 'https://yugipedia.com/api.php'
sets_query_url = '?action=ask&query=[[Category:TCG%20Set%20Card%20Lists||OCG%20Set%20Card%20Lists]]'
lists_query_url = '?action=query&prop=revisions&rvprop=content&format=json&titles='

# Lists

## Attributes list to split monsters query
attributes = [
    'DIVINE', 
    'LIGHT', 
    'DARK', 
    'WATER', 
    'EARTH', 
    'FIRE', 
    'WIND'
]

## Abreviations dictionaries
rarity_dict = {
    'c': 'Common', 
    'r': 'Rare', 
    'sr': 'Super Rare', 
    'ur': 'Ultra Rare', 
    'utr': 'Ultimate Rare', 
    'n': 'Normal', 
    'nr': 'Normal Rare', 
    'sp': 'Short Print', 
    'ssp': 'Super Short Print', 
    'hfr': 'Holofoil Rare', 
    'scr': 'Secret Rare', 
    'uscr': 'Ultra Secret Rare', 
    'scur': 'Secret Ultra Rare', 
    'pscr': 'Prismatic Secret Rare', 
    'hgr': 'Holographic Rare', 
    'gr': 'Ghost Rare', 
    'pr': 'Parallel Rare', 
    'npr': 'Normal Parallel Rare', 
    'pc': 'Parallel Common', 
    'spr': 'Super Parallel Rare', 
    'upr': 'Ultra Parallel Rare', 
    'dnpr': 'Duel Terminal Normal Parallel Rare', 
    'dpc': 'Duel Terminal Parallel Common', 
    'drpr': 'Duel Terminal Rare Parallel Rare', 
    'dspr': 'Duel Terminal Super Parallel Rare', 
    'dupr': 'Duel Terminal Ultra Parallel Rare', 
    'DScPR': 'Duel Terminal Secret Parallel Rare', 
    'gur': 'Gold Rare', 
    'escr': 'Extra Secret Rare', 
    'ggr': 'Ghost/Gold Rare', 
    'shr': 'Shatterfoil Rare', 
    'cr': 'Collector\'s Rare', 
    'altr': 'Starlight Rare', 
    'str': 'Starlight Rare', 
    'gr': 'Ghost Rare', 
    'gscr': 'Gold Secret Rare', 
    'sfr': 'Starfoil Rare', 
    '20scr': '20th Secret Rare', 
    'dscpr': 'Duel Terminal Secret Parallel Rare', 
    'dnrpr': 'Duel Terminal Normal Rare Parallel Rare',
    'kcc': 'Kaiba Corporation Common' 

}

regions_dict = {
    'EN':'English', 
    'NA': 'North American English',
    'EU':'European English', 
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

## Styling dictionaries
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

# Functions

## Generate Markdown header
def header(name=None):
    if name is None:
        try: 
            name = ipynbname.name()
        except:
            name = ''
            
    with open('../Assets/header.md') as f:
        header = f.read()
        header = header.replace('@TIMESTAMP@', datetime.now().astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M %Z"))
        header = header.replace('@NOTEBOOK@', name)
        return Markdown(header)

def footer():
    with open('../Assets/footer.md') as f:
        footer = f.read()
        footer = footer.replace('@TIMESTAMP@', datetime.now().astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M %Z"))
        return Markdown(footer)
    
## API call functions

### Cards
def card_query(_password = True, _card_type = True, _property = True, _primary = True, _secondary = True, _attribute = True, _monster_type = True, _stars = True, _atk = True, _def = True, _scale = True, _link = True, _arrows = True, _effect_type = True, _archseries = True, _category = True, _tcg = True, _ocg = True, _date = True, _page_name = True):
    search_string = f'|?English%20name=Name'
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
    if _category:
        search_string += '|?category'
    if _tcg:
        search_string += '|?TCG%20status'
    if _ocg:
        search_string += '|?OCG%20status'
    if _date:
        search_string += '|?Modification%20date'
    if _page_name:
        search_string += '|?Page%20name'
    
    return search_string

def fetch_st(st_query, st, cg='CG', step = 5000, limit = 5000, debug=False):
    st = st.capitalize()
    valid_st = {'Spell', 'Trap'}
    if st not in valid_st:
        raise ValueError("results: st must be one of %r." % valid_st)
    
    cg = cg.upper()
    valid_cg = {'CG', 'TCG', 'OCG', 'BOTH'}
    if cg not in valid_cg:
        raise ValueError("results: CG must be one of %r." % valid_st) 
    elif cg=='BOTH':
        cg='CG'
        
    print(f'Downloading {st}s')
    st_df = fetch_card(st_query, f'{cg}%20{st}%20Cards', step=step, limit=limit, debug=debug)
            
    if debug:
        print('- Total')
              
    print(f'{len(st_df.index)} results\n')
    
    return st_df

def fetch_monster(monster_query, cg='CG', step = 5000, limit = 5000, debug=False):
    print('Downloading monsters')
    monster_df = pd.DataFrame()
    iterator = tqdm(attributes, leave = False, unit='attribute')
    for att in iterator:
        iterator.set_description(att)
        if debug:
            tqdm.write(f"- {att}")
        
        temp_df = fetch_card(monster_query, f'{cg}%20monsters', step=step, limit=limit, extra_filter=f'[[Attribute::{att}]]', iterator=iterator, debug=debug)
        monster_df = pd.concat([monster_df, temp_df], ignore_index=True, axis=0) 
    
    if debug:
        print('- Total')
        
    print(f'{len(monster_df.index)} results\n')
    
    return monster_df

def fetch_card(query, concept, step=5000, limit=5000, extra_filter='', iterator=None, debug=False):
    df=pd.DataFrame()
    i = 0
    complete = False
    while not complete:
        if iterator is not None:
            iterator.set_postfix(it=i+1)
            
        response = pd.read_json(f'{api_url}?action=ask&query=[[Concept:{concept}]]{extra_filter}{query}|limit%3D{step}|offset={i*step}|order%3Dasc&format=json')
        result = extract_printouts(response)
        formatted_df = format_df(result)
        df = pd.concat([df, formatted_df], ignore_index=True, axis=0)           

        if debug:
            tqdm.write(f'Iteration {i+1}: {len(formatted_df.index)} results')

        if len(formatted_df.index)<step or i*step>=limit:
            complete = True
        else:
            i+=1
            
    return df

def fetch_errata(errata, limit = 1000):
    errata = errata.lower()
    valid = {'name', 'type'}
    if errata not in valid:
        raise ValueError("results: errata must be one of %r." % valid)
        
    print(f'Downloading {errata} errata')
    errata_query_df = pd.read_json(f'{api_url}?action=ask&query=[[Category:Cards%20with%20{"card%20"+errata if errata == "type" else errata}%20errata]]|limit={limit}|order%3Dasc&format=json')
    errata_keys = list(errata_query_df['query']['results'].keys())
    errata_df = pd.DataFrame(True, index = [i.split(':')[1].strip() for i in errata_keys if 'Card Errata:' in i], columns = [f'{errata.capitalize()} errata'])
    
    print(f'{len(errata_df.index)} results\n')
    
    return errata_df

### Sets
def get_set_titles(limit=5000):
    df = pd.read_json(f'{api_url}{sets_query_url}|limit%3D{limit}|order%3Dasc&format=json')
    keys = list(df['query']['results'].keys())
    return keys

def fetch_set_lists(titles, debug=False):  # Separate formating function
    if debug:
        print(f'{len(titles)} set lists requested')
    
    titles = up.quote('|'.join(titles))
    
    set_lists_df = pd.DataFrame(columns = ['Set','Card number','Name','Rarity','Print','Quantity','Region'])   
    success = 0
    error = 0

    df = pd.read_json(f'{api_url}{lists_query_url}{titles}')
    contents = df['query']['pages'].values()
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
                            region = argument.string[argument.string.index('=')+1:]
                        elif 'rarities=' in argument:
                            rarity = tuple(rarity_dict.get(i.strip().lower(), string.capwords(i.strip())) for i in argument.string[argument.string.index('=')+1:].split(','))
                        elif 'print=' in argument:
                            card_print = argument.string[argument.string.index('=')+1:]
                        elif 'qty=' in argument:
                            qty = argument.string[argument.string.index('=')+1:]
                        elif 'description=' in argument:
                            desc = argument.string[argument.string.index('=')+1:]
                        elif 'options=' in argument:
                            opt = argument.string[argument.string.index('=')+1:]
                        else:
                            set_list = argument.string[2:-1]
                            lines = set_list.split('\n')

                            list_df = pd.DataFrame([x.split(';') for x in lines])
                            list_df = list_df[~list_df[0].str.contains('!:')]
                            list_df = list_df.applymap(lambda x: x.split('//')[0] if x is not None else x)
                            list_df = list_df.applymap(lambda x: x.strip() if x is not None else x)
                            list_df.replace(r'^\s*$', None, regex = True, inplace = True)

                    if opt != 'noabbr':
                        set_df['Card number'] = list_df[0]
                        set_df['Name'] = list_df[1]
                    else: 
                        set_df['Name'] = list_df[0]

                    if len(list_df.columns)>2: # and rare in str
                        set_df['Rarity'] = list_df[2].apply(lambda x: tuple([rarity_dict.get(y.strip().lower(), string.capwords(y.strip())) for y in x.split(',')]) if x is not None else rarity)
                    else:
                        set_df['Rarity'] = [rarity for _ in set_df.index]

                    if len(list_df.columns)>3 :
                        if card_print is not None: # and new/reprint in str
                            set_df['Print'] = list_df[3].apply(lambda x: x if x is not None else card_print)
                            if len(list_df.columns)>4 and qty is not None:
                                set_df['Quantity'] = list_df[4].apply(lambda x: x if x is not None else qty)
                        elif qty is not None:
                            set_df['Quantity'] = list_df[3].apply(lambda x: x if x is not None else qty)
                    
                    set_df['Name'] = set_df['Name'].apply(lambda x: x.strip('\u200e').split(' (')[0] if x is not None else x)
                    set_df['Set'] = title.split("(")[0].strip()
                    set_df['Quantity'] = pd.to_numeric(set_df['Quantity'])
                    set_df['Region'] = region.upper()
                    set_lists_df = pd.concat([set_lists_df, set_df], ignore_index=True)
                    success+=1
                    
        else:
            if debug:
                print(f"Error! No content for \"{content['title']}\"")
            error+=1
    
    if debug:
        print(f'{success} set lists received - {error} errors')
        print('-------------------------------------------------')
    
    return set_lists_df, success, error

def fetch_all_set_lists(step = 50, debug=False):
    # Get list of sets
    keys = get_set_titles()

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

def fetch_set_info(sets, step=15, debug=False):
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
        response = pd.read_json(f'{api_url}?action=askargs&conditions={titles}&printouts={ask}&format=json')
        formatted_response = extract_printouts(response)
        formatted_df = format_df(formatted_response)
        if debug:
            tqdm.write(f'Iteration {i}\n{len(formatted_df)} set properties downloaded - {step-len(formatted_df)} errors')
            tqdm.write('-------------------------------------------------')
        
        set_info_df = pd.concat([set_info_df, formatted_df])

    set_info_df = set_info_df.convert_dtypes()
    set_info_df.sort_index(inplace = True)
    
    print(f'{"Total:" if debug else ""}{len(set_info_df)} set properties received - {len(sets)-len(set_info_df)} errors')
    
    return set_info_df

## Formating functions
def extract_printouts(df):
    df = pd.DataFrame(df['query']['results']).transpose()
    df = pd.DataFrame(df['printouts'].values.tolist(), index = df['printouts'].keys())
    return df
    
def format_df(input_df, input_errata_df=None):
    df = pd.DataFrame(index=input_df.index)
    
    # Cards
    if 'Name' in input_df.columns:
        df['Name'] = input_df['Name'].dropna().apply(lambda x: x[0].strip('\u200e'))
    if 'Password' in input_df.columns:
        df['Password'] = input_df['Password'].dropna().apply(lambda x: x[0] if len(x)>0 else np.nan)
    if 'Card type' in input_df.columns:
        df['Card type'] = input_df['Card type'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Property' in input_df.columns:
        df['Property'] = input_df['Property'].dropna().apply(lambda x: x[0] if len(x)>0 else np.nan)
    if 'Primary type' in input_df.columns:
        df['Primary type'] = input_df['Primary type'].dropna().apply(lambda x: [i['fulltext'] for i in x] if len(x)>0 else []).apply(lambda y: list(filter(lambda z: z != 'Pendulum Monster', y)) if len(y)>0 else []).apply(lambda y: list(filter(lambda z: z != 'Effect Monster', y))[0] if len(y)>1 else (y[0] if len(y)>0 else np.nan))
    if 'Secondary type' in input_df.columns:
        df['Secondary type'] = input_df['Secondary type'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Attribute' in input_df.columns:
        df['Attribute'] = input_df['Attribute'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Monster type' in input_df.columns:
        df['Monster type'] = input_df['Monster type'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Level/Rank' in input_df.columns:
        df['Level/Rank'] = input_df['Level/Rank'].dropna().apply(lambda x: x[0] if len(x)>0 else np.nan)
    if 'ATK' in input_df.columns:
        df['ATK'] = input_df['ATK'].dropna().apply(lambda x: x[0] if len(x)>0 else np.nan)
    if 'DEF' in input_df.columns:
        df['DEF'] = input_df['DEF'].dropna().apply(lambda x: x[0] if len(x)>0 else np.nan)
    if 'Pendulum Scale' in input_df.columns:
        df['Pendulum Scale'] = input_df['Pendulum Scale'].dropna().apply(lambda x: str(x[0]) if len(x)>0 else np.nan)
    if 'Link' in input_df.columns:
        df['Link'] = input_df['Link'].dropna().apply(lambda x: str(x[0]) if len(x)>0 else np.nan)
    if 'Link Arrows' in input_df.columns:
        df['Link Arrows'] = input_df['Link Arrows'].dropna().apply(lambda x: tuple([arrows_dict[i] for i in sorted(x)]) if len(x)>0 else np.nan)
    if 'Effect type' in input_df.columns:
        df['Effect type'] = input_df['Effect type'].dropna().apply(lambda x: tuple(sorted([i['fulltext'] for i in x])) if len(x)>0 else np.nan)
    if 'Archseries' in input_df.columns:
        df['Archseries'] = input_df['Archseries'].dropna().apply(lambda x: tuple(sorted(x)) if len(x)>0 else np.nan)
    if 'Category' in input_df.columns:
        df['Artwork'] = input_df['Category'].dropna().apply(lambda x: [i['fulltext'] for i in x] if len(x)>0 else np.nan).apply(format_artwork)
    if 'TCG status' in input_df.columns:
        df['TCG status'] = input_df['TCG status'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'OCG status' in input_df.columns:
        df['OCG status'] = input_df['OCG status'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Page name' in input_df.columns:
        df['Page name'] = input_df['Page name'].dropna().apply(lambda x: x[0] if len(x)>0 else np.nan)
    # Sets
    if 'Series' in input_df.columns:
        df['Series'] = input_df['Series'].apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Set type' in input_df.columns:
        df['Set type'] = input_df['Set type'].apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Cover card' in input_df.columns:
        df['Cover card'] = input_df['Cover card'].apply(lambda x: tuple(sorted([y['fulltext'] for y in x])) if len(x)>0 else np.nan)
    # Date columns    
    df = df.join(input_df.filter(like=' date').applymap(lambda x: pd.to_datetime(x[0]['timestamp'], unit = 's', errors = 'coerce') if len(x)>0 else np.nan))
    ##################
    return df

### Cards
def format_artwork(row):
    result = tuple()
    if 'Category:OCG/TCG cards with alternate artworks' in row:
        result += ('Alternate',)
    if 'Category:OCG/TCG cards with edited artworks' in row:
        result += ('Edited',)
    if result == tuple():
        return np.nan
    else:
        return result

def format_errata(row):
    result = tuple()
    if row['Name errata']:
        result += ('Name',)
    if row['Type errata']:
        result += ('Type',)
    if result == tuple():
        return np.nan
    else:
        return result 
    
def merge_errata(input_df, input_errata_df, drop=False):
    if 'Page name' in input_df.columns:
        input_df['Errata'] = input_errata_df.merge(input_df['Page name'].dropna(), right_on = 'Page name', left_index = True).apply(format_errata,axis = 1)
        if drop:
            input_df.drop('Page name',axis=1,inplace=True)
    else:
        print('Error! No \"page\" name column to join errata')
    
    return input_df

### Sets
def merge_set_info(input_df, input_info_df):
    if all([col in input_df.columns for col in ['Set', 'Region']]):
        input_df['Release'] = input_df[['Set','Region']].apply(lambda x: input_info_df[regions_dict[x['Region']]+' release date'][x['Set']] if (x['Region'] in regions_dict.keys() and x['Set'] in input_info_df.index) else np.nan, axis = 1)
        input_df['Release'] = pd.to_datetime(input_df['Release'].astype(str), errors='coerce') # Bug fix
        input_df = input_df.merge(input_info_df.loc[:,:'Modification date'], left_on = 'Set', right_index = True, how = 'outer', indicator = True).reset_index(drop=True) 
        print('Set properties merged')
    else:
        print('Error! No \"Set\" and/or \"Region\" column(s) to join set info')
        
    return input_df

## Changelog
def generate_changelog(previous_df, current_df, col):
    changelog = previous_df.merge(current_df,indicator = True, how='outer').loc[lambda x : x['_merge']!='both'].sort_values(col, ignore_index=True)
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
        changelog = changelog.loc[rows_to_keep]
    
    if changelog.empty:
        print('No changes')
        
    return changelog

## Plotting functions
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2,(y1-y2)/2,v2)
    adjust_yaxis(ax1,(y2-y1)/2,v1)

def adjust_yaxis(ax,ydif,v):
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

def generate_rate_grid(dy, ax, xlabel = 'Date', size="150%", pad=0, colors=None, cumsum=True): 
    
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
    
def rate_subplots(df, figsize = None, title='', xlabel='Date', colors=None, cumsum=True, bg = None, vlines = None):
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
    
def rate_plot(dy, figsize = (16,6), title=None, xlabel = 'Date', colors=None, cumsum=True, bg = None, vlines = None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharey=True, sharex=True)
    fig.suptitle(f'{title if title is not None else dy.index.name.capitalize()}{f" by {dy.columns.name.lower()}" if dy.columns.name is not None else ""}')
    
    axes = generate_rate_grid(dy, ax, size='100%', colors = colors, cumsum=cumsum)
    for i, ax in enumerate(axes[:2]):
        if bg is not None and all(col in bg.columns for col in ['begin','end']):
            bg = bg.copy()
            bg['end'].fillna( dy.index.max(), inplace=True)
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

# Frontend shortcuts
def save_notebook():
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    print("Notebook saved to disk")