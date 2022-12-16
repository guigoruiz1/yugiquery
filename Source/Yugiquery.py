# Imports

import ipynbname
import glob
import os
import string
import calendar
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import urllib.parse as up
import wikitextparser as wtp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FixedLocator
import matplotlib.dates as mdates
from matplotlib_venn import venn2
from datetime import datetime, timezone
from ast import literal_eval
from IPython.display import Markdown

# API variables

api_url = 'https://yugipedia.com/api.php'
sets_query_url = '?action=ask&query=[[Category:TCG%20Set%20Card%20Lists||OCG%20Set%20Card%20Lists]]|limit%3D5000|order%3Dasc&format=json'
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

card_colors = {
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
    'Level': '#f1a41f'
}

# Functions

## Generate Markdown header
def header(name=None):
    if name is None:
        try: 
            name = ipynbname.name()
        except:
            name = ''

    header = (open('../Assets/header.md').read())
    header = header.replace('@DATE@', datetime.now().astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M %Z"))
    header = header.replace('@NOTEBOOK@', name)
    return Markdown(header)

## API call functions
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

def fetch_spell(spell_query, step = 5000, limit = 5000):
    print('Downloading Spells')
    spell_df = pd.DataFrame()
    for i in range(int(limit/step)):
        df = pd.read_json(f'{api_url}?action=ask&query=[[Concept:CG%20Spell%20Cards]]{spell_query}|limit%3D{step}|offset={i*step}|order%3Dasc&format=json')
        df = extract_results(df)
        print(f'Iteration {i+1}: {len(df.index)} results')
        spell_df = pd.concat([spell_df, df], ignore_index=True, axis=0)
        if len(df.index)<step:
            break
                
    print(f'- Total\n{len(spell_df.index)} results\n')
    
    return spell_df

def fetch_trap(trap_query, step = 5000, limit = 5000):
    print('Downloading Traps')
    trap_df = pd.DataFrame()
    for i in range(int(limit/step)):    
        df = pd.read_json(f'{api_url}?action=ask&query=[[Concept:CG%20Trap%20Cards]]{trap_query}|limit%3D{step}|offset={i*step}|order%3Dasc&format=json')
        df = extract_results(df)
        print(f'Iteration {i+1}: {len(df.index)} results')
        trap_df = pd.concat([trap_df, df], ignore_index=True, axis=0)
        if len(df.index)<step:
            break
                
    print(f'- Total\n{len(trap_df.index)} results\n')
    
    return trap_df

def fetch_monster(monster_query, step = 5000, limit = 5000):
    print('Downloading Monsters')
    monster_df = pd.DataFrame()
    for att in attributes:
        print(f"- {att}")
        for i in range(int(limit/step)):
            df = pd.read_json(f'{api_url}?action=ask&query=[[Concept:CG%20monsters]][[Attribute::{att}]]{monster_query}|limit%3D{step}|offset={i*step}|order%3Dasc&format=json')
            df = extract_results(df)
            print(f'Iteration {i+1}: {len(df.index)} results')
            monster_df = pd.concat([monster_df, df], ignore_index=True, axis=0)
            if len(df.index)<step:
                break
        
    print(f'- Total\n{len(monster_df.index)} results\n')
    
    return monster_df

def fetch_name_errata(limit = 1000):
    print('Downloading name errata')
    name_query_df = pd.read_json(f'{api_url}?action=ask&query=[[Category:Cards%20with%20name%20errata]]|limit={limit}|order%3Dasc&format=json')
    name_keys = list(name_query_df['query']['results'].keys())
    name_df = pd.DataFrame(True, index = [i.split(':')[1].strip() for i in name_keys if 'Card Errata:' in i], columns = ['Name errata'])
    
    print(f'- Total\n{len(name_df.index)} results\n')
    
    return name_df

def fetch_type_errata(limit = 1000):
    print('Downloading type errata')
    type_query_df = pd.read_json(f'{api_url}?action=ask&query=[[Category:Cards%20with%20card%20type%20errata]]|limit={limit}|order%3Dasc&format=json')
    type_keys = list(type_query_df['query']['results'].keys())
    type_df = pd.DataFrame(True, index = [i.split(':')[1].strip() for i in type_keys if 'Card Errata:' in i], columns = ['Type errata'])
    
    print(f'- Total\n{len(type_df.index)} results\n')
    
    return type_df

## Cards formatting functions
def extract_results(df):
    df = pd.DataFrame(df['query']['results']).transpose()
    df = pd.DataFrame(df['printouts'].values.tolist(), index = df['printouts'].keys())
    return df

def extract_artwork(row):
    result = tuple()
    if 'Category:OCG/TCG cards with alternate artworks' in row:
        result += ('Alternate',)
    if 'Category:OCG/TCG cards with edited artworks' in row:
        result += ('Edited',)
    if result == tuple():
        return np.nan
    else:
        return result

def concat_errata(row):
    result = tuple()
    if row['Name errata']:
        result += ('Name',)
    if row['Type errata']:
        result += ('Type',)
    if result == tuple():
        return np.nan
    else:
        return result 
    
def format_df(input_df, input_errata_df=None):
    df = pd.DataFrame()
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
        df['Artwork'] = input_df['Category'].dropna().apply(lambda x: [i['fulltext'] for i in x] if len(x)>0 else np.nan).apply(extract_artwork)
    # Erratas column
    if input_errata_df is not None and 'Page name' in input_df.columns:
        df['Errata'] = input_errata_df.merge(input_df['Page name'].dropna().apply(lambda x: x[0]).rename('Name'), right_on = 'Name', left_index = True).apply(concat_errata,axis = 1)
    #################
    if 'TCG status' in input_df.columns:
        df['TCG status'] = input_df['TCG status'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'OCG status' in input_df.columns:
        df['OCG status'] = input_df['OCG status'].dropna().apply(lambda x: x[0]['fulltext'] if len(x)>0 else np.nan)
    if 'Modification date' in input_df.columns:
        df['Modification date'] = input_df['Modification date'].dropna().apply(lambda x: pd.Timestamp(int(x[0]['timestamp']), unit='s').ctime() if len(x)>0 else np.nan)
    
    return df

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
        new_entries = changelog['Version'][nunique['Version'] == 1].dropna(axis=0, how='all').index
        rows_to_keep = true_changes.union(new_entries).unique()
        changelog = changelog.loc[rows_to_keep]
    
    if changelog.empty:
        print('No changes')
        
    return changelog

## Plotting functions
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

def rate_plot(dy, xlabel = 'Date', title=None, size="50%", pad=0, figsize = (16,8)):
    
    warnings.filterwarnings( "ignore", category = UserWarning, message = "This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")
    
    y = dy.fillna(0).cumsum()
    fig, ax_top = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharey=True, sharex=True)
    divider = make_axes_locatable(ax_top)
    ax_bottom = divider.append_axes("bottom", size=size, pad=pad)
    ax_top.figure.add_axes(ax_bottom)
    
    axes = [ax_top, ax_bottom]
    if len(dy.columns)==1:
        ax_bottom_2 = ax_bottom.twinx()
        
        ax_top.plot(y, label = "Cummulative")
        ax_bottom_2.plot(dy.resample('Y').sum(), label = "Yearly rate", style='--')
        ax_bottom.plot(dy.resample('M').sum(), label = "Monthly rate")
        
        ax_bottom_2.set_ylabel(f'Yearly {dy.index.name.lower()} rate')
        
        ax_top.legend(loc='upper left')
        ax_bottom.legend(loc='upper left')
        ax_bottom_2.legend(loc='upper right')
    else:
        dy = dy.resample('Y').sum()
        ax_top.stackplot(y.index, y.values.T, labels = y.columns)
        ax_bottom.stackplot(dy.index, dy.values.T)
        ax_top.legend(loc='upper left')
    
    fig.suptitle(f'{", ".join(dy.columns) if (title is None) else title} {dy.index.name.lower()}s{f" by {dy.columns.name.lower()}" if dy.columns.name is not None else ""}')
        
    ax_top.set_ylabel(f'Cumulative {dy.index.name.lower()}s')
    ax_bottom.set_ylabel(f'Monthly {dy.index.name.lower()} rate')
    ax_top.set_xticklabels([])
    ax_bottom.set_xlabel(xlabel)
    
    for ax in axes:
        ax.set_xlim([y.index.min()-pd.Timedelta(weeks=13),y.index.max()+pd.Timedelta(weeks=52)])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid()
    
    if len(dy.columns)==1:
        align_yaxis(ax_bottom, 0, ax_bottom_2, 0)
        l = ax_bottom.get_ylim()
        l2 = ax_bottom_2.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(ax_bottom.get_yticks())
        ax_bottom_2.yaxis.set_major_locator(FixedLocator(ticks))
        ax_bottom_2.yaxis.set_minor_locator(AutoMinorLocator())
          
    plt.tight_layout()        
    plt.show()
    
def rate_subplots(df, title=None, xlabel='Date', figsize = (16,80)):
    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=figsize, sharex=True)
    axes[0].set_title(f'{", ".join(df.columns) if (title is None) else title} {df.index.name.lower()}s{f" by {df.columns.name.lower()}" if df.columns.name is not None else ""}')
    axes[-1].set_xlabel('Date')

    twinx = []
    by_month = df.resample('M').sum()
    by_year = df.resample('Y').sum()
    # test.index = test.index+pd.Timedelta(days=1) # Bug fix
    cmap = plt.cm.tab20
    c=0
    for i, col in enumerate(df.columns):
        temp_ax = axes[i].twinx()

        axes[i].plot(by_month[col], color=cmap(2*c), label = 'Monthly')
        axes[i].yaxis.set_major_locator(MaxNLocator(4, integer=True))
        axes[i].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i].set_ylabel(col)
        axes[i].legend(loc='upper left')
        axes[i].grid()

        temp_ax.plot(by_year[col], color = cmap(2*c+1), ls='--', label='Yearly')
        temp_ax.legend(loc='upper right')
        temp_ax.grid() 
        twinx.append(temp_ax)

        c+=1
        if c==int(len(cmap.colors)/2):
            c=0

    for ax_left, ax_right in zip(axes,twinx):
        ax_right.set_ylim(bottom=0.)
        align_yaxis(ax_left, 0, ax_right, 0)
        l = ax_left.get_ylim()
        l2 = ax_right.get_ylim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
        ticks = f(ax_left.get_yticks())
        ax_right.yaxis.set_major_locator(FixedLocator(ticks))
        ax_right.yaxis.set_minor_locator(AutoMinorLocator())


    plt.tight_layout()
    plt.show()