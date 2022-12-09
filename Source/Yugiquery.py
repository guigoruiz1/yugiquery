# Imports

import ipynbname
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import urllib.parse as up
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib_venn import venn2
from datetime import datetime, timezone
from ast import literal_eval
from IPython.display import Markdown

# API variables

api_url = 'https://yugipedia.com/api.php'
sets_query_url = '?action=ask&query=[[Category:Set%20Card%20Lists]]|limit%3D5000|order%3Dasc&format=json'
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