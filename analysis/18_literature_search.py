# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import datetime
import io
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# PubMed records per publication year that mention O-GlcNAc in the title or abstract
#   Query: the terms below in [tiab], excluding errata, retraction notices and expressions of
#   concern ([pt]); one esearch request per publication year ([dp]), from 1984 (discovery of
#   O-GlcNAc) to the last complete year. PubMed reads Greek letters as their names (β as beta).
#   Early papers also used other terms, so early counts are minimums.
#   Outputs: data_processed/literature/pubmed_oglcnac_per_year.tsv (query and retrieval date
#   in the header) and the supplementary figure (PDF + RGB TIFF at 300 dpi; 85 mm wide, 8 pt).
# ---------------------------------------------------------------------------
TERMS = ['"O-GlcNAc"[tiab]', '"O-GlcNAcylation"[tiab]', '"O-GlcNAcylated"[tiab]',
         '"O-linked N-acetylglucosamine"[tiab]', '"O-linked beta-N-acetylglucosamine"[tiab]',
         '"O-linked N-acetyl-glucosamine"[tiab]', '"O-linked GlcNAc"[tiab]', '"O-beta-GlcNAc"[tiab]',
         '"O-linked beta-N-acetylglucosaminylation"[tiab]', '"O-GlcNAcase"[tiab]']
NOTICES = ['"Published Erratum"[pt]', '"Retraction Notice"[pt]', '"Expression of Concern"[pt]']
QUERY = '(' + ' OR '.join(TERMS) + ') NOT (' + ' OR '.join(NOTICES) + ')'
FIRST_YEAR, LAST_YEAR = 1984, datetime.date.today().year - 1
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
OUT_DIR, FIG_DIR = 'data_processed/literature', 'results/figures/literature'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

retrieved = datetime.date.today().isoformat()
rows = []
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    r = requests.get(EUTILS, params={'db': 'pubmed', 'term': f'({QUERY}) AND {year}[dp]',
                                     'retmode': 'json', 'retmax': 0}, timeout=60)
    r.raise_for_status()
    rows.append({'year': year, 'records': int(r.json()['esearchresult']['count'])})
    time.sleep(0.4)
counts = pd.DataFrame(rows)

with open(f'{OUT_DIR}/pubmed_oglcnac_per_year.tsv', 'w', encoding='utf-8') as f:
    f.write(f'# PubMed query: ({QUERY}) AND <year>[dp]\n# Retrieved: {retrieved}\n')
    counts.to_csv(f, sep='\t', index=False)
print(f'Retrieved {retrieved}; total {FIRST_YEAR}-{LAST_YEAR}: {counts["records"].sum():,}')
print(counts.set_index('year').loc[[1984, 2005, 2015, LAST_YEAR]].to_string())

# Figure
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(85 / 25.4, 55 / 25.4), layout='constrained')
ax.bar(counts['year'], counts['records'], width=0.8, color='#2a78d6')
pad = 0.06 * counts['records'].max()
for year in (2005, 2015, LAST_YEAR):
    n = int(counts.loc[counts['year'] == year, 'records'].iloc[0])
    top = counts.loc[counts['year'].between(year - 1, year + 1), 'records'].max()
    if top > n:   # a neighbouring bar is taller: label above it, with a line down to this bar
        ax.annotate(f'{n}', xy=(year, n), xytext=(year, top + pad), ha='center', va='bottom', color='#4a4945',
                                        arrowprops={'arrowstyle': '-', 'color': '#4a4945', 'lw': 0.6, 'linestyle': ':', 'shrinkA': 1, 'shrinkB': 0})
    else:
        ax.annotate(f'{n}', (year, n), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom',
                    color='#4a4945')
ax.set_xlabel('Publication year')
ax.set_ylabel('PubMed records')
ax.set_xlim(FIRST_YEAR - 1, LAST_YEAR + 1)
ax.set_xticks(range(1985, LAST_YEAR + 1, 5))
ax.tick_params(colors='#4a4945')

name = f'{FIG_DIR}/figure_supplementary_pubmed_records'
fig.savefig(f'{name}.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{name}.tif', dpi=(300, 300))
plt.show()
