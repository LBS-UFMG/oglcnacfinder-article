"""Crossref lookup of DOIs for tools that are only named in the papers (titles are printed for manual verification)."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')
# Crossref asks API clients for a contact address: set it in the environment variable CONTACT_EMAIL
CONTACT = os.environ.get('CONTACT_EMAIL')
HEADERS = {'User-Agent': f'oglcnac-survey (mailto:{CONTACT})' if CONTACT else 'oglcnac-survey'}
QUERIES = {
    'dbOGAP / O-GlcNAcScan (Wang 2011)': 'dbOGAP integrated bioinformatics resource protein O-GlcNAcylation BMC Bioinformatics 2011',
    'YinOYang (Gupta & Brunak 2002)': 'Gupta Brunak Prediction of glycosylation across the human proteome and the correlation to protein function Pacific Symposium Biocomputing 2002',
    'Wu et al. 2014 (Lee group)': 'Wu 2014 BMC Bioinformatics O-GlcNAcylation sites substrate motifs Lee',
    'MusiteDeep (2020)': 'MusiteDeep deep-learning based webserver protein post-translational modification site prediction visualization 2020',
    'SPRINT-Gly': 'SPRINT-Gly predicting N- and O-linked glycosylation sites of human and mouse proteins',
    'GlycoMine_PU': 'GlycoMine PU positive unlabelled learning glycosylation site prediction',
    'EnsembleGly': 'Glycosylation site prediction using ensembles of Support Vector Machine classifiers',
    'GPP': 'Prediction of glycosylation sites using random forests GPP Hamby Hirst',
    'CKSAAP_OGlySite': 'Prediction of mucin-type O-glycosylation sites in mammalian proteins using the composition of k-spaced amino acid pairs',
    'GlycoPP': 'GlycoPP webserver prediction N- and O-glycosites prokaryotic protein sequences',
}
for name, q in QUERIES.items():
    url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.bibliographic': q, 'rows': 2})
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        items = json.load(r)['message']['items']
    print(f'\n## {name}')
    for i in items:
        year = (i.get('issued', {}).get('date-parts') or [[None]])[0][0]
        print(f"  {year} | {(i.get('container-title') or [''])[0]} | {i.get('DOI')} | {(i.get('title') or [''])[0][:130]}")
    time.sleep(1)
