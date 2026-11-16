"""Find bibliographic records, abstracts and URLs for the leads found in the PDFs (PTG-PLM and other glycosylation tools)."""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / 'survey_leads.json'
# Crossref asks API clients for a contact address: set it in the environment variable CONTACT_EMAIL
CONTACT = os.environ.get('CONTACT_EMAIL')
HEADERS = {'User-Agent': f'oglcnac-survey (mailto:{CONTACT})' if CONTACT else 'oglcnac-survey'}

# bibliographic queries built from the references as printed in the PDFs
QUERIES = {
    'PTG-PLM (Alkuhlani 2022, Axioms 11:469)': 'Alkuhlani PTG-PLM protein language model glycosylation Axioms',
    'GlycoMine (Li 2015, Bioinformatics 31:1411)': 'GlycoMine machine learning glycosylation site prediction Bioinformatics 2015',
    'GlycoMine-struct (Li 2016, Sci Rep 6:34595)': 'GlycoMine struct glycosylation structural features Scientific Reports 2016',
    'NetOGlyc-4.0 (Steentoft 2013)': 'Precision mapping of the human O-GalNAc glycoproteome SimpleCell NetOGlyc4.0',
    'GlycoEP (Chauhan 2013, PLoS ONE)': 'GlycoEP prediction glycosites eukaryotic proteins Chauhan',
    'DAPPLE 2 (Trost 2016, JPR 15:2760)': 'DAPPLE 2 post-translational modification sites homology Trost',
    'Zhu 2022 (JBCB 20:2150029)': 'O-glycosylation site prediction Homo sapiens support vector machine J Bioinform Comput Biol',
    'Alkuhlani 2023 (O-glycosylation, cited by Khalid)': 'Alkuhlani O-glycosylation site prediction protein language model',
}


def crossref(q):
    url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.bibliographic': q, 'rows': 3})
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        items = json.load(r)['message']['items']
    return [{'doi': i.get('DOI'), 'title': (i.get('title') or [''])[0], 'journal': (i.get('container-title') or [''])[0],
             'year': (i.get('issued', {}).get('date-parts') or [[None]])[0][0]} for i in items]


def europepmc(doi):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': f'DOI:"{doi}"', 'resultType': 'core', 'format': 'json'})
    with urllib.request.urlopen(url, timeout=60) as r:
        res = json.load(r)['resultList']['result']
    return res[0] if res else {}


out = {}
for name, q in QUERIES.items():
    hits = crossref(q)
    print(f'\n## {name}')
    for h in hits:
        print(f"  crossref: {h['year']} | {h['journal']} | {h['doi']} | {h['title'][:120]}")
    top = hits[0] if hits else {}
    rec = europepmc(top['doi']) if top.get('doi') else {}
    abstract = re.sub('<[^>]+>', '', rec.get('abstractText', ''))
    urls = sorted(set(re.findall(r'https?://[^\s,;)\]]+', abstract)))
    out[name] = {'crossref_hits': hits, 'pmcid': rec.get('pmcid', ''), 'abstract': abstract, 'urls': urls}
    print('  PMC:', rec.get('pmcid', '-'), '| URLs in abstract:', urls or '-')
    print('  ', abstract[:900])
    time.sleep(1)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('->', OUT)
