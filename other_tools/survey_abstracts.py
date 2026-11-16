"""Retrieve abstracts of candidate O-GlcNAc site predictors and extract URLs (servers, code)."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / 'survey_abstracts.json'

DOIS = {
    # dedicated O-GlcNAc site predictors
    'O-GlcNAcPRED (2013)': '10.1039/c3mb70326f',
    'two-layered ML with OGT motifs (2015)': '10.1186/1471-2105-16-s18-s10',
    'PGlcS (2015)': '10.1016/j.jtbi.2015.06.026',
    'LM-OGlcNAc-Site (2023)': '10.3390/ijms242116000',
    'Seber & Braatz RNN (2024)': '10.1016/j.compchemeng.2024.108818',
    'DeepO-GlcNAc (2024)': '10.3389/fcell.2024.1456728',
    'Khalid et al. ESM-2 (2024)': '10.1371/journal.pone.0316215',
    'O-GlcNAcPRED-DL (2024)': '10.1021/acs.jproteome.3c00458',
    # reviews / evaluations (sources for the tool list)
    'review: prediction unattained objective (2021)': '10.2147/aabc.s294867',
    'evaluation: reliability of computational prediction (2014)': '10.1016/j.bbapap.2013.12.002',
    'chapter: computational prediction (2018)': '10.1007/978-1-4939-7717-8_14',
    'OGT target sequence compositional bias (2023)': '10.1016/j.jbc.2023.104629',
    # multi-PTM / general O-glycosylation predictors: do they cover O-GlcNAc?
    'human O-linked glycosylation, stacked PLM (2024)': '10.1093/bioinformatics/btae643',
    'O-glycosylation, spatial local environment (2025)': '10.1093/bioinformatics/btaf034',
    'CLASPP unified PTM (2026)': '10.1371/journal.pcbi.1014616',
    'MTPrompt-PTM (2025)': '10.3390/biom15060843',
    'PTM-ssMP web server (2018)': '10.7150/ijbs.24121',
    'multiple kernel SVM PTM (2017)': '10.7717/peerj.3261',
}


def europepmc(doi):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': f'DOI:"{doi}"', 'resultType': 'core', 'format': 'json'})
    with urllib.request.urlopen(url, timeout=60) as r:
        res = json.load(r)['resultList']['result']
    return res[0] if res else {}


out = {}
for name, doi in DOIS.items():
    rec = europepmc(doi)
    abstract = re.sub('<[^>]+>', '', rec.get('abstractText', ''))
    urls = sorted(set(re.findall(r'https?://[^\s,;)\]]+', abstract)))
    out[name] = {'doi': doi, 'title': rec.get('title', ''), 'year': rec.get('pubYear', ''),
                 'journal': rec.get('journalInfo', {}).get('journal', {}).get('title', ''),
                 'pmcid': rec.get('pmcid', ''), 'abstract': abstract, 'urls': urls}
    print(f'\n## {name} | {rec.get("pubYear", "")} | {doi} | PMC: {rec.get("pmcid", "-")}')
    print('URLs:', urls or '-')
    print(abstract[:1300])
    time.sleep(1)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('->', OUT)
