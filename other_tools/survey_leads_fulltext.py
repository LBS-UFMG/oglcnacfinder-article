"""For each lead (DOIs confirmed by survey_leads.py): abstract, full-text URLs, glycan type covered (O-GlcNAc vs O-GalNAc), availability sentences."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / 'survey_leads_fulltext.json'
HEADERS = {'User-Agent': 'Mozilla/5.0 (literature survey)'}

DOIS = {
    'PTG-PLM (2022)': '10.3390/axioms11090469',
    'GlycoMine (2015)': '10.1093/bioinformatics/btu852',
    'GlycoMine-struct (2016)': '10.1038/srep34595',  # Sci Rep 6:34595, article number as printed in O-GlcNAcPRED-II / DeepO-GlcNAc
    'NetOGlyc-4.0 (2013)': '10.1038/emboj.2013.79',
    'GlycoEP (2013)': '10.1371/journal.pone.0067008',
    'DAPPLE 2 (2016)': '10.1021/acs.jproteome.6b00304',
    'Captor, Zhu et al. (2021)': '10.1142/s0219720021500293',
    'Alkuhlani O-glycosylation (2023)': '10.21608/ijicis.2023.160986.1218',
}
AVAIL = re.compile(r'(availab|github|gitlab|zenodo|figshare|web ?server|webserver|freely accessible|source code|http)', re.I)
GLYC = {'O-GlcNAc': re.compile(r'O-?GlcNAc', re.I), 'O-GalNAc': re.compile(r'O-?GalNAc|mucin', re.I)}


def get(url, accept=None):
    h = dict(HEADERS)
    if accept:
        h['Accept'] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read()


def europepmc(doi):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': f'DOI:"{doi}"', 'resultType': 'core', 'format': 'json'})
    res = json.loads(get(url))['resultList']['result']
    return res[0] if res else {}


def crossref_title(doi):
    try:
        m = json.loads(get(f'https://api.crossref.org/works/{urllib.parse.quote(doi)}'))['message']
        return (m.get('title') or [''])[0], (m.get('container-title') or [''])[0]
    except Exception as exc:
        return f'crossref error: {exc}', ''


def fulltext(rec, doi):
    """PMC XML when available, else the publisher landing page (open-access HTML)."""
    if rec.get('pmcid'):
        try:
            root = ET.fromstring(get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{rec['pmcid']}/fullTextXML"))
            text = ' '.join(' '.join(root.itertext()).split())
            links = {e.get('{http://www.w3.org/1999/xlink}href') for e in root.iter() if e.get('{http://www.w3.org/1999/xlink}href')}
            return 'PMC', text, links
        except Exception:
            pass
    try:
        html = get(f'https://doi.org/{doi}').decode('utf-8', 'replace')
        text = ' '.join(re.sub(r'<script.*?</script>|<style.*?</style>|<[^>]+>', ' ', html, flags=re.S).split())
        links = set(re.findall(r'href="(https?://[^"]+)"', html))
        return 'publisher', text, links
    except Exception as exc:
        return f'none ({exc})', '', set()


out = {}
for name, doi in DOIS.items():
    title, journal = crossref_title(doi)
    rec = europepmc(doi)
    abstract = re.sub('<[^>]+>', '', rec.get('abstractText', ''))
    source, text, links = fulltext(rec, doi)
    links = sorted(l for l in links | set(re.findall(r'https?://[^\s,;)\]"<]+', text + ' ' + abstract))
                   if l and not re.search(r'doi\.org|creativecommons|ncbi\.nlm|pubmed|europepmc|scholar\.google|crossref|orcid|twitter|facebook|linkedin|mdpi\.com/(?!.*s1)|oup\.com|wiley|springer|nature\.com|plos\.org|acs\.org|worldscientific', l))
    sentences = re.split(r'(?<=[.!?])\s+', text or abstract)
    avail = [s for s in sentences if AVAIL.search(s) and len(s) < 500][:10]
    counts = {k: len(p.findall(text or abstract)) for k, p in GLYC.items()}
    out[name] = {'doi': doi, 'title': title, 'journal': journal, 'pmcid': rec.get('pmcid', ''), 'fulltext_source': source,
                 'abstract': abstract, 'mentions': counts, 'links': links, 'availability_sentences': avail}
    print(f'\n## {name} | {doi} | {journal}\n  title: {title}\n  full text: {source} | mentions: {counts}')
    print('  abstract:', abstract[:1500] or '-')
    print('  links:', links[:20])
    for s in avail:
        print('   -', s[:300])
    time.sleep(1)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('->', OUT)
