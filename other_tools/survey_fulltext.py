"""From PMC full texts: URLs, data/code availability statements, and O-GlcNAc coverage of each candidate tool."""
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
PMC = {
    'OGTSite (2015)': 'PMC4682369',
    'LM-OGlcNAc-Site (2023)': 'PMC10650050',
    'DeepO-GlcNAc (2024)': 'PMC11500328',
    'Khalid et al. ESM-2 (2024)': 'PMC11687694',
    'Stack-OglyPred-PLM (2024)': 'PMC11552629',
    'O-glycosylation spatial env (2025)': 'PMC11814488',
    'CLASPP (2026)': 'PMC13475974',
    'MTPrompt-PTM (2025)': 'PMC12190693',
    'PTM-ssMP (2018)': 'PMC6036757',
    'multiple kernel SVM (2017)': 'PMC5410141',
    'review 2021 (tools evaluated)': 'PMC8197665',
}
AVAIL = re.compile(r'(availab|github|zenodo|figshare|web ?server|webserver|freely accessible|source code)', re.I)
OGLC = re.compile(r'O-?GlcNAc', re.I)

out = {}
for name, pmc in PMC.items():
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML'
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
    except Exception as exc:
        print(f'\n## {name} ({pmc}): full text not available ({exc})')
        continue
    text = ' '.join(' '.join(root.itertext()).split())
    links = sorted({e.get('{http://www.w3.org/1999/xlink}href') for e in root.iter() if e.get('{http://www.w3.org/1999/xlink}href')}
                   | set(re.findall(r'https?://[^\s,;)\]]+', text)))
    links = [l for l in links if not re.search(r'doi\.org|creativecommons|ncbi\.nlm|pubmed|europepmc|scholar\.google|crossref', l or '')]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    avail = [s for s in sentences if AVAIL.search(s)][:8]
    n_oglc = len(OGLC.findall(text))
    out[name] = {'pmc': pmc, 'links': links, 'availability_sentences': avail, 'oglcnac_mentions': n_oglc}
    print(f'\n## {name} ({pmc}) | O-GlcNAc mentions in full text: {n_oglc}')
    print('links:', links[:15])
    for s in avail:
        print('  -', s[:300])
    if 'review' in name:
        tools = re.findall(r'\b([A-Z][\w-]*(?:PRED|Scan|Site|Yang|Pred|GlcNAc\w*)[\w.-]*)\b', text)
        print('  tool-like names:', sorted(set(tools))[:40])
    time.sleep(1)
(HERE / 'survey_fulltext.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
