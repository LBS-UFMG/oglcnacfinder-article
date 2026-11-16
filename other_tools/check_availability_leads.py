"""Live availability check for the servers and repositories of the leads (URLs as printed in the papers)."""
import json
import ssl
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / f'availability_leads_{date.today().isoformat()}.json'
HEADERS = {'User-Agent': 'Mozilla/5.0 (availability check)'}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE  # several academic servers have expired certificates

SERVERS = {
    'GlycoMine (paper URL)': 'http://www.structbioinfor.org/Lab/GlycoMine/',
    'GlycoMine-struct (paper URL)': 'http://glycomine.erc.monash.edu/Lab/GlycoMine_Struct/',
    'GlycoEP (paper URL)': 'http://www.imtech.res.in/raghava/glycoep/',
    'DAPPLE 2 (paper URL)': 'http://saphire.usask.ca/saphire/dapple2',
    'NetOGlyc-4.0 (paper URL)': 'http://www.cbs.dtu.dk/services/NetOGlyc-4.0/',
    'Mauri et al. 2021 OGP (GitLab)': 'https://gitlab.in2p3.fr/cmsb-public/OGP',
}
REPOS = {
    'PTG-PLM': 'Alhasanalkuhlani/PTG-PLM',
    'Captor (Zhu et al. 2021)': 'YanZhu06/Captor',
}


def http(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            body = r.read(6000).decode('utf-8', 'replace')
            title = body.split('<title>')[1].split('</title>')[0].strip() if '<title>' in body else ''
            return {'status': r.status, 'final_url': r.geturl(), 'title': ' '.join(title.split())[:100]}
    except Exception as exc:
        return {'status': 'unreachable', 'error': str(exc)[:150]}


def github(repo):
    try:
        req = urllib.request.Request(f'https://api.github.com/repos/{repo}', headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            m = json.load(r)
        req = urllib.request.Request(f'https://api.github.com/repos/{repo}/contents', headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            files = [f['name'] for f in json.load(r)]
        return {'exists': True, 'pushed_at': m['pushed_at'][:10], 'archived': m['archived'], 'description': m['description'], 'top_level': files}
    except Exception as exc:
        return {'exists': False, 'error': str(exc)[:150]}


result = {'servers': {}, 'repos': {}}
for name, url in SERVERS.items():
    result['servers'][name] = {'url': url, **http(url)}
    print(f'{name:34s} {result["servers"][name]}')
for name, repo in REPOS.items():
    result['repos'][name] = {'repo': repo, **github(repo)}
    print(f'{name:34s} {result["repos"][name]}')
    time.sleep(1)
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
print('->', OUT)
