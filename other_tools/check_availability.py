"""Check whether the web servers and code repositories of O-GlcNAc site predictors are reachable."""
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / f'availability_{date.today().isoformat()}.json'
HEADERS = {'User-Agent': 'Mozilla/5.0 (availability check)'}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE  # several academic servers have expired certificates

SERVERS = {
    'YinOYang 1.2': 'https://services.healthtech.dtu.dk/services/YinOYang-1.2/',
    'OGTSite': 'http://csb.cse.yzu.edu.tw/OGTSite/',
    'O-GlcNAcPRED-II': 'http://121.42.167.206/OGlcPred/',
    'PTM-ssMP': 'http://bioinformatics.ustc.edu.cn/PTM-ssMP/index/',
    'LM-OGlcNAc-Site (server)': 'http://kcdukkalab.org/LMOGlcNAcSite',
    'DeepO-GlcNAc': 'http://124.220.189.245:8000/',
    'O-GlcNAcPRED-DL': 'https://oglcnac.org/pred_dl/',
    'Khalid et al. (Kaggle data)': 'https://www.kaggle.com/datasets/esm2oglcnac/o-glcnac-prediction-model',
}
REPOS = {
    'LM-OGlcNAc-Site': 'KCLabMTU/LM-OGlcNAc-Site',
    'Stack-OglyPred-PLM': 'PakhrinLab/Stack-OglyPred-PLM',
    'O-GlcNAc spatial env (Hong et al. 2025)': 'pseokyoung/o-glcnac-prediction',
}


def http(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            body = r.read(4000).decode('utf-8', 'replace')
            title = body.split('<title>')[1].split('</title>')[0].strip() if '<title>' in body else ''
            return {'status': r.status, 'final_url': r.geturl(), 'title': title[:100]}
    except Exception as exc:
        return {'status': 'unreachable', 'error': str(exc)[:150]}


def github(repo):
    try:
        req = urllib.request.Request(f'https://api.github.com/repos/{repo}', headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            m = json.load(r)
        return {'exists': True, 'pushed_at': m['pushed_at'][:10], 'archived': m['archived'], 'stars': m['stargazers_count']}
    except Exception as exc:
        return {'exists': False, 'error': str(exc)[:150]}


def github_search(q):
    req = urllib.request.Request('https://api.github.com/search/repositories?' + urllib.parse.urlencode({'q': q}), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return [(i['full_name'], i['description']) for i in json.load(r)['items'][:5]]


result = {'servers': {}, 'repos': {}}
for name, url in SERVERS.items():
    result['servers'][name] = {'url': url, **http(url)}
    print(f'{name:32s} {result["servers"][name]}')
for name, repo in REPOS.items():
    result['repos'][name] = {'repo': repo, **github(repo)}
    print(f'{name:32s} {result["repos"][name]}')
    time.sleep(1)
for q in ['O-GlcNAcylation RNN Seber', 'oglcnac rnn', 'DeepO-GlcNAc', 'O-GlcNAcPRED-DL', 'OGlcNAc prediction']:
    hits = github_search(q)
    result.setdefault('github_search', {})[q] = hits
    print(f'GitHub search "{q}": {hits}')
    time.sleep(3)
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
print('->', OUT)
