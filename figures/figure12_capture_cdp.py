"""High-resolution screenshots of the O-GlcNAcFinder web server with headless Chrome (DevTools protocol).

usage: python cdp_shot.py <url> <out.png> [--width 1280] [--height 900] [--scale 2] [--wait 8] [--js "<js before shot>"]
         [--full]   (capture the whole page height)
"""
import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websocket

ap = argparse.ArgumentParser()
ap.add_argument('url'); ap.add_argument('out')
ap.add_argument('--width', type=int, default=1280); ap.add_argument('--height', type=int, default=900)
ap.add_argument('--scale', type=float, default=2); ap.add_argument('--wait', type=float, default=8)
ap.add_argument('--js', default=''); ap.add_argument('--full', action='store_true')
a = ap.parse_args()

CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
PORT = 9333
profile = tempfile.mkdtemp(prefix='cdp_')
proc = subprocess.Popen([CHROME, '--headless=new', f'--remote-debugging-port={PORT}', f'--user-data-dir={profile}',
                         '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars',
                         f'--window-size={a.width},{a.height}', 'about:blank'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json'))
            page = next(t for t in tabs if t['type'] == 'page'); break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=120)
    n = [0]

    def cmd(method, **params):
        n[0] += 1; ws.send(json.dumps({'id': n[0], 'method': method, 'params': params}))
        while True:
            m = json.loads(ws.recv())
            if m.get('id') == n[0]:
                return m.get('result', {})

    cmd('Emulation.setDeviceMetricsOverride', width=a.width, height=a.height, deviceScaleFactor=a.scale, mobile=False)
    cmd('Page.enable')
    cmd('Page.navigate', url=a.url)
    time.sleep(a.wait)
    if a.js:
        print('js:', cmd('Runtime.evaluate', expression=a.js, returnByValue=True).get('result', {}).get('value'))
        time.sleep(6)
    params = {'format': 'png', 'captureBeyondViewport': a.full}
    if a.full:
        h = cmd('Runtime.evaluate', expression='document.documentElement.scrollHeight', returnByValue=True)['result']['value']
        params['clip'] = {'x': 0, 'y': 0, 'width': a.width, 'height': h, 'scale': 1}
    data = cmd('Page.captureScreenshot', **params)['data']
    open(a.out, 'wb').write(base64.b64decode(data))
    print('saved', a.out)
    ws.close()
finally:
    proc.kill()
