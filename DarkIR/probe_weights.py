import requests, json
H = {'User-Agent': 'Mozilla/5.0'}
for repo in ['cidautai/DarkIR', 'Fundacion-Cidaut/DarkIR']:
    for ep in ['releases', 'releases/latest']:
        try:
            r = requests.get('https://api.github.com/repos/%s/%s' % (repo, ep), headers=H, timeout=30)
            print('==', repo, ep, r.status_code)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, list):
                    for rel in d:
                        for a in rel.get('assets', []):
                            print('  ', rel.get('tag_name'), a['name'], a.get('size'), a.get('browser_download_url'))
                else:
                    for a in d.get('assets', []):
                        print('  ', d.get('tag_name'), a['name'], a.get('size'), a.get('browser_download_url'))
        except Exception as e:
            print('ERR', repo, type(e).__name__, str(e)[:60])
# HF full-text search
try:
    r = requests.get('https://huggingface.co/api/quicksearch?q=DarkIR_1k&type=model', headers=H, timeout=30)
    print('hf qs:', r.status_code, r.text[:300])
except Exception as e:
    print('hf ERR', e)
