import requests, collections, time
H = {'User-Agent': 'Mozilla/5.0'}

def list_files(repo):
    r = requests.get('https://huggingface.co/api/datasets/' + repo, headers=H, timeout=60)
    return [s['rfilename'] for s in r.json().get('siblings', [])]

for repo in ['okhater/lolv2-synthetic']:
    files = list_files(repo)
    print('===', repo, 'total', len(files))
    c = collections.Counter()
    for f in files:
        p = f.split('/'); c['/'.join(p[:2])] += 1
    for k, v in sorted(c.items()):
        print('   %-25s %d' % (k, v))
    # pick one test input for speed test
    sample = next(f for f in files if f.startswith('Test/Input/'))
    for base in ['https://huggingface.co/datasets/%s/resolve/main/' % repo,
                 'https://hf-mirror.com/datasets/%s/resolve/main/' % repo]:
        url = base + sample
        try:
            t = time.time()
            r = requests.get(url, headers=H, timeout=60, stream=True)
            n = 0
            for ch in r.iter_content(1 << 16):
                n += len(ch)
            dt = time.time() - t
            print('   %-55s status=%s %.1fKB %.2fMB/s' % (base.split('/')[2], r.status_code, n/1024, n/1024/1024/dt))
        except Exception as e:
            print('   ', base.split('/')[2], 'ERR', type(e).__name__, str(e)[:60])
