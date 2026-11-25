"""Download LOLv2-real and LOLv2-synth test pairs from HuggingFace mirrors
and place them in the layout expected by dataset_all_LOL.py:

    data/datasets/LOL-v2/Real_captured/test/{Low,Normal}/*.png
    data/datasets/LOL-v2/Synthetic/test/{Low,Normal}/*.png
"""
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

H = {'User-Agent': 'Mozilla/5.0'}
REPOS = {
    'real': 'okhater/lolv2-real',
    'synth': 'okhater/lolv2-synthetic',
}
DEST_ROOT = r'd:\Dark\DarkIR\data\datasets\LOL-v2'
WORKERS = 16
RETRIES = 8
BASES = [
    'https://huggingface.co/datasets/%s/resolve/main/',
    'https://hf-mirror.com/datasets/%s/resolve/main/',
]


def list_files(repo):
    apis = ['https://huggingface.co/api/datasets/' + repo,
            'https://hf-mirror.com/api/datasets/' + repo]
    last = None
    for attempt in range(6):
        for api in apis:
            try:
                r = requests.get(api, headers=H, timeout=60)
                if r.status_code == 200:
                    return [s['rfilename'] for s in r.json().get('siblings', [])]
                last = 'status %s' % r.status_code
            except Exception as e:
                last = '%s: %s' % (type(e).__name__, str(e)[:60])
        time.sleep(3 + attempt * 2)
    raise RuntimeError('cannot list ' + repo + ' -> ' + str(last))


def need_files(files):
    return [f for f in files
            if (f.startswith('Test/Input/') or f.startswith('Test/GT/'))
            and f.lower().endswith('.png')]


def local_path(split, hf_path):
    sub = 'Real_captured' if split == 'real' else 'Synthetic'
    if hf_path.startswith('Test/Input/'):
        kind = 'Low'
    elif hf_path.startswith('Test/GT/'):
        kind = 'Normal'
    else:
        return None
    base = os.path.basename(hf_path)
    return os.path.join(DEST_ROOT, sub, 'test', kind, base)


def dl_one(split, repo, hf_path):
    out = local_path(split, hf_path)
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return hf_path, True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(RETRIES):
        for base_tpl in BASES:
            url = (base_tpl % repo) + hf_path
            try:
                r = requests.get(url, headers=H, stream=True, timeout=120)
                if r.status_code != 200:
                    continue
                tmp = out + '.tmp'
                got = 0
                with open(tmp, 'wb') as f:
                    for c in r.iter_content(256 * 1024):
                        if c:
                            f.write(c)
                            got += len(c)
                if got < 1000:
                    continue
                os.replace(tmp, out)
                return hf_path, True
            except Exception:
                continue
        time.sleep(1 + attempt)
    return hf_path, False


def main():
    todo = []
    for split, repo in REPOS.items():
        files = list_files(repo)
        nfs = need_files(files)
        print('[list] %-6s %-25s test files=%d' % (split, repo, len(nfs)))
        for f in nfs:
            todo.append((split, repo, f))
    print('total to download:', len(todo))
    t0 = time.time()
    done = ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(dl_one, s, repo, f): (s, f) for s, repo, f in todo}
        for fut in as_completed(futs):
            done += 1
            f, success = fut.result()
            ok += 1 if success else 0
            if done % 20 == 0 or done == len(todo):
                el = time.time() - t0
                print('  %4d/%4d ok=%d  %.1fs elapsed' % (done, len(todo), ok, el), flush=True)
    print('DONE ok=%d/%d' % (ok, len(todo)))
    for split in REPOS:
        for kind in ['Low', 'Normal']:
            d = os.path.join(DEST_ROOT, 'Real_captured' if split == 'real' else 'Synthetic', 'test', kind)
            n = len([x for x in os.listdir(d)]) if os.path.isdir(d) else 0
            print('  %-6s %-7s -> %d' % (split, kind, n))


if __name__ == '__main__':
    main()
