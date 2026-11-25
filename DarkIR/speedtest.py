import requests, time

fid = '1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r'
urls = {
    'usercontent': 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t' % fid,
    'uc':          'https://drive.google.com/uc?export=download&id=%s&confirm=t' % fid,
    'docs_uc':     'https://docs.google.com/uc?export=download&id=%s&confirm=t' % fid,
}
for name, u in urls.items():
    try:
        s = requests.Session()
        s.headers.update({'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-4194303'})
        t = time.time()
        r = s.get(u, stream=True, timeout=60, allow_redirects=True)
        n = 0
        for c in r.iter_content(262144):
            n += len(c)
            if n >= 4 * 1024 * 1024:
                break
        dt = time.time() - t
        ctype = r.headers.get('Content-Type')
        print('%-12s status=%s got=%.2fMB in %.1fs -> %.2f MB/s ctype=%s' % (
            name, r.status_code, n / 1024 / 1024, dt, n / 1024 / 1024 / dt, ctype))
    except Exception as e:
        print('%-12s ERR %s' % (name, e))
