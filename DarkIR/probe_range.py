import requests, time
fid = '1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r'
u = 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t' % fid
for label, rng in [('late-range', 'bytes=4700000000-4701048575'), ('mid-range', 'bytes=1000000000-1001048575')]:
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0', 'Range': rng})
    try:
        r = s.get(u, stream=True, timeout=60)
        n = 0
        for c in r.iter_content(262144):
            n += len(c)
        print(label, 'status=', r.status_code, 'got=', n, 'ctype=', r.headers.get('Content-Type'))
    except Exception as e:
        print(label, 'ERR', e)
    time.sleep(2)
