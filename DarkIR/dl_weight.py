import requests, time

FOLDER = 'https://cidautes-my.sharepoint.com/:f:/g/personal/alvgar_cidaut_es/Epntbl4SucFNpeIT_jyYZ-cB9BamMbacbyq_svrkMCpShA?e=XB9YBB'
DL = 'https://cidautes-my.sharepoint.com/personal/alvgar_cidaut_es/_layouts/15/download.aspx?UniqueId=029ba74c-3fcd-4402-9abe-8dfc414f263f'
OUT = r'd:\Dark\DarkIR\models\DarkIR_1k_cr_mt.pt'
H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'}

s = requests.Session()
s.headers.update(H)
r0 = s.get(FOLDER, timeout=60, allow_redirects=True)
print('folder:', r0.status_code, 'cookies=', list(s.cookies.keys()))

for attempt in range(3):
    r = s.get(DL, timeout=120, allow_redirects=True, stream=True)
    ct = r.headers.get('Content-Type', '')
    print('dl attempt', attempt, 'status', r.status_code, 'ct', ct, 'final', r.url[:90])
    if ct.startswith('text/html'):
        print('  still html/login, body len', len(r.content))
        r.close()
        time.sleep(5)
        continue
    data = b''
    for c in r.iter_content(1 << 20):
        if c:
            data += c
    print('  bytes', len(data), 'head', data[:8])
    if len(data) > 1000000:
        open(OUT, 'wb').write(data)
        print('SAVED', OUT, len(data))
        break
    r.close()
