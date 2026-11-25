import requests, re, json
url = 'https://cidautes-my.sharepoint.com/:f:/g/personal/alvgar_cidaut_es/Epntbl4SucFNpeIT_jyYZ-cB9BamMbacbyq_svrkMCpShA?e=XB9YBB'
H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'}
s = requests.Session()
r = s.get(url, headers=H, timeout=60)
t = r.text
open(r'd:\Dark\onedrive_page.html', 'w', encoding='utf-8', errors='ignore').write(t)
print('status', r.status_code, 'len', len(t))
for kw in ['.pt', 'downloadUrl', 'FileLeafRef', 'ServerRedirected', 'apiUrl', 'listData', 'DriveItem', 'children', 'webUrl', '__INITIAL']:
    print(kw, '->', t.count(kw))
# print snippets around .pt
for m in re.finditer(r'.{60}\.pt.{60}', t):
    print('SNIP:', m.group(0)[:200])
# look for any https with /personal/.../Documents
for m in set(re.findall(r'https?://[^"\'\\s<>]+', t)):
    if ('download' in m.lower() or '.pt' in m.lower() or 'Documents' in m) and len(m) < 300:
        print('URL:', m)
