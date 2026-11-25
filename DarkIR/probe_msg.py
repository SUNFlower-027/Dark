import re
import requests

fid = '1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r'
u = 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t' % fid
s = requests.Session()
s.headers.update({'User-Agent': 'Mozilla/5.0'})
r = s.get(u, stream=True, timeout=60)
print('status', r.status_code, 'ctype', r.headers.get('Content-Type'))
txt = r.text
# strip tags for a readable snippet
plain = re.sub(r'<script[\s\S]*?</script>', ' ', txt)
plain = re.sub(r'<style[\s\S]*?</style>', ' ', plain)
plain = re.sub(r'<[^>]+>', ' ', plain)
plain = re.sub(r'\s+', ' ', plain).strip()
print('LEN', len(txt))
print('SNIPPET:', plain[:600])
