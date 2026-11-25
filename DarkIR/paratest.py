"""Test aggregate download speed with N parallel range connections."""
import requests, time
from concurrent.futures import ThreadPoolExecutor

fid = '1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r'
URL = 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t' % fid
CHUNK = 4 * 1024 * 1024  # 4MB each
N = 8

def fetch(i):
    a = i * CHUNK
    b = a + CHUNK - 1
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=%d-%d' % (a, b)})
    r = s.get(URL, stream=True, timeout=120)
    n = 0
    for c in r.iter_content(262144):
        n += len(c)
    return n

t = time.time()
with ThreadPoolExecutor(max_workers=N) as ex:
    sizes = list(ex.map(fetch, range(N)))
dt = time.time() - t
total = sum(sizes)
print('aggregate: %.2f MB in %.1fs -> %.2f MB/s over %d conns' % (
    total / 1024 / 1024, dt, total / 1024 / 1024 / dt, N))
