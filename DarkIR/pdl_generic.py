"""Generic parallel resumable range downloader.

Usage: python pdl_generic.py <url> <out_path> [workers] [chunk_mb]
"""
import os
import sys
import time
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = sys.argv[1]
OUT = sys.argv[2]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CHUNK = int(sys.argv[4]) * 1024 * 1024 if len(sys.argv) > 4 else 8 * 1024 * 1024
RETRIES = 6

PARTS_DIR = OUT + ".parts"
os.makedirs(PARTS_DIR, exist_ok=True)


def probe_total():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
    r = s.get(URL, stream=True, timeout=60)
    cr = r.headers.get("Content-Range", "")
    if "/" in cr:
        return int(cr.split("/")[-1])
    cl = r.headers.get("Content-Length")
    return int(cl) if cl else None


TOTAL = probe_total()
if not TOTAL:
    print("Could not determine total size")
    sys.exit(1)
n_chunks = (TOTAL + CHUNK - 1) // CHUNK
print("total=%.2f MB, chunks=%d" % (TOTAL / 1024 / 1024, n_chunks), flush=True)


def part_path(i):
    return os.path.join(PARTS_DIR, "p%05d" % i)


def chunk_range(i):
    a = i * CHUNK
    b = min(a + CHUNK - 1, TOTAL - 1)
    return a, b, b - a + 1


def is_done(i):
    p = part_path(i)
    return os.path.exists(p) and os.path.getsize(p) == chunk_range(i)[2]


def download_chunk(i):
    a, b, need = chunk_range(i)
    if is_done(i):
        return i, need, True
    for attempt in range(RETRIES):
        try:
            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0", "Range": "bytes=%d-%d" % (a, b)})
            r = s.get(URL, stream=True, timeout=180)
            if r.status_code not in (200, 206):
                raise RuntimeError("status %s" % r.status_code)
            tmp = part_path(i) + ".tmp"
            got = 0
            with open(tmp, "wb") as f:
                for c in r.iter_content(512 * 1024):
                    if c:
                        f.write(c)
                        got += len(c)
            if got != need:
                raise RuntimeError("size %d != %d" % (got, need))
            os.replace(tmp, part_path(i))
            return i, need, True
        except Exception:
            if attempt == RETRIES - 1:
                return i, need, False
            time.sleep(2 + attempt * 2)
    return i, need, False


todo = [i for i in range(n_chunks) if not is_done(i)]
done_bytes = (n_chunks - len(todo)) * CHUNK
t0 = time.time()
last = [t0, done_bytes]
failed = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(download_chunk, i): i for i in todo}
    for k, fut in enumerate(as_completed(futs), 1):
        i, need, ok = fut.result()
        if ok:
            done_bytes += need
        else:
            failed.append(i)
        now = time.time()
        if now - last[0] >= 8 or k == len(todo):
            spd = (done_bytes - last[1]) / (now - last[0]) / 1024 / 1024
            print("  %.1f%%  %.1f/%.1f MB  %.2f MB/s  failed=%d" % (
                done_bytes / TOTAL * 100, done_bytes / 1024 / 1024,
                TOTAL / 1024 / 1024, spd, len(failed)), flush=True)
            last[0], last[1] = now, done_bytes

if failed:
    print("FAILED:", failed[:20])
    sys.exit(1)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "wb") as out:
    for i in range(n_chunks):
        with open(part_path(i), "rb") as f:
            shutil.copyfileobj(f, out, 8 * 1024 * 1024)
size = os.path.getsize(OUT)
print("final %d bytes (expected %d)" % (size, TOTAL))
if size == TOTAL:
    shutil.rmtree(PARTS_DIR, ignore_errors=True)
    print("OK:", OUT)
else:
    print("SIZE MISMATCH")
    sys.exit(1)
