"""Parallel, resumable, patient chunked downloader for a Google Drive file.

Google Drive applies a per-file anonymous download quota ("Too many users have
viewed or downloaded this file recently").  When that page is returned the
request is retried later instead of aborting, so the job can simply be left
running in the background until the quota window reopens.
"""
import os
import sys
import time
import shutil
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

FILE_ID = "1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r"
TOTAL = 5936812008
URL = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t" % FILE_ID
OUT = r"d:\Dark\LOLBlur_test.zip"
PARTS_DIR = r"d:\Dark\dl_parts"
LOG = os.path.join(PARTS_DIR, "download.log")
CHUNK = 16 * 1024 * 1024
WORKERS = int(os.environ.get("WORKERS", "3"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "20000"))

os.makedirs(PARTS_DIR, exist_ok=True)
n_chunks = (TOTAL + CHUNK - 1) // CHUNK


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def part_path(i):
    return os.path.join(PARTS_DIR, "part_%05d" % i)


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
    for attempt in range(MAX_ATTEMPTS):
        quota = False
        try:
            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0",
                              "Range": "bytes=%d-%d" % (a, b)})
            r = s.get(URL, stream=True, timeout=180)
            if r.status_code not in (200, 206):
                raise RuntimeError("http %s" % r.status_code)
            if r.headers.get("Content-Type", "").startswith("text/html"):
                raise RuntimeError("quota page")
            tmp = part_path(i) + ".tmp"
            got = 0
            with open(tmp, "wb") as f:
                for c in r.iter_content(512 * 1024):
                    if not c:
                        continue
                    f.write(c)
                    got += len(c)
            if got != need:
                raise RuntimeError("size mismatch got=%d need=%d" % (got, need))
            os.replace(tmp, part_path(i))
            return i, need, True
        except Exception as e:
            msg = str(e)
            quota = "quota page" in msg
            if attempt == MAX_ATTEMPTS - 1:
                log("chunk %d FAILED: %s" % (i, msg))
                return i, need, False
            time.sleep(35 if quota else min(20, 4 + 2 * attempt))
    return i, need, False


def main():
    todo = [i for i in range(n_chunks) if not is_done(i)]
    total_done = n_chunks - len(todo)
    log("chunks total=%d, remaining=%d, workers=%d" % (n_chunks, len(todo), WORKERS))
    if not todo:
        log("nothing to download")
    else:
        state = {"done": total_done, "failed": 0}
        stop = threading.Event()
        t0 = time.time()

        def heartbeat():
            last = state["done"]
            while not stop.wait(120):
                now = time.time()
                done = state["done"]
                spd = (done - last) * CHUNK / (now - t0) / 1024 / 1024
                log("progress %.1f%%  %.2f/%.2f GB  %.2f MB/s  done_chunks=%d failed=%d  elapsed=%.0fmin" % (
                    done * CHUNK / TOTAL * 100, done * CHUNK / 1024**3, TOTAL / 1024**3,
                    spd, done, state["failed"], (now - t0) / 60))

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()

        failed = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(download_chunk, i): i for i in todo}
            for fut in as_completed(futs):
                i, need, ok = fut.result()
                if ok:
                    state["done"] += 1
                else:
                    failed.append(i)
                    state["failed"] += 1
                    log("chunk %d failed permanently" % i)
        stop.set()
        if failed:
            log("FAILED chunks: %s count=%d" % (failed[:20], len(failed)))
            sys.exit(1)

    log("all chunks done, concatenating -> %s" % OUT)
    with open(OUT, "wb") as out:
        for i in range(n_chunks):
            with open(part_path(i), "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
    size = os.path.getsize(OUT)
    log("final size = %d bytes (expected %d)" % (size, TOTAL))
    if size == TOTAL:
        shutil.rmtree(PARTS_DIR, ignore_errors=True)
        log("DOWNLOAD OK")
    else:
        log("SIZE MISMATCH")
        sys.exit(1)


if __name__ == "__main__":
    main()
