"""Fetch the missing tail of LOLBlur test.zip with large, byte-resumable ranges.

Google Drive throttles anonymous downloads by a byte window, so every allowed
request should carry as many bytes as possible.  Only two byte ranges are
missing:

  A: chunk 248        -> [4_160_749_568, 4_177_526_784)   16 MB
  B: chunks 294..353  -> [4_932_501_504, 5_936_812_008)  0.94 GiB

Range B is split across workers; each worker streams into its own file and
resumes from the current file size when a connection drops or is throttled.
"""
import os
import sys
import time
import shutil
import threading
import requests

FILE_ID = "1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r"
URL = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t" % FILE_ID
TOTAL = 5936812008
CHUNK = 16 * 1024 * 1024
OUT = r"d:\Dark\LOLBlur_test.zip"
PARTS_DIR = r"d:\Dark\dl_parts"
LOG = os.path.join(PARTS_DIR, "tail.log")

SEG_A = ("A", 248 * CHUNK, 249 * CHUNK - 1)
B_START, B_END = 294 * CHUNK, TOTAL - 1
N_B_WORKERS = int(os.environ.get("WORKERS", "3"))

os.makedirs(PARTS_DIR, exist_ok=True)


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def seg_file(name):
    return os.path.join(PARTS_DIR, "tail_%s.bin" % name)


def segments():
    segs = [SEG_A]
    span = B_END - B_START + 1
    per = span // N_B_WORKERS
    for k in range(N_B_WORKERS):
        a = B_START + k * per
        b = B_END if k == N_B_WORKERS - 1 else a + per - 1
        segs.append(("B%d" % k, a, b))
    return segs


def fetch_segment(seg):
    name, start, end = seg
    path = seg_file(name)
    need_total = end - start + 1
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= need_total:
            return name, True
        cur = start + have
        try:
            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0",
                              "Range": "bytes=%d-%d" % (cur, end)})
            r = s.get(URL, stream=True, timeout=120)
            if r.status_code not in (200, 206):
                raise RuntimeError("http %s" % r.status_code)
            if r.headers.get("Content-Type", "").startswith("text/html"):
                raise RuntimeError("quota page")
            with open(path, "ab") as f:
                for c in r.iter_content(1024 * 1024):
                    if not c:
                        continue
                    f.write(c)
                    have += len(c)
                    if have >= need_total:
                        break
        except Exception as e:
            msg = str(e)
            quota = "quota page" in msg
            log("segment %s stalled at %d/%d (%.1f%%) : %s" % (
                name, have, need_total, have * 100.0 / need_total, msg))
            time.sleep(45 if quota else 8)


def assemble():
    log("assembling final zip -> %s" % OUT)
    with open(OUT, "wb") as out:
        for i in range(0, 248):
            with open(os.path.join(PARTS_DIR, "part_%05d" % i), "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
        with open(seg_file("A"), "rb") as f:
            shutil.copyfileobj(f, out, 8 * 1024 * 1024)
        for i in range(249, 294):
            with open(os.path.join(PARTS_DIR, "part_%05d" % i), "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
        for k in range(N_B_WORKERS):
            with open(seg_file("B%d" % k), "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
    size = os.path.getsize(OUT)
    log("final size = %d (expected %d)" % (size, TOTAL))
    if size != TOTAL:
        log("SIZE MISMATCH")
        sys.exit(1)
    log("DOWNLOAD OK")
    if os.environ.get("AUTO_TEST", "1") == "1":
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        steps = [
            [sys.executable, "prepare_lolblur.py"],
            [sys.executable, "evaluate_full.py", "-p", "./options/test/LOLBlur.yml"],
        ]
        for cmd in steps:
            log("run: " + " ".join(cmd))
            rc = subprocess.run(cmd, cwd=here).returncode
            log("exit code = %d" % rc)


def main():
    segs = segments()
    todo = [s for s in segs
            if (os.path.getsize(seg_file(s[0])) if os.path.exists(seg_file(s[0])) else 0)
            < (s[2] - s[1] + 1)]
    log("segments=%d pending=%d workers=%d" % (len(segs), len(todo), len(todo)))
    if not todo:
        assemble()
        return

    stop = threading.Event()
    t0 = time.time()

    def heartbeat():
        while not stop.wait(120):
            have = 0
            need = 0
            for name, a, b in segs:
                need += b - a + 1
                p = seg_file(name)
                have += min(os.path.getsize(p) if os.path.exists(p) else 0, b - a + 1)
            log("tail progress %.1f%%  %.1f MB / %.1f MB  %.2f MB/s  elapsed %.0f min" % (
                have * 100.0 / need, have / 1024**2, need / 1024**2,
                have / 1024**2 / max(1e-9, time.time() - t0) / 60,
                (time.time() - t0) / 60))

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            for name, ok in ex.map(fetch_segment, segs):
                log("segment %s complete=%s" % (name, ok))
    finally:
        stop.set()
    assemble()


if __name__ == "__main__":
    main()
