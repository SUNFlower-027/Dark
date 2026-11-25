"""Robust Google Drive file downloader bypassing gdown's confirm/limit issues."""
import os
import re
import sys
import time
import requests

FILE_ID = "1cDLedBs5lPqveOvAtMrIkYqOlmw55N7r"
OUT = r"d:\Dark\LOLBlur_test.zip"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})

def get_confirm_token(text):
    m = re.search(r'name="confirm"\s+value="([^"]+)"', text)
    if m:
        return m.group(1)
    m = re.search(r'confirm=([0-9A-Za-z_\-]+)', text)
    if m:
        return m.group(1)
    return None

def candidate_urls(fid):
    return [
        f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={fid}&confirm=t",
        f"https://drive.usercontent.google.com/u/0/uc?id={fid}&export=download",
    ]

def try_download(url, out):
    print("GET", url)
    r = session.get(url, stream=True, timeout=120, allow_redirects=True)
    print("  status", r.status_code, "ctype", r.headers.get("Content-Type"))
    ctype = r.headers.get("Content-Type", "")
    if "text/html" in ctype:
        # possibly a confirmation page
        text = r.text[:200000]
        token = get_confirm_token(text)
        print("  got HTML, confirm token =", token)
        if token:
            sep = "&" if "?" in url else "?"
            url2 = f"{url}{sep}confirm={token}"
            print("  retry with confirm token")
            r = session.get(url2, stream=True, timeout=120, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "")
            print("  status", r.status_code, "ctype", ctype)
        if "text/html" in ctype:
            snippet = re.sub(r"<[^>]+>", " ", text)[:300]
            print("  STILL HTML. snippet:", snippet)
            return False
    total = int(r.headers.get("Content-Length", 0))
    print(f"  size = {total/(1024*1024):.1f} MB" if total else "  size unknown")
    done = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(1024 * 512):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            if total and done % (20 * 1024 * 1024) < 1024 * 512:
                print(f"    {done/(1024*1024):.1f}/{total/(1024*1024):.1f} MB")
    print(f"  finished {done/(1024*1024):.1f} MB -> {out}")
    return done > 1024 * 1024

for url in candidate_urls(FILE_ID):
    try:
        if try_download(url, OUT):
            if os.path.exists(OUT) and os.path.getsize(OUT) > 50 * 1024 * 1024:
                print("OK downloaded:", os.path.getsize(OUT) / (1024*1024), "MB")
                sys.exit(0)
    except Exception as e:
        print("  ERROR:", e)
    time.sleep(2)

print("ALL ATTEMPTS FAILED")
sys.exit(1)

