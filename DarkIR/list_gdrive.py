"""Recursively list a public Google Drive folder tree WITHOUT downloading."""
import sys
import requests
from gdown.download_folder import (
    _parse_embedded_folder_view,
    _GoogleDriveFile,
    _extract_folder_id,
)

FOLDER_URL = "https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX"

sess = requests.Session()
stats = {"folders": 0, "files": 0}

def walk(folder_id, depth=0, path=""):
    try:
        name, children = _parse_embedded_folder_view(
            sess=sess, folder_id=folder_id, verify=True, timeout=60
        )
    except Exception as e:
        print("  " * depth + f"[ERROR] {e}")
        return
    stats["folders"] += 1
    print("  " * depth + f"DIR  {name}/   ({len(children)} items)")
    for cid, cname, ctype in children:
        if ctype == _GoogleDriveFile.TYPE_FOLDER:
            walk(cid, depth + 1, path + "/" + cname)
        else:
            stats["files"] += 1
            if depth <= 1:
                print("  " * (depth + 1) + f"file {cname}  [id={cid}]")
            elif stats["files"] <= 6 or stats["files"] % 500 == 0:
                print("  " * (depth + 1) + f"file {cname}")
    if depth == 0:
        print(f"\n== total: {stats['folders']} folders, {stats['files']} files ==")

fid = _extract_folder_id(url=FOLDER_URL)
walk(fid)
