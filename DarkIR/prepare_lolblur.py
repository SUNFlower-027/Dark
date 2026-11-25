"""Extract the LOLBlur test split into the layout expected by dataset_LOLBlur.py.

Expected result:
    data/datasets/LOLBlur/test/low_blur_noise/<clip>/*.png
    data/datasets/LOLBlur/test/high_sharp_scaled/<clip>/*.png
"""
import os
import sys
import zipfile
import collections

ZIP_PATH = r"d:\Dark\LOLBlur_test.zip"
DEST = r"d:\Dark\DarkIR\data\datasets\LOLBlur"
NEEDED = ("test/low_blur_noise/", "test/high_sharp_scaled/")


def main():
    if not os.path.exists(ZIP_PATH):
        sys.exit("zip not found: %s" % ZIP_PATH)
    size = os.path.getsize(ZIP_PATH)
    print("zip size = %.2f GB" % (size / 1024**3))
    if size != 5936812008:
        sys.exit("unexpected zip size %d != 5936812008" % size)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        print("entries =", len(names))
        counts = collections.Counter()
        for n in names:
            for k in NEEDED:
                if n.startswith(k):
                    counts[k] += 1
        for k in NEEDED:
            print("  %-26s %d entries" % (k, counts[k]))

        members = [n for n in names if n.startswith(NEEDED) and not n.endswith("/")]
        print("extracting %d images -> %s" % (len(members), DEST))
        done = 0
        for n in members:
            zf.extract(n, DEST)
            done += 1
            if done % 300 == 0:
                print("  %d/%d" % (done, len(members)), flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
