"""Inspect ZIP local headers from a partially-downloaded chunked file."""
import os
import struct
import collections

PARTS_DIR = r"d:\Dark\dl_parts"
CHUNK = 16 * 1024 * 1024
TOTAL = 5936812008


class PrefixStream:
    def __init__(self, allowed):
        self.allowed = allowed  # set of chunk indices available
        self.pos = 0

    def avail_end(self):
        n = 0
        for i in range(len(self.allowed)):
            if i not in self.allowed:
                break
            n = min(TOTAL, (i + 1) * CHUNK)
        return n

    def read(self, size):
        end = min(self.pos + size, self.avail_end())
        if end <= self.pos:
            return b""
        out = b""
        p = self.pos
        while p < end:
            ci = p // CHUNK
            off = p % CHUNK
            fp = os.path.join(PARTS_DIR, "part_%05d" % ci)
            with open(fp, "rb") as f:
                f.seek(off)
                take = min(end - p, CHUNK - off)
                out += f.read(take)
            p += take
        self.pos = end
        return out


def main():
    parts = [i for i in range(354)
             if os.path.exists(os.path.join(PARTS_DIR, "part_%05d" % i))
             and os.path.getsize(os.path.join(PARTS_DIR, "part_%05d" % i))
             == min(CHUNK, TOTAL - i * CHUNK)]
    s = PrefixStream(set(parts))
    print("available prefix bytes =", s.avail_end())
    entries = []
    while True:
        hdr = s.read(30)
        if len(hdr) < 30:
            print("EOF of available prefix at", s.pos)
            break
        sig = struct.unpack("<I", hdr[:4])[0]
        if sig != 0x04034B50:
            print("bad signature %08X at" % sig, s.pos - 30)
            break
        (ver, flags, method, mt, md, crc, csize, usize,
         nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr[4:30])
        name = s.read(nlen).decode("utf-8", "replace")
        extra = s.read(elen)
        entries.append((name, flags, method, csize, usize, s.pos))
        if flags & 0x08 and csize == 0:
            print("data descriptor encountered at", name)
            break
        s.pos += csize

    print("entries parsed:", len(entries))
    dirs = collections.Counter(os.path.dirname(n) for n, *_ in entries)
    for d, c in dirs.most_common(20):
        print("  %-40s %d" % (d, c))
    print("--- first 12 ---")
    for e in entries[:12]:
        print("   ", e[0], "method=%d csize=%d usize=%d" % (e[2], e[3], e[4]))
    print("--- last 6 ---")
    for e in entries[-6:]:
        print("   ", e[0], "method=%d csize=%d usize=%d" % (e[2], e[3], e[4]))


if __name__ == "__main__":
    main()
