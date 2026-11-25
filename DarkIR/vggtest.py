import requests, time

cands = {
    'pytorch_official': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'nju_mirror':       'https://mirror.nju.edu.cn/pytorch/models/vgg16-397923af.pth',
    'aliyun_models':    'https://mirrors.aliyun.com/pytorch-models/vgg16-397923af.pth',
    'sjtu_mirror':      'https://mirror.sjtu.edu.cn/pytorch-models/vgg16-397923af.pth',
}
for name, u in cands.items():
    try:
        s = requests.Session()
        s.headers.update({'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-2097151'})
        t = time.time()
        r = s.get(u, stream=True, timeout=30, allow_redirects=True)
        n = 0
        for c in r.iter_content(262144):
            n += len(c)
            if n >= 2 * 1024 * 1024:
                break
        dt = time.time() - t
        print('%-16s status=%s got=%.2fMB %.2fs -> %.2f MB/s' % (
            name, r.status_code, n / 1024 / 1024, dt, n / 1024 / 1024 / max(dt, 1e-6)))
    except Exception as e:
        print('%-16s ERR %s' % (name, str(e)[:80]))
