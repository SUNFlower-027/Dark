"""Generate report assets: param count, GPU latency, and LOLv2-real comparison figures."""
import os, sys, csv, time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r'd:\Dark\DarkIR')
os.chdir(r'd:\Dark\DarkIR')

from options.options import parse
from evaluate_full import build_model, load_weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[env] device =', DEVICE, '|', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu')

opt = parse('./options/test/v2real_lsrw.yml')
WPATH = r'd:\Dark\DarkIR\models\DarkIR_1k_cr_mt.pt'
model = build_model(opt)
n_params = sum(p.numel() for p in model.parameters())
print('[model] params = %.4f M  (paper DarkIR-m = 3.322 M)' % (n_params / 1e6))
model = load_weights(model, WPATH)
model = model.to(DEVICE).eval()

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

DATA = r'd:\Dark\DarkIR\data\datasets\LOL-v2\Real_captured\test'
LOW = os.path.join(DATA, 'Low')
NORMAL = os.path.join(DATA, 'Normal')

# ---- pick best / median / worst PSNR from the real per-image CSV ----
rows = []
with open(r'd:\Dark\DarkIR\eval_results\lolv2_real_per_image.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows.append((r['image'], float(r['psnr']), float(r['ssim']), float(r['lpips'])))
rows.sort(key=lambda x: x[1])  # ascending PSNR
picks = [('worst', rows[0]), ('median', rows[len(rows) // 2]), ('best', rows[-1])]
print('[picks]')
for tag, (name, p, s, l) in picks:
    print('  %-7s %-12s PSNR=%.3f SSIM=%.4f LPIPS=%.4f' % (tag, name, p, s, l))

# ---- latency benchmark on a real LOLv2-real image (representative resolution) ----
low0 = to_tensor(Image.open(os.path.join(LOW, picks[0][1][0])).convert('RGB')).unsqueeze(0).to(DEVICE)
H, W = low0.shape[2], low0.shape[3]
print('[speed] benchmark resolution = %dx%d' % (W, H))
with torch.no_grad():
    for _ in range(20):
        _ = model(low0)
    torch.cuda.synchronize()
    N = 100
    t0 = time.time()
    for _ in range(N):
        _ = model(low0)
    torch.cuda.synchronize()
    dt = time.time() - t0
mean_ms = dt / N * 1000
fps = N / dt
print('[speed] %d iters: total %.2fs -> mean %.2f ms/img (%.1f FPS) @ %dx%d' % (N, dt, mean_ms, fps, W, H))

# ---- generate 3 comparison figures: Input | DarkIR Output | Reference ----
OUT = r'd:\Dark\report2_work\figs'
os.makedirs(OUT, exist_ok=True)

def try_font(size):
    for p in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\arial.ttf']:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

font = try_font(22)
font_sm = try_font(18)

def infer(low_path):
    low = to_tensor(Image.open(low_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(low)
        if isinstance(out, (tuple, list)):
            out = out[0]
    out = out.clamp(0, 1).squeeze(0).cpu()
    return to_pil(out)

FIGH = 360
def resize_h(img, h):
    w = int(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)

manifest = []
for tag, (name, p, s, l) in picks:
    inp = Image.open(os.path.join(LOW, name)).convert('RGB')
    ref = Image.open(os.path.join(NORMAL, name)).convert('RGB')
    out = infer(os.path.join(LOW, name))
    a, b, c = resize_h(inp, FIGH), resize_h(out, FIGH), resize_h(ref, FIGH)
    gap = 16
    label_h = 44
    Wtot = a.width + b.width + c.width + gap * 2
    canvas = Image.new('RGB', (Wtot, FIGH + label_h), 'white')
    x = 0
    for im, lab in [(a, 'Input (Low-light)'), (b, 'DarkIR Output'), (c, 'Reference (Normal)')]:
        canvas.paste(im, (x, label_h))
        draw = ImageDraw.Draw(canvas)
        tw = draw.textlength(lab, font=font)
        draw.text((x + im.width // 2 - tw // 2, 10), lab, fill='black', font=font)
        x += im.width + gap
    # per-image metric caption bottom-right of output panel
    cap = 'PSNR %.2f / SSIM %.3f' % (p, s)
    draw = ImageDraw.Draw(canvas)
    draw.text((a.width + gap + 6, FIGH + label_h - 26), cap, fill=(40, 40, 40), font=font_sm)
    fn = 'cmp_%s_%s' % (tag, name)
    canvas.save(os.path.join(OUT, fn))
    print('[fig] saved', fn, canvas.size)
    manifest.append((fn, name, tag, p, s, l))

print('\n=== ASSET SUMMARY ===')
print('params_M=%.4f' % (n_params / 1e6))
print('latency_ms=%.2f fps=%.1f res=%dx%d' % (mean_ms, fps, W, H))
for m in manifest:
    print('figure:', m)
