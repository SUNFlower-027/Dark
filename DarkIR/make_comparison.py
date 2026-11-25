"""Generate side-by-side comparison figures: input (low-light) vs DarkIR output."""
from PIL import Image, ImageDraw
import os

base = r'd:\Dark\DarkIR'
inputs = os.path.join(base, 'images', 'inputs')
results = os.path.join(base, 'images', 'results')
out_dir = os.path.join(base, 'comparison')
os.makedirs(out_dir, exist_ok=True)

names = [f for f in os.listdir(inputs) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for name in names:
    inp = Image.open(os.path.join(inputs, name)).convert('RGB')
    out = Image.open(os.path.join(results, name)).convert('RGB')
    # normalize to same height
    H = 420
    def resize_to_h(img, h):
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)
    a = resize_to_h(inp, H)
    b = resize_to_h(out, H)
    gap = 20
    label_h = 40
    W = a.width + b.width + gap
    canvas = Image.new('RGB', (W, H + label_h), 'white')
    canvas.paste(a, (0, label_h))
    canvas.paste(b, (a.width + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((a.width // 2 - 60, 12), 'Input (Low-light)', fill='black')
    draw.text((a.width + gap + b.width // 2 - 70, 12), 'DarkIR Output', fill='black')
    save_path = os.path.join(out_dir, 'cmp_' + name)
    canvas.save(save_path)
    print('saved', save_path, canvas.size)
