"""Full LOLv2 (real + synthetic) test-set evaluation for DarkIR-m.

Metrics match the official testing.py / utils/test_utils.py exactly:
  - PSNR  = 20 * log10(1 / sqrt(MSE)), averaged over images
  - SSIM  = pytorch_msssim.ssim(enhanced, high, data_range=1), averaged
  - LPIPS = lpips.LPIPS(net='vgg'), averaged
No clamping, identical to the official evaluation loop.

Expected layout (downloaded by download_lolv2.py):
    data/datasets/LOL-v2/Real_captured/test/{Low,Normal}/*.png
    data/datasets/LOL-v2/Synthetic/test/{Low,Normal}/*.png

Paper reference (DarkIR-m, multitask checkpoint DarkIR_1k_cr_mt.pt):
    LOLv2-real : PSNR 23.87 | SSIM 0.880 | LPIPS 0.186
    LOLv2-synth: PSNR 25.54 | SSIM 0.934 | LPIPS 0.058
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pytorch_msssim
from lpips import LPIPS
from torch.utils.data import Dataset, DataLoader

from evaluate_full import build_model, load_weights

PAPER = {
    'real':  (23.87, 0.880, 0.186),
    'synth': (25.54, 0.934, 0.058),
}

DATA_ROOT = r'd:\Dark\DarkIR\data\datasets\LOL-v2'


class PairDataset(Dataset):
    def __init__(self, low_dir, normal_dir):
        # Pair by sorted index, identical to the official MyDataset_Crop loader
        # (sorted(images_low), sorted(images_high), then index pairing).
        self.low = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)
                           if f.lower().endswith('.png')])
        self.normal = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                              if f.lower().endswith('.png')])
        n = min(len(self.low), len(self.normal))
        self.pairs = list(zip(self.low[:n], self.normal[:n]))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_p, high_p = self.pairs[idx]
        low = self.to_tensor(Image.open(low_p).convert('RGB'))
        high = self.to_tensor(Image.open(high_p).convert('RGB'))
        return high, low, os.path.basename(low_p)


def evaluate_split(model, lpips_model, calc_ssim, split, device, save_csv):
    sub = 'Real_captured' if split == 'real' else 'Synthetic'
    base = os.path.join(DATA_ROOT, sub, 'test')
    ds = PairDataset(os.path.join(base, 'Low'), os.path.join(base, 'Normal'))
    print('\n[%s] paired images = %d (low=%d normal=%d)' % (
        split, len(ds), len(ds.low), len(ds.normal)))
    psnrs, ssims, lpipses = [], [], []
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for idx in range(len(ds)):
            high, low, name = ds[idx]
            high = high.unsqueeze(0).to(device)
            low = low.unsqueeze(0).to(device)
            out = model(low)
            if isinstance(out, (tuple, list)):
                out = out[0]
            mse = torch.mean((high - out) ** 2)
            psnr = (20 * torch.log10(1. / torch.sqrt(mse))).item()
            ssim = calc_ssim(out, high).item()
            lp = torch.mean(lpips_model(out, high)).item()
            psnrs.append(psnr); ssims.append(ssim); lpipses.append(lp)
            rows.append((idx, name, psnr, ssim, lp))
            if (idx + 1) % 20 == 0 or idx + 1 == len(ds):
                el = time.time() - t0
                eta = el / (idx + 1) * (len(ds) - idx - 1)
                print('  [%4d/%4d] PSNR=%.4f SSIM=%.4f LPIPS=%.4f | avg %.4f/%.4f/%.4f | %.1fs eta %.0fs' % (
                    idx + 1, len(ds), psnr, ssim, lp,
                    np.mean(psnrs), np.mean(ssims), np.mean(lpipses), el, eta), flush=True)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or '.', exist_ok=True)
        with open(save_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['index', 'image', 'psnr', 'ssim', 'lpips'])
            w.writerows(rows)
    return np.mean(psnrs), np.mean(ssims), np.mean(lpipses), len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--config', default='./options/test/v2real_lsrw.yml')
    ap.add_argument('--split', choices=['real', 'synth', 'both'], default='both')
    ap.add_argument('--weights', default=None)
    args = ap.parse_args()

    from options.options import parse
    opt = parse(args.config)
    wpath = args.weights or opt['save']['path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[env] device =', device, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    print('[weights]', wpath)

    model = build_model(opt)
    n_params = sum(p.numel() for p in model.parameters())
    print('[model] params = %.3f M' % (n_params / 1e6))
    model = load_weights(model, wpath)
    model = model.to(device).eval()

    calc_ssim = lambda a, b: pytorch_msssim.ssim(a, b, data_range=1.)
    lpips_model = LPIPS(net='vgg', verbose=False).to(device)

    splits = ['real', 'synth'] if args.split == 'both' else [args.split]
    results = {}
    for s in splits:
        csv_path = './eval_results/lolv2_%s_per_image.csv' % s
        psnr, ssim, lp, n = evaluate_split(model, lpips_model, calc_ssim, s, device, csv_path)
        results[s] = (psnr, ssim, lp, n)

    print('\n================ LOLv2 FINAL (DarkIR-m, %s) ================' % os.path.basename(wpath))
    print('%-10s %8s %8s %8s %8s  | paper  %8s %8s %8s' % ('split', 'PSNR', 'SSIM', 'LPIPS', 'n', 'PSNR', 'SSIM', 'LPIPS'))
    for s in splits:
        psnr, ssim, lp, n = results[s]
        ppsnr, pssim, plp = PAPER[s]
        print('%-10s %8.4f %8.4f %8.4f %8d  |        %8.2f %8.3f %8.3f   (dPSNR %+.3f)' % (
            s, psnr, ssim, lp, n, ppsnr, pssim, plp, psnr - ppsnr))
    print('====================================================')


if __name__ == '__main__':
    main()
