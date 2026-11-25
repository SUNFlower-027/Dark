"""Full LOLBlur test-set evaluation for DarkIR (Windows-friendly, no DDP).

Metrics match the official testing.py / utils/test_utils.py exactly:
  - PSNR  = 20 * log10(1 / sqrt(MSE)), averaged over images
  - SSIM  = pytorch_msssim.ssim(enhanced, high, data_range=1), averaged
  - LPIPS = lpips.LPIPS(net='vgg'), averaged
No clamping, identical to the official evaluation loop.

Usage:
  python evaluate_full.py -p ./options/test/LOLBlur.yml [-n LIMIT]
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_msssim
from lpips import LPIPS
from torch.utils.data import DataLoader

from options.options import parse
from archs.DarkIR import DarkIR
from data.dataset_reader.dataset_LOLBlur import main_dataset_lolblur


def build_model(opt):
    net = opt['network']
    model = DarkIR(
        img_channel=net['img_channels'],
        width=net['width'],
        middle_blk_num_enc=net['middle_blk_num_enc'],
        middle_blk_num_dec=net['middle_blk_num_dec'],
        enc_blk_nums=net['enc_blk_nums'],
        dec_blk_nums=net['dec_blk_nums'],
        dilations=net['dilations'],
        extra_depth_wise=net['extra_depth_wise'],
    )
    return model


def load_weights(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    weights = ckpt['params'] if isinstance(ckpt, dict) and 'params' in ckpt else ckpt
    cleaned = {k[7:] if k.startswith('module.') else k: v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print('[load] missing=%d unexpected=%d' % (len(missing), len(unexpected)))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--config', default='./options/test/LOLBlur.yml')
    ap.add_argument('-n', '--limit', type=int, default=0, help='evaluate first N images only (0 = all)')
    ap.add_argument('--save_csv', default=r'./eval_results/per_image_metrics.csv')
    args = ap.parse_args()

    opt = parse(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[env] device =', device, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')

    model = build_model(opt)
    n_params = sum(p.numel() for p in model.parameters())
    print('[model] params = %.3f M' % (n_params / 1e6))
    model = load_weights(model, opt['save']['path'])
    model = model.to(device).eval()

    # Reuse the official dataset class -> identical pairing/order logic
    loader, _ = main_dataset_lolblur(
        rank=1,
        test_path=opt['datasets']['val']['test_path'],
        batch_size_test=1,
        verbose=True,
        num_workers=0,
        world_size=1,
    )
    ds = loader.dataset
    print('[data] test images =', len(ds))
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print('[data] evaluating', n, 'images')

    calc_ssim = lambda a, b: pytorch_msssim.ssim(a, b, data_range=1.)
    lpips_model = LPIPS(net='vgg', verbose=False).to(device)

    psnrs, ssims, lpipses = [], [], []
    rows = []
    t0 = time.time()
    os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)

    with torch.no_grad():
        for idx in range(n):
            high, low = ds[idx]          # dataset returns (high, low)
            high = high.unsqueeze(0).to(device)
            low = low.unsqueeze(0).to(device)

            out = model(low)             # no clamp, same as official
            if isinstance(out, (tuple, list)):
                out = out[0]

            mse = torch.mean((high - out) ** 2)
            psnr = (20 * torch.log10(1. / torch.sqrt(mse))).item()
            ssim = calc_ssim(out, high).item()
            lp = torch.mean(lpips_model(out, high)).item()

            psnrs.append(psnr); ssims.append(ssim); lpipses.append(lp)
            rows.append((idx, os.path.basename(ds.imgs_low[idx]), psnr, ssim, lp))

            if (idx + 1) % 20 == 0 or idx + 1 == n:
                el = time.time() - t0
                eta = el / (idx + 1) * (n - idx - 1)
                print('  [%4d/%4d] PSNR=%.4f SSIM=%.4f LPIPS=%.4f | avg PSNR=%.4f SSIM=%.4f LPIPS=%.4f | %.1fs eta %.0fs' % (
                    idx + 1, n, psnr, ssim, lp,
                    np.mean(psnrs), np.mean(ssims), np.mean(lpipses), el, eta), flush=True)

    with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index', 'image', 'psnr', 'ssim', 'lpips'])
        w.writerows(rows)

    print('\n================ FINAL ================')
    print('  images : %d' % n)
    print('  PSNR   : %.4f' % np.mean(psnrs))
    print('  SSIM   : %.4f' % np.mean(ssims))
    print('  LPIPS  : %.4f' % np.mean(lpipses))
    print('  total  : %.1f s' % (time.time() - t0))
    print('  csv    :', args.save_csv)
    print('=======================================')


if __name__ == '__main__':
    main()
