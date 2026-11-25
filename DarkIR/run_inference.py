"""
DarkIR standalone inference (Windows-compatible, no DDP / NCCL required).

Usage:
    python run_inference.py -p ./options/test/LOLBlur.yml -i ./images/inputs -o ./images/results

Restores every image in the input folder and saves results to the output folder.
"""
import os
import glob
import time
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from options.options import parse
from archs.DarkIR import DarkIR


def build_model(net_opt, device):
    model = DarkIR(
        img_channel=net_opt['img_channels'],
        width=net_opt['width'],
        middle_blk_num_enc=net_opt['middle_blk_num_enc'],
        middle_blk_num_dec=net_opt['middle_blk_num_dec'],
        enc_blk_nums=net_opt['enc_blk_nums'],
        dec_blk_nums=net_opt['dec_blk_nums'],
        dilations=net_opt['dilations'],
        extra_depth_wise=net_opt['extra_depth_wise'],
    )
    return model.to(device)


def load_weights(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # official checkpoints store weights under 'params' (saved without DDP prefix)
    if isinstance(ckpt, dict) and 'params' in ckpt:
        weights = ckpt['params']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        weights = ckpt['model_state_dict']
    else:
        weights = ckpt
    # strip a possible 'module.' prefix (in case the checkpoint was DDP-wrapped)
    cleaned = {}
    for k, v in weights.items():
        nk = k[7:] if k.startswith('module.') else k
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f'[load] missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print('  missing keys (first 5):', missing[:5])
    if unexpected:
        print('  unexpected keys (first 5):', unexpected[:5])
    return model


def main():
    parser = argparse.ArgumentParser(description='DarkIR inference')
    parser.add_argument('-p', '--config', type=str, default='./options/test/LOLBlur.yml')
    parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs')
    parser.add_argument('-o', '--out_path', type=str, default='./images/results')
    args = parser.parse_args()

    opt = parse(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device = {device}')
    if torch.cuda.is_available():
        print(f'[env] GPU = {torch.cuda.get_device_name(0)}')

    model = build_model(opt['network'], device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] params = {n_params/1e6:.3f} M')

    weight_path = opt['save']['path']
    print(f'[load] weights = {weight_path}')
    model = load_weights(model, weight_path)
    model.eval()

    os.makedirs(args.out_path, exist_ok=True)
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    files = sorted(glob.glob(os.path.join(args.inp_path, '*')))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    print(f'[data] found {len(files)} input images in {args.inp_path}')

    padder = 8
    t0 = time.time()
    with torch.no_grad():
        for idx, f in enumerate(files, 1):
            img = Image.open(f).convert('RGB')
            t = to_tensor(img).unsqueeze(0).to(device)
            _, _, H, W = t.shape
            ph = (padder - H % padder) % padder
            pw = (padder - W % padder) % padder
            if ph or pw:
                t = F.pad(t, (0, pw, 0, ph), value=0)
            out = model(t, side_loss=False)
            out = torch.clamp(out, 0., 1.)
            out = out[:, :, :H, :W]
            out_path = os.path.join(args.out_path, os.path.basename(f))
            to_pil(out.squeeze(0).cpu()).save(out_path)
            print(f'  [{idx}/{len(files)}] saved {os.path.basename(f)}  ({H}x{W})')
    print(f'[done] {len(files)} images restored in {time.time()-t0:.2f}s -> {args.out_path}')


if __name__ == '__main__':
    main()
