from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch


def _draw_circle(h: int, w: int, cy: int, cx: int, r: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)


def _draw_box(h: int, w: int, y0: int, x0: int, y1: int, x1: int) -> torch.Tensor:
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def make_sample(channels: int, h: int, w: int, style_shift: bool = False):
    image = torch.zeros((channels, h, w), dtype=torch.float32)
    sem = torch.full((h, w), fill_value=2, dtype=torch.long)

    circle = _draw_circle(h, w, h // 3, w // 3, max(3, min(h, w) // 6))
    box = _draw_box(h, w, h // 2, w // 2, h - 4, w - 4)

    sem[circle] = 0
    sem[box] = 1

    image += 0.05 * torch.randn_like(image)
    image[0][circle] += 1.0
    image[1][box] += 1.0
    image[2] += (sem == 2).float() * 0.25

    if style_shift:
        image = image.roll(shifts=1, dims=0)
        image = torch.sign(image) * torch.abs(image).pow(0.8)
        image = 0.8 * image

    inst = {
        'labels': torch.tensor([0, 1], dtype=torch.long),
        'masks': torch.stack([circle, box]).float(),
    }
    return {'image': image, 'sem_labels': sem, 'inst_targets': inst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--num-samples', type=int, default=16)
    ap.add_argument('--channels', type=int, default=8)
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--target-style-shift', action='store_true')
    args = ap.parse_args()

    src = [make_sample(args.channels, args.height, args.width, style_shift=False) for _ in range(args.num_samples)]
    tgt = [make_sample(args.channels, args.height, args.width, style_shift=args.target_style_shift) for _ in range(args.num_samples)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'src': src, 'tgt': tgt}, args.output)
    print(f'Saved toy panoptic dataset to {args.output}')


if __name__ == '__main__':
    main()
