import torch
import clip
import os
import numpy as np
import lpips
from PIL import Image, ImageFilter

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
loss_fn = lpips.LPIPS(net='vgg').cuda()

dir_list = os.listdir()
vecs = {}
vals = {}
for exp in ['baseline', 'clip', 'clipmut']:
    print(f'calculating {exp}')
    vecs[exp] = {}
    vals[exp] = []
    for dir in dir_list:
        # if dir != "a smiling _cat":
        #     continue
        vecs[exp][dir] = []
        if not os.path.isdir(dir):
            continue
        pngs = sorted(os.listdir(os.path.join(dir, exp)))
        max_val = 0
        for i, im in enumerate(pngs):
            im_path = os.path.join(dir, exp, im)
            image = preprocess(Image.open(im_path)).unsqueeze(0).to(device)
            vecs[exp][dir].append(image)
            if len(vecs[exp][dir]) == 2:
                with torch.no_grad():
                    var = loss_fn(vecs[exp][dir][0], vecs[exp][dir][1])
                vecs[exp][dir].pop(0)
                max_val = 0
                vals[exp].append(var.mean())

l = len(vals['baseline'])
sum(vals['baseline']) / l, sum(vals['clip']) / l, sum(vals['clipmut']) / l, l/99