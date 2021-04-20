#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.datasets import CIFAR10
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from torch.utils.data import DataLoader
import foolbox as fb
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

outdir = 'transformer_linf_imgs'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = create_model('vit_base_patch16_384', pretrained=True)
model.reset_classifier(num_classes=10)
transforms = Compose([Resize(384), ToTensor(), Normalize(mean=(0.5,0.5, 0.5), std=(0.5, 0.5, 0.5))])
cifar10_test = CIFAR10(root='./datasets/test/', download=True, train=False, transform=transforms)

test_dl = DataLoader(cifar10_test, batch_size=8)

model = model.to(device)
sd = torch.load('cifar10_vit.pth', map_location='cuda:0')

model.load_state_dict(sd)


model = model.to(device)



def prod(x):
    pr = 1.0
    for p in x.shape:
        pr*=p
    return pr

numparams = 0
for params in model.parameters():
    numparams += prod(params)
    #print(params.shape)

def get_img(x):
    tmp=  x[:,...].detach().cpu().numpy().transpose(1,2,0)
    return tmp

model.eval();

att_model = fb.PyTorchModel(model, bounds=(-1, 1), device=device)
epsilons_max = [
        # 0.0005,
        # 0.001,
        # 0.0015,
        # 0.002,
        # 0.003,
        # 0.005,
        0.01,
        0.02,
        0.03,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
#attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=5000, abort_early=True)
#epsilons = []
f = open('accuracies.csv', 'w')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for eps_max in epsilons_max:
    step_size = eps_max/2.0
    attack = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.5)
    successes = []
    epsilons = []
    print(f'{eps_max}')
    for idx, (img, label) in enumerate(test_dl):
        # if idx <= 21:
        #     continue
        img = img.to(device)
        bs, ch, sx, sy = img.shape
        label = label.to(device)
        #print(idx)
        #print(list(model.parameters())[0].device)
        raw, clipped, succ = attack(att_model, img, criterion=fb.criteria.Misclassification(label), epsilons=eps_max)
        eps = np.linalg.norm((img - clipped).view(bs, -1).cpu().detach().numpy(), axis=1, ord=np.inf)
        epsilons.extend(eps)
        #epsilons.append(img - clipped[])
        successes.extend(succ.cpu().detach().numpy())
        # if idx % 100 == 0:
        #     for i in range(img.shape[0]):
        #         fig = plt.figure(figsize=(20,10))
        #         ax1 = fig.add_subplot(131)
        #         img_np = get_img(img[i,...])
        #         clipped_np = get_img(clipped[i])
        #         ax1.imshow(( img_np + 1)/2.0)
        #         ax2 = fig.add_subplot(132)
        #         ax2.imshow((clipped_np+1)/2.0)
        #         ax3 = fig.add_subplot(133)
        #         ax3.imshow(np.abs(img_np - clipped_np)*100)
        #         ax1.set_title(succ.cpu().detach().numpy()[i])
        #         ax3.set_title(eps[i])
        #         plt.savefig(f'{outdir}/{eps_max}_{idx}_{i}.png')
        #         plt.close()
    #sys.exit()
    np.save(f'{outdir}/epsilons_linf_{eps_max}', epsilons)
    np.save(f'{outdir}/successes_linf_{eps_max}', successes)
    robust_accuracy = np.sum(successes)/10000.0
    f.write(f'{eps_max}, {robust_accuracy}\n')
