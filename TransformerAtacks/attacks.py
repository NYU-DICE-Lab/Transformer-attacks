#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.datasets import CIFAR10, ImageNet
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from torch.utils.data import DataLoader, Subset
import os
import sys
import argparse
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import foolbox as fb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


mtype_dict = {'vit384': 'vit_base_patch16_384', 'vit224':'vit_base_patch16_224', 'wide-resnet':'wide_resnet101_2', 'deit':None}
att_type_dict = {'pgdlinf': fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.033, steps=40, random_start=True), 
                'pgdl2' : fb.attacks.L2ProjectedGradientDescentAttack(steps=40, random_start=True)
                }


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outdir', help='Output directory',default= 'results/')
#parser.add_argument('-m', '--model', help='Model path')
parser.add_argument('-mt', '--mtype', help='Model type', choices=['vit384', 'vit224', 'wide-resnet', 'deit'], default='vit224')
parser.add_argument('-dpath', help='Data path', default = '/data/datasets/Imagenet/val')
parser.add_argument('--gpu', help='gpu to use', default=0, type=int)
parser.add_argument('-at', '--att_type', help='Attack type', choices=['pgdl2', 'pgdlinf'], default='pgdlinf')

args = parser.parse_args()

outdir = args.outdir
mtype = args.mtype
att_type = args.att_type

model_name = mtype_dict[mtype]
if model_name is None:
    raise Exception(f'{mtype}: No such model type found')


device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
model = create_model(model_name, pretrained=True)
#model.reset_classifier(num_classes=10)
config = resolve_data_config({}, model=model)
transforms = create_transform(**config)
print(config)

#transforms = Compose([Resize(384), ToTensor(), Normalize(mean=(0.5,0.5, 0.5), std=(0.5, 0.5, 0.5))])
#cifar10_test = CIFAR10(root='./datasets/test/', download=True, train=False, transform=transforms)
indices = np.load('imagenet_indices.npy')
imagenet_val = Subset(ImageNet(root=args.dpath, split='val', transform=transforms), indices)
test_dl = DataLoader(imagenet_val, batch_size=8)

model = model.to(device)
#sd = torch.load('cifar10_vit.pth', map_location='cuda:0')
#model.load_state_dict(sd)


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

att_model = fb.PyTorchModel(model, bounds=(-2.1179, 2.6400), device=device)
eps_vals = [1/255.0, 2/255.0, 4/255.0, 8/255.0, 16/255.0, 32/255.0, 64/255.0, 128/255.0]
#attack = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.033, steps=40, random_start=True)
#attack = fb.attacks.L2ProjectedGradientDescentAttack(steps=40, random_start=True))
attack = att_type_dict[att_type]
if 'l2' in att_type:
    size = 224 # TODO make it variable later
    eps_vals = [i * 224 * np.sqrt(3) for i in eps_vals]
#epsilons = []
f = open('accuracies.csv', 'w')
if not os.path.exists(outdir):
    os.makedirs(outdir)
clean_acc = 0.0
for idx, (img, label) in enumerate(test_dl):
    img = img.to(device)
    #print(img.min(), img.max())
    label = label.to(device)
    pred = torch.argmax(model(img), dim=1)
    clean_acc += torch.eq(pred, label).sum()
print(clean_acc/1000.0)

corrects = np.zeros(8)
epsilons = []
for idx, (img, label) in enumerate(test_dl):
#     if idx < 14:
#         continue
    img = img.to(device)
    bs, ch, sx, sy = img.shape
    label = label.to(device)
    #print(list(model.parameters())[0].device)
    raw, clipped, succ = attack(att_model, img, criterion=fb.criteria.Misclassification(label), epsilons=eps_vals,)
    #print(succ)
    corrects += succ.cpu().numpy().sum(axis=1)
    print(corrects)
    #print(len(clipped))
    if idx%20 == 0:
        for i in range(len(clipped)):
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(131)
            img_np = get_img(img[0,...])
            #print(clipped[i].shape)
            clipped_np = get_img(clipped[i][0,...])
            ax1.imshow(( img_np + 1)/2.0)
            ax2 = fig.add_subplot(132)
            ax2.imshow((clipped_np+1)/2.0)
            ax3 = fig.add_subplot(133)
            ax3.imshow(np.abs(img_np - clipped_np)*100)
            ax1.set_title(succ.cpu().detach().numpy()[i])
            ax3.set_title(eps_vals[i])
            plt.savefig(f'{outdir}/{idx}_{i}.png'.format(idx, i))
            plt.close()
        #epsilons.append(img - clipped[])
    #sys.exit()
print(corrects)
rob_acc = 1 - corrects/1000.0#1000.0
print(f'robust accuracies: {rob_acc}')
