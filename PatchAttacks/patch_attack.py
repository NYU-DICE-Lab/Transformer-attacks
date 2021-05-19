#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.datasets import CIFAR10, ImageNet
from timm.models import create_model# ,apply_test_time_pool, load_checkpoint, is_model, list_models
#from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
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
import logging
from pathlib import Path
from torch import autograd
import copy


def get_img(x):
    tmp=  x[:,...].detach().cpu().numpy().transpose(1,2,0)
    return tmp

def prod(x):
    pr = 1.0
    for p in x.shape:
        pr*=p
    return pr

class Patcher(nn.Module):
    """
    Differentiable module that exposes the various patches as different parameters
    Note: Currently built for a single image
    ## TODO: Figure out how to select patches for a batch
    """
    def __init__(self, img, patch_size):
        super().__init__()
        bs, ch, sx, sy = img.shape
        self.patches = nn.ParameterList()
        self.patch_size = patch_size
        self.sx = sx
        self.sy = sy
        for i in range(0, sx, patch_size):
            for j in range(0, sy, patch_size):
                self.patches.append(nn.Parameter(img[:, :, i:i+patch_size, j:j+patch_size]))
        self.img = img
    
    def get_patches(self):
        return self.patches

    def forward(self, img):
        for i in range(0, self.sx, self.patch_size):
            for j in range(0, self.sy, self.patch_size):
                self.img[:, :, i:i+patch_size, j:j+patch_size]
        return self.img

mtype_dict = {'vit384': 'vit_base_patch16_384', 'vit224':'vit_base_patch16_224', 'wide-resnet':'wide_resnet101_2', 'deit':None, 'bit_152_4':'resnetv2_152x4_bitm'}
att_type_dict = {'pgdlinf': fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.033, steps=40, random_start=True), 
                'pgdl2' : fb.attacks.L2ProjectedGradientDescentAttack(steps=40, random_start=True)
                }

def get_patches(img, patch_size=16):
    bs, ch, sx, sy = img.size()
    patches = []
    for i in range(0, sx, patch_size):
        for j in range(0, sy, patch_size):
            patches.append(img[:,:,i:i+patch_size, j:j+patch_size])
    return patches

def reconstruct_img(patches, img):
    bs, ch, sx ,sy = img.shape
    _,_,patch_size, _ = patches[0].shape
    recon = torch.zeros((bs, ch, sx, sy), device=device)
    k = 0
    for i in range(0, sx, patch_size):
        for j in range(0, sy, patch_size):
            recon[:,:,i:i+patch_size, j:j+patch_size] = patches[k]
            k+=1
    return recon

# def MultiPatchGDAttack(model, img, label, loss=nn.CrossEntropyLoss(), iterations=40, device=torch.device('cuda:0'), *args, **kwargs):
#     #img = nn.Parameter(img).to(device)
#     base_img = copy.deepcopy(img)
#     img = img.to(device)
#     img.requires_grad_(True)
#     bs, ch, sx, sy = img.size()
#     label = label.to(device)
#     patch_size=16
#    # patcher = Patcher(img, patch_size=16)
#     #patches = get_patches(img, patch_size=16)  
#     #max_patch_idx = torch.argmax(torch.tensor([torch.norm(i, p='fro') for i in patches]))
#     l2_norms = {}
#     max_val = np.inf
#     max_i, max_j = 0,0 
#     for i in range(0, sx, patch_size):
#         for j in range(0, sy, patch_size):
#             l2_norms[(i,j)] = torch.norm(img[:,:,i:i+patch_size, j:j+patch_size])
#             if l2_norms[(i,j)] < max_val:
#                 max_i = i
#                 max_j = j
#                 max_val = l2_norms[(i,j)]
#     lr = 0.033
#     succ = 0.0
#     #opt = torch.optim.SGD(params=patches.parameters(), lr=0.033)
#     for i in range(iterations):
#         #img_recon = reconstruct_img(patches, img)
#         pred = model(img)
#         loss_val = loss(pred, label)
#         #print(loss_val)
#         #grad_val = autograd.grad(loss_val, img[:,:, max_i:max_i+patch_size, max_j:max_j+patch_size])
#         grad_val = autograd.grad(loss_val, img)
#         #print(grad_val[0].max())
#         logging.debug(grad_val[0].shape)
#         img[:,:,max_i:max_i+patch_size, max_j:max_j+patch_size] += grad_val[0][:,:,max_i:max_i+patch_size, max_j:max_j+patch_size]
#         #old_patch = copy.deepcopy(patches[max_patch_idx])
#         #patches[max_patch_idx] += grad_val[0]
#         #print((old_patch - patches[max_patch_idx]).max())
        
#         with torch.no_grad():
#             #recon_img = reconstruct_img(patches, img)
#             #print(torch.abs(recon_img  - img).max())
#             pred2 = model(img)
#             #print(pred2, pred, label)
#             if torch.argmax(pred2, dim=1) != torch.argmax(pred, dim=1):
#                     logging.info('Image broken at iteration {i}')
#                     succ = 1.0
#                     break
#     return succ, base_img, img, max_i*sx+max_j

def SinglePatchGDAttack(model, img, label, loss=nn.CrossEntropyLoss(), iterations=40, device=torch.device('cuda:0'), *args, **kwargs):
    #img = nn.Parameter(img).to(device)
    base_img = copy.deepcopy(img)
    img = img.to(device)
    img.requires_grad_(True)
    bs, ch, sx, sy = img.size()
    label = label.to(device)
    patch_size=16
   # patcher = Patcher(img, patch_size=16)
    #patches = get_patches(img, patch_size=16)  
    #max_patch_idx = torch.argmax(torch.tensor([torch.norm(i, p='fro') for i in patches]))
    l2_norms = {}
    max_val = 0.0
    max_i, max_j = 0,0 
    for i in range(0, sx, patch_size):
        for j in range(0, sy, patch_size):
            l2_norms[(i,j)] = torch.norm(img[:,:,i:i+patch_size, j:j+patch_size])
            if l2_norms[(i,j)] > max_val:
                max_i = i
                max_j = j
                max_val = l2_norms[(i,j)]
    lr = 0.033
    succ = 0.0
    #opt = torch.optim.SGD(params=patches.parameters(), lr=0.033)
    for i in range(iterations):
        #img_recon = reconstruct_img(patches, img)
        pred = model(img)
        loss_val = loss(pred, label)
        #print(loss_val)
        #grad_val = autograd.grad(loss_val, img[:,:, max_i:max_i+patch_size, max_j:max_j+patch_size])
        grad_val = autograd.grad(loss_val, img)
        #print(grad_val[0].max())
        logging.debug(grad_val[0].shape)
        img[:,:,max_i:max_i+patch_size, max_j:max_j+patch_size] += lr * grad_val[0][:,:,max_i:max_i+patch_size, max_j:max_j+patch_size]
        #old_patch = copy.deepcopy(patches[max_patch_idx])
        #patches[max_patch_idx] += grad_val[0]
        #print((old_patch - patches[max_patch_idx]).max())
        
        with torch.no_grad():
            #recon_img = reconstruct_img(patches, img)
            #print(torch.abs(recon_img  - img).max())
            pred2 = model(img)
            #print(pred2, pred, label)
            if torch.argmax(pred2, dim=1) != torch.argmax(pred, dim=1):
                    logging.info('Image broken at iteration {i}')
                    succ = 1.0
                    break
    return succ, base_img, img, max_i*sx+max_j

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', help='Output directory',default= 'results/')
    #parser.add_argument('-m', '--model', help='Model path')
    parser.add_argument('-mt', '--mtype', help='Model type', choices=['vit384', 'vit224', 'wide-resnet', 'deit', 'bit_152_4'], default='vit224')
    parser.add_argument('-dpath', help='Data path', default = '/data/datasets/Imagenet/val')
    parser.add_argument('--gpu', help='gpu to use', default=0, type=int)
    parser.add_argument('-at', '--att_type', help='Attack type', choices=['pgdl2', 'pgdlinf', 'gd'], default='pgdlinf')
    parser.add_argument('-it', '--iter', help='No. of iterations', type=int, default=40)
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mtype = args.mtype
    att_type = args.att_type
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, filename=outdir / 'run.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    model_name = mtype_dict[mtype]
    logging.info(f'Running {att_type} for {model_name} for imagenet')
    if model_name is None:
        raise Exception(f'{mtype}: No such model type found')

    model = create_model(model_name, pretrained=True)
    #model.reset_classifier(num_classes=10)
    config = resolve_data_config({}, model=model)
    transforms = create_transform(**config)
    #print(config)
    #print(model)
    #cifar10_test = CIFAR10(root='./datasets/test/', download=True, train=False, transform=transforms)
    indices = np.load('imagenet_indices.npy')
    imagenet_val = Subset(ImageNet(root=args.dpath, split='val', transform=transforms), indices)
    test_dl = DataLoader(imagenet_val, batch_size=1)

    model = model.to(device)
    #sd = torch.load('cifar10_vit.pth', map_location='cuda:0')
    #model.load_state_dict(sd)
    numparams = 0
    for params in model.parameters():
        numparams += prod(params)

    model.eval();

    # att_model = fb.PyTorchModel(model, bounds=(-2.1179, 2.6400), device=device)
    # eps_vals = [1/255.0, 2/255.0, 4/255.0, 8/255.0, 16/255.0, 32/255.0, 64/255.0, 128/255.0]
    # #attack = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.033, steps=40, random_start=True)
    # #attack = fb.attacks.L2ProjectedGradientDescentAttack(steps=40, random_start=True))
    # attack = att_type_dict[att_type]
    # if 'l2' in att_type:
    #     size = 224 # TODO make it variable later
    #     eps_vals = [i * 224 * np.sqrt(3) for i in eps_vals]
    # eps_vals = np.array(eps_vals)
    # #epsilons = []
    # f = open('accuracies.csv', 'w')
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    clean_acc = 0.0
    for idx, (img, label) in enumerate(test_dl):
        img = img.to(device)
        #print(img.min(), img.max())
        label = label.to(device)
        pred = torch.argmax(model(img), dim=1)
        clean_acc += torch.eq(pred, label).sum()
    logging.info(f'Clean accuracy for imagenet subset:{clean_acc/1000.0}')

    #corrects = np.zeros(8)
    #epsilons = []
    attack_succ = 0.0
    for idx, (img, label) in enumerate(test_dl):
        # if idx > 1:
        #     break
        img = img.to(device)
        bs, ch, sx, sy = img.shape
        label = label.to(device)
        succ, img, attack_img, patch_idx = SinglePatchGDAttack(model, img, label, loss=nn.CrossEntropyLoss(), iterations=args.iter, device=device)
        attack_succ += succ
        if True:
            eps = torch.norm(img - attack_img, p="fro")
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(131)
            img_np = get_img(img[0,...])
            #print(clipped[i].shape)
            clipped_np = get_img(attack_img[0,...])
            ax1.imshow((img_np + 1)/2.0)
            ax2 = fig.add_subplot(132)
            ax2.imshow((clipped_np+1)/2.0)
            ax3 = fig.add_subplot(133)
            ax3.imshow(np.abs(img_np - clipped_np)*100)
            ax1.set_title(succ)
            ax3.set_title(f'{idx}, {succ}, {patch_idx}, {eps:.04f}')
            plt.savefig(f'{outdir}/{idx}_{succ}.png')
            plt.close()
        #epsilons.append(img - clipped[])
    #sys.exit()
    print(attack_succ)
    rob_acc = 1 - attack_succ/idx
    logging.info(f'Robust accuracy:{rob_acc}')