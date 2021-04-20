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

outdir = './trainedmodels/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('vit_base_patch16_384', pretrained=True)
model.reset_classifier(num_classes=10)
transforms = Compose([Resize(384), ToTensor(), Normalize(mean=(0.5,0.5, 0.5), std=(0.5, 0.5, 0.5))])
cifar10 = CIFAR10(root='./datasets/train/', download=True, train=True, transform=transforms)
cifar10_test = CIFAR10(root='./datasets/test/', download=True, train=False, transform=transforms)
train_dl = DataLoader(cifar10, batch_size=16, drop_last=True)
test_dl = DataLoader(cifar10_test, batch_size=16, drop_last=True)


model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.003)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)

train_iter = iter(train_dl)
for idx in range(10000):
    model.train()
    if idx > 10000:
        break
    try:
        img, label = next(train_iter)
    except:
        train_iter = iter(train_dl)
        img, label = next(train_iter)
    optim.zero_grad()
    img = img.to(device)
    label = label.to(device)
    pred = model(img)
    loss_val = loss_fn(pred, label)
    loss_val.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    torch.save(model.state_dict(), outdir + 'cifar10_vit.pth')
    if idx % 100 == 0:
        model.eval()
        correct = 0.0
        with torch.no_grad():
            for timg, tlabel in test_dl:
                timg = timg.to(device)
                tlabel = tlabel.to(device)
                tout = model(timg)
                loss_t = loss_fn(tout, tlabel)
                correct += (tlabel == torch.argmax(tout, dim=1)).sum().item()
            #print(correct, len(test_dl.dataset))
            acc = correct/len(test_dl.dataset)
            sched.step()
        
        print(f'Idx:{idx}, Train_loss:{loss_val.item()}, Test loss:{loss_t.item()}, test accuracy:{acc:.2f}')
    