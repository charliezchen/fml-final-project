from torch import nn
import torch
import numpy as np
from torchvision import datasets, transforms
import torchvision
from IPython import embed as e

class Ensemble(nn.Module):
    def __init__(self, m1, m2, m3):
        super(Ensemble, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
    
    def forward(self, X):
        if self.m2 is None:
            return self.m1(X)
        else:
            return 1/ 3 * (self.m1(X) + self.m2(X))
        logits = 0
        for model in self.models:
            logits += model(X)
        
        return logits / len(self.models)

if __name__=='__main__':
    from models.wideresnet import *

    device = 'cuda'
    m1 = WideResNet()
    m1.load_state_dict(torch.load("model-cifar-wideResNet/model-wideres-epoch7.pt"))

    m2 = WideResNet()
    m2.load_state_dict(torch.load("model-cifar-alp-separation-beta0-lam1-batch320/model-wideres-epoch6.pt"))

    m3 = WideResNet()
    

    ensemble = Ensemble(m1, m2)
    # ensemble.to(device)

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, **kwargs)

    for i in train_loader:
        s=i[0]
        break
    ensemble(s)

    e() or b
