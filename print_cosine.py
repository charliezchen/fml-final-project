# Adapted from eval.py and trades.py & train_trades_cifar10.py
import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from IPython import embed as e
from tqdm import tqdm

from torch.autograd import Variable


import sys
sys.path.insert(0,'..')

from models.wideresnet import *
from models.resnet import *
from ensemble import Ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='./model_test.pt')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./log_cosine_file.txt')
    parser.add_argument('--perturb-steps', default=10,
                    help='perturb number of steps')
    parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
    parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
    

    parser.add_argument('--single', action='store_true', default=False)
    

    args = parser.parse_args()

    # load model

    m1 = WideResNet()
    ckpt = torch.load(args.model + "-0.pt")
    m1.load_state_dict(ckpt)

    m2 = WideResNet()
    ckpt = torch.load(args.model + "-1.pt")
    m2.load_state_dict(ckpt)

    m3 = WideResNet()
    ckpt = torch.load(args.model + "-2.pt")
    m3.load_state_dict(ckpt)
    
    # model = Ensemble([m2])
    model = Ensemble(m1, m2, m3)
    # model.cuda()
    
    device = 'cuda'

    ms = [m1, m2, m3]

    
    [m.to(device) for m in ms]
    model.to(device)
    [m.eval() for m in ms]
    model.eval()


    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)

    natual_cos = []
    adv_cos = []
    for x_natural, y in tqdm(test_loader):
        # print(batch.shape)
        x_natural = x_natural.to(device)

        criterion_kl = nn.KLDivLoss(size_average=False)
        batch_size = len(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(args.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        ls = [m(x_natural) for m in ms]
        adv_ls = [m(x_adv) for m in ms]

        tmp_cos = []
        tmp_adv_cos = []
        for i in range(len(ls)):
            for j in range(i+1, len(ls)):
                tmp_cos.append(F.cosine_similarity(ls[i], ls[j]).detach().cpu().numpy())
                tmp_adv_cos.append(F.cosine_similarity(adv_ls[i], adv_ls[j]).detach().cpu().numpy())
        
        natual_cos.append(np.mean(tmp_cos))
        adv_cos.append(np.mean(tmp_adv_cos))


    print("The average cosine of the model on the dataset")
    print("natural cosine:")
    print(np.mean(natual_cos))
    print("adversarial cosine:")
    print(np.mean(adv_cos))