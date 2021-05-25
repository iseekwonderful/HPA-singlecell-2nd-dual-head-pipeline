import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.xresnet import *


class Mish(nn.Module):
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class MultiTailPub(nn.Module):
    def __init__(self, pretrained='imagenet', model='mt_se_resnext50_32x4d', dropout=True):
        super().__init__()
        print('[ √ ] Model using dropout: {}'.format(dropout))
        if model in ['mt_resnet18', 'mt_resnet34', 'mt_resnet50', 'mt_se_resnext50_32x4d',
                     'mt_se_resnext101_32x4d', 'mt_senet154', 'mt_densenet161', 'mt_densenet121']:
            self.net = pretrainedmodels.__dict__[model[3:]](pretrained=pretrained)
            self.in_features = self.net.last_linear.in_features
        else:
            raise Exception('This model not supported!')
        if dropout:
            self.gra_tail = nn.Sequential(
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                # nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(),
                nn.Linear(self.in_features, 168)
            )
        else:
            self.gra_tail = nn.Sequential(
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                # nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0),
                nn.Linear(self.in_features, 168)
            )
        if dropout:
            self.vow_tail = nn.Sequential(
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                # nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(),
                nn.Linear(self.in_features, 11)
            )
        else:
            self.vow_tail = nn.Sequential(
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                # nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0),
                nn.Linear(self.in_features, 11)
            )
        if dropout:
            self.con_tail = nn.Sequential(
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                # nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(),
                nn.Linear(self.in_features, 7)
            )
        else:
            self.con_tail = nn.Sequential(
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                # nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0),
                nn.Linear(self.in_features, 7)
            )

    def forward(self, x):
        x = self.net.features(x)
        return torch.cat([self.gra_tail(x), self.vow_tail(x), self.con_tail(x)], 1)

    def infer(self, x):
        return self.net.features(x)

    def freeze_backbone(self):
        for x in self.net.parameters():
            x.requires_grad = False

    def unfreeze_all(self):
        for x in self.parameters():
            x.requires_grad = True

    def freeze_top_layer(self, k):
        if k > 5:
            raise Exception('No suh large amount of layers')
        for i in range(k):
            for x in getattr(self.net, 'layer{}'.format(i)).parameters():
                x.requires_grad = True

