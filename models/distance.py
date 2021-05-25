import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pretrainedmodels

from models.xresnet import *
from models.margins import ArcModule
import torch
from models.public import GeM, Mish


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


class ArcModel(nn.Module):
    '''
    Feature -> AvgPool -> Flatten -> ArcModule

    '''
    def __init__(self, pretrained='imagenet', model='arc_se_resnext50_32x4d',
                 dropout=True, channel_size=512, out_feature=168):
        super().__init__()
        print('[ W ] Only output first class, make sure set loss weight to [1, 0, 0]!')
        if model in ['arc_resnet18', 'arc_resnet34', 'arc_resnet50', 'arc_se_resnext50_32x4d',
                     'arc_se_resnext101_32x4d', 'arc_senet154']:
            self.net = pretrainedmodels.__dict__[model[4:]](pretrained=pretrained)
            self.in_features = self.net.last_linear.in_features
        else:
            raise Exception('This model not supported!')
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.margin = ArcModule(in_features=self.in_features, out_features=out_feature, m=0.5)
        # arcFace layers
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 5 * 8, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        x = self.net.features(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        features = F.normalize(x)
        if labels is not None:
            return self.margin(features, labels)
        return features

    def infer(self, x):
        return self.forward(x)


class ArcModelOriginal(nn.Module):
    '''
    # original implementation
    Feature -> batchnorm -> dropout -> Flatten -> linear1 -> bartchnorm -> Arcmodule


    '''
    def __init__(self, pretrained='imagenet', model='arc_se_resnext50_32x4d',
                 dropout=True, channel_size=512, out_feature=168):
        super().__init__()
        print('[ W ] Only output first class, make sure set loss weight to [1, 0, 0]!')
        if model in ['arc_resnet18', 'arc_resnet34', 'arc_resnet50', 'arc_se_resnext50_32x4d',
                     'arc_se_resnext101_32x4d', 'arc_senet154']:
            self.net = pretrainedmodels.__dict__[model[4:]](pretrained=pretrained)
            self.in_features = self.net.last_linear.in_features
        else:
            raise Exception('This model not supported!')
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.margin = ArcModule(in_features=self.in_features, out_features=out_feature, m=0.5)
        # arcFace layers
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 5 * 8, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        features = self.net.features(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            return self.margin(features, labels)
        return features

    def infer(self, x):
        return self.forward(x)


class ArcModelTail(nn.Module):
    '''
                Mish(),
                nn.Conv2d(self.in_features, self.in_features, (1, 1)),
                nn.BatchNorm2d(self.in_features),
                GeM(),
                nn.Flatten(),
                nn.Dropout(),
                nn.Linear(self.in_features, 168)

    # original implementation
    Feature -> Mish -> Conv2D -> batchnorm -> GeM -> Normalize -> ArcModule

    '''

    def __init__(self, pretrained='imagenet', model='arc_se_resnext50_32x4d',
                 dropout=True, channel_size=512, out_feature=168):
        super().__init__()
        print('[ W ] Only output first class, make sure set loss weight to [1, 0, 0]!')
        if model in ['art_resnet18', 'art_resnet34', 'art_resnet50', 'art_se_resnext50_32x4d',
                     'art_se_resnext101_32x4d', 'art_senet154']:
            self.net = pretrainedmodels.__dict__[model[4:]](pretrained=pretrained)
            self.in_features = self.net.last_linear.in_features
        else:
            raise Exception('This model not supported!')
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.margin = ArcModule(in_features=self.in_features, out_features=out_feature, m=0.5)
        # arcFace layers
        self.gra_tail = nn.Sequential(
            Mish(),
            nn.Conv2d(self.in_features, self.in_features, (1, 1)),
            nn.BatchNorm2d(self.in_features),
            GeM(),
            nn.Flatten(),
        )

    def forward(self, x, labels=None):
        features = self.net.features(x)
        features = self.gra_tail(features)
        features = F.normalize(features)
        if labels is not None:
            return self.margin(features, labels)
        return features

    def infer(self, x):
        return self.forward(x)
