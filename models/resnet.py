import pretrainedmodels
from torch import nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class CustomResnetModel(nn.Module):
    def __init__(self, backbone, pretrained='imagenet', out_features=81313, dropout=0.5):
        super().__init__()
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.net = pretrainedmodels.__dict__[backbone](pretrained=pretrained)
        self.net.conv1 = nn.Conv2d(4, self.net.conv1.out_channels, kernel_size=self.net.conv1.kernel_size,
                                   stride=self.net.conv1.stride, padding=self.net.conv1.padding, bias=False)

        # self.net.last_linear = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        # )
        self.last_linear = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt):
        x = self.net.features(x)
        pooled = nn.Flatten()(self.pool(x))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


class DenseNet(nn.Module):
    def __init__(self, backbone, pretrained='imagenet', out_features=81313, dropout=0.5):
        super().__init__()
        if backbone in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            self.net = pretrainedmodels.__dict__[backbone](pretrained=pretrained)
        self.net.features.conv0 = nn.Conv2d(4, self.net.features.conv0.out_channels, kernel_size=self.net.features.conv0.kernel_size,
                                   stride=self.net.features.conv0.stride, padding=self.net.features.conv0.padding, bias=False)

        self.last_linear = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt):
        x = self.net.features(x)
        pooled = nn.Flatten()(self.pool(x))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


class CustomSenet(nn.Module):
    def __init__(self, name='se_resnext50_32x4d', pretrained='imagenet', out_features=19, dropout=0.5):
        super().__init__()
        self.net = pretrainedmodels.__dict__[name](pretrained=pretrained)
        self.net.layer0.conv1 = nn.Conv2d(4, self.net.layer0.conv1.out_channels, kernel_size=self.net.layer0.conv1.kernel_size,
                                   stride=self.net.layer0.conv1.stride, padding=self.net.layer0.conv1.padding, bias=False)
        self.last_linear = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=self.net.last_linear.in_features, out_features=out_features)
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt):
        x = self.net.features(x)
        pooled = nn.Flatten()(self.pool(x))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


class CustomDenseNet(nn.Module):
    def __init__(self, name='densenet161', pretrained='imagenet'):
        super().__init__()
        self.net = pretrainedmodels.__dict__[name](pretrained=pretrained)
        self.fc = nn.Linear(in_features=self.net.last_linear.in_features, out_features=186)

    def forward(self, x):
        x = self.net.features(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc(x)
        return x