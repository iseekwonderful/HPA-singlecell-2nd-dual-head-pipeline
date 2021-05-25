import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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


class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

    @property
    def net(self):
        return self.model


class JakiroResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model

