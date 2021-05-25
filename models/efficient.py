import torch
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
from geffnet.conv2d_layers import Conv2dSame


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


# class EfficinetNet(nn.Module):
#     def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313, dropout=0.5, feature_dim=512):
#         super().__init__()
#         self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
#                                     pretrained=(pretrained == 'imagenet'))
#         print(name)
#         self.feature_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=feature_dim)
#         self.last_linear = nn.Linear(in_features=feature_dim, out_features=out_features)
#         self.pool = GeM()
#         self.dropout = dropout
#
#     def forward(self, x, infer=False):
#         x = self.model.features(x)
#         f = self.feature_linear(nn.Flatten()(self.pool(x)))
#         if infer:
#             return self.pool(x)
#         else:
#             f = nn.ReLU()(f)
#             if self.dropout:
#                 return self.last_linear(nn.Dropout(self.dropout)(f))
#             else:
#                 return self.last_linear(f)

class EfficinetNet(nn.Module):
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313, dropout=0.5, feature_dim=512):
        super().__init__()
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
                                    pretrained=(pretrained == 'imagenet'))
        self.model.conv_stem = Conv2dSame(4, self.model.conv_stem.out_channels, kernel_size=(3, 3), stride=(2, 2), bias=False)
        print(name)
        # self.feature_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=feature_dim)
        self.last_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt=16):
        x = self.model.features(x)
        pooled = nn.Flatten()(self.pool(x))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))
        # if infer:
        #     return self.pool(x)
        # else:
        #     f = nn.ReLU()(f)
        #     if self.dropout:
        #         return self.last_linear(nn.Dropout(self.dropout)(f))
        #     else:
        #         return self.last_linear(f)
