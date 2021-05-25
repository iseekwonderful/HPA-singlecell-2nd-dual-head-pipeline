import math
import torch
import pretrainedmodels
import torch.nn.functional as F

from torchvision import models
from torch import nn
from functools import partial
from sklearn.metrics import recall_score
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.parameter import Parameter



model_dict = {
    'd121': models.densenet121,
    'd161': models.densenet161,
    'd201': models.densenet201,
    'd169': models.densenet169,
    'x50': pretrainedmodels.se_resnext50_32x4d,
    'x101': pretrainedmodels.se_resnext101_32x4d,
    's154': pretrainedmodels.senet154,
}


class Backbone(nn.Module):
    def __init__(self, name, pretrained=True, copy_weight=True):
        nn.Module.__init__(self)
        self.name = name
        if name in ['d121', 'd161', 'd201', 'd169']:
            self.net = model_dict[name](pretrained=pretrained)

        elif name in ['x50', 'x101', 's154']:
            self.net = model_dict[name](pretrained='imagenet')
            setattr(self.net, 'features', nn.Sequential(
                self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4
            ))
            setattr(self.net, 'classifier', self.net.last_linear)

        elif name in ['eb1', 'eb2', 'eb3', 'eb4', 'eb5', 'eb6']:
            self.net = model_dict[name]()
            if not copy_weight:
                print('[W] Weight already copied, fuck u!')

            # setattr(self.net, 'feature', self.net.extract_features)
            setattr(self.net, 'classifier', self.net._fc)

    def forward(self, x):
        if self.name in ['eb1', 'eb2', 'eb3', 'eb4', 'eb5', 'eb6']:
            return self.net.extract_features(x)
        else:
            return self.net.features(x)


def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output


class MarginLinear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=10008,  s=64., m=0.5):
        super(MarginLinear, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label, is_infer = False):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)

        if not is_infer:
            output[idx_, label] = cos_theta_m[idx_, label]

        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output


class MarginHead(nn.Module):

    def __init__(self, num_class=10008, emb_size = 2048, s=64., m=0.5):
        super(MarginHead,self).__init__()
        self.fc = MarginLinear(embedding_size=emb_size, classnum=num_class , s=s, m=m)

    def forward(self, fea, label, is_infer):
        fea = l2_norm(fea)
        logit = self.fc(fea, label, is_infer)
        return logit


class DistLinear(nn.Module):
    # the basic cosine distance
    def __init__(self, indim, outdim):
        super(DistLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)

        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)

        return scores


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=10, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class CosineSe50(nn.Module):
    def __init__(self, out_features=5, backbone='x50'):
        super(CosineSe50, self).__init__()
        self.backbone = Backbone(name=backbone)
        in_features = self.backbone.net.last_linear.in_features
        self.top = DistLinear(in_features, out_features)

    def forward(self, x):
        x = self.backbone(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        x = nn.Dropout()(x)
        dis = self.top(x)
        return dis

    def infer(self, x):
        x = self.backbone(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        return x


class CosineD121(nn.Module):
    def __init__(self, out_features=168, backbone='d121'):
        super(CosineD121, self).__init__()
        self.backbone = Backbone(name=backbone)
        in_features = self.backbone.net.classifier.in_features
        self.top = DistLinear(in_features, out_features)

    def forward(self, x):
        x = self.backbone(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        return self.top(x)

    def infer(self, x):
        x = self.backbone(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        return x


class ArcFaceD121(nn.Module):
    def __init__(self, channel_size=512, out_feature=168, dropout=0.5, backbone='d121'):
        super(ArcFaceD121, self).__init__()
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.backbone = Backbone(name=backbone)
        self.in_features = self.backbone.net.classifier.in_features
        self.margin = ArcModule(in_features=self.channel_size, out_features=out_feature)
        # arcFace layers
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 16 * 16, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        features = self.backbone.net.features(x)
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


class ArcFaceD121PoolFc(nn.Module):
    def __init__(self, channel_size=512, out_features=168, backbone='d121'):
        super(ArcFaceD121PoolFc, self).__init__()
        self.channel_size = channel_size
        self.out_feature = out_features
        self.backbone = Backbone(name=backbone)
        self.in_features = self.backbone.net.classifier.in_features
        print('[W] out_feature is {}'.format(out_features))
        self.margin = ArcModule(in_features=self.channel_size, out_features=out_features)
        # arcFace layers
        self.fc1 = nn.Linear(self.in_features, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc1(x)
        x = self.bn2(x)
        features = F.normalize(x)
        if labels is not None:
            return self.margin(features, labels)
        return features

    def infer(self, x):
        return self.forward(x)


class ArcFaceFc(nn.Module):
    def __init__(self, channel_size=512, out_feature=168, backbone='x50'):
        super(ArcFaceFc, self).__init__()
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.backbone = Backbone(name=backbone)
        self.in_features = self.backbone.net.classifier.in_features
        self.margin = ArcModule(in_features=self.channel_size, out_features=out_feature)
        # arcFace layers
        self.fc1 = nn.Linear(self.in_features, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc1(x)
        x = self.bn2(x)
        features = F.normalize(x)
        if labels is not None:
            return self.margin(features, labels)
        return features

    def infer(self, x):
        return self.forward(x)


# class EfficientArcFace(nn.Module):
#     def __init__(self, channel_size=512, out_features=168, backbone='d121'):
#         super(EfficientArcFace, self).__init__()
#         self.channel_size = channel_size
#         self.out_feature = out_features
#         self.backbone = Backbone(name=backbone)
#         self.in_features = self.backbone.net.classifier.in_features
#         print('[W] out_feature is {}'.format(out_features))
#         self.margin = ArcModule(in_features=self.channel_size, out_features=out_features)
#         # arcFace layers
#         self.fc1 = nn.Linear(self.in_features, self.channel_size)
#         self.bn2 = nn.BatchNorm1d(self.channel_size)
#
#     def forward(self, x, labels=None):
#         x = self.backbone(x)
#         x = F.avg_pool2d(x, x.shape[2:])
#         x = x.view(x.shape[0], x.shape[1])
#         x = self.fc1(x)
#         x = self.bn2(x)
#         features = F.normalize(x)
#         if labels is not None:
#             return self.margin(features, labels)
#         return features
#
#     def infer(self, x):
#         return self.forward(x)


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


class EfficinetNetArcFace(nn.Module):
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313,
                 dropout=0.5, feature_dim=512, m=0.1):
        super().__init__()
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
                                    pretrained=(pretrained == 'imagenet'))
        print(name)
        self.feature_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=feature_dim)
        self.last_linear = nn.Linear(in_features=feature_dim, out_features=out_features)
        self.pool = GeM()
        self.margin = ArcModule(in_features=feature_dim, out_features=out_features, m=m)
        self.dropout = dropout

    def forward(self, x, label):
        x = self.model.features(x)
        f = self.feature_linear(nn.Flatten()(self.pool(x)))
        features = F.normalize(f)
        if label is None:
            return features
        else:
            if self.dropout:
                return self.margin(nn.Dropout(self.dropout)(features), label)
            else:
                return self.margin(features, label)


class EfficinetNetCosine(nn.Module):
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313,
                 dropout=0.5, feature_dim=512, m=0.1):
        super().__init__()
        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
                                    pretrained=(pretrained == 'imagenet'))
        print(name)
        self.feature_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=feature_dim)
        self.last_linear = nn.Linear(in_features=feature_dim, out_features=out_features)
        self.pool = GeM()
        self.margin = DistLinear(indim=feature_dim, outdim=out_features)
        self.dropout = dropout

    def forward(self, x, label):
        x = self.model.features(x)
        f = self.feature_linear(nn.Flatten()(self.pool(x)))
        features = F.normalize(f)
        if label is None:
            return features
        else:
            if self.dropout:
                return self.margin(nn.Dropout(self.dropout)(features))
            else:
                return self.margin(features)
