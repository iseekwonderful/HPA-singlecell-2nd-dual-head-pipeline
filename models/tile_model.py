import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from torch.nn.parameter import Parameter
import efficientnet_pytorch



class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class AvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)

    def forward(self, x): return self.ap(x)


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


# class TileModel(nn.Module):
#     def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True, activate=False):
#         super().__init__()
#         m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
#         self.enc = nn.Sequential(*list(m.children())[:-2])
#         nc = list(m.children())[-1].in_features
#         self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*nc,512),
#                             Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
#
#     def forward(self, *x):
#         shape = x[0].shape
#         n = len(x)
#         x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
#         #x: bs*N x 3 x 128 x 128
#         x = self.enc(x)
#         #x: bs*N x C x 4 x 4
#         shape = x.shape
#         #concatenate the output for tiles into a single map
#         x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
#           .view(-1,shape[1],shape[2]*n,shape[3])
#         #x: bs x C x N*4 x 4
#         x = self.head(x)
#         #x: bs x n
#         return x


# class TileModel(nn.Module):
#     def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
#         super().__init__()
#         if arch in ['resnet18_ssl', 'resnet50_ssl', 'resnext50_32x4d_ssl', 'resnext101_32x4d_ssl', 'resnext101_32x8d_ssl', 'resnext101_32x16d_ssl']:
#             m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
#             self.enc = nn.Sequential(*list(m.children())[:-2])
#             nc = list(m.children())[-1].in_features
#         elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
#             m = pretrainedmodels.__dict__[arch](pretrained=('imagenet' if pre else None))
#             self.enc = m.features
#             nc = m.last_linear.in_features
#         self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*nc,512),
#                             Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
#
#     def forward(self, *x):
#         shape = x[0].shape
#         n = len(x)
#         x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
#         #x: bs*N x 3 x 128 x 128
#         x = self.enc(x)
#         #x: bs*N x C x 4 x 4
#         shape = x.shape
#         #concatenate the output for tiles into a single map
#         x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
#           .view(-1,shape[1],shape[2]*n,shape[3])
#         #x: bs x C x N*4 x 4
#         x = self.head(x)
#         #x: bs x n
#         return x


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


class TileModel(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True, pool_method='concat'):
        super().__init__()
        # print('[ i ] Using encoder: {}, output: {}, pool: {}, pretrained: {}'.format(
        #     arch, n, pool_method, pre
        # ))
        if arch in ['resnet18_ssl', 'resnet50_ssl', 'resnext50_32x4d_ssl', 'resnext101_32x4d_ssl', 'resnext101_32x8d_ssl', 'resnext101_32x16d_ssl']:
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
            self.enc = nn.Sequential(*list(m.children())[:-2])
            nc = list(m.children())[-1].in_features
        elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            m = pretrainedmodels.__dict__[arch](pretrained=('imagenet' if pre else None))
            self.enc = m.features
            nc = m.last_linear.in_features
        elif arch in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b3']:
            if pre:
                m = efficientnet_pytorch.EfficientNet.from_pretrained(arch)
            else:
                m = efficientnet_pytorch.EfficientNet.from_name(arch)
            self.enc = m.extract_features
            self.body = m
            nc = m._fc.in_features
        self.pool_method = pool_method
        if pool_method == 'concat':
            self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*nc,512),
                                Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        elif pool_method == 'gem':
            self.head = nn.Sequential(GeM(),nn.Flatten(),nn.Linear(nc,512),
                                Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        elif pool_method == 'avg':
            self.head = nn.Sequential(AvgPool2d(),nn.Flatten(),nn.Linear(nc,512),
                                Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        elif pool_method == 'concatGeM':
            self.pool, self.gem = AdaptiveConcatPool2d(), GeM()
            self.head = nn.Sequential(nn.Flatten(),nn.Linear(3 * nc,512),
                                Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        else:
            raise Exception('Unknown Poll: {}'.format(pool_method))
    def concat_head(self, x):
        a = self.pool(x)
        b = self.gem(x)
        x = torch.cat([a, b], 1)
        x = self.head(x)
        return x

    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        if self.pool_method == 'concatGeM':
            x = self.concat_head(x)
        else:
            x = self.head(x)
        #x: bs x n
        return x


class ConcatModel(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True, pool_method='concat'):
        super().__init__()
        # print('[ i ] Using encoder: {}, output: {}, pool: {}, pretrained: {}'.format(
        #     arch, n, pool_method, pre
        # ))
        if arch in ['resnet18_ssl', 'resnet50_ssl', 'resnext50_32x4d_ssl', 'resnext101_32x4d_ssl', 'resnext101_32x8d_ssl', 'resnext101_32x16d_ssl']:
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
            self.enc = nn.Sequential(*list(m.children())[:-2])
            nc = list(m.children())[-1].in_features
        elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            m = pretrainedmodels.__dict__[arch](pretrained=('imagenet' if pre else None))
            self.enc = m.features
            nc = m.last_linear.in_features
        elif arch in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b3']:
            if pre:
                m = efficientnet_pytorch.EfficientNet.from_pretrained(arch)
            else:
                m = efficientnet_pytorch.EfficientNet.from_name(arch)
            self.enc = m.extract_features
            self.body = m
            nc = m._fc.in_features
        self.pool_method = pool_method
        if pool_method == 'concat':
            self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(2*nc,512),
                                Mish(),
                                      # nn.BatchNorm1d(512),
                                      nn.Dropout(0.5),nn.Linear(512,n))
        elif pool_method == 'gem':
            self.head = nn.Sequential(GeM(),nn.Flatten(),nn.Linear(nc,512),
                                Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        elif pool_method == 'concatGeM':
            self.pool, self.gem = AdaptiveConcatPool2d(), GeM()
            self.head = nn.Sequential(nn.Flatten(),nn.Linear(3 * nc,512),
                                Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        else:
            raise Exception('Unknown Poll: {}'.format(pool_method))

    def concat_head(self, x):
        a = self.pool(x)
        b = self.gem(x)
        x = torch.cat([a, b], 1)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.enc(x)
        if self.pool_method == 'concatGeM':
            x = self.concat_head(x)
        else:
            x = self.head(x)
        return x


class SLModel(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True, pool_method='concat'):
        super().__init__()
        # print('[ i ] Using encoder: {}, output: {}, pool: {}, pretrained: {}'.format(
        #     arch, n, pool_method, pre
        # ))
        if arch in ['resnet18_ssl', 'resnet50_ssl', 'resnext50_32x4d_ssl', 'resnext101_32x4d_ssl',
                    'resnext101_32x8d_ssl', 'resnext101_32x16d_ssl']:
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
            self.enc = nn.Sequential(*list(m.children())[:-2])
            nc = list(m.children())[-1].in_features
        elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            m = pretrainedmodels.__dict__[arch](pretrained=('imagenet' if pre else None))
            self.enc = m.features
            nc = m.last_linear.in_features
        elif arch in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b3', 'efficientnet-b5', 'efficientnet-b6']:
            if pre:
                m = efficientnet_pytorch.EfficientNet.from_pretrained(arch)
            else:
                m = efficientnet_pytorch.EfficientNet.from_name(arch)
            self.enc = m.extract_features
            self.body = m
            nc = m._fc.in_features
        self.pool_method = pool_method
        if pool_method == 'concat':
            self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Dropout(0.5),nn.Linear(2*nc,n))
        elif pool_method == 'gem':
            self.head = nn.Sequential(GeM(),nn.Flatten(),nn.Dropout(0.5),nn.Linear(nc,n))
        elif pool_method == 'avg':
            self.head = nn.Sequential(AvgPool2d(),nn.Flatten(),nn.Dropout(0.5),nn.Linear(nc,n))
        elif pool_method == 'concatGeM':
            self.pool, self.gem = AdaptiveConcatPool2d(), GeM()
            self.head = nn.Sequential(nn.Flatten(),nn.Dropout(0.5),nn.Linear(3*nc,n))
        else:
            raise Exception('Unknown Poll: {}'.format(pool_method))

    def concat_head(self, x):
        a = self.pool(x)
        b = self.gem(x)
        x = torch.cat([a, b], 1)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.enc(x)
        if self.pool_method == 'concatGeM':
            x = self.concat_head(x)
        else:
            x = self.head(x)
        return x