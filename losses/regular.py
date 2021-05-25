from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss, SmoothL1Loss
from losses.bi_tempered_loss import bi_tempered_logistic_loss
from utils import *
import torch.nn.functional as F
from torch import nn
import torch

from torch.autograd import Variable


# ICME.2019Skeleton-BasedActionRecognitionwithSynchronousLocalandNon-LocalSpatio-TemporalLearningandFrequencyAttention.pdf
# Soft-margin focal loss
def criterion_margin_focal_binary_cross_entropy(weight_pos=2, gamma=2):
    def _criterion_margin_focal_binary_cross_entropy(logit, truth):
        # weight_pos=2
        weight_neg=1
        # gamma=2
        margin=0.2
        em = np.exp(margin)

        logit = logit.view(-1)
        truth = truth.view(-1)
        log_pos = -F.logsigmoid( logit)
        log_neg = -F.logsigmoid(-logit)

        log_prob = truth*log_pos + (1-truth)*log_neg
        prob = torch.exp(-log_prob)
        margin = torch.log(em +(1-em)*prob)

        weight = truth*weight_pos + (1-truth)*weight_neg
        loss = margin + weight*(1 - prob) ** gamma * log_prob

        loss = loss.mean()
        return loss
    return _criterion_margin_focal_binary_cross_entropy


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    '''
    copy from fastai

    '''
    def __init__(self, eps:float=0.1, reduction='mean'):
        super().__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def bi_tempered_loss(t1, t2, smooth):
    def _bi(output, truth):
        label = torch.zeros_like(output).to(output.device)
        for i in range(label.shape[0]):
            label[i, truth[i]] = 1
        return bi_tempered_logistic_loss(output, label, t1, t2, smooth).mean()
    return _bi


def ce(weight=None):
    print('[ √ ] Using CE loss, weight is {}'.format(weight))
    ce = CrossEntropyLoss(weight=torch.tensor(weight).cuda())

    def _ce_loss(output, truth):
        return ce(output, truth)
    return _ce_loss


def ce_oheb():
    def _ce_loss(output, truth):
        ce = CrossEntropyLoss(reduction='none')
        r = ce(output, truth).view(-1)
        return r.topk(output.shape[0] - 2, largest=False)[0].mean()
    return _ce_loss


def focal_loss(gamma=2):
    def _focal_loss(output, truth):
        focal = FocalLoss(gamma=gamma)
        return focal(output, truth)
    return _focal_loss


def mse():
    # print('[ √ ] Using loss MSE')
    def _mse_loss(output, truth):
        mse = MSELoss()
        return mse(output, truth)
    return _mse_loss


def bce(reduction='mean'):
    # print('[ √ ] Using loss MSE')
    def _bce_loss(output, truth):
        bce = BCEWithLogitsLoss(reduction=reduction)
        return bce(output, truth)
    return _bce_loss


def bce_ohem(ratio=0.5):
    def _bce_loss(output, truth):
        bce = BCEWithLogitsLoss(reduction='none')
        r = bce(output, truth).view(-1)
        return r.topk(int(r.shape[0] * ratio))[0].mean()
    return _bce_loss


def bce_mse(ratio=(0.5, 0.5)):
    print('SpecialLoss: BCE and SUM(1) MSE')
    def _bce_mse_loss(output, truth):
        bce = BCEWithLogitsLoss()
        mse = MSELoss()
        b = bce(output, truth)
        m = mse(torch.sigmoid(output).sum(1), truth.sum(1))
        return ratio[0] * b + ratio[1] * m, b, m
    return _bce_mse_loss



def sl1(k):
    # print('[ √ ] Using loss MSE')
    def _sl1_loss(output, truth):
        _sl1 = SmoothL1Loss()
        return _sl1(k * output, k * truth)
    return _sl1_loss



def mae():
    # print('[ √ ] Using loss MSE')
    def _mae_loss(output, truth):
        mae = L1Loss()
        return mae(output, truth)
    return _mae_loss


# def label_smooth_ce(eps=0.1):
#     def _ce_loss(output, gt):
#         ce = LabelSmoothingCrossEntropy(eps=eps)
#         return ce(output, gt)
#     return _ce_loss

def label_smooth_ce(eps=0.1, reduction='mean'):
    def _ce_loss(output, gt):
        ce = LabelSmoothingCrossEntropy(eps=eps, reduction=reduction)
        return ce(torch.cat([output, 1-output], 1), gt.view(-1).long())
    return _ce_loss


def label_smooth_ce_ohem(eps=0.1, pa=0.5, bs=64):

    def _ce_loss(output, gt):
        ce = LabelSmoothingCrossEntropy(eps=eps, reduction='none')
        k = min(int(pa * bs), output.size(0))
        return ce(output, gt).topk(k=k)
    return _ce_loss


def class_balanced_ce(class_weight, weight=(1, 1, 1)):
    print('[ √ ] class balanced loss. the weight of loss is {}'.format(weight))

    def _ce_loss(output, gt, gt2, gt3):
        gra_ce, vow_ce, con_ce = (CrossEntropyLoss(weight=class_weight[0]),
                                  CrossEntropyLoss(weight=class_weight[1]),
                                  CrossEntropyLoss(weight=class_weight[2]),)
        gra = gra_ce(output[:, :GRAPHEME_ROOT], gt)
        vow = vow_ce(output[:, GRAPHEME_ROOT:GRAPHEME_ROOT+VOWEL_DIACRITIC], gt2)
        con = con_ce(output[:, -CONSONANT_DIACRITIC:], gt3)
        return weight[0] * gra + weight[1] * vow + weight[2] * con, gra, vow, con
    return _ce_loss


