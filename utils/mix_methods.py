
import torch
import torch.nn as nn
import imp
import numpy as np
import utils
import os
import torch.nn.functional as F
import random
import copy


def mini_forward(model, x):
    try:
        conv4 = model.model(x)
        x = nn.AdaptiveAvgPool2d(1)(conv4)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.5)(x)
        x = model.fc(x)
    except:
        conv4 = model.module.model(x)
        x = nn.AdaptiveAvgPool2d(1)(conv4)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.5)(x)
        x = model.module.fc(x)
    return x, conv4, None


def get_spm(input, target, image_size, model):
    imgsize = image_size
    bs = input.size(0)
    with torch.no_grad():

        output, fms, _ = mini_forward(model, input)
        # if 'inception' in conf.netname:
        #     clsw = model.module.fc
        # else:
        #     clsw = model.module.classifier
        try:
            clsw = model.fc
        except:
            clsw = model.module.fc
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps, clslogit


def snapmix(input,target,image_size,beta,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if True:
        wfmaps,_ = get_spm(input,target,image_size,model)
        bs = input.size(0)
        lam = np.random.beta(beta, beta)
        lam1 = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()


def as_cutmix(input,target,image_size,beta,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if True:
        bs = input.size(0)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]

        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = utils.rand_bbox(input.size(), lam)

        if (bby2_1-bby1_1)*(bbx2_1-bbx1_1) > 4 and  (bby2-bby1)*(bbx2-bbx1)>4:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            # adjust lambda to exactly match pixel ratio
            lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a *= torch.ones(input.size(0))
    lam_b = 1 - lam_a

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

def cutmix(input,target,image_size,beta,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    target_b = target.clone()

    if r < True:
        bs = input.size(0)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a *= torch.ones(input.size(0))

    lam_b = 1 - lam_a


    return input,target,target_b,lam_a.cuda(),lam_b.cuda()



def cutout(input,target,image_size=None,beta=None,model=None):

    r = np.random.rand(1)
    lam = torch.ones(input.size(0)).cuda()
    target_b = target.clone()
    lam_a = lam
    lam_b = 1-lam

    if True:
        bs = input.size(0)
        lam = 0.75
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = 0

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()


def mixup(input,target,image_size,beta,model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    bs = input.size(0)
    target_a = target
    target_b = target

    if True:
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        lam = np.random.beta(beta, beta)
        lam_a = lam_a*lam
        input = input * lam + input[rand_index] * (1-lam)

    lam_b = 1 - lam_a

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()
