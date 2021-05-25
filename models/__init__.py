from models.resnet import *
from models.xresnet import *
from models.public import MultiTailPub
from models.margins import CosineSe50, EfficinetNetArcFace, EfficinetNetCosine
from models.distance import ArcModel, ArcModelTail
from models.tile_model import TileModel, ConcatModel, SLModel
from models.resnetd import JakiroResNet200D
from models.efficient import EfficinetNet
from configs import Config


def get_model(cfg: Config, pretrained='imagenet'):
    if cfg.model.name in ['resnet200d', 'resnet152d', 'resnet101d', 'resnet50d', 'resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a RANZCRResNet200D, mdl_name: {cfg.model.name}, pool: {pool}')
        return JakiroResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['cos_tf_efficientnet_b5', 'cos_tf_efficientnet_b0', 'cos_tf_efficientnet_b7']:
        return EfficinetNetCosine(name=cfg.model.name[4:], pretrained=pretrained,
                                   m=cfg.model.param.get('dropout', 0.1),
                                   out_features=cfg.model.out_feature, dropout=cfg.model.param['dropout'])
    if cfg.model.name in ['arc_tf_efficientnet_b5', 'arc_tf_efficientnet_b7']:
        return EfficinetNetArcFace(name=cfg.model.name[4:], pretrained=pretrained,
                                   m=cfg.model.param.get('dropout', 0.1),
                                   out_features=cfg.model.out_feature, dropout=cfg.model.param['dropout'])
    elif cfg.model.name in ['tf_efficientnet_b5', 'efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b0', 'tf_efficientnet_b7',
                            'tf_efficientnet_l2_ns', 'tf_efficientnet_b6_ns']:
        return EfficinetNet(name=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature, dropout=cfg.model.param['dropout'])
    elif cfg.model.name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return CustomResnetModel(backbone=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature)
    elif cfg.model.name in ['densenet161', 'densenet121', 'densenet201', 'densenet169']:
        return DenseNet(backbone=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature)
    elif cfg.model.name in ['XResNet', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152',
           'xresnet18_deep', 'xresnet34_deep', 'xresnet50_deep']:
        return globals()[cfg.model.name](c_out=11)
    elif cfg.model.name in ['se_resnext50_32x4d', 'se_resnext101_32x4d']:
        return CustomSenet(name=cfg.model.name, pretrained=pretrained, out_features=cfg.model.out_feature)
    elif cfg.model.name in ['arc_resnet18', 'arc_resnet34', 'arc_resnet50', 'arc_se_resnext50_32x4d',
                     'arc_se_resnext101_32x4d', 'arc_senet154']:
        return ArcModel(pretrained=pretrained, model=cfg.model.name)
    elif cfg.model.name in ['art_resnet18', 'art_resnet34', 'art_resnet50', 'art_se_resnext50_32x4d',
                     'art_se_resnext101_32x4d', 'art_senet154']:
        return ArcModelTail(pretrained=pretrained, model=cfg.model.name)
    elif cfg.model.name in ['CosineSe50']:
        return CosineSe50()
