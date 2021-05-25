from losses.regular import ce, class_balanced_ce, label_smooth_ce, label_smooth_ce_ohem, mse, mae, bce, sl1, bce_mse
from losses.regular import focal_loss, bce_ohem, criterion_margin_focal_binary_cross_entropy, ce_oheb, bi_tempered_loss
import numpy as np
import pickle as pk
import os


def get_loss(cfg):
    return globals().get(cfg.loss.name)(**cfg.loss.param)


def get_class_balanced_weighted(df, betas):
    '''
    generate class balanced weight

    :param df:
    :param betas:
    :return:
    '''
    weight = df.grapheme_root.value_counts().sort_index().values
    weight = (1 - betas[0]) / (1 - np.power(betas[0], weight))
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../debug/weight.pkl', 'wb') as fp:
        pk.dump([weight], fp)
    return weight


def get_log_weight(df):
    weight = (1 / np.log1p(df.label.value_counts().sort_index())).values
    return weight

