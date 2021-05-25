import albumentations as A
import os
from dataloaders.gridmask import GridMask
from dataloaders.albu_augmix import RandomAugMix, SingleCutOut


def get_tfms(name):
    path = os.path.dirname(os.path.realpath(__file__)) + '/../configs/augmentation/{}.yaml'.format(name)
    return A.load(path, data_format='yaml')


if __name__ == '__main__':
    get_tfms('basic_gridmask_cutout.yaml')
