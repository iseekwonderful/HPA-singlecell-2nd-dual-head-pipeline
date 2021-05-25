import numpy as np
import random
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop, RandomApply,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import cv2
from dataloaders.augmix import augment_and_mix
from dataloaders.gridmask import GridMask
import albumentations
from dataloaders.albu_augmix import RandomAugMix


def erosion(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.erode(img, kernel, iterations=1)
    return img


def diation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def do_rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


class BBox:
    def __init__(self, scale=(0.6, 1.4), rotate=(-10, 10), p_dia=0.3, p_ero=0.3):
        self.scale = scale
        self.rotate = rotate
        self.p_dia = p_dia
        self.p_ero = p_ero

    def __call__(self, img):
        img = np.array(img)[:, :, 0]
        x_, y_ = np.where(img > 32)
        bbox = x_.min(), x_.max(), y_.min(), y_.max()
        bbx = img[bbox[0]:bbox[1], bbox[2]: bbox[3]]
        # scale
        factor = np.random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # check size limit:
        if int(bbx.shape[1] * factor) > 236 or int(bbx.shape[0] * factor) > 137:
               factor = min(236 / bbx.shape[1], 137 / bbx.shape[0])
        bbx = cv2.resize(bbx, ( int(bbx.shape[1] * factor), int(bbx.shape[0] * factor) ))
        width, height = bbx.shape[1], bbx.shape[0]
        x, y = random.randint(0, 137 - height), random.randint(0, 236 - width)
        all_new = np.zeros((137, 236), dtype=np.uint8)
        all_new[x:x+height, y:y+width] = bbx
        # rotate
        rot = do_rotate(all_new, np.random.randint(self.rotate[0], self.rotate[1]))
        if np.random.random() < self.p_dia:
            rot = diation(rot)
        if np.random.random() < self.p_ero:
            rot = erosion(rot)
        return Image.fromarray(np.repeat(rot[:, :, np.newaxis], 3, axis=2))


class CutOut:
    def __init__(self):
        pass

    def __call__(self, image):
        def _cut(img):
            sz = img.size
            #             w, h = random.randint(0, sz[0] // 2), random.randint(0, sz[1] // 2)
            w = random.randint(min(sz) // 4, min(sz) // 2)
            x, y = random.randint(0, sz[0] - w), random.randint(0, sz[1] - w)
            arr = np.array(img)
            arr[y: y + w, x: x + w] = 0
            return Image.fromarray(arr)

        return _cut(image)


class WarmCutOut:
    def __init__(self):
        pass

    def __call__(self, image):
        def _cut(img):
            sz = img.size
            #             w, h = random.randint(0, sz[0] // 2), random.randint(0, sz[1] // 2)
            w = random.randint(min(sz) // 8, min(sz) // 4)
            x, y = random.randint(0, sz[0] - w), random.randint(0, sz[1] - w)
            arr = np.array(img)
            arr[y: y + w, x: x + w] = 0
            return Image.fromarray(arr)

        return _cut(image)


class AugMix:
    def __call__(self, image):
        return augment_and_mix(image)


class TGridMask:
    def __init__(self):
        self.transforms_train = albumentations.Compose([
            albumentations.OneOf([
                GridMask(num_grid=3, mode=0, rotate=15),
                GridMask(num_grid=3, mode=1, rotate=15),
                GridMask(num_grid=3, mode=2, rotate=15),
            ], p=1)
        ])

    def __call__(self, image):
        image = np.array(image)
        image = self.transforms_train(image=image)['image']
        return Image.fromarray(image)


class NoRotGridMask:
    def __init__(self):
        self.transforms_train = albumentations.Compose([
            albumentations.OneOf([
                # GridMask(num_grid=(5, 8), mode=0, rotate=0),
                GridMask(num_grid=(3, 7), mode=1, rotate=5),
                GridMask(num_grid=(3, 7), mode=2, rotate=-5),
            ], p=1)
        ])

    def __call__(self, image):
        image = np.array(image)
        image = self.transforms_train(image=image)['image']
        return Image.fromarray(image)


class RotGridMaskV1:
    def __init__(self):
        self.transforms_train = albumentations.Compose([
            albumentations.OneOf([
                GridMask(num_grid=(5, 8), mode=0, rotate=0),
                GridMask(num_grid=(3, 7), mode=2, rotate=-5),
            ], p=1)
        ])

    def __call__(self, image):
        image = np.array(image)
        image = self.transforms_train(image=image)['image']
        return Image.fromarray(image)


class RotGridMaskV2:
    def __init__(self):
        self.transforms_train = albumentations.Compose([
            albumentations.OneOf([
                GridMask(num_grid=3, mode=0),
                GridMask(num_grid=3, mode=1),
                GridMask(num_grid=3, mode=2),
            ], p=1)
        ])

    def __call__(self, image):
        image = np.array(image)
        image = self.transforms_train(image=image)['image']
        return Image.fromarray(image)


class TAugMix:
    def __init__(self):
        self.transforms_train = albumentations.Compose([
            RandomAugMix(severity=5, width=2, p=1.),
        ])

    def __call__(self, image):
        image = np.array(image)
        image = self.transforms_train(image=image)['image'].astype(np.uint8)
        return Image.fromarray(image)


cutout_gridmask_augmix = lambda x: Compose([
    RandomChoice([
        CutOut(),
        TGridMask(),
        TAugMix()
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


cutout_gridmask_augmix_no_rot = lambda x: Compose([
    RandomChoice([
        CutOut(),
        NoRotGridMask(),
        TAugMix()
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


cutout_gridmask_augmix_v1 = lambda x: Compose([
    RandomChoice([
        CutOut(),
        RotGridMaskV1(),
        TAugMix()
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


cutout_gridmask_augmix_v2 = lambda x: Compose([
    RandomChoice([
        CutOut(),
        RotGridMaskV1(),
        TAugMix()
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


seq_augmix_oneof_gridmask_augmix = lambda x: Compose([
    TAugMix(),
    RandomChoice([
        CutOut(),
        TGridMask(),
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])



gridmask_augmix = lambda x: Compose([
    RandomChoice([
        TGridMask(),
        TAugMix()
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


baseline_transform = lambda x: Compose([
    # Resize(x),
    #     RandomChoice([
    # #         RandomHorizontalFlip(),
    #         RandomAffine(90),
    # #         RandomVerticalFlip(),
    #     ]),
    # Resize((256, 256)),
    CutOut(),
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


gridmask_transformation = lambda x: Compose([
    TGridMask(),
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


augmix_gridmask_transformation = lambda x: Compose([
    TAugMix(),
    TGridMask(),
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


cutout_gridmask_mixed_transformation = lambda x: Compose([
    RandomChoice([TGridMask(), CutOut()]),
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


bbox_transform_p025 = lambda x: Compose([
    # Resize(x),
    RandomApply([BBox()], p=0.25),
    CutOut(),
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


bbox_transform_p05 = lambda x: Compose([
    # Resize(x),
    RandomApply([BBox()], p=0.5),
    CutOut(),
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])



cutmix_transformation = lambda x: Compose([
    RandomChoice([
        ColorJitter(0.75, 0.75, 0.75),
        RandomRotation(10)
    ]),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])


augmix_transformation = lambda x: Compose([
    AugMix(),
])


train_transform = lambda x: Compose([
    Resize(x),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])

test_transform = lambda x: Compose([
    Resize(x),
    ToTensor(),
    Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
])

None_trainsform = lambda x: Compose([
    ToTensor(),
    Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def get_tfms(name):
    return globals()[name]


if __name__ == '__main__':
    get_tfms('baseline_transform')