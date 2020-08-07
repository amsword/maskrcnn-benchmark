# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
        adaptive = cfg.INPUT.COLORJITTER_ADAPTIVE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        adaptive = None

    to_bgr255 = cfg.INPUT.TO_BGR255
    from qd.qd_pytorch import (DictTransformMaskNormalize,
            DictTransformMaskColorJitter,
            DictTransformMaskToTensor,
            DictTransformMaskRandomHorizontalFlip,
            )
    normalize_transform = DictTransformMaskNormalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
            )
    color_jitter = DictTransformMaskColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            adaptive=adaptive,
            )
    to_tensor = DictTransformMaskToTensor()
    flipper = DictTransformMaskRandomHorizontalFlip(flip_prob)

    from qd.qd_pytorch import DictTransformMaskResize
    if is_train:
        if cfg.INPUT.TRAIN_RESIZER != '':
            from qd.qd_common import load_from_yaml_str
            from qd.qd_common import execute_func
            resizer = execute_func(load_from_yaml_str(cfg.INPUT.TRAIN_RESIZER))
        elif not cfg.INPUT.USE_FIXED_SIZE_AUGMENTATION:
            resizer = DictTransformMaskResize(min_size, max_size,
                                              cfg.INPUT.MIN_SIZE_ON_ITER,
                                              cfg.INPUT.TREAT_MIN_AS_MAX,
                                              )
        else:
            from qd.qd_yolov2pt import DictResizeAndPlaceForMaskRCNN
            resizer = DictResizeAndPlaceForMaskRCNN(cfg)
    else:
        resizer = DictTransformMaskResize(min_size, max_size,
                cfg.INPUT.MIN_SIZE_ON_ITER)

    from qd.qd_pytorch import DictTransformCompose
    transform = DictTransformCompose(
            [
                color_jitter,
                resizer,
                flipper,
                to_tensor,
                normalize_transform,
            ]
        )
    return transform

