# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    if '\n' in cfg.MODEL.META_ARCHITECTURE:
        from qd.qd_common import load_from_yaml_str
        param = load_from_yaml_str(cfg.MODEL.META_ARCHITECTURE)
        from qd.qd_common import execute_func
        meta_arch = execute_func(param)
    else:
        meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
