# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
#nms_max = amp.float_function(_C.nms_max)
nms_max = _C.nms_max
hnms = _C.hnms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
