# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
# load just to register the ops
from maskrcnn_benchmark import _C

nms = torch.ops.roi_ops.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
