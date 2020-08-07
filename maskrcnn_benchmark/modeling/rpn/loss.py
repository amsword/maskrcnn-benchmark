# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


def get_kind_to_num_info(labels, all_anchor_sizes_each_pyramid):
    each_py_kind_to_num = None
    for l, anchor_sizes_each_py in zip(labels, all_anchor_sizes_each_pyramid):
        indicators = [['ignore', l == -1],
                      ['background', l == 0],
                      ['positive', l > 0]]
        for indicator in indicators:
            indicator.append(indicator[1].sum())

        if each_py_kind_to_num is None:
            from collections import defaultdict
            each_py_kind_to_num = [defaultdict(int) for _ in anchor_sizes_each_py]
        start = 0
        for idx_py, anchor_size in enumerate(anchor_sizes_each_py):
            for note, indicator, total in indicators:
                curr_indicator = indicator[start: (start + anchor_size)].sum()
                each_py_kind_to_num[idx_py][note] += curr_indicator
            start += anchor_size
    return each_py_kind_to_num

class BoxListIoUModule(torch.nn.Module):
    def forward(self, target, anchor):
        return boxlist_iou(target, anchor)

class RPNLossComputationModule(torch.nn.Module):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        super().__init__()
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']
        self.num_valid_iter = 0
        self.boxlist_iou = BoxListIoUModule()

        self.num_call = 0
        # kind = ignore; background; positive; all means each pyramid
        #self.all_kind_to_num = None

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = self.boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        if len(target) == 0:
            dummy_bbox = torch.zeros((len(matched_idxs), 4),
                    dtype=torch.float32, device=matched_idxs.device)
            from maskrcnn_benchmark.structures.bounding_box import BoxList
            matched_targets = BoxList(dummy_bbox, target.size, target.mode)
            from future.utils import viewitems
            for k, v in viewitems(target.extra_fields):
                if len(v) == 0:
                    if k == 'labels':
                        matched_targets.add_field(k,
                                torch.zeros(len(matched_idxs),
                                    dtype=v.dtype,
                                    device=v.device),
                                )
                    else:
                        # the following seems incorrect.
                        matched_targets.add_field(k, v)
                else:
                    raise Exception('we have no idea of how to deal with '
                            'non-empty fields')
        else:
            matched_targets = target[matched_idxs.clamp(min=0)]
            if (self.num_valid_iter % 300) == 0:
                # only print debug information
                with torch.no_grad():
                    max_val, _ = match_quality_matrix.max(dim=1)
                    import logging
                    logging.info('GT->Anchor: num={}; Mean={}; min={}; max={}'.format(
                        len(max_val),
                        max_val.mean(),
                        max_val.min(),
                        max_val.max()))
            self.num_valid_iter += 1

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image.has_field('tightness'):
                targets_per_image = targets_per_image[targets_per_image.get_field('tightness') > 0.9]
                self.copied_fields.append('tightness')
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def forward(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        #print_log = (self.num_call % 100) == 0
        self.num_call += 1
        #if print_log:
            #all_anchor_sizes_each_pyramid = [[len(a) for a in anchors_per_image]
                #for anchors_per_image in anchors]
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        #if print_log:
            #all_kind_to_num = get_kind_to_num_info(labels, all_anchor_sizes_each_pyramid)
            #from qd.qd_common import print_table
            #print_table(all_kind_to_num)
            #if self.all_kind_to_num is None:
                #self.all_kind_to_num = all_kind_to_num
            #else:
                #for kind_to_num, self_kind_to_num in zip(all_kind_to_num, self.all_kind_to_num):
                    #for kind, num in kind_to_num.items():
                        #self_kind_to_num[kind] += num
            #print_table(self.all_kind_to_num)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']
        self.num_valid_iter = 0

        self.num_call = 0
        # kind = ignore; background; positive; all means each pyramid
        self.all_kind_to_num = None

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        if len(target) == 0:
            dummy_bbox = torch.zeros((len(matched_idxs), 4),
                    dtype=torch.float32, device=matched_idxs.device)
            from maskrcnn_benchmark.structures.bounding_box import BoxList
            matched_targets = BoxList(dummy_bbox, target.size, target.mode)
            from future.utils import viewitems
            for k, v in viewitems(target.extra_fields):
                if len(v) == 0:
                    if k == 'labels':
                        matched_targets.add_field(k,
                                torch.zeros(len(matched_idxs),
                                    dtype=v.dtype,
                                    device=v.device),
                                )
                    else:
                        # the following seems incorrect.
                        matched_targets.add_field(k, v)
                else:
                    raise Exception('we have no idea of how to deal with '
                            'non-empty fields')
        else:
            matched_targets = target[matched_idxs.clamp(min=0)]
            #if (self.num_valid_iter % 300) == 0:
                # only print debug information
                #with torch.no_grad():
                    #max_val, _ = match_quality_matrix.max(dim=1)
                    #import logging
                    #logging.info('GT->Anchor: num={}; Mean={}; min={}; max={}'.format(
                        #len(max_val),
                        #max_val.mean(),
                        #max_val.min(),
                        #max_val.max()))
            self.num_valid_iter += 1

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image.has_field('tightness'):
                targets_per_image = targets_per_image[targets_per_image.get_field('tightness') > 0.9]
                self.copied_fields.append('tightness')
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        #print_log = (self.num_call % 100) == 0
        #self.num_call += 1
        #if print_log:
            #all_anchor_sizes_each_pyramid = [[len(a) for a in anchors_per_image]
                #for anchors_per_image in anchors]

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        #if print_log:
            #with torch.no_grad():
                #all_kind_to_num = get_kind_to_num_info(labels, all_anchor_sizes_each_pyramid)
                #from qd.qd_common import print_table
                #print_table(all_kind_to_num)
                #if self.all_kind_to_num is None:
                    #self.all_kind_to_num = all_kind_to_num
                #else:
                    #for kind_to_num, self_kind_to_num in zip(all_kind_to_num, self.all_kind_to_num):
                        #for kind, num in kind_to_num.items():
                            #self_kind_to_num[kind] += num
                #print_table(self.all_kind_to_num)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def create_rpn_matcher(cfg):
    if cfg.MODEL.RPN.MATCHER_TYPE == 'default':
        matcher = Matcher(
            cfg.MODEL.RPN.FG_IOU_THRESHOLD,
            cfg.MODEL.RPN.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
    elif cfg.MODEL.RPN.MATCHER_TYPE == 'single':
        from maskrcnn_benchmark.modeling.single_matcher import SingleMatcher
        matcher = SingleMatcher(cfg.MODEL.RPN.BG_IOU_THRESHOLD)
    else:
        raise NotImplementedError
    return matcher

def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = create_rpn_matcher(cfg)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
