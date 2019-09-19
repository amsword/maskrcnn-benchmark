# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False,
        classification_loss_type='CE',
        num_classes=81,
        boundingbox_loss_type='SL1',
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.classification_loss_type = classification_loss_type
        if self.classification_loss_type == 'CE':
            self._classifier_loss = F.cross_entropy
        elif self.classification_loss_type == 'BCE':
            from qd.qd_pytorch import BCEWithLogitsNegLoss
            self._classifier_loss = BCEWithLogitsNegLoss()
        elif self.classification_loss_type.startswith('IBCE'):
            param = map(float, self.classification_loss_type[4:].split('_'))
            from qd.qd_pytorch import IBCEWithLogitsNegLoss
            self._classifier_loss = IBCEWithLogitsNegLoss(*param)
        elif self.classification_loss_type == 'MCEB':
            from qd.qd_pytorch import MCEBLoss
            self._classifier_loss = MCEBLoss()
        else:
            assert self.classification_loss_type.startswith('tree')
            raise NotImplementedError('not tested')
            _, tree_file = list(classification_loss_type.split('$'))[1]
            from mtorch.softmaxtree_loss import SoftmaxTreeWithLoss
            self._classifier_loss = SoftmaxTreeWithLoss(
                tree_file,
                ignore_label=-1, # this is dummy value since this will not happend
                loss_weight=1,
                valid_normalization=True,
            ).cuda()

        self.num_classes = num_classes
        if boundingbox_loss_type == 'SL1':
            self.weight_box_loss = False
        else:
            assert boundingbox_loss_type.startswith('WSL1')
            self.weight_box_loss = True
            bbs = boundingbox_loss_type.split('_')
            assert bbs[0] == 'WSL1'
            valid_iou_lower = 0.1
            if len(bbs) == 2:
                valid_iou_lower = float(bbs[1])
            else:
                assert len(bbs) == 1, 'not implemented'
            from qd.layers.smooth_l1_loss import SmoothL1LossWithIgnore
            self.box_loss = SmoothL1LossWithIgnore(beta=1, size_average=False,
                    valid_iou_lower=valid_iou_lower)

    def create_all_bkg_labels(self, num, device):
        if self.classification_loss_type in ['CE']:
            return torch.zeros(num,
                dtype=torch.float32,
                device=device)
        elif self.classification_loss_type in ['BCE'] or \
                self.classification_loss_type.startswith('IBCE'):
            return torch.zeros((num, self.num_classes),
                dtype=torch.float32,
                device=device)
        else:
            raise NotImplementedError(self.classification_loss_type)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", 'tightness'])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        if len(target) == 0:
            dummy_bbox = torch.zeros((len(matched_idxs), 4),
                    dtype=torch.float32, device=matched_idxs.device)
            from maskrcnn_benchmark.structures.bounding_box import BoxList
            matched_targets = BoxList(dummy_bbox, target.size, target.mode)
            matched_targets.add_field('labels', self.create_all_bkg_labels(
                len(matched_idxs), matched_idxs.device))
            matched_targets.add_field('tightness', torch.zeros(len(matched_idxs),
                        device=matched_idxs.device))
        else:
            matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        if self.weight_box_loss:
            matched_iou = torch.full((len(matched_idxs),), -1., device=matched_idxs.device)
            matched_iou[matched_idxs >= 0] = match_quality_matrix[matched_idxs[matched_idxs >= 0],
                    matched_idxs >= 0]
            matched_targets.add_field('matched_iou', matched_iou)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        matched_ious = []
        regression_tightness = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler
            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            regression_tightness.append(matched_targets.get_field('tightness'))
            if matched_targets.has_field('matched_iou'):
                matched_ious.append(matched_targets.get_field('matched_iou'))

        result = {'labels': labels,
                'regression_targets': regression_targets,
                'regression_tightness': regression_tightness}

        if len(matched_ious) > 0:
            assert len(matched_ious) == len(regression_targets)
            result['matched_ious'] = matched_ious

        return result

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        prepare_info = self.prepare_targets(proposals, targets)
        labels, regression_targets = prepare_info['labels'], prepare_info['regression_targets']
        regression_tightness = prepare_info['regression_tightness']
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for i, (labels_per_image, regression_targets_per_image, proposals_per_image, rt) in enumerate(zip(
            labels, regression_targets, proposals, regression_tightness
        )):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field('regression_tightness', rt)
            if 'matched_ious' in prepare_info:
                proposals_per_image.add_field(
                        'matched_ious', prepare_info['matched_ious'][i])

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        regression_tightness = cat(
            [proposal.get_field("regression_tightness") for proposal in proposals], dim=0
        )

        classification_loss = self._classifier_loss(class_logits, labels)

        if labels.dim() == 1:
            # get indices that correspond to the regression targets for
            # the corresponding ground truth labels, to be used with
            # advanced indexing
            sampled_pos_inds_subset = torch.nonzero((labels > 0) &
                    (regression_tightness > 0.9)).squeeze(1)
            if sampled_pos_inds_subset.numel() == 0:
                box_loss = torch.tensor(0., device=device)
            else:
                labels_pos = labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor(
                        [0, 1, 2, 3], device=device)
                sampled_box_regression = box_regression[sampled_pos_inds_subset[:, None], map_inds]
                sampled_box_target = regression_targets[sampled_pos_inds_subset]
                if not self.weight_box_loss:
                    box_loss = smooth_l1_loss(
                        sampled_box_regression,
                        sampled_box_target,
                        size_average=False,
                        beta=1,
                    )
                else:
                    matched_ious = cat(
                        [proposal.get_field("matched_ious") for proposal in proposals], dim=0
                    )
                    sampled_ious = matched_ious[sampled_pos_inds_subset]
                    box_loss = self.box_loss(sampled_box_regression,
                            sampled_box_target,
                            sampled_ious)

                box_loss = box_loss / labels.numel()
        else:
            assert labels.dim() == 2
            x = torch.nonzero((labels > 0) & (regression_tightness > 0.9)[:, None])
            if x.numel() == 0:
                box_loss = torch.tensor(0., device=device)
            else:
                sampled_pos_inds_subset = x[:, 0]
                if self.num_classes == labels.shape[1]:
                    labels_pos = x[:, 1]
                else:
                    raise Exception('we should never reached here')
                    # the first one is background
                    assert self.num_classes == labels.shape[1] + 1
                    labels_pos = x[:, 1] + 1
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([0, 1, 2, 3], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor(
                        [0, 1, 2, 3], device=device)
                sampled_box_regression = box_regression[sampled_pos_inds_subset[:, None], map_inds]
                sampled_box_target = regression_targets[sampled_pos_inds_subset]
                if not self.weight_box_loss:
                    box_loss = smooth_l1_loss(
                        sampled_box_regression,
                        sampled_box_target,
                        size_average=False,
                        beta=1,
                    )
                else:
                    matched_ious = cat(
                        [proposal.get_field("matched_ious") for proposal in proposals], dim=0
                    )
                    sampled_ious = matched_ious[sampled_pos_inds_subset]
                    box_loss = self.box_loss(sampled_box_regression,
                            sampled_box_target,
                            sampled_ious)
                box_loss = box_loss / labels.shape[0]
        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    classification_loss_type = cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_LOSS
    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg,
        classification_loss_type,
        num_classes,
        boundingbox_loss_type=cfg.MODEL.ROI_BOX_HEAD.BOUNDINGBOX_LOSS_TYPE,
    )

    return loss_evaluator
