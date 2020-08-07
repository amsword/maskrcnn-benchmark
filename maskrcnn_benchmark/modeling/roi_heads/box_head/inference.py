# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
#from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms_no_convert_back
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_softnms

class BoxListSort(nn.Module):
    def __init__(self, max_proposals, score_field):
        super(BoxListSort, self).__init__()
        self.max_proposals = max_proposals
        self.score_field = score_field

    def forward(self, x):
        if len(x.bbox) <= self.max_proposals:
            return x
        _, idx = torch.topk(x.get_field(self.score_field),
                self.max_proposals)
        return x[idx]

class BoxListNMS(nn.Module):
    def __init__(self, thresh, max_proposals, score_field):
        super(BoxListNMS, self).__init__()
        self.thresh = thresh
        self.max_proposals = max_proposals
        self.score_field = score_field
        self.input_mode = 'xyxy'

    def forward(self, x):
        return  boxlist_nms_no_convert_back(x,
                self.thresh,
                max_proposals=self.max_proposals,
                score_field=self.score_field)

class BoxListSoftNMS(nn.Module):
    def __init__(self, thresh, max_proposals, score_field,
            score_thresh):
        super(BoxListSoftNMS, self).__init__()
        self.thresh = thresh
        self.score_field = score_field
        self.input_mode = 'xyxy'
        if max_proposals == -1:
            max_proposals = 10000000000;
        self.max_proposals = max_proposals
        self.score_thresh = score_thresh

    def forward(self, x):
        return boxlist_softnms(x, self.thresh,
                threshold=self.score_thresh,
                max_box=self.max_proposals,
                score_field=self.score_field)

class BoxListComposeHNMS(nn.Module):
    def __init__(self, nms_policy, max_proposals, score_field):
        super(BoxListComposeHNMS, self).__init__()
        from qd.hnms import MultiHashNMSAnyKPt
        if nms_policy.NUM == 0:
            hnms1 = None
        else:
            hnms1 = MultiHashNMSAnyKPt(
                    num=nms_policy.NUM,
                    w0=nms_policy.WH0,
                    h0=nms_policy.WH0,
                    alpha=nms_policy.ALPHA,
                    gamma=nms_policy.GAMMA,
                    rerank=False)
        if nms_policy.NUM2 == 0:
            hnms2 = None
        else:
            hnms2 = MultiHashNMSAnyKPt(
                    num=nms_policy.NUM2,
                    w0=nms_policy.WH0,
                    h0=nms_policy.WH0,
                    alpha=nms_policy.ALPHA2,
                    gamma=nms_policy.GAMMA2,
                    rerank=True,
                    rerank_iou=nms_policy.THRESH2)
        if nms_policy.COMPOSE_FINAL_RERANK:
            if nms_policy.COMPOSE_FINAL_RERANK_TYPE == 'softnms':
                rerank = BoxListSoftNMS(
                    nms_policy.THRESH,
                    score_thresh=0.,
                    max_proposals=max_proposals,
                    score_field=score_field)
            elif nms_policy.COMPOSE_FINAL_RERANK_TYPE == 'nms':
                rerank = BoxListNMS(nms_policy.THRESH,
                        max_proposals=max_proposals,
                        score_field=score_field)
            elif nms_policy.COMPOSE_FINAL_RERANK_TYPE == 'sort':
                rerank = BoxListSort(max_proposals, score_field)
            else:
                raise NotImplementedError(nms_policy.COMPOSE_FINAL_RERANK_TYPE)
        else:
            rerank = None
        self.hnms1 = hnms1
        self.hnms2 = hnms2
        self.rerank = rerank
        self.nms_policy = nms_policy

        self.score_field = score_field
        self.max_proposals = max_proposals
        self.input_mode = 'cxywh'

    def forward(self, boxlist):
        if self.hnms1 is not None or self.hnms2 is not None:
            #origin_mode = boxlist.mode
            boxlist = boxlist.convert('cxywh')
            rects = boxlist.bbox
            scores = boxlist.get_field(self.score_field)
        if self.hnms1 is not None and self.hnms2 is not None:
            keep = self.hnms1(rects, scores)
            keep2 = self.hnms2(rects[keep], scores[keep])
            keep = keep[keep2]
        elif self.hnms1 is not None:
            keep = self.hnms1(rects, scores)
        elif self.hnms2 is not None:
            keep = self.hnms2(rects, scores)
        else:
            keep = None

        if self.rerank is not None:
            if keep is not None:
                boxlist = boxlist[keep]
            boxlist = self.rerank(boxlist)
        else:
            if self.max_proposals > 0 and len(keep) > self.max_proposals:
                _, idx = scores[keep].topk(self.max_proposals)
                keep = keep[idx]
            boxlist = boxlist[keep]
        return boxlist

def create_nms_func(nms_policy, score_thresh=0.,
        score_field='scores', max_proposals=-1):
    if nms_policy.TYPE == 'nms':
        #from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
        #return lambda x: boxlist_nms(x, nms_policy.THRESH,
                #max_proposals=max_proposals, score_field=score_field)
        return BoxListNMS(nms_policy.THRESH,
                max_proposals=max_proposals, score_field=score_field)
    elif nms_policy.TYPE == 'softnms':
        #from maskrcnn_benchmark.structures.boxlist_ops import boxlist_softnms
        #if max_proposals == -1:
            #max_proposals = 10000000000;
        #return lambda x: boxlist_softnms(x, nms_policy.THRESH,
                #threshold=score_thresh, max_box=max_proposals)
        return BoxListSoftNMS(nms_policy.THRESH,
                max_proposals=max_proposals,
                score_field=score_field,
                score_thresh=score_thresh)
    elif nms_policy.TYPE == 'single_hnms':
        raise Exception
    elif nms_policy.TYPE == 'multi_hnms_any':
        raise Exception
    elif nms_policy.TYPE == 'multi_hnms_all':
        raise Exception
    elif nms_policy.TYPE == 'multi_id_hnms_any':
        raise Exception
    elif nms_policy.TYPE == 'multi_rerank_hnms_any':
        raise Exception
    elif nms_policy.TYPE == 'multi_rerank_hnms_anyk':
        raise Exception
    elif nms_policy.TYPE == 'compose_hnms_anyk':
        raise Exception
    elif nms_policy.TYPE == 'compose_hnms_pt':
        return BoxListComposeHNMS(nms_policy, max_proposals,
                score_field)

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        classification_activate='softmax',
        nms_policy=None,
        cfg=None,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        if nms_policy is not None:
            if nms_policy.TYPE != 'nms':
                logging.info('apply {} rather than standard nms'.format(nms_policy.TYPE))
            elif nms_policy.THRESH != self.nms:
                logging.info('nms threshold = {}'.format(nms_policy.THRESH))
        self.nms_func = create_nms_func(nms_policy, score_thresh)
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.nms_on_max_conf_agnostic = cfg.MODEL.ROI_HEADS.NMS_ON_MAX_CONF_AGNOSTIC
        if not self.cls_agnostic_bbox_reg:
            assert not self.nms_on_max_conf_agnostic
        self.classification_activate = classification_activate
        if classification_activate == 'softmax':
            self.logits_to_prob = lambda x: F.softmax(x, -1)
            self.cls_start_idx = 1
        elif classification_activate == 'sigmoid':
            self.logits_to_prob = torch.nn.Sigmoid()
            self.cls_start_idx = 0
        elif classification_activate == 'tree':
            from qd.layers import SoftMaxTreePrediction
            self.logits_to_prob = SoftMaxTreePrediction(
                    tree=cfg.MODEL.ROI_BOX_HEAD.TREE_0_BKG,
                    pred_thresh=self.score_thresh)
            self.cls_start_idx = 1
        else:
            raise NotImplementedError()
        self.output_feature = cfg.TEST.OUTPUT_FEATURE
        if self.output_feature:
            # needed to extract features when they have not been pooled yet
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, boxes, features):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        if self.output_feature:
            if len(features.shape) > 2:
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
        class_logits, box_regression = x
        class_prob = self.logits_to_prob(class_logits)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        decode2cxywh = False
        if decode2cxywh:
            proposals = self.box_coder.decode2cxywh(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
            mode = 'cxywh'
        else:
            proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
            mode = 'xyxy'

        if self.cls_agnostic_bbox_reg and not self.nms_on_max_conf_agnostic:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        if self.output_feature:
            features = features.split(boxes_per_image, dim=0)

        results = []
        for i, (prob, boxes_per_img, image_shape) in enumerate(zip(
            class_prob, proposals, image_shapes
        )):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape,
                    mode=mode)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if self.nms_on_max_conf_agnostic:
                boxlist = self.filter_results_nms_on_max(boxlist, num_classes)
            else:
                feature = features[i] if self.output_feature else None
                boxlist = self.filter_results(boxlist, num_classes,
                        feature)
                #boxlist = self.filter_results_parallel(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape, mode):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode=mode)
        boxlist.add_field("scores", scores)
        return boxlist

    def prepare_empty_boxlist(self, boxlist):
        device = boxlist.bbox.device
        boxlist_empty = BoxList(torch.zeros((0,4)).to(device), boxlist.size,
                mode='xyxy')
        boxlist_empty.add_field("scores", torch.Tensor([]).to(device))
        boxlist_empty.add_field("labels", torch.full((0,), -1,
                dtype=torch.int64, device=device))
        return boxlist_empty

    def filter_results_parallel(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist

        # cpu version is faster than gpu. revert it to gpu only by verifying

        boxlist = boxlist.to('cpu')

        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        all_cls_boxlist_for_class = []
        for j in range(self.cls_start_idx, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            if len(inds) == 0:
                continue
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            all_cls_boxlist_for_class.append((j, boxlist_for_class))

        all_boxlist_for_class = [boxlist_for_class for _, boxlist_for_class in
            all_cls_boxlist_for_class]
        from qd.qd_common import parallel_map

        all_boxlist_for_class = parallel_map(self.nms_func, all_boxlist_for_class)

        for i, boxlist_for_class in enumerate(all_boxlist_for_class):
            j = all_cls_boxlist_for_class[i][0]
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        if len(result) > 0:
            result = cat_boxlist(result)
        else:
            return self.prepare_empty_boxlist(boxlist)

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

    def filter_results_nms_on_max(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist

        # cpu version is faster than gpu. revert it to gpu only by verifying
        boxlist = boxlist.to('cpu')

        boxes = boxlist.bbox.reshape(-1, 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        result = []
        max_scores, _ = scores[:, self.cls_start_idx:].max(dim=1, keepdim=False)
        keep = (max_scores > self.score_thresh).nonzero().squeeze(-1)
        if len(keep) == 0:
            return self.prepare_empty_boxlist(boxlist)
        boxes, scores, max_scores = boxes[keep], scores[keep], max_scores[keep]

        boxlist = BoxList(boxes, boxlist.size, mode=boxlist.mode)
        boxlist.add_field("scores", max_scores)
        boxlist.add_field('original_scores', scores)
        boxlist = self.nms_func(boxlist)

        scores = boxlist.get_field('original_scores')
        all_idxrow_idxcls = (scores[:, self.cls_start_idx:] > self.score_thresh).nonzero()
        all_idxrow_idxcls[:, 1] += self.cls_start_idx

        boxes = boxlist.bbox
        boxes = boxes[all_idxrow_idxcls[:, 0]]
        if boxes.dim() == 1:
            boxes = boxes[None, :]
        labels = all_idxrow_idxcls[:, 1]
        scores =  scores[all_idxrow_idxcls[:, 0], all_idxrow_idxcls[:, 1]]
        result = BoxList(boxes, boxlist.size, mode=boxlist.mode)
        result.add_field("labels", labels)
        result.add_field("scores", scores)

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


    def filter_results(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist

        # cpu version is faster than gpu. revert it to gpu only by verifying
        boxlist = boxlist.to('cpu')

        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(self.cls_start_idx, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            if len(inds) == 0:
                continue
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size,
                    mode=boxlist.mode)
            boxlist_for_class.add_field("scores", scores_j)
            if feature is not None:
                boxlist_for_class.add_field('box_features', feature[inds])
            boxlist_for_class = self.nms_func(boxlist_for_class)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)
        if len(result) > 0:
            result = cat_boxlist(result)
        else:
            return self.prepare_empty_boxlist(boxlist)

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    classification_activate = cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE
    nms_policy = cfg.MODEL.ROI_HEADS.NMS_POLICY

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        classification_activate=classification_activate,
        nms_policy=nms_policy,
        cfg=cfg,
    )
    return postprocessor
