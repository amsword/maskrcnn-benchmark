import torch


def register_custom_op():
    # experimenting custom op registration.
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze
    import torch.onnx.symbolic_helper as sym_help
    @parse_args('v', 'v', 'f')
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([2000], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args('v', 'v', 'f', 'i', 'i', 'i')
    def symbolic_roi_align(g, x, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
        srois = squeeze(g, select(g, rois, 1, g.op('Constant', value_t=torch.tensor([0], dtype=torch.long))), 1)
        batch_indices = g.op("Cast", srois, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        return g.op('RoiAlign', x, rois, batch_indices, spatial_scale_f=spatial_scale, output_height_i=pooled_height,
                    output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('roi_ops::nms', symbolic_multi_label_nms, 10)
    register_custom_op_symbolic('roi_ops::roi_align_forward', symbolic_roi_align, 10)
