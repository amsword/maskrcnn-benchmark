1. read the image
    ```python
    original = cv2.imread('/home/jianfw/data/sample_images/TaylorSwift.jpg',
            cv2.IMREAD_COLOR)
    # assume the size of original is 900x1600x3
    ```

2. preprocessing the image
    ```python
    if cfg.INPUT.TO_BGR255: # this is True
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
    
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,  #[102.9801, 115.9465, 122.7717]
        std=cfg.INPUT.PIXEL_STD, # [1.0, 1.0, 1.0]
    )
    
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(self.min_image_size), # min_image_size = 800
            T.ToTensor(), # after this, the size of the image is 3xHxW
            to_bgr_transform,
            normalize_transform,
        ]
    )
    image = transforms(original_image)
    ```

3. convert to ImageList
    - find an image size so that both the height and the width are minimal to be
larger than or equal to the original image size and is divisible by 32. 
    - create a convas of the image size
    - put the image on the top-left corner of the canvas
    - the type of ImageList contains two fields.
        - tensors: the enlarged tensor to contain all the images. In inference,
        it is 1 image. 
        - image_sizes: it is the original image size
    ```python
    # SIZE_DIVISIBILITY = 32
    image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
    ```
    The implementation of to_image_list(). Only the key component is listed here
    ```python
    def to_image_list(tensors, size_divisible=0):
        if isinstance(tensors, torch.Tensor) and size_divisible > 0:
            tensors = [tensors]
    
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    
        if size_divisible > 0:
            import math
    
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)
    
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    
        image_sizes = [im.shape[-2:] for im in tensors]
    
        return ImageList(batched_imgs, image_sizes)
    ```

4. feed the image to the model.backbone.body.stem
    ```python
    (stem): StemWithFixedBatchNorm(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): FrozenBatchNorm2d()
    )
    class FrozenBatchNorm2d(nn.Module):
        def __init__(self, n):
            super(FrozenBatchNorm2d, self).__init__()
            self.register_buffer("weight", torch.ones(n))
            self.register_buffer("bias", torch.zeros(n))
            self.register_buffer("running_mean", torch.zeros(n))
            self.register_buffer("running_var", torch.ones(n))
    
        def forward(self, x):
            # effectively: weight * (x - mean) / sqrt(var) + bias for each element
            scale = self.weight * self.running_var.rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
    ```

5. feed the output of stem to multiple stages. Each stage increases the
   channels but decrease the spatial resolution. Each stage is named
   'layer{}'.format(i) where i belongs to [1-4].
    ```python
    ipdb> pp self.model.backbone.body.layer1
    Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    class BottleneckWithFixedBatchNorm(nn.Module):
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu_(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
            out = F.relu_(out)
    
            out0 = self.conv3(out)
            out = self.bn3(out0)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = F.relu_(out)
    
            return out
    ```
    Note the first residual block of higher stages (2 - 4) use stride=2 to decrease the spatial resolution. The channels is also increased
    ```python
    (layer2): Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (3): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    (layer3): Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (3): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (4): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (5): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    (layer4): Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    ```
6. The output of each stage (layer1, layer2, layer3, layer4) are collected together. 
    ```python
    class ResNet(nn.Module):
        def forward(self, x):
            outputs = []
            x = self.stem(x)
            for stage_name in self.stages: # layer1, layer2, layer3, layer4
                x = getattr(self, stage_name)(x)
                if self.return_features[stage_name]:
                    outputs.append(x)
            return outputs
    ```

7. The next is to feed the data to FPN
    ```python
    ipdb> pp self.model.backbone.fpn
    FPN(
      (fpn_inner1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fpn_inner2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fpn_inner3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fpn_inner4): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (top_blocks): LastLevelMaxPool()
    )
    class FPN(nn.Module):
        def forward(self, x):
            """
            Arguments:
                x (list[Tensor]): feature maps for each feature level.
            Returns:
                results (tuple[Tensor]): feature maps after FPN layers.
                    They are ordered from highest resolution first.
            """
            last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
            results = []
            results.append(getattr(self, self.layer_blocks[-1])(last_inner))
            for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
            ):
                inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
                inner_lateral = getattr(self, inner_block)(feature)
                # TODO use size instead of scale to make it robust to different sizes
                # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
                # mode='bilinear', align_corners=False)
                last_inner = inner_lateral + inner_top_down
                results.insert(0, getattr(self, layer_block)(last_inner))
    
            if self.top_blocks is not None:
                last_results = self.top_blocks(results[-1])
                results.extend(last_results)
    
            return tuple(results)

    class LastLevelMaxPool(nn.Module):
        def forward(self, x):
            return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
    ```
    Note that the output of FPN will have 5 rather than 4 features because
         of `top_blocks`.
8. Next module is the RPNModule
    ```python
    ipdb> pp self.model.rpn
    RPNModule(
      (anchor_generator): AnchorGenerator(
        (cell_anchors): BufferList()
      )
      (head): RPNHead(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
      )
      (box_selector_train): RPNPostProcessor()
      (box_selector_test): RPNPostProcessor()
    )
    class RPNModule(torch.nn.Module):
        """
        Module for RPN computation. Takes feature maps from the backbone and RPN
        proposals and losses. Works for both FPN and non-FPN.
        """
        def forward(self, images, features, targets=None):
            """
            Arguments:
                images (ImageList): images for which we want to compute the predictions
                features (list[Tensor]): features computed from the images that are
                    used for computing the predictions. Each tensor in the list
                    correspond to different feature levels
                targets (list[BoxList): ground-truth boxes present in the image (optional)
    
            Returns:
                boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                    image.
                losses (dict[Tensor]): the losses for the model during training. During
                    testing, it is an empty dict.
            """
            objectness, rpn_box_regression = self.head(features)
            anchors = self.anchor_generator(images, features)
    
            if self.training:
                return self._forward_train(anchors, objectness, rpn_box_regression, targets)
            else:
                return self._forward_test(anchors, objectness, rpn_box_regression)
    
        def _forward_test(self, anchors, objectness, rpn_box_regression):
            boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
            if self.cfg.MODEL.RPN_ONLY:
                # For end-to-end models, the RPN proposals are an intermediate state
                # and don't bother to sort them in decreasing score order. For RPN-only
                # models, the proposals are the final output and we return them in
                # high-to-low confidence order.
                inds = [
                    box.get_field("objectness").sort(descending=True)[1] for box in boxes
                ]
                boxes = [box[ind] for box, ind in zip(boxes, inds)]
            return boxes, {}
    ```
    - `self.head(features)`: We have 5 feature maps. Each feature map has 256
    channels with different spatial resolution. Each feature map is passed to a
    shared head module. Note that the head is shared across the 5 feature maps.
    It is not optimized for each individual feature map. For each head, the
    featrure map is first processed by a conv layer with 256 channels and 3x3
    as kernel size. Then, the classification prediction and teh bounding box
    regression output is generated by the 1x1 conv layer. Each feature map is
    responsible for only 1 scale, but 3 different aspect ratios. Thus, the
    channel for the classification is 3 and for the bounding box is $3\times 4$.
    The output is a tuple with 2 elements. The first is a list of the
    classification and the second one is the bounding box regression. Both
    lists have the same length with the features.
    - `anchors = self.anchor_generator(images, features)`: Only the shapes are
    used. The feature itself is not used. If there are B images, the return
    value is also a list with B elements, each of which is a list of BoxList
    for each feature map.
    ```python
    class AnchorGenerator(nn.Module):

        def forward(self, image_list, feature_maps):
            grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
            anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
            anchors = []
            for i, (image_height, image_width) in enumerate(image_list.image_sizes):
                anchors_in_image = []
                for anchors_per_feature_map in anchors_over_all_feature_maps:
                    boxlist = BoxList(
                        anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                    )
                    self.add_visibility_to(boxlist)
                    anchors_in_image.append(boxlist)
                anchors.append(anchors_in_image)
            return anchors

        def grid_anchors(self, grid_sizes):
            anchors = []
            # self.strides = [4, 8, 16, 32, 64]
            for size, stride, base_anchors in zip(
                grid_sizes, self.strides, self.cell_anchors
            ):
                grid_height, grid_width = size
                device = base_anchors.device
                shifts_x = torch.arange(
                    0, grid_width * stride, step=stride, dtype=torch.float32, device=device
                )
                shifts_y = torch.arange(
                    0, grid_height * stride, step=stride, dtype=torch.float32, device=device
                )
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
                shift_x = shift_x.reshape(-1)
                shift_y = shift_y.reshape(-1)
                shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

                anchors.append(
                    (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
                )

            return anchors
    ```
    - `self._forward_test(anchors, objectness, rpn_box_regression)`: 
       ```python
       class RPNPostProcessor(torch.nn.Module):
           def forward(self, anchors, objectness, box_regression, targets=None):
               """
               Arguments:
                   anchors: list[list[BoxList]] - Each element corresponds to one
               image. Each element is for multiple features
                   objectness: list[tensor] - Each element corresponds to one feature
               map
                   box_regression: list[tensor] - Each element corresponds to one
               feature map
       
               Returns:
                   boxlists (list[BoxList]): the post-processed anchors, after
                       applying box decoding and NMS
               """
               sampled_boxes = []
               num_levels = len(objectness)
               anchors = list(zip(*anchors))
               for a, o, b in zip(anchors, objectness, box_regression):
                   # one example: a=(BoxList,); o=1x3x200x360; b=1x12x200x360
                   sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))
       
               boxlists = list(zip(*sampled_boxes))
               boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
       
               if num_levels > 1:
                   boxlists = self.select_over_all_levels(boxlists)
       
               # append ground-truth bboxes to proposals
               if self.training and targets is not None:
                   boxlists = self.add_gt_proposals(boxlists, targets)
       
               return boxlists
       
           def forward_for_single_feature_map(self, anchors, objectness, box_regression):
               """
               Arguments:
                   anchors: list[BoxList]
                   objectness: tensor of size N, A, H, W
                   box_regression: tensor of size N, A * 4, H, W
               """
               device = objectness.device
               N, A, H, W = objectness.shape
       
               # put in the same format as anchors
               objectness = objectness.permute(0, 2, 3, 1).reshape(N, -1)
               objectness = objectness.sigmoid()
               box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
               box_regression = box_regression.reshape(N, -1, 4)
       
               num_anchors = A * H * W
               
               # self.pre_nms_top_n = 1000
               pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
               objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
       
               batch_idx = torch.arange(N, device=device)[:, None]
               box_regression = box_regression[batch_idx, topk_idx]
       
               image_shapes = [box.size for box in anchors]
               concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
               concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
       
               proposals = self.box_coder.decode(
                   box_regression.view(-1, 4), concat_anchors.view(-1, 4)
               )
       
               proposals = proposals.view(N, -1, 4)
       
               result = []
               for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
                   boxlist = BoxList(proposal, im_shape, mode="xyxy")
                   boxlist.add_field("objectness", score)
                   boxlist = boxlist.clip_to_image(remove_empty=False)
                   boxlist = remove_small_boxes(boxlist, self.min_size)
                   boxlist = boxlist_nms(
                       boxlist,
                       self.nms_thresh,
                       max_proposals=self.post_nms_top_n,
                       score_field="objectness",
                   )
                   result.append(boxlist)
               return result
       class BoxCoder(object):
           def decode(self, rel_codes, boxes):
               """
               From a set of original boxes and encoded relative box offsets,
               get the decoded boxes.
       
               Arguments:
                   rel_codes (Tensor): encoded boxes
                   boxes (Tensor): reference boxes.
               """
       
               boxes = boxes.to(rel_codes.dtype)
       
               TO_REMOVE = 1  # TODO remove
               widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
               heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
               ctr_x = boxes[:, 0] + 0.5 * widths
               ctr_y = boxes[:, 1] + 0.5 * heights
       
               wx, wy, ww, wh = self.weights
               dx = rel_codes[:, 0::4] / wx
               dy = rel_codes[:, 1::4] / wy
               dw = rel_codes[:, 2::4] / ww
               dh = rel_codes[:, 3::4] / wh
       
               # Prevent sending too large values into torch.exp()
               dw = torch.clamp(dw, max=self.bbox_xform_clip)
               dh = torch.clamp(dh, max=self.bbox_xform_clip)
       
               pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
               pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
               pred_w = torch.exp(dw) * widths[:, None]
               pred_h = torch.exp(dh) * heights[:, None]
       
               pred_boxes = torch.zeros_like(rel_codes)
               # x1
               pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
               # y1
               pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
               # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
               pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
               # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
               pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
       
               return pred_boxes
       ```

9. After RPNModule, we have a bunch of proposals. The next is the CombinedROIHeads, which contains the box head and the mask head. Let's first focus on the box head. 
    ```python
    ipdb> pp self.roi_heads.box
    ROIBoxHead(
      (feature_extractor): FPN2MLPFeatureExtractor(
        (pooler): Pooler(
          (poolers): ModuleList(
            (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2)
            (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2)
            (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2)
            (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2)
          )
        )
        (fc6): Linear(in_features=12544, out_features=1024, bias=True)
        (fc7): Linear(in_features=1024, out_features=1024, bias=True)
      )
      (predictor): FPNPredictor(
        (cls_score): Linear(in_features=1024, out_features=81, bias=True)
        (bbox_pred): Linear(in_features=1024, out_features=324, bias=True)
      )
      (post_processor): PostProcessor()
    )
    class ROIBoxHead(torch.nn.Module):
        def forward(self, features, proposals, targets=None):
            """
            Arguments:
                features (list[Tensor]): feature-maps from possibly several levels
                proposals (list[BoxList]): proposal boxes
                targets (list[BoxList], optional): the ground-truth targets.
    
            Returns:
                x (Tensor): the result of the feature extractor
                proposals (list[BoxList]): during training, the subsampled proposals
                    are returned. During testing, the predicted boxlists are returned
                losses (dict[Tensor]): During training, returns the losses for the
                    head. During testing, returns an empty dict.
            """
    
            if self.training:
                # Faster R-CNN subsamples during training the proposals with a fixed
                # positive / negative ratio
                with torch.no_grad():
                    proposals = self.loss_evaluator.subsample(proposals, targets)
    
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = self.feature_extractor(features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x)
    
            if not self.training:
                result = self.post_processor((class_logits, box_regression), proposals)
                return x, result, {}
    
            loss_classifier, loss_box_reg = self.loss_evaluator(
                [class_logits], [box_regression]
            )
            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            )
    
    class FPN2MLPFeatureExtractor(nn.Module):
        def forward(self, x, proposals):
            # ipdb> [y.shape for y in x]
            # [torch.Size([1, 256, 200, 360]), torch.Size([1, 256, 100, 180]), torch.Size([1, 256, 50, 90]), torch.Size([1, 256, 25, 45]), torch.Size([1, 256, 13, 23])]
            # ipdb> pp proposals
            # [BoxList(num_boxes=1000, image_width=1422, image_height=800, mode=xyxy)]
            x = self.pooler(x, proposals)
            x = x.view(x.size(0), -1)
    
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
    
    class Pooler(nn.Module):
        def forward(self, x, boxes):
            """
            Arguments:
                x (list[Tensor]): feature maps for each level
                boxes (list[BoxList]): boxes to be used to perform the pooling operation.
            Returns:
                result (Tensor)
            """
            num_levels = len(self.poolers)
            rois = self.convert_to_roi_format(boxes)
            if num_levels == 1:
                return self.poolers[0](x[0], rois)
    
            levels = self.map_levels(boxes)
    
            num_rois = len(rois)
            num_channels = x[0].shape[1]
            output_size = self.output_size[0]
    
            dtype, device = x[0].dtype, x[0].device
            result = torch.zeros(
                (num_rois, num_channels, output_size, output_size),
                dtype=dtype,
                device=device,
            )
            for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
                idx_in_level = torch.nonzero(levels == level).squeeze(1)
                rois_per_level = rois[idx_in_level]
                result[idx_in_level] = pooler(per_level_feature, rois_per_level)
    
            return result

    class PostProcessor(nn.Module):
        """
        From a set of classification scores, box regression and proposals,
        computes the post-processed boxes, and applies NMS to obtain the
        final results
        """
    
        def forward(self, x, boxes):
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
            class_logits, box_regression = x
            class_prob = F.softmax(class_logits, -1)
    
            # TODO think about a representation of batch of boxes
            image_shapes = [box.size for box in boxes]
            boxes_per_image = [len(box) for box in boxes]
            concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
    
            proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
    
            num_classes = class_prob.shape[1]
    
            proposals = proposals.split(boxes_per_image, dim=0)
            class_prob = class_prob.split(boxes_per_image, dim=0)
    
            results = []
            for prob, boxes_per_img, image_shape in zip(
                class_prob, proposals, image_shapes
            ):
                boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = self.filter_results(boxlist, num_classes)
                results.append(boxlist)
            return results
        def prepare_boxlist(self, boxes, scores, image_shape):
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
            boxlist = BoxList(boxes, image_shape, mode="xyxy")
            boxlist.add_field("scores", scores)
            return boxlist

        def filter_results(self, boxlist, num_classes):
            """Returns bounding-box detection results by thresholding on scores and
            applying non-maximum suppression (NMS).
            """
            # unwrap the boxlist to avoid additional overhead.
            # if we had multi-class NMS, we could perform this directly on the boxlist
            boxes = boxlist.bbox.reshape(-1, num_classes * 4)
            scores = boxlist.get_field("scores").reshape(-1, num_classes)

            device = scores.device
            result = []
            # Apply threshold on detection probabilities and apply NMS
            # Skip j = 0, because it's the background class
            inds_all = scores > self.score_thresh
            for j in range(1, num_classes):
                inds = inds_all[:, j].nonzero().squeeze(1)
                scores_j = scores[inds, j]
                boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms, score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
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
    ```
