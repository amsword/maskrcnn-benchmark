1. read the image
    ```
    original = cv2.imread('/home/jianfw/data/sample_images/TaylorSwift.jpg',
            cv2.IMREAD_COLOR)
    # assume the size of original is 900x1600x3
    ```

2. preprocessing the image
    ```
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
    ```
    # SIZE_DIVISIBILITY = 32
    image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
    ```
    The implementation of to_image_list(). Only the key component is listed here
    ```
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
    ```
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
    ```
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
    other stages. Note the first residual block to decrease the spatial resolution
    and increase the channels.
    ```
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
    ```
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
    ```
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
8. Next module is the RPNModule
    ```
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
    ```
