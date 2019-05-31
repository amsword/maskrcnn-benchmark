from tqdm import tqdm
import torch
import json
import torchvision.transforms as transforms
import logging

from maskrcnn_benchmark.structures.bounding_box import BoxList

from qd.qd_pytorch import TSVSplitImage
from qd.tsv_io import TSVDataset


def merge_class_names_by_location_id(anno):
    if any('location_id' in a for a in anno):
        assert all('location_id' in a for a in anno)
        location_id_rect = [(a['location_id'], a) for a in anno]
        from qd.qd_common import list_to_dict
        location_id_to_rects = list_to_dict(location_id_rect, 0)
        merged_anno = []
        for _, rects in location_id_to_rects.items():
            import copy
            r = copy.deepcopy(rects[0])
            r['class'] = [r['class']]
            r['class'].extend((rects[i]['class'] for i in range(1,
                len(rects))))
            merged_anno.append(r)
        return merged_anno
    else:
        assert all('location_id' not in a for a in anno)
        for a in anno:
            a['class'] = [a['class']]
        return anno

class MaskTSVDataset(TSVSplitImage):

    """Docstring for MaskTSVDataset. """

    def __init__(self, data, split, version=0, transforms=None,
            multi_hot_label=False,
            cache_policy=None, labelmap=None,
            remove_images_without_annotations=True):
        # we will not use the super class's transform, but uses transforms
        # instead
        super(MaskTSVDataset, self).__init__(data, split, version=version,
                cache_policy=cache_policy, transform=None, labelmap=labelmap)
        self.transforms = transforms
        self.use_seg = False
        dataset = TSVDataset(data)
        assert dataset.has(split, 'hw')
        from qd.qd_pytorch import TSVSplitProperty
        self.hw_tsv = TSVSplitProperty(data, split, t='hw')
        self.all_key_hw = None
        self.dataset = dataset
        if remove_images_without_annotations:
            from qd.process_tsv import load_key_rects
            self.ensure_load_key_hw()
            key_rects = load_key_rects(dataset.iter_data(split, t='label',
                version=version))
            self.shuffle = [i for i, ((key, rects), (_, (h, w))) in enumerate(zip(key_rects,
                self.all_key_hw)) if self.will_non_empty(rects, w, h) > 0]
        else:
            self.shuffle = None
        self.multi_hot_label = multi_hot_label

    def ensure_load_key_hw(self):
        if self.all_key_hw is None:
            self.all_key_hw = []
            logging.info('loading hw')
            for i in tqdm(range(len(self.hw_tsv))):
                key, h, w = self.read_key_hw(i)
                self.all_key_hw.append((key, [h, w]))

    @property
    def id_to_img_map(self):
        self.ensure_load_key_hw()
        return {i: key for i, (key, _) in
                    enumerate(self.all_key_hw)}

    def get_keys(self):
        return [self.read_key_hw(i)[0] for i in range(len(self.hw_tsv))]

    def _tsvcol_to_label(self, col):
        anno = json.loads(col)
        return anno

    def will_non_empty(self, anno, w, h):
        # coco data has this kind of property
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]
        anno = [a for a in anno if a['class'] in self.label_to_idx]
        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (w, h), mode="xyxy")
        target = target.clip_to_image(remove_empty=True)
        return len(target) > 0

    def __getitem__(self, idx):
        if not self.multi_hot_label:
            return self.get_item_for_softmax(idx)
        else:
            return self.get_item_multi_hot_label(idx)

    def get_item_multi_hot_label(self, idx):
        if self.shuffle:
            idx = self.shuffle[idx]
        cv_im, anno, key = super(MaskTSVDataset, self).__getitem__(idx)
        # we randomly select the max_box results. in the future, we should put it
        # in the Transform
        max_box = 300

        img = transforms.ToPILImage()(cv_im)

        # coco data has this kind of property
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

        anno = [a for a in anno if a['class'] in self.label_to_idx]

        anno = merge_class_names_by_location_id(anno)
        # it will occupy 10G if it is 300 for one image.
        if len(anno) > max_box:
            import random
            random.shuffle(anno)
            anno = anno[:max_box]

        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")
        # the first one is for background
        classes = torch.zeros((len(anno), len(self.label_to_idx)))
        for i, obj in enumerate(anno):
            for c in obj['class']:
                classes[i, self.label_to_idx[c]] = 1
        target.add_field("labels", classes)

        assert not self.use_seg, 'not tested'
        #masks = [obj["segmentation"] for obj in anno]
        #from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
        #masks = SegmentationMask(masks, img.size)
        #target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_item_for_softmax(self, idx):
        if self.shuffle:
            idx = self.shuffle[idx]
        cv_im, anno, key = super(MaskTSVDataset, self).__getitem__(idx)
        # we randomly select the max_box results. in the future, we should put it
        # in the Transform
        max_box = 300
        # it will occupy 10G if it is 300 for one image.
        if len(anno) > max_box:
            import random
            random.shuffle(anno)
            anno = anno[:max_box]
        if any('location_id' in a for a in anno):
            # in this case, all locations should be unique for now.
            assert len(set(a['location_id'] for a in anno)) == len(anno)

        img = transforms.ToPILImage()(cv_im)

        # coco data has this kind of property
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

        anno = [a for a in anno if a['class'] in self.label_to_idx]

        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        # 0 is the background
        classes = [self.label_to_idx[obj["class"]] + 1 for obj in anno]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if self.use_seg:
            masks = [obj["segmentation"] for obj in anno]
            from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        if self.shuffle:
            return len(self.shuffle)
        else:
            return super(MaskTSVDataset, self).__len__()

    def get_img_info(self, index):
        if self.shuffle:
            index = self.shuffle[index]
        key, h, w = self.read_key_hw(index)
        result = {'height': h, 'width': w}
        return result

    def read_key_hw(self, idx_after_shuffle):
        key, str_hw = self.hw_tsv[idx_after_shuffle]
        hw = [int(x) for x in str_hw.split(' ')]
        return key, hw[0], hw[1]


