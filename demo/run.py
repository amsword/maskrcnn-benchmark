from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os
import sys
sys.path.append('/home/jianfw/code/quickdetection/scripts')

from qd_common import get_mpi_rank, get_mpi_size


def mpi_predict():
    mpi_rank = get_mpi_rank()
    mpi_size = get_mpi_size()
    
    curr_gpu = mpi_rank // 8
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(curr_gpu)
    
    config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    #cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    coco_demo = COCODemo(
    	cfg,
    	min_image_size=800,
    	confidence_threshold=0.5,
    	)
    import cv2
    
    
    from process_image import show_image, draw_bb
    from qd_common import img_from_base64, json_dump
    from tsv_io import tsv_writer, TSVDataset
    from qd_util import job_split
    from tqdm import tqdm
    
    output_format = \
        'home/jianfw/code/quickdetection/output/OpenImageV4_448_X_MaskRCNNBenchmark/snapshot/model_iter_0.caffemodel.{}.{}.predict'
    data = 'OpenImageV4_448'
    dataset = TSVDataset(data)
    debug = True
    split = 'test'
    total = dataset.num_rows(split)
    all_jobs = job_split(total, mpi_size)
    curr_jobs = all_jobs[mpi_rank]
    
    output_format = output_format + '.' + str(mpi_rank) + '.' + str(mpi_size)
    
    def gen_rows():
        for key, str_rects, str_im in tqdm(dataset.iter_data(split,
            filter_idx=curr_jobs)):
            im = img_from_base64(str_im)
            predictions = coco_demo.compute_prediction(im)
            predictions = coco_demo.select_top_predictions(predictions)
            scores = predictions.get_field("scores").tolist()
            labels = predictions.get_field("labels").tolist()
            labels = [coco_demo.CATEGORIES[i] for i in labels]
            boxes = predictions.bbox
            rects = []
            import torch
            for box, score, label in zip(boxes, scores, labels):
                box = box.to(torch.int64)
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            
                r = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                rect = {'class': label, 'conf': score, 'rect': r}
                rects.append(rect)
            if debug:
                import numpy as np
                im_mask = np.copy(im)
                draw_bb(im_mask, [r['rect'] for r in rects], [r['class'] for r in rects],
                        [r['conf'] for r in rects])
                import json
                origin_rects = json.loads(str_rects)
                draw_bb(im, [r['rect'] for r in origin_rects], 
                        [r['class'] for r in origin_rects])
                from process_image import show_images
                show_images([im, im_mask], 1, 2)
            for r in rects:
                r['from'] = 'maskrcnn_benchmark.demo'
            yield key, json_dump(rects)
    tsv_writer(gen_rows(), output_format.format(data, split))

def single_predict():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config_file = "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    #cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    coco_demo = COCODemo(
    	cfg,
    	min_image_size=800,
    	confidence_threshold=0.5,
    	)
    import cv2
    from process_image import show_image, draw_bb
    
    im = cv2.imread('/home/jianfw/data/sample_images/TaylorSwift.jpg',
            cv2.IMREAD_COLOR)
    predictions = coco_demo.compute_prediction(im)
    predictions = coco_demo.select_top_predictions(predictions)
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [coco_demo.CATEGORIES[i] for i in labels]
    boxes = predictions.bbox
    rects = []
    import torch
    for box, score, label in zip(boxes, scores, labels):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    
        r = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        rect = {'class': label, 'conf': score, 'rect': r}
        rects.append(rect)
    import numpy as np
    im_mask = np.copy(im)
    draw_bb(im_mask, [r['rect'] for r in rects], [r['class'] for r in rects],
            [r['conf'] for r in rects])
    import json
    from process_image import show_images
    show_images([im, im_mask], 1, 2)


if __name__ == '__main__':
    #mpi_predict()
    single_predict()
    pass
    


