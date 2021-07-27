import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import VOCDataset
from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer


@torch.no_grad()
def run_demo(cfg, backbone_ckpt, boxhead_ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    backbone = build_backbone(cfg)
    boxhead = build_box_head(cfg)
    backbone = backbone.to(device)
    boxhead = boxhead.to(device)
    backbone_checkpointer = CheckPointer(backbone, 'backbone_last_checkpoint.txt', save_dir=cfg.OUTPUT_DIR)
    boxhead_checkpointer = CheckPointer(boxhead, 'boxhead_last_checkpoint.txt', save_dir=cfg.OUTPUT_DIR)
    backbone_checkpointer.load(backbone_ckpt, use_latest=backbone_ckpt is None)
    boxhead_checkpointer.load(boxhead_ckpt, use_latest=boxhead_ckpt is None)
    backbone_weight_file = backbone_ckpt if backbone_ckpt else backbone_checkpointer.get_checkpoint_file()
    print('Loaded backbone weights from {}'.format(backbone_weight_file))
    boxhead_weight_file = boxhead_ckpt if boxhead_ckpt else boxhead_checkpointer.get_checkpoint_file()
    print('Loaded boxhead weights from {}'.format(boxhead_weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    backbone.eval()
    boxhead.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        features = backbone(images.to(device))
        result = boxhead(features)[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="configs/vgg_ssd300_voc0712_feature_alignment.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--backbone_ckpt", type=str, default='outputs/vgg_ssd300_voc0712_feature_alignment_domain_lambda0.5/'
                                                             'backbone_000500.pth', help="Trained weights.")
    parser.add_argument("--boxhead_ckpt", type=str, default='outputs/vgg_ssd300_voc0712_feature_alignment_domain_lambda0.5/'
                                                             'boxhead_000500.pth', help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             backbone_ckpt=args.backbone_ckpt,
             boxhead_ckpt=args.boxhead_ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
