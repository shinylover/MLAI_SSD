import argparse
import logging
import os

import torch
import torch.utils.data

from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.utils import dist_util
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger


def evaluation(cfg, backbone_ckpt, boxhead_ckpt, distributed):
    logger = logging.getLogger("SSD.inference")

    backbone = build_backbone(cfg)
    boxhead = build_box_head(cfg)

    backbone_checkpointer = CheckPointer(backbone, 'backbone_last_checkpoint.txt', save_dir=cfg.OUTPUT_DIR, logger=logger)
    boxhead_checkpointer = CheckPointer(boxhead, 'boxhead_last_checkpoint.txt', save_dir=cfg.OUTPUT_DIR, logger=logger)

    device = torch.device(cfg.MODEL.DEVICE)
    backbone = backbone.to(device)
    boxhead = boxhead.to(device)
    backbone_checkpointer.load(backbone_ckpt, use_latest=backbone_ckpt is None)
    boxhead_checkpointer.load(boxhead_ckpt, use_latest=boxhead_ckpt is None)
    do_evaluation(cfg, backbone, boxhead, distributed)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="configs/vgg_ssd300_voc0712_baseline.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--boxhead_ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, backbone_ckpt=args.ckpt, boxhead_ckpt=args.boxhead_ckpt, distributed=distributed)


if __name__ == '__main__':
    main()
