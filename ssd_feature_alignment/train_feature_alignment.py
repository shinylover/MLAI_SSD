import argparse
import logging
import os

import torch

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader, make_target_data_loader
from ssd.engine.feature_alignment_trainer import do_train
from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.modeling.domain_discriminator import DomainDiscriminator
from ssd.solver.build import make_optimizer, make_adam_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool


#我们的方法，通过对抗思想进行特征的对齐
#与baseline相比，此时新增了两个域分类器
#一个域分类器第1维表示源域，第二维表示目标域
#对应的，第二个域分类器第1维表示目标域，第二维表示源域
#训练需要将baseline的模型参数先加载进来
#对于第一个域分类器，能够将源域的样本分到第一维，将目标域的样本分到第二维
#对于特征提取器而言，它希望提取的特征能够尽可能骗过域分类器，将源域的分到第二维，同样的，将目标域分到第一维
#第二个域分类器一样的思想
def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    backbone = build_backbone(cfg)
    boxhead = build_box_head(cfg)
    domaindiscriminator = DomainDiscriminator()#域判别器

    device = torch.device(cfg.MODEL.DEVICE)
    backbone = backbone.to(device)
    boxhead = boxhead.to(device)
    domaindiscriminator = domaindiscriminator.to(device)

    backbone_lr = cfg.SOLVER.BACKBONELR
    backbone_optimizer = make_optimizer(cfg, backbone, backbone_lr)
    boxhead_lr = cfg.SOLVER.BOXHEADLR
    boxhead_optimizer = make_optimizer(cfg, boxhead, boxhead_lr)
    domaindiscriminator_lr = cfg.SOLVER.DOMAINDISCRIMINATORLR
    domaindiscriminator_optimizer = make_adam_optimizer(cfg, domaindiscriminator, domaindiscriminator_lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    backbone_scheduler = make_lr_scheduler(cfg, backbone_optimizer, milestones)
    boxhead_scheduler = make_lr_scheduler(cfg, boxhead_optimizer, milestones)
    domaindiscriminator_scheduler = make_lr_scheduler(cfg, domaindiscriminator_optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    backbone_checkpointer = CheckPointer(backbone, 'backbone_last_checkpoint.txt', backbone_optimizer,
                                         backbone_scheduler, cfg.OUTPUT_DIR, save_to_disk,
                                         logger)
    boxhead_checkpointer = CheckPointer(boxhead, 'boxhead_last_checkpoint.txt', boxhead_optimizer, boxhead_scheduler,
                                        cfg.OUTPUT_DIR, save_to_disk,
                                        logger)
    domaindiscriminator_checkpointer = CheckPointer(domaindiscriminator, 'domaindiscriminator_last_checkpoint.txt',
                                                    domaindiscriminator_optimizer, domaindiscriminator_scheduler,
                                                  cfg.OUTPUT_DIR, save_to_disk, logger)

    backbone_checkpointer.load('outputs/vgg_ssd300_voc0712_baseline/backbone_final.pth')
    boxhead_checkpointer.load('outputs/vgg_ssd300_voc0712_baseline/boxhead_final.pth')

    extra_backbone_checkpoint_data = backbone_checkpointer.load()
    extra_boxhead_checkpoint_data = boxhead_checkpointer.load()
    extra_domaindiscriminator_checkpoint_data = domaindiscriminator_checkpointer.load()
    arguments.update(extra_backbone_checkpoint_data)
    arguments.update(extra_boxhead_checkpoint_data)
    arguments.update(extra_domaindiscriminator_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    source_train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter,
                                           start_iter=arguments['iteration'], drop_last=False)
    target_train_loader = make_target_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter,
                                           start_iter=arguments['iteration'], drop_last=False)

    backbone, boxhead, domaindiscriminator = do_train(cfg, backbone, boxhead, domaindiscriminator, source_train_loader,
            target_train_loader, backbone_optimizer, boxhead_optimizer, domaindiscriminator_optimizer,
            backbone_checkpointer, boxhead_checkpointer, domaindiscriminator_checkpointer, device, arguments, args)
    return backbone, boxhead, domaindiscriminator


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="configs/vgg_ssd300_voc0712_feature_alignment.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=100, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=100, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    backbone, boxhead, domaindiscriminator = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, backbone, boxhead, distributed=args.distributed)


if __name__ == '__main__':
    main()
