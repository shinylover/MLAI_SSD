import argparse
import logging
import os

import torch

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.base_trainer import do_train
from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

#baseline训练，即直接利用ssd在voc上进行训练并在clipart上测试

def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')#日志文件
    backbone = build_backbone(cfg)#模型框架，即vgg
    boxhead = build_box_head(cfg)#模型的输出层

    device = torch.device(cfg.MODEL.DEVICE)#显卡
    backbone = backbone.to(device)#将框架部署到显卡上
    boxhead = boxhead.to(device)#将输出层部署到显卡上

    backbone_lr = cfg.SOLVER.BACKBONELR#框架的学习率
    backbone_optimizer = make_optimizer(cfg, backbone, backbone_lr)#框架优化器
    boxhead_lr = cfg.SOLVER.BOXHEADLR#输出层的学习率
    boxhead_optimizer = make_optimizer(cfg, boxhead, boxhead_lr)#输出层的优化器

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]#学习率的变化
    backbone_scheduler = make_lr_scheduler(cfg, backbone_optimizer, milestones)#初始学习率0.001，在80000的时候衰减至0.0001，
    # 10000的时候衰减至0.00001
    boxhead_scheduler = make_lr_scheduler(cfg, boxhead_optimizer, milestones)

    arguments = {"iteration": 0}#从0代开始
    #保存文件
    save_to_disk = dist_util.get_rank() == 0
    backbone_checkpointer = CheckPointer(backbone, 'backbone_last_checkpoint.txt', backbone_optimizer, backbone_scheduler, cfg.OUTPUT_DIR, save_to_disk,
                                         logger)
    boxhead_checkpointer = CheckPointer(boxhead, 'boxhead_last_checkpoint.txt', boxhead_optimizer, boxhead_scheduler, cfg.OUTPUT_DIR, save_to_disk,
                                         logger)

    extra_backbone_checkpoint_data = backbone_checkpointer.load()#文件加载
    extra_boxhead_checkpoint_data = boxhead_checkpointer.load()
    arguments.update(extra_backbone_checkpoint_data)#更新文件中的东西
    arguments.update(extra_boxhead_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus#迭代次数
    source_train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter,
                                           start_iter=arguments['iteration'], drop_last=False)#数据集加载

    backbone, boxhead = do_train(cfg, backbone, boxhead, source_train_loader, backbone_optimizer,
                                 boxhead_optimizer, backbone_scheduler, boxhead_scheduler, backbone_checkpointer,
                                 boxhead_checkpointer, device, arguments, args)#训练框架和输出层
    return backbone, boxhead


def main():#主函数
    #解析参数
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="configs/vgg_ssd300_voc0712_baseline.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
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

    backbone, boxhead = train(cfg, args)#训练

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, backbone, boxhead, distributed=args.distributed)


if __name__ == '__main__':
    main()
