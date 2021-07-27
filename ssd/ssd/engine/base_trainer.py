import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger

#baseline的训练器

def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):

    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

#训练
def do_train(cfg, backbone,
             boxhead,
             source_data_loader,
             backbone_optimizer,
             boxhead_optimizer,
             backbone_scheduler,
             boxhead_scheduler,
             backbone_checkpointer,
             boxhead_checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    backbone.train()
    boxhead.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    for iteration, (source_images, source_targets, _) in enumerate(source_data_loader, start_iter):#数据集导入
        iteration = iteration + 1
        arguments["iteration"] = iteration

        source_images = source_images.to(device)#图像
        source_targets = source_targets.to(device)#标签

        source_features = backbone(source_images)#提取特征
        reg_loss, cls_loss = boxhead(source_features, source_targets)#计算回归损失和分类损失
        detection_loss = reg_loss + cls_loss#总的损失
        meters.update(reg_loss=reg_loss)
        meters.update(cls_loss=cls_loss)
        meters.update(detection_loss=detection_loss)

        backbone_optimizer.zero_grad()#梯度置0
        boxhead_optimizer.zero_grad()
        detection_loss.backward()#损失回传
        backbone_optimizer.step()#进行优化
        boxhead_optimizer.step()
        backbone_scheduler.step()#学习率变化
        boxhead_scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    '{meters}',
                ]).format(
                    iter=iteration,
                    meters=str(meters),
                )
            )

            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/detection_loss', detection_loss, global_step=global_step)


        if iteration % args.save_step == 0:
            backbone_checkpointer.save("backbone_{:06d}".format(iteration), **arguments)
            boxhead_checkpointer.save("boxhead_{:06d}".format(iteration), **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, backbone, boxhead, distributed=args.distributed, iteration=iteration)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            backbone.train()
            boxhead.train()

    backbone_checkpointer.save("backbone_final", **arguments)
    boxhead_checkpointer.save("boxhead_final", **arguments)
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return backbone, boxhead
