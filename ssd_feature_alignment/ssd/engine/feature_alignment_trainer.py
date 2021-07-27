import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.modeling.domain_discriminator import domain_loss
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger

class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


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


def do_train(cfg, backbone,
             boxhead,
             domaindiscriminator,
             source_data_loader,
             target_data_loader,
             backbone_optimizer,
             boxhead_optimizer,
             domaindiscriminator_optimizer,
             backbone_checkpointer,
             boxhead_checkpointer,
             domaindiscriminator_checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    backbone.train()
    boxhead.train()
    domaindiscriminator.train()
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
    for iteration, (source_images, source_targets, _) in enumerate(source_data_loader, start_iter):#源域数据集加载
        iteration = iteration + 1
        arguments["iteration"] = iteration

        target_images, _, _ = IterLoader(target_data_loader).next_one()#目标图像加载

        source_images = source_images.to(device)
        source_targets = source_targets.to(device)
        target_images = target_images.to(device)#目标域的图像

        source_features = backbone(source_images)#源域特征
        target_features = backbone(target_images)#目标域特征
        #将源域和目标域的第0个特征图，也就是conv4_3出来的特征图送进两个域分类器
        source_cls = domaindiscriminator(source_features[0])
        target_cls = domaindiscriminator(target_features[0])
        d_real_loss = domain_loss(source_cls, real=True)
        d_fake_loss = domain_loss(target_cls, real=False)

        d_loss = cfg.SOLVER.LAMBDA * ((d_real_loss + d_fake_loss) / 2)
        meters.update(d_loss=d_loss)

        domaindiscriminator_optimizer.zero_grad()
        d_loss.backward()
        domaindiscriminator_optimizer.step()

        source_features = backbone(source_images)
        target_features = backbone(target_images)
        target_cls = domaindiscriminator(target_features[0])
        reg_loss, cls_loss = boxhead(source_features, source_targets)
        g_loss = domain_loss(target_cls, real=True)
        detection_loss = reg_loss + cls_loss
        ssd_loss = detection_loss + cfg.SOLVER.LAMBDA * g_loss

        meters.update(reg_loss=reg_loss)
        meters.update(cls_loss=cls_loss)
        meters.update(detection_loss=detection_loss)
        meters.update(g_loss=g_loss)
        meters.update(ssd_loss=ssd_loss)

        #更新框架，其实也可以说是更新特征提取器

        backbone_optimizer.zero_grad()
        boxhead_optimizer.zero_grad()
        ssd_loss.backward()
        backbone_optimizer.step()
        boxhead_optimizer.step()


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
                summary_writer.add_scalar('losses/d_loss', d_loss, global_step=global_step)
                summary_writer.add_scalar('losses/g_loss', g_loss, global_step=global_step)


        if iteration % args.save_step == 0:
            backbone_checkpointer.save("backbone_{:06d}".format(iteration), **arguments)
            boxhead_checkpointer.save("boxhead_{:06d}".format(iteration), **arguments)
            domaindiscriminator_checkpointer.save("domaindiscriminator_{:06d}".format(iteration), **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, backbone, boxhead, distributed=args.distributed, iteration=iteration)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            backbone.train()
            boxhead.train()

    backbone_checkpointer.save("backbone_final", **arguments)
    boxhead_checkpointer.save("boxhead_final", **arguments)
    domaindiscriminator_checkpointer.save("domaindiscriminator_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return backbone, boxhead, domaindiscriminator

