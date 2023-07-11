# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (images, templates_, boxes_, masks_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (images, templates_, masks_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images = images.to(device, non_blocking=True)
        temps = []
        for temp in templates_:
            temps.append(temp.to(device, non_blocking=True))

        masks = []
        for mask in masks_:
            masks.append(mask.to(device, non_blocking=True))
        
        if args.amp:
            with torch.cuda.amp.autocast():
                loss, pred = model(images, temps, masks)
        else:
            loss, pred = model(images, temps, masks)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            import pdb; pdb.set_trace()
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss/accum_iter
        if args.amp:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            if not args.amp:
                optimizer.step()
            
            optimizer.zero_grad()

        torch.cuda.synchronize()

        model.module.iter_update()
        
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}