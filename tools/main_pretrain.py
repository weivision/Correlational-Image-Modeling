# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import GaussianBlur, Solarization, Cutout


def get_args_parser():
    parser = argparse.ArgumentParser('Self-Supervised Image Modeling Pre-Training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_per_epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--gear', default='mae', type=str, metavar='GEAR',
                        help='Name of gear to train')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # MAE parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # CIM parameters
    parser.add_argument('--context_size', default=176, type=int, help='size of context image')
    parser.add_argument('--template_size', default=64, type=int, help='size of template image')
    parser.add_argument('--template_num', default=1, type=int, help='number of template image')
    parser.add_argument('--rotaton_max_degree', default=1, type=int, help='maximal degree of image rotation')
    parser.add_argument('--cutout_num', default=1, type=int, help='number of cuts on image')
    parser.add_argument('--cutout_size', default=32, type=int, help='size of cuts on image')
    parser.add_argument('--context_min_scale', type=float, default=0.2,
                        help='minimal scale of context image')
    parser.add_argument('--template_min_scale', type=float, default=0.2,
                        help='minimal scale of template image')
    parser.add_argument('--template_max_scale', type=float, default=1.0,
                        help='maximal scale of template image')
    parser.add_argument('--template_min_ratio', type=float, default=1.0/3.0,
                        help='minimal ratio of template image') 
    parser.add_argument('--template_max_ratio', type=float, default= 3.0/1.0,
                        help='maximal scale of template image')
    parser.add_argument('--common_aug', action='store_true', default=False, dest='common_aug')
    parser.add_argument('--template_aug', action='store_true', default=False, dest='template_aug')
    parser.add_argument('--sigma_cont', type=float, default=1.0,
                        help='loss weight for contrastive loss')
    parser.add_argument('--sigma_corr', type=float, default=1.0,
                        help='loss weight for correlation loss')
    # Optimizer parameters
    parser.add_argument('--amp', action='store_true', default=False, dest='FP16')
    parser.set_defaults(amp=False)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=5.0,
                        help='clip gradient norm (default: 5.0)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument(
        "--port", default=None, type=str, help="port used to set up distributed training"
    )
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # define augmentation
    if args.gear == "cim":
        search = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(args.context_min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),])
        if args.common_aug:
            common = transforms.Compose([
                    transforms.RandomApply(
                            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8  # not strengthened
                        ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomApply([Solarization()], p=0.2),
            ])
        else:
            common = None

        if args.template_aug:
            template = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
            ])
        else:
            template = None

        post_context = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Cutout(cutout_num=args.cutout_num, cutout_size=args.cutout_size)
        ])

        post_template = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Cutout(cutout_num=args.cutout_num, cutout_size=args.cutout_size)
        ])

        transform_train = {
            "search": search,
            "common": common,
            "template": template,
            "post_context": post_context,
            "post_template": post_template,
        }
    
    else:
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    

    if args.gear == "cim":
        dataset_train = CorrlationDataset(
            os.path.join(args.data_path, 'train'),
            search_size=args.input_size,
            context_size=args.context_size,
            template_size=args.template_size,
            template_num=args.template_num,
            scale=(args.template_min_scale, args.template_max_scale),
            ratio=(args.template_min_ratio, args.template_max_ratio), #(3.0 / 4.0, 4.0 / 3.0),
            degree=args.rotaton_max_degree,
            transform=transform_train
        )
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.gear == "mae":
        import models.mae as models
        from engines.pretrain_mae import train_one_epoch
        model_name = args.gear + "_" +args.model
        model = models.__dict__[model_name](
            norm_pix_loss=args.norm_pix_loss, 
            img_size=args.input_size,
        )
    elif args.gear == "cim":
        import models.cim as models
        from engines.pretrain_cim import train_one_epoch
        model_name = args.gear + "_" +args.model
        model = models.__dict__[model_name](
            img_size=args.input_size,
            context_size=args.context_size,
            template_size=args.template_size,
            accum_iter=args.accum_iter,
            sigma_cont=args.sigma_cont,
            sigma_corr=args.sigma_corr,
        )
    else:
        raise ValueError("Not supported type of gear: {}!".format(args.gear))

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    model.module.train_update(
        epochs=args.epochs, 
        iter_per_epoch= len(data_loader_train), 
        cur_epoch=args.start_epoch
    )
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # import pdb; pdb.set_trace()
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_per_epochs == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
