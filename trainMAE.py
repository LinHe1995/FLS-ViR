import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import json
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from model_mae import MaskedAutoencoderViT
from functools import partial
from configs import get_trainmae_config

from utils import myLoss

from util.mask_transform import MaskTransform
from util.datasets import ImageListFolder
import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable
import math
import sys


best_acc = 0
argsm = get_trainmae_config()

bl_loss = 1.
fl_loss = 0.

def initialize(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.kaiming_normal_(m.bias.data.unsqueeze(0))
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.kaiming_normal_(m.bias.data.unsqueeze(0))
        if isinstance(m, nn.LayerNorm):
            if len(m.weight.data.shape) < 2:
                torch.nn.init.kaiming_normal_(m.weight.data.unsqueeze(0))
            else:
                torch.nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.kaiming_normal_(m.bias.data.unsqueeze(0))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, _ = batch

        samples = images.to(device, non_blocking=True)

        if args.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        else:
            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        global fl_loss
        fl_loss = loss

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:

            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class trainMAE(object):
    def __init__(self):
        self.args = argsm.parse_args()
        print(f"-----------{self.args.project_name}-----------")

        is_use_cuda = self.args.is_use_cuda and torch.cuda.is_available()
        if is_use_cuda:
            torch.cuda.manual_seed(self.args.seed)
        else:
            torch.manual_seed(self.args.seed)
        self.device = torch.device("cuda" if is_use_cuda else "cpu")
        kwargs = {'num_workers': 0, 'pin_memory': True} if is_use_cuda else {}
        print("Create Dataloader")

        transform_train = MaskTransform(self.args)
        self.images_path = os.path.join(self.args.data_dir)
        self.labels_path = os.path.join(self.args.data_dir, "images")
        dataset_train = ImageListFolder(self.images_path, transform=transform_train, ann_file=self.labels_path)
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

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )
        print('Create Model')
        self.model = MaskedAutoencoderViT(
            patch_size=self.args.patch_size, seq_len=self.seq_len, embed_dim=self.args.emb_dim, depth=self.args.num_layers,
            num_heads=self.args.num_heads,
            decoder_embed_dim=self.args.de_emb_dim, decoder_depth=self.args.de_num_layers,
            decoder_num_heads=self.args.de_num_heads,
            mlp_ratio=self.args.de_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(self.device).float()

        initialize(self.model)

        self.model_without_ddp = self.model

        if self.args.is_resume and self.args.pretrained_weight:
            model_dict = self.model.state_dict()
            print("loading the pretrained weight")
            checkpoint = torch.load(self.args.pretrained_weight, map_location=self.device)
            pretrained_dict = checkpoint['model']
            del_key = {k for k, _ in model_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in del_key and np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=True)
            print("Restoring the weight from pretrained-weight file \nFinished to load the weight")

        if is_use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(
                torch.cuda.device_count()))

            cudnn.benchmark = True
            print("Use", torch.cuda.device_count(), 'gpus')


        print("Establish the loss, optimizer and learning_rate function")

        self.criterion = myLoss().to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.args.lr,
                                   weight_decay=self.args.weight_decay,
                                   momentum=self.args.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)

        print(f"Start training for {self.args.epochs} epochs")

        start_time = time.time()
        loss_scaler = NativeScaler()
        log_dir = self.args.model_dir + self.args.project_name
        if os.path.exists(log_dir) is False:
            os.mkdir(log_dir)
        if global_rank == 0 and log_dir is not None and self.args.is_save:
            log_writer = SummaryWriter(log_dir=log_dir)
        else:
            log_writer = None

        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch, data_loader_train,loss_scaler,log_writer,log_dir)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def train(self, epoch, data_loader_train,loss_scaler,log_writer,log_dir):
        if False:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            self.model, data_loader_train,
            self.optimizer, self.device, epoch, loss_scaler,
            log_writer=log_writer,
            args=self.args
        )
        global bl_loss
        if log_dir and fl_loss < bl_loss and (epoch > 300 or epoch + 1 == self.args.epochs):
            bl_loss = fl_loss
            misc.save_model(
                args=self.args, model=self.model, model_without_ddp=self.model_without_ddp,
                optimizer=self.optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if self.args.model_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(self.args.model_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":

    trainMAE()

