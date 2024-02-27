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
from tqdm import tqdm

from model_FLSVIR import FLSReLM
from model_FLSVIR_ft import FLSReLM_ft
from functools import partial
from configs import get_train_config,get_trainmae_config
from augdata import Dataset, MixupDataset
from utils import Color_print
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
args = get_train_config()
argsm = get_trainmae_config()

str_acc = ''
str_loss = ''
str_name=''

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

class train(object):
    def __init__(self):
        self.args = args.parse_args()
        print(f"-----------{self.args.project_name}-----------")
        is_use_cuda = self.args.is_use_cuda
        self.device = "cuda"
        print("self.device ",self.device)
        kwargs = {'num_workers': 0, 'pin_memory': True} if is_use_cuda else {}

        print("Create Dataloader")
        self.images_path = os.path.join(self.args.data_dir, "images")
        self.labels_path = os.path.join(self.args.data_dir, "image_class_labels")
        self.annotation_lines = self.get_image_label()

        np.random.seed(10101)
        np.random.shuffle(self.annotation_lines)
        np.random.seed(None)
        self.num_val = int(len(self.annotation_lines) * self.args.val_num)
        self.num_train = len(self.annotation_lines) - self.num_val

        self.train_loader = DataLoader(
            MixupDataset(Dataset(self.annotation_lines[:self.num_train], type='train')),
            batch_size=self.args.batch_size, shuffle=False, **kwargs)

        self.test_loader = DataLoader(
            Dataset(self.annotation_lines[self.num_train + 1:], type='test'),
            batch_size=self.args.batch_size, shuffle=False, **kwargs)

        print('Create Model')

        self.model = FLSReLM(
            emb_dim=self.args.emb_dim,
            mlp_dim=self.args.mlp_dim,
            seq_len=self.seq_len,
            num_layers=self.args.num_layers,
            num_heads=self.args.num_heads,
            num_classes=self.args.num_class,
            image_size=(self.args.image_size, self.args.image_size),
            patch_size=(self.args.patch_size, self.args.patch_size),
            dropout_rate=self.args.dropout_rate,
            attn_dropout_rate=self.args.attn_dropout_rate,
            layer_dropout=0.0,
            reverse_thres=0.0,
            use_scale_norm=False,
            use_rezero=False
        ).to(self.device).float()

        initialize(self.model)
        
        model_dict = self.model.state_dict()
        print("loading the pretrained weight")
        checkpoint = torch.load(self.args.pretrained_weight, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if np.shape(model_dict[k[7:]]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=True)
        print("Restoring the weight from pretrained-weight file \nFinished to load the weight")
        
        print("Establish the loss, optimizer and learning_rate function")
        #self.criterion = nn.CrossEntropyLoss().to(self.device)
        #self.criterion = nn.LogSoftmax()#.to(self.device)
        self.criterion = myLoss().to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.args.lr,
                                   weight_decay=self.args.weight_decay,
                                   momentum=self.args.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)

        print("Start training")
        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            facc = open(self.args.model_dir+'accfile', "a")
            floss = open(self.args.model_dir + 'lossfile', "a")
            facc.write(str_acc)
            floss.write(str_loss)
            facc.close()
            floss.close()
        torch.cuda.empty_cache()
        Color_print("finish model training")

    def train(self, epoch):

        global str_loss

        self.model.train()

        average_loss = []
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}/{self.args.epochs}')

        local_str_loss = ''
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            average_loss.append(loss.item()) 
            self.optimizer.step()
            pbar.set_description(f'Train Epoch: {epoch}/{self.args.epochs} loss: {np.mean(average_loss)}')
            local_str_loss = str(np.mean(average_loss))
            self.scheduler.step()
        str_loss = local_str_loss + "\n"

    def test(self, epoch):

        global best_acc
        global str_acc

        self.model.eval()
        test_loss = 0
        correct = torch.zeros(1).squeeze().to(self.device)
        total = torch.zeros(1).squeeze().to(self.device)
        average_loss = []
        pbar = tqdm(self.test_loader, desc=f'Test Epoch{epoch}/{self.args.epochs}', mininterval=0.3)
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss = self.criterion(output, target).item()
                average_loss.append(test_loss)
                pred = torch.max(output, 1)[1]
                target = torch.max(target, 1)[1]
                correct += (pred == target).sum()
                total += len(target)
                pbar.set_description(
                    f'Test  Epoch: {epoch}/{self.args.epochs} ')
                predict_acc = correct / total
        percentage = round(predict_acc.item(), 4) * 100
        print(f"\n预测准确率:{percentage}% ")
        str_acc = str(percentage) + "\n"

        if self.args.is_save and predict_acc > best_acc:
            best_acc = predict_acc
            self.save_model(epoch, average_loss, predict_acc, correct, total)

    def get_image_label(self):
        images = []
        labels = []
        with open(self.images_path) as f:
            for line in f.readlines():
                if ").tif" in line.split()[-1]:
                    images.append(line.split()[-2]+" "+line.split()[-1])
                else:
                    images.append(line.split()[-1])
        with open(self.labels_path) as f:
            for line in f.readlines():
                labels.append(line.split()[-1])

        lines = []

        for image, label in zip(images, labels):
            lines.append(
                self.args.data_dir + str(image) + '*' + str(label))

        return lines

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def save_model(self, epoch, average_loss, predict_acc, correct, total):
        if not os.path.isdir(self.args.model_dir + self.args.project_name) and self.args.is_save:
            os.mkdir(self.args.model_dir + self.args.project_name)
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': round(np.mean(average_loss), 2)
        },
            self.args.model_dir + self.args.project_name + f'/Epoch-{epoch}-Test_loss-{round(np.mean(average_loss), 4)}.pth')
        percentage = round(predict_acc.item(), 4) * 100
        print(
            f"\n预测准确率:{percentage}% "
            f"预测数量:{correct}/{total},"
            f"保存路径:{self.args.model_dir + self.args.project_name}/Epoch-{epoch}-Test_loss-{round(np.mean(average_loss), 4)}.pth'")

class finetune(object):
    def __init__(self):
        self.args = args.parse_args()
        print(f"-----------{self.args.project_name}-----------")
        is_use_cuda = self.args.is_use_cuda
        self.device = "cuda"
        print("self.device ",self.device)
        kwargs = {'num_workers': 0, 'pin_memory': True} if is_use_cuda else {}

        print("Create Dataloader")
        self.images_path = os.path.join(self.args.data_dir, "images")
        self.labels_path = os.path.join(self.args.data_dir, "image_class_labels")
        self.annotation_lines = self.get_image_label()

        np.random.seed(10101)
        np.random.shuffle(self.annotation_lines)
        np.random.seed(None)
        self.num_val = int(len(self.annotation_lines) * self.args.val_num)
        self.num_train = len(self.annotation_lines) - self.num_val

        self.train_loader = DataLoader(
            MixupDataset(Dataset(self.annotation_lines[:self.num_train], type='train')),
            batch_size=self.args.batch_size, shuffle=False, **kwargs)

        self.test_loader = DataLoader(
            Dataset(self.annotation_lines[self.num_train + 1:], type='test'),
            batch_size=self.args.batch_size, shuffle=False, **kwargs)

        print('Create Model')

        self.model = FLSReLM_ft(
            emb_dim=self.args.emb_dim,
            mlp_dim=self.args.mlp_dim,
            seq_len=self.seq_len,
            num_layers=self.args.num_layers,
            num_heads=self.args.num_heads,
            num_classes=self.args.num_class,
            image_size=(self.args.image_size, self.args.image_size),
            patch_size=(self.args.patch_size, self.args.patch_size),
            dropout_rate=self.args.dropout_rate,
            attn_dropout_rate=self.args.attn_dropout_rate,
            layer_dropout=0.0,
            reverse_thres=0.0,
            use_scale_norm=False,
            use_rezero=False
        ).to(self.device).float()

        initialize(self.model)
        
        model_dict = self.model.state_dict()
        print("loading the pretrained weight")
        checkpoint = torch.load(self.args.pretrained_weight, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if np.shape(model_dict[k[7:]]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=True)
        print("Restoring the weight from pretrained-weight file \nFinished to load the weight")
        
        for name, parm in self.model.named_parameters():
            if "new" in name:
                print("unfreeze "+ name)
            elif "classifier" in name:
                print("unfreeze "+ name)
            elif "norm.weight" == name:
                print("unfreeze "+ name)
            elif "norm.bias" == name:
                print("unfreeze "+ name)
            else:
                parm.requires_grad = False

        print("Establish the loss, optimizer and learning_rate function")
        #self.criterion = nn.CrossEntropyLoss().to(self.device)
        #self.criterion = nn.LogSoftmax()#.to(self.device)
        self.criterion = myLoss().to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.args.lr,
                                   weight_decay=self.args.weight_decay,
                                   momentum=self.args.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)

        print("Start training")
        for epoch in range(1, self.args.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            facc = open(self.args.model_dir+'accfile', "a")
            floss = open(self.args.model_dir + 'lossfile', "a")
            facc.write(str_acc)
            floss.write(str_loss)
            facc.close()
            floss.close()
        torch.cuda.empty_cache()
        Color_print("finish model training")

    def train(self, epoch):

        global str_loss

        self.model.train()

        average_loss = []
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}/{self.args.epochs}')

        local_str_loss = ''
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            average_loss.append(loss.item()) 
            self.optimizer.step()
            pbar.set_description(f'Train Epoch: {epoch}/{self.args.epochs} loss: {np.mean(average_loss)}')
            local_str_loss = str(np.mean(average_loss))
            self.scheduler.step()
        str_loss = local_str_loss + "\n"

    def test(self, epoch):

        global best_acc
        global str_acc

        self.model.eval()
        test_loss = 0
        correct = torch.zeros(1).squeeze().to(self.device)
        total = torch.zeros(1).squeeze().to(self.device)
        average_loss = []
        pbar = tqdm(self.test_loader, desc=f'Test Epoch{epoch}/{self.args.epochs}', mininterval=0.3)
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss = self.criterion(output, target).item()
                average_loss.append(test_loss)
                pred = torch.max(output, 1)[1]
                target = torch.max(target, 1)[1]
                correct += (pred == target).sum()
                total += len(target)
                pbar.set_description(
                    f'Test  Epoch: {epoch}/{self.args.epochs} ')
                predict_acc = correct / total
        percentage = round(predict_acc.item(), 4) * 100
        print(f"\n预测准确率:{percentage}% ")
        str_acc = str(percentage) + "\n"

        if self.args.is_save and predict_acc > best_acc:
            best_acc = predict_acc
            self.save_model(epoch, average_loss, predict_acc, correct, total)

    def get_image_label(self):
        images = []
        labels = []
        with open(self.images_path) as f:
            for line in f.readlines():
                if ").tif" in line.split()[-1]:
                    images.append(line.split()[-2]+" "+line.split()[-1])
                else:
                    images.append(line.split()[-1])
        with open(self.labels_path) as f:
            for line in f.readlines():
                labels.append(line.split()[-1])

        lines = []

        for image, label in zip(images, labels):
            lines.append(
                self.args.data_dir + str(image) + '*' + str(label))

        return lines

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def save_model(self, epoch, average_loss, predict_acc, correct, total):
        if not os.path.isdir(self.args.model_dir + self.args.project_name) and self.args.is_save:
            os.mkdir(self.args.model_dir + self.args.project_name)
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': round(np.mean(average_loss), 2)
        },
            self.args.model_dir + self.args.project_name + f'/Epoch-{epoch}-Test_loss-{round(np.mean(average_loss), 4)}.pth')
        percentage = round(predict_acc.item(), 4) * 100
        print(
            f"\n预测准确率:{percentage}% "
            f"预测数量:{correct}/{total},"
            f"保存路径:{self.args.model_dir + self.args.project_name}/Epoch-{epoch}-Test_loss-{round(np.mean(average_loss), 4)}.pth'")


if __name__ == "__main__":
    #finetune()
    train()

