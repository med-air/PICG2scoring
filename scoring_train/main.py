from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.optim import SGD, lr_scheduler, AdamW
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler
from collections import Counter

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from mean import get_mean_std

# from dataset import get_training_data, get_validation_data, get_inference_data
from datasets.prostate_lesion import Prostate_lesionDataset
from datasets.prostate_lesion_public_np import Prostate_lesionDataset_public
from datasets.prostate_lesion_public_np_ins import Prostate_lesionDataset_public_np_ins
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
import inference

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets
    
def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()


    # if opt.pretrain_path is not None:
    #     opt.n_finetune_classes = opt.n_classes
    #     opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    if opt.dilation_flag:
        opt.dilation = [1,1,2,4]
        opt.stride = [2,1,1]
    else:
        opt.dilation = [1,1,1,1]
        opt.stride = [2,2,2]

    
    if opt.inference:
        if opt.distributed:
            opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

            if opt.dist_rank == 0:
                print(opt)
                with (opt.result_path / 'opts_inf.json').open('w') as opt_file:
                    json.dump(vars(opt), opt_file, default=json_serial)
        else:
            print(opt)
            with (opt.result_path / 'opts_inf.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        if opt.distributed:
            opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

            if opt.dist_rank == 0:
                print(opt)
                with (opt.result_path / 'opts.json').open('w') as opt_file:
                    json.dump(vars(opt), opt_file, default=json_serial)
        else:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)


    return opt


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler




def get_train_utils(opt, model_parameters):
    if opt.used_dataset == 'public':
        train_data = Prostate_lesionDataset_public(opt, phase='train')
        if opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            if opt.datasampler:
                targets = train_data.train_label_list
                count_target = Counter(targets)
                # print(count_target)
                weight_target = []
                for i in range(len(targets)):
                    if targets[i] == 0:
                        weight_target.append(1./count_target[0])
                    elif targets[i] == 1:
                        weight_target.append(1./count_target[1])
                    elif targets[i] == 2:
                        weight_target.append(1./count_target[2])
                    elif targets[i] == 3:
                        weight_target.append(1./count_target[3])
                    elif targets[i] == 4:
                        weight_target.append(1./count_target[4])
                    else:
                        print(i)
                        raise ValueError("error data sampler")
                assert len(weight_target) == len(targets)

                    

                # class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
                # print(len(class_sample_count))
                # print(class_sample_count)
                # weight = 1. / class_sample_count
                # # print(targets)
                # samples_weight = np.array([weight[t] for t in targets])
                samples_weight = np.array(weight_target)
                samples_weight = torch.from_numpy(samples_weight)
                samples_weight = samples_weight.double()
                # train_sampler = ImbalancedDatasetSampler(train_data)
                train_sampler  = WeightedRandomSampler(samples_weight, len(samples_weight))
            else:

                train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=opt.batch_size,
                                                    shuffle=(train_sampler is None),
                                                    num_workers=opt.n_threads,
                                                    pin_memory=True,
                                                    sampler=train_sampler,
                                                    worker_init_fn=worker_init_fn)
    else:

            train_data = Prostate_lesionDataset(opt, phase='train')
            if opt.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_data)
            else:
                if opt.datasampler:
                    train_sampler = ImbalancedDatasetSampler(train_data)
                else:

                    train_sampler = None
            train_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=opt.batch_size,
                                                    shuffle=(train_sampler is None),
                                                    num_workers=opt.n_threads,
                                                    pin_memory=True,
                                                    sampler=train_sampler,
                                                    worker_init_fn=worker_init_fn)


    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    if opt.optimizer == 'sgd':
        optimizer = SGD(model_parameters,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        dampening=dampening,
                        weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)

        assert opt.lr_scheduler in ['plateau', 'multistep', 'ExponentialLR']
        assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
        if opt.lr_scheduler == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.plateau_patience)
        elif opt.lr_scheduler == 'ExponentialLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                opt.multistep_milestones)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr = opt.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 20,gamma=0.5) 

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    if opt.used_dataset == 'public':
        if opt.used_ins:
            val_data = Prostate_lesionDataset_public_np_ins(opt, phase='test')
        else:
            val_data = Prostate_lesionDataset_public(opt, phase='test')

        if opt.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=opt.n_threads,
                                                pin_memory=True,
                                                sampler=val_sampler,
                                                worker_init_fn=worker_init_fn)

    else:
            val_data = Prostate_lesionDataset(opt, phase='test')

            if opt.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data, shuffle=False)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data,
                                                    batch_size=opt.batch_size,
                                                    shuffle=False,
                                                    num_workers=opt.n_threads,
                                                    pin_memory=True,
                                                    sampler=val_sampler,
                                                    worker_init_fn=worker_init_fn)
        

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc', 'mse'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    if opt.used_dataset == 'public':
        inference_data = Prostate_lesionDataset_public(opt, phase='inference')

        inference_loader = torch.utils.data.DataLoader(
                inference_data,
                batch_size=opt.inference_batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True,
                worker_init_fn=worker_init_fn)
    else:
            inference_data = Prostate_lesionDataset(opt, phase='inference')

            inference_loader = torch.utils.data.DataLoader(
                inference_data,
                batch_size=opt.inference_batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True,
                worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        inf_logger = Logger(opt.result_path / 'inf.log',
                            ['image_name', 'target', 'output_value', 'output', 'acc', 'mse', "mae"])
        inf_json = opt.result_path / 'inf.json'
    else:
        inf_logger = None


    return inference_loader, inf_logger, inf_json


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)

    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()

    if opt.is_master_node:
        print(model)

    if opt.loss_select == 'BalanceCE':
        if opt.n_classes == 4 and opt.data_name == 'PI-RADS':
            PI_weight = [99/32, 1.0, 99/71, 99/19]
            PI_weight = torch.Tensor(PI_weight).to(opt.device)
            criterion = CrossEntropyLoss(weight=PI_weight).to(opt.device)
        elif opt.n_classes == 6 and opt.data_name == 'ISUP':
            # ISUP_weight = [1.0, 150/27, 150/20, 150/11, 150/10, 150/3]
            # ISUP_weight = [1.0, 5.0, 5.0, 10.0, 10.0, 10.0]
            ISUP_weight = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
            ISUP_weight = torch.Tensor(ISUP_weight).to(opt.device)
            criterion = CrossEntropyLoss(weight=ISUP_weight).to(opt.device)
        else:
            raise ValueError("error data in BalanceCE")
    elif opt.loss_select == "multiFocal":
        if opt.n_classes == 5:
            PI_weight = torch.Tensor(opt.focalweight).to(opt.device)
            criterion = MultiClassFocalLossWithAlpha(alpha=PI_weight, gamma=opt.focalgamma).to(opt.device)

        else:
            raise ValueError("error data in multiFocal")

            

    elif opt.loss_select == "CE":
        criterion = CrossEntropyLoss().to(opt.device)
    
    else:
        raise ValueError("error loss function")
    if opt.kd_loss == 'KL':
        criterion2 = KLDivLoss(reduction="batchmean").to(opt.device)
    elif opt.kd_loss == "CE":
        criterion2 = CrossEntropyLoss().to(opt.device)
    else:
        raise ValueError("error KD loss")

    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    current_val_acc = 0
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, criterion2, opt.loss_weight, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, opt, tb_writer, opt.distributed)



        if not opt.no_val:
            prev_val_loss, prev_val_acc, prev_val_mse= val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed)
            if not opt.no_train:
                if (prev_val_acc >= current_val_acc and opt.is_master_node and i >100) or (prev_val_acc >= 0.5 and opt.is_master_node and i >100):
                    save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                    save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                    scheduler)
                    current_val_acc = prev_val_acc




        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    if opt.inference:
        inference_loader, inf_logger, inf_json = get_inference_utils(opt)


        inference.inference(inference_loader, model, inf_logger, inf_json, opt.device)


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
