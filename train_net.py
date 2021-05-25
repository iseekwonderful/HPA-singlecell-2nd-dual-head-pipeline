import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss, get_class_balanced_weighted, get_log_weight
from losses.regular import class_balanced_ce
from optimizers import get_optimizer
from distributed_train import basic_train, basic_validate
from scheduler import get_scheduler
from utils import load_matched_state
from configs import Config
from torch.utils.tensorboard import SummaryWriter
from basic_train import tta_validate
import torch
import random
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model, SyncBatchNorm
from pathlib import Path


import warnings
warnings.filterwarnings('ignore')


def train(gpu, cfg: Config):
    # reset seed for each process
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(gpu)
    if gpu == 0 and not cfg.dpp.mode == 'valid':
        result_path = prepare_for_result(cfg)
        writer = SummaryWriter(log_dir=result_path)
        cfg.dump_json(result_path / 'config.json')
    elif cfg.dpp.mode == 'valid':
        result_path = Path(cfg.train.dir) / cfg.basic.id
        mode, ckp = cfg.dpp.mode, cfg.dpp.checkpoint
        cfg = Config.load(result_path / 'config.json')
        cfg.dpp.mode, cfg.dpp.checkpoint = mode, ckp
    else:
        result_path = None
        writer = None
    # init basic elements
    rank = cfg.dpp.rank * cfg.dpp.nodes + gpu
    word_size = cfg.dpp.gpus * cfg.dpp.nodes
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=cfg.dpp.gpus * cfg.dpp.nodes,
        rank=rank
    )
    train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
    # should not do like this - -, maybe we can, since works fine
    train_ds, valid_ds = train_dl.dataset, valid_dl.dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=word_size, rank=rank)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_ds, num_replicas=word_size, rank=rank)
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.transform.num_preprocessor,
        pin_memory=True,
        sampler=train_sampler)
    valid_dl = torch.utils.data.DataLoader(
        dataset=valid_ds,
        batch_size=cfg.train.batch_size // 2,
        shuffle=False,
        num_workers=cfg.transform.num_preprocessor,
        pin_memory=False,
        sampler=valid_sampler)
    if gpu == 0:
        print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
    model = get_model(cfg).cuda(gpu)
    # if necessary load checkpoint
    if cfg.dpp.mode == 'train':
        if cfg.model.from_checkpoint and not cfg.model.from_checkpoint == 'none':
            if gpu == 0:
                print('[ ! ] Loading checkpoint from {}.'.format(cfg.model.from_checkpoint))
            try:
                load_matched_state(model, torch.load(cfg.model.from_checkpoint,
                                                     map_location={'cuda:0': 'cuda:{}'.format(gpu)}))
            except:
                print('[ ! ] Loading model failed, consider checking the saving method.')
                # this issue comes when we save a model with 'module' of DDP or DP
                st = torch.load(cfg.model.from_checkpoint, map_location={'cuda:0': 'cuda:{}'.format(gpu)})
                nst = {}
                for k, v in st.items():
                    nst[k.replace('module.', '')] = v
                model.load_state_dict(nst)
    elif cfg.dpp.mode == 'valid':
        if not cfg.dpp.checkpoint:
            raise Exception('Validation please provide a path')
        print('[ ! ] Loading checkpoint from {}.'.format(cfg.dpp.checkpoint))
        load_matched_state(model, torch.load(cfg.dpp.checkpoint,
                                             map_location={'cuda:0': 'cuda:{}'.format(gpu)}))
    if cfg.loss.name == 'weighted_ce_loss':
        # if we use weighted ce loss, we load the loss here.
        weights = torch.Tensor(cfg.loss.param['weight']).cuda()
        loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_func = get_loss(cfg)
    optimizer = get_optimizer(model, cfg)
    if gpu == 0:
        print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))
    if not cfg.basic.amp == 'None':
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
    if not cfg.scheduler.name == 'none':
        scheduler = get_scheduler(cfg, optimizer, len(train_dl))
    else:
        scheduler = None
    if cfg.dpp.sb:
        model = convert_syncbn_model(model)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu], find_unused_parameters=True)
    if cfg.dpp.mode == 'train':
        basic_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer, gpu, loss_func)
    elif cfg.dpp.mode == 'valid':
        basic_validate(model, valid_dl, loss_func, cfg, gpu)
    else:
        raise Exception('Unknown mode!')


if __name__ == '__main__':
    args, cfg = parse_args(mode='mp')
    cfg.dpp.gpus = len(cfg.basic.GPU)
    # print(cfg.dpp.sb)
    if not cfg.dpp.sb:
        print('[ x ] DPP without SyncBN warning!')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8877'
    mp.spawn(train, nprocs=cfg.dpp.gpus, args=(cfg,))


