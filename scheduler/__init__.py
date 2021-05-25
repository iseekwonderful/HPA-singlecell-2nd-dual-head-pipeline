from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
from configs import Config
from copy import deepcopy


class MileStone:
    def __init__(self, milestone, lrs, optimizer):
        self.milestone = milestone
        self.lrs = lrs
        self.optimizer = optimizer

    def step(self, epoch):
        for i, m in enumerate(self.milestone):
            if epoch < m:
                print('[ i ] epoch: {} lr set to : {}'.format(epoch, self.lrs[i]))
                self.optimizer.param_groups[0]['lr'] = float(self.lrs[i])
                return
        else:
            raise Exception('cannot find lr for epoch: {}'.format(epoch))


def get_scheduler(cfg: Config, optim, batch_per_epoch):
    if cfg.scheduler.name == 'OneCycleLR':
        return OneCycleLR(
            optimizer=optim, epochs=cfg.train.num_epochs, steps_per_epoch=batch_per_epoch, **cfg.scheduler.param)
    elif cfg.scheduler.name == 'CyclicLR':
        param = deepcopy(cfg.scheduler.param)
        if 'T' in param:
            param['step_size_up'] = (batch_per_epoch * param['T']) // 2
            param['step_size_down'] = batch_per_epoch * param['T'] - param['step_size_up']
            del param['T']
        print('[ i ] Cyclic lr scheduler, ', param)
        return CyclicLR(optimizer=optim, **param)
    elif cfg.scheduler.name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer=optim, T_max=cfg.train.num_epochs * batch_per_epoch, eta_min=0, last_epoch=-1)
    elif cfg.scheduler.name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer=optim, **cfg.scheduler.param)
    elif cfg.scheduler.name == 'StepLR':
        return MileStone(optimizer=optim, lrs=cfg.scheduler.param['lrs'], milestone=cfg.scheduler.param['milestone'])
    else:
        raise Exception('scheduler not found')
