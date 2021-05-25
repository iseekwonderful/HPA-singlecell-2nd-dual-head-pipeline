from utils import *
import tqdm
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from configs import Config
import torch
from utils import rand_bbox
import torch.distributed as dist
import pickle as pk
from path import Path
import os
from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
from utils.metric import macro_multilabel_auc
import neptune

try:
    from apex import amp
except:
    raise Exception('While training distributed, apex is required!')
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import time

def to_hex(image_id) -> str:
    return '{0:0{1}x}'.format(image_id, 12)


def gather_list_and_concat(list_of_nums):
    tensor = torch.Tensor(list_of_nums).cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def gather_tensor_and_concat(tensor):
    tensor = tensor.cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, gpu, val_loss):
    t = time.time()
    if cfg.loss.weight_type == 'predefined':
        w = torch.tensor(cfg.loss.weight_value).cuda(gpu)
    if gpu == 0:
        print('[ √ ] DistributedDataParallel training, clip_grad: {}, amp: {}'.format(
            cfg.train.clip, cfg.basic.amp
        ))
    try:
        for epoch in range(cfg.train.num_epochs):
            img_size = cfg.transform.size
            if epoch == 0 and cfg.train.freeze_start_epoch:
                if gpu == 0:
                    print('[ W ] Freeze backbone layer')
                # only fit arcface-efficient model
                for x in model.module.model.parameters():
                    x.requires_grad = False
            if epoch == 1 and cfg.train.freeze_start_epoch:
                if gpu == 0:
                    print('[ W ] Unfreeze backbone layer')
                for x in model.module.model.parameters():
                    x.requires_grad = True
            train_dl.sampler.set_epoch(epoch)
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            results = []
            predicted, truth = [], []
            # tq = tqdm.tqdm(train_dl)
            losses, length = [], len(train_dl)
            bce_loss, mse_loss = [], []
            basic_lr = optimizer.param_groups[0]['lr']
            for i, (ipt, lbl) in enumerate(train_dl):
                # warmup
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                ipt = ipt.cuda()
                lbl = lbl.cuda()
                # cutmix
                r = np.random.rand(1)
                if cfg.train.combine_mix:
                    for ii, e in enumerate(cfg.train.combine_p):
                        if r < e:
                            idx = ii
                            break
                    else:
                        raise Exception('Fatal no mix method')
                    method = cfg.train.combine_list[idx]
                    # print(f'[DEBUG] p: {r}, method: {method}')
                    if method == 'cutmix':
                        input, target_a, target_b, lam_a, lam_b = cutmix(ipt, lbl, img_size, cfg.train.beta, model)
                    elif method == 'as_cutmix':
                        input, target_a, target_b, lam_a, lam_b = as_cutmix(ipt, lbl, img_size, cfg.train.beta, model)
                    elif method == 'snapmix':
                        input, target_a, target_b, lam_a, lam_b = snapmix(ipt, lbl, img_size, cfg.train.beta, model)
                    elif method == 'mixup':
                        input, target_a, target_b, lam_a, lam_b = mixup(ipt, lbl, img_size, cfg.train.beta, model)
                    elif method == 'cutout':
                        input, target_a, target_b, lam_a, lam_b = cutout(ipt, lbl, img_size, cfg.train.beta, model)
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        output = model(input, lbl)
                    else:
                        output = model(ipt)
                    loss = (loss_func(output, target_a) * torch.tensor(
                        lam_a).cuda().float() +
                            loss_func(output, target_b) * torch.tensor(
                                lam_b).cuda().float())
                    loss = loss.mean()
                    if cfg.optimizer.step > 1:
                        loss = loss / cfg.optimizer.step
                    # print(loss)
                    losses.append(loss.item())
                elif cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
                    input, target_a, target_b, lam_a, lam_b = cutmix(ipt, lbl, img_size, cfg.train.beta, model)
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        output = model(input, lbl)
                    else:
                        output = model(ipt)
                    if cfg.loss.weight_type == 'predefined':
                        loss_a = torch.mul(loss_func(output, target_a), w).mean()
                        loss_b = torch.mul(loss_func(output, target_b), w).mean()
                        loss = (loss_a * torch.tensor(
                            lam_a).cuda().float() +
                                loss_b * torch.tensor(
                                    lam_b).cuda().float())
                    else:
                        loss = (loss_func(output, target_a) * torch.tensor(
                            lam_a).cuda().float() +
                                loss_func(output, target_b) * torch.tensor(
                                    lam_b).cuda().float())
                    # if cfg.loss.weight_type == 'predefined':
                    #     loss = torch.mul(loss, w)
                    loss = loss.mean()
                    if cfg.optimizer.step > 1:
                        loss = loss / cfg.optimizer.step
                    # print(loss)
                    losses.append(loss.item())
                else:
                    if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                        output = model(ipt, lbl)
                    else:
                        output = model(ipt)
                    # loss = loss_func(output, lbl)
                    loss = loss_func(output, lbl)
                    if cfg.loss.weight_type == 'predefined':
                        loss = torch.mul(loss, w)
                    if not len(loss.shape) == 0:
                        loss = loss.mean()
                    losses.append(loss.item())
                    # print(loss)
                # if AMP
                if not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # if AMP end
                # backward
                if i % cfg.optimizer.step == 0:
                    if cfg.train.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                # lr scheduler
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    # TODO maybe, a bug
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                        scheduler.step()
                # lr scheduler end
                results.append({
                    'step': i,
                    'loss': loss.item(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr']
                })
                losses.append(loss.item())
                # record step
                stp = 50 if not cfg.basic.debug else 2
                if gpu == 0 and i % stp == 0 and not i == 0:
                    print('Time: {:7.2f}, Epoch: {:3d}, Iter: {:3d} / {:3d}, loss: {:.4f}, lr: {:.6f}'.format(
                        time.time() - t, epoch, i, length, np.array(losses).mean(), optimizer.param_groups[0]['lr']))
            # for debug only
            try:
                # validate_loss, valid_accuracy, valid_gap, df = basic_validate(model, valid_dl, val_loss, cfg, gpu)
                validate_loss, accuracy, auc = basic_validate(model, valid_dl, val_loss, cfg, gpu)
                if gpu == 0:
                    print(' [ √ ] Validation, epoch: {} loss: {:.4f} accuracy: {:.4f} auc: {:.4f}'.format(
                        epoch, validate_loss, accuracy, auc))
                    train_state = pd.DataFrame(results)
                    if writer:
                        writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), train_state.loss.mean(), epoch)
                        writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
                        writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss, epoch)
                        writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
                        writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc, epoch)
                        # naptune
                        # neptune.log_metric('train_f{}/loss'.format(cfg.experiment.run_fold), train_state.loss.mean())
                        # neptune.log_metric('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'])
                        # neptune.log_metric('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss)
                        # neptune.log_metric('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy)
                        # neptune.log_metric('valid_f{}/auc'.format(cfg.experiment.run_fold), auc)

                    with open(save_path / 'train.log', 'a') as fp:
                        fp.write('{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                            epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), validate_loss, accuracy, auc))
            except:
                raise
            if save_path:
                try:
                    state_dict = model.module.state_dict()
                except AttributeError:
                    state_dict = model.state_dict()
                torch.save(state_dict, save_path / 'checkpoints/f{}_epoch-{}-{:.4f}.pth'.format(
                    cfg.experiment.run_fold, epoch, accuracy))
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    except KeyboardInterrupt:
        if gpu == 0:
            print('[ X ] Ctrl + c, QUIT')
            torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


def basic_validate(mdl, dl, loss_func, cfg, gpu):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (ipt, lbl) in enumerate(dl):
            ipt = ipt.cuda()
            lbl = lbl.cuda()
            if 'arc' in cfg.model.name or 'cos' in cfg.model.name:
                output = mdl(ipt, lbl)
            else:
                output = mdl(ipt)
            # loss = loss_func(output, lbl)
            loss = loss_func(output, lbl)
            if not len(loss.shape) == 0:
                loss = loss.mean()
            losses.append(loss.item())
            predicted.append(torch.sigmoid(output.cpu()).numpy())
            truth.append(lbl.cpu().numpy())
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        truth = np.concatenate(truth)
        val_loss = np.array(losses).mean()
        accuracy = ((predicted > 0.5) == truth).sum().astype(np.float) / truth.shape[0] / truth.shape[1]
        # print(df.shape, predicted.shape)
        # df['prediction'] = predicted
        # df['truth'] = np.concatenate(truth)
        val_losses = gather_list_and_concat([val_loss])
        accuracies = gather_list_and_concat([accuracy])
        collected_loss = val_losses.cpu().numpy().mean()
        collected_accuracy = accuracies.cpu().numpy().mean()
        predicted = gather_tensor_and_concat(torch.tensor(predicted)).cpu()
        truth = gather_tensor_and_concat(torch.tensor(truth)).cpu()
        auc = macro_multilabel_auc(truth, predicted, gpu=gpu)

        return collected_loss, collected_accuracy, auc


def tta_validate(mdl, dl, loss_func, tta):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        tq = tqdm.tqdm(dl)
        for i, (ipt, lbl) in enumerate(tq):
            ipt = [x.cuda() for x in ipt]
            lbl = lbl.cuda().long()
            output = mdl(*ipt)
            loss = loss_func(output, lbl)
            losses.append(loss.item())
            predicted.append(output.cpu().numpy())
            truth.append(lbl.cpu().numpy())
            # loss, gra, vow, con = loss_func(output, GRAPHEME, VOWEL, CONSONANT)
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        length = dl.dataset.df.shape[0]
        res = np.zeros_like(predicted[:length, :])
        for i in range(tta):
            res += predicted[i * length: (i + 1) * length]
        res = res / length
        pred = torch.softmax(torch.tensor(res), 1).argmax(1).numpy()
        tru = np.concatenate(truth)[:length]
        val_loss, val_kappa = (np.array(losses).mean(),
                               cohen_kappa_score(tru, pred, weights='quadratic'))
        print('Validation: loss: {:.4f}, kappa: {:.4f}'.format(
            val_loss, val_kappa
        ))
        df = dl.dataset.df.reset_index().drop('index', 1).copy()
        df['prediction'] = pred
        df['truth'] = tru
        return val_loss, val_kappa, df
