import torch
from torch.nn import functional as F
from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from torch.optim.lr_scheduler import _LRScheduler
import math
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os.path as osp

import numpy as np


@MODEL_REGISTRY.register()
class ESRModel(BaseModel):
    def __init__(self, opt):
        super(ESRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            use_ema = self.opt.get('use_ema', False)
            if use_ema:
                param_key = 'params_ema'
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.reset_momentum_iter = self.opt['train'].get('reset_momentum_iter', None)
        self.use_amp = self.opt.get('use_amp', False)
        if self.use_amp:
            logger = get_root_logger()
            logger.info('Use mixed precision training.')
            self.scaler = torch.cuda.amp.GradScaler()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('wave_opt'):
            self.cri_wave = build_loss(train_opt['wave_opt']).to(self.device)
        else:
            self.cri_wave = None
        
        if train_opt.get('mesa_opt'):
            start_ratio = train_opt['mesa_opt'].pop('start_ratio', 0.33)
            self.mesa_start_iter = int(start_ratio * train_opt['total_iter'])
            self.cri_mesa = build_loss(train_opt['mesa_opt']).to(self.device)
        else:
            self.cri_mesa = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_freq is None and self.cri_wave is None:
            raise ValueError('Pixel, perceptual and frequency losses are None.')

        self.gradient_clip = train_opt.get('gradient_clip', None)
                
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def reset_momentums(self):
        for w in self.optimizer_g.state_dict()['state'].values():
            w['step'] = torch.zeros_like(w['step'])
            w['exp_avg'] = torch.zeros_like(w['exp_avg'])
            w['exp_avg_sq'] = torch.zeros_like(w['exp_avg_sq'])

    def optimize_parameters(self, current_iter):
        if not self.use_amp:
            self.optimizer_g.zero_grad()
            self.output = self.net_g(self.lq)

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # wavelet-based frequency loss
            if self.cri_wave:
                l_wave = self.cri_wave(self.output, self.gt)
                l_total += l_wave
                loss_dict['l_wave'] = l_wave

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style
            
            if self.cri_mesa:
                if current_iter >= self.mesa_start_iter:
                    with torch.no_grad():
                        output_emas = self.net_g_ema(self.lq)
                    l_mesa = self.cri_mesa(self.output, output_emas)
                    l_total += l_mesa
                    loss_dict['l_mesa'] = l_mesa
                else:
                    l_mesa = torch.zeros(1, device=self.device)
                    loss_dict['l_mesa'] = l_mesa

            l_total.backward()
            
            if self.gradient_clip is not None:
                clip_val = self.gradient_clip
                if current_iter > 50000:
                    clip_val = clip_val / 2
                # current_lr = self.schedulers[0].get_last_lr()[0]
                # clip_val = clip_val * current_lr
                # test this for later
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_val)
            
            self.optimizer_g.step()
        
        else:
            with torch.cuda.amp.autocast():
                self.optimizer_g.zero_grad()
                self.output = self.net_g(self.lq)

                l_total = 0
                loss_dict = OrderedDict()
                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # wavelet-based frequency loss
                if self.cri_wave:
                    l_wave = self.cri_wave(self.output, self.gt)
                    l_total += l_wave
                    loss_dict['l_wave'] = l_wave

                # perceptual loss
                if self.cri_perceptual:
                    l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                    if l_percep is not None:
                        l_total += l_percep
                        loss_dict['l_percep'] = l_percep
                    if l_style is not None:
                        l_total += l_style
                        loss_dict['l_style'] = l_style
                
                if self.cri_mesa:
                    if current_iter >= self.mesa_start_iter:
                        with torch.no_grad():
                            output_emas = self.net_g_ema(self.lq)
                        l_mesa = self.cri_mesa(self.output, output_emas)
                        l_total += l_mesa
                        loss_dict['l_mesa'] = l_mesa
                    else:
                        l_mesa = torch.zeros(1, device=self.device)
                        loss_dict['l_mesa'] = l_mesa
                
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
        if self.reset_momentum_iter is not None:
            if current_iter % self.reset_momentum_iter == 0:
                logger = get_root_logger()
                logger.info(f'Reset momentums for net_g at iteration {current_iter}')
                self.reset_momentums()
                
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g'].get('window_size')
        if window_size is not None:
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    # self.output = self.net_g_ema(img)
                    net = self.get_bare_model(self.net_g_ema)
                    self.output = net(img)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    # self.output = self.net_g(img)
                    net = self.get_bare_model(self.net_g)
                    self.output = net(img)
                self.net_g.train()

            self.output = self.output[:, :, :h * scale, :w * scale]
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    # self.output = self.net_g_ema(self.lq)
                    net = self.get_bare_model(self.net_g_ema)
                    self.output = net(self.lq)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    # self.output = self.net_g(self.lq)
                    net = self.get_bare_model(self.net_g)
                    self.output = net(self.lq)
        self.net_g.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
