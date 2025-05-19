""" Utilities """
import os
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
from numpy import random
import math
import utils
from scipy.signal import find_peaks, welch
from scipy import signal
from scipy.fft import fft
# from SNiC import ideal_bandpass, normalize_psd, IPR_SSL, EMD_SSL, torch_power_spectral_density, SNR_SSL

args = utils.get_args()


class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre_lable, gt_lable):
        if len(gt_lable.shape) == 3:
            M, N, A = gt_lable.shape
            gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
            pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=-1))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=-1))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=-1) / (aPow * bPow + 0.001)
        loss = 1 - torch.sum(pearson, dim=0) / (gt_lable.shape[0])
        '''
        _, psd = torch_power_spectral_density(gt_lable, fps=30, low_hz=40 / 60, high_hz=180 / 60,
                                              normalize=True, bandpass=True, device=gt_lable.device)
        _, psd_pre = torch_power_spectral_density(pre_lable, fps=30, low_hz=40 / 60, high_hz=180 / 60,
                                                  normalize=True, bandpass=True, device=gt_lable.device)
        l_psd = F.kl_div(psd_pre.log(), psd, reduction='mean')
        loss += l_psd'''
        return loss



class SP_loss(nn.Module):
    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super(SP_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
                           requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred=None, flag=None):  # all variable operation
        fps = 30

        hr = gt.clone()

        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t * tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t * tmp), dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx





def get_loss(bvp_pre, resp_pre, hr_pre, rr_pre, spo_pre, bvp_gt, resp_gt, hr_gt, rr_gt, spo_gt, dataName, loss_bvp, loss_resp, loss_hr,
             loss_rr, loss_spo, args, inter_num, loss_res=None):
    k = 2.0 / (1.0 + np.exp(-10.0 * inter_num / args.max_iter)) - 1.0
    if dataName == 'PhysDrive':
        # l_bvp = loss_bvp[0](bvp_pre, bvp_gt) #+ 0.1 * loss_bvp[1](bvp_pre, bvp_gt)
        #l_resp = loss_resp[0](resp_pre, resp_gt) #+ 0.1 * loss_resp[1](resp_pre, resp_gt)
        l_bvp = loss_bvp[0](bvp_pre, bvp_gt) #+ 0.1 * loss_bvp[1](bvp_pre, bvp_gt)
        l_hr =  loss_hr(torch.squeeze(hr_pre), hr_gt) / 10
        l_rr = loss_rr(torch.squeeze(rr_pre), rr_gt) / 10
        l_spo = loss_spo(torch.squeeze(spo_pre), spo_gt) / 10
        loss = (l_bvp + l_rr + l_hr) / 3
        # loss = l_hr
        if loss_res is not None:
            # loss_res['bvp'].append(l_bvp.item())
            #loss_res['resp'].append(l_resp.item())
            loss_res['hr'].append(l_hr.item())
            loss_res['rf'].append(l_rr.item())
            loss_res['spo'].append(l_spo.item())
            loss_res['all'].append(loss.item())
    if dataName == 'BUAA':
        l_bvp = loss_bvp[0](bvp_pre, bvp_gt) + 0.1 * loss_bvp[1](bvp_pre, bvp_gt)
        #l_resp = loss_resp[0](resp_pre, resp_gt) #+ 0.1 * loss_resp[1](resp_pre, resp_gt)
        l_hr = k * loss_hr(torch.squeeze(hr_pre), hr_gt) / 10
        # l_rr = k * loss_rr(torch.squeeze(rr_pre), rr_gt) / 10
        # l_spo = k * loss_spo(torch.squeeze(spo_pre), spo_gt) / 10
        loss = (l_bvp + l_hr) / 2
        if loss_res is not None:
            loss_res['bvp'].append(l_bvp.item())
            #loss_res['resp'].append(l_resp.item())
            loss_res['hr'].append(l_hr.item())
            # loss_res['rf'].append(l_rr.item())
            # loss_res['spo'].append(l_spo.item())
            loss_res['all'].append(loss.item())

    return loss, loss_res
