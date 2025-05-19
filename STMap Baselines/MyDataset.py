# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
from scipy.signal import find_peaks, butter, filtfilt
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import rr_cal
import random
import utils
import torch
import neurokit2 as nk
from typing import List, Dict
from scipy.signal import butter
from scipy.sparse import spdiags

normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class Data_DG(Dataset):
    def __init__(self, root_dir, dataName, STMap, frames_num, args, transform=None, domain_label=None, datalist=None,
                 peoplelist=None, output_people=False):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name = STMap
        self.frames_num = int(frames_num)
        if datalist is None:
            self.datalist = os.listdir(root_dir)
            self.datalist = list(sorted(self.datalist))
        else:
            self.datalist = datalist

        self.output_people = output_people
        if output_people:
            self.peoplelist = list(peoplelist)

        self.num = len(self.datalist)
        self.domain_label = domain_label
        self.transform = transform
        self.args = args

        if self.args.pt == 'vit':
            self.transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
            self.transform_aug = transforms.Compose([
                transforms.Resize(size=(224, 224)),  # transforms.Resize(size=(64, 256)),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
                # transforms.RandomGrayscale(p=0.1),
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(size=(64, 256)),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 #                      std=[0.229, 0.224, 0.225])
                                                 ])
            self.transform_aug = transforms.Compose([
                transforms.Resize(size=(64, 256)),  # transforms.Resize(size=(64, 256)),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
                # transforms.RandomGrayscale(p=0.1),
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):
        if self.dataName == 'PhysDrive':
            resp_name = 'Label/RESP.mat'
            resp_path = os.path.join(nowPath, resp_name)
            resp = scio.loadmat(resp_path)['RESP']
            resp = np.array(resp.astype('float32')).reshape(-1)
            resp = resp[Step_Index:Step_Index + self.frames_num]
            resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
            resp = resp.astype('float32')
            try:
                rr = nk.rsp_rate(resp, sampling_rate=30, method="xcorr")
            except:
                rr= utils.rr_cal(resp, 30)
            rr = rr.astype('float32')

            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt = utils.hr_cal(bvp)
            gt = np.array(gt)
            gt = gt.astype('float32')

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')

            return gt, bvp, sp, rr, resp

        elif self.dataName == 'PURE':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'UBFC':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

        return gt, bvp

    def __getitem__(self, idx):
        img_name = 'STMap'
        STMap_name = self.STMap_Name
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        index_nowPath = nowPath
        nowPath = str(temp['Path'][0])
        nowPath = nowPath.replace('/remote-home/hao.lu', '/home/jywang')
        # nowPath = nowPath.replace('/home/haolu', '/remote-home/hao.lu')
        Step_Index = int(temp['Step_Index'])
        people_i = nowPath.split('/')[-1]
        # get HR value and bvp signal
        if self.dataName == 'PhysDrive':
            gt, bvp, sp, rr, resp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['PURE', 'VIPL', 'VV100', 'HMPC-Dv1']:
            gt, bvp, sp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['HCW', 'V4V']:
            gt, bvp, rf = self.getLabel(nowPath, Step_Index)
        else:
            gt, bvp = self.getLabel(nowPath, Step_Index)
        # get STMap
        STMap_Path = os.path.join(nowPath, img_name)
        feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))
        With, Max_frame, _ = feature_map.shape
        # get original map
        map_ori = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
        # get augmented map
        Spatial_aug_flag = 0
        Temporal_aug_flag = 0
        Step_Index_aug = Step_Index
        if self.args.spatial_aug_rate > 0:
            if (random.uniform(0, 100) / 100.0) < self.args.spatial_aug_rate:
                temp_ratio = (1.0 * random.uniform(0, 100) / 100.0)
                Index = np.arange(With)
                if temp_ratio < 0.3:
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.6:
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.9:
                    np.random.shuffle(Index[random.randint(0, With - 1):random.randint(0, With - 1)])
                    map_aug = map_ori[Index]
                else:
                    np.random.shuffle(Index)
                    map_aug = map_ori[Index]
                Spatial_aug_flag = 1
            else:
                map_aug = map_ori

        if ((Spatial_aug_flag == 0) and (self.args.temporal_aug_rate > 0)):
            if Step_Index + self.frames_num + 30 < Max_frame:
                if (random.uniform(0, 100) / 100.0) < self.args.temporal_aug_rate:
                    Step_Index_aug = int(random.uniform(0, 29) + Step_Index)
                    map_aug = feature_map[:, Step_Index_aug:Step_Index_aug + self.frames_num, :]
                    Temporal_aug_flag = 1
                else:
                    map_aug = map_ori
            else:
                map_aug = map_ori

        if ((Spatial_aug_flag == 0) and (Temporal_aug_flag == 0)):
            map_aug = map_ori

        if self.dataName == 'PhysDrive':
            gt_aug, bvp_aug, sp_aug, rr_aug, resp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['PURE', 'VIPL', 'VV100', 'HMPC-Dv1']:
            gt_aug, bvp_aug, sp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['HCW', 'V4V']:
            gt_aug, bvp_aug, rf_aug = self.getLabel(nowPath, Step_Index_aug)
        else:
            gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)

        for c in range(map_ori.shape[2]):
            for r in range(map_ori.shape[0]):
                map_ori[r, :, c] = 255 * ((map_ori[r, :, c] - np.min(map_ori[r, :, c])) / \
                                          (0.00001 + np.max(map_ori[r, :, c]) - np.min(map_ori[r, :, c])))

        for c in range(map_aug.shape[2]):
            for r in range(map_aug.shape[0]):
                map_aug[r, :, c] = 255 * ((map_aug[r, :, c] - np.min(map_aug[r, :, c])) / \
                                          (0.00001 + np.max(map_aug[r, :, c]) - np.min(map_aug[r, :, c])))

        # if self.domain_label is not None:
        # domain_label = np.full((map_ori.shape[0], 1), np.inf)

        map_ori = Image.fromarray(np.uint8(map_ori))
        map_aug = Image.fromarray(np.uint8(map_aug))

        map_ori = self.transform(map_ori)
        map_aug = self.transform_aug(map_aug)

        if self.output_people:
            domain_label = self.peoplelist.index(people_i)
        else:
            domain_label = self.domain_label

        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            return (map_ori, bvp, gt, sp, gt, map_aug, bvp_aug, gt, sp, gt, domain_label)
        elif self.dataName in ['HCW', 'V4V']:
            return (map_ori, bvp, gt, 0, rf, map_aug, bvp_aug, gt, 0, rf, domain_label)
        elif self.dataName in ['PhysDrive']:
            return (map_ori, bvp, gt, sp, resp, rr, map_aug, bvp_aug, gt, sp, resp_aug, rr, domain_label)
        else:
            return (map_ori, bvp, gt, 0, bvp, 0, map_aug, bvp_aug, gt, 0, bvp, 0, domain_label)


def CrossValidation(root_dir, fold_num=5, fold_index=0, test_percent=20):
    datalist = os.listdir(root_dir)
    #datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    fold_size = round(((num / fold_num) - 2))
    test_fold_num = int(test_percent / 100 * 5)
    train_size = num - fold_size
    test_index = datalist[fold_index * fold_size:fold_index * fold_size + fold_size * test_fold_num - 1]
    train_index = datalist[0:fold_index * fold_size] + datalist[fold_index * fold_size + fold_size * test_fold_num:]
    return train_index, test_index


def group_samples(root_dir, conditions, samples=None):
    """
    根据给定的实验条件对样本进行分组。

    每个样本字符串格式: sub_file + '_' + seq + '_' + i
      - sub_file: 三字符, 分别表示 车型(car), 性别(gender), 时间(time)
          * car: 'A','B','C'
          * gender: 'M','F'
          * time: 'Z','H','Y','W'
      - seq: 两字符, 分别表示 路况(difficulty), 说话状态(speech)
          * difficulty: 'A','B','C'
          * speech: '1' (不说话), '2' (说话)
      - i: 索引或其他信息，可忽略

    :param samples: 样本列表
    :param conditions: 实验条件列表, 最多两个. 可选值:
                       'car'       (车型),
                       'gender'    (性别),
                       'time'      (时段),
                       'difficulty'(路况),
                       'speech'    (说话状态)
    :return: 字典, 键为由条件值拼接的分组名, 值为该组对应的样本列表
    """
    if samples is None:
        samples = os.listdir(root_dir)
    # 支持的条件
    allowed = {'car', 'gender', 'time', 'difficulty', 'speech'}
    if len(conditions) > 2:
        raise ValueError("最多只能指定两个实验条件")
    for cond in conditions:
        if cond not in allowed:
            raise ValueError(f"未知的实验条件: {cond}")

    def parse_sample(s: str) -> Dict[str, str]:
        # 拆分为 sub_file, seq, 后缀
        parts = s.split('_', 2)
        if len(parts) < 2:
            raise ValueError(f"样本格式不正确: {s}")
        sub_file, seq = parts[0], parts[1]
        if len(sub_file) != 4 or len(seq) != 2:
            raise ValueError(f"sub_file 或 seq 长度不符合要求: {s}")
        return {
            'car': sub_file[0],
            'gender': sub_file[1],
            'time': sub_file[2],
            'difficulty': seq[0],
            'speech': seq[1]
        }

    groups: Dict[str, List[str]] = {}

    for sample in samples:
        attrs = parse_sample(sample)
        # 生成分组键
        if conditions:
            key_vals = [attrs[c] for c in conditions]
            key = '_'.join(key_vals)
        else:
            key = 'all'
        groups.setdefault(key, []).append(sample)

    return groups


def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    print('Now processing' + root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in [s for s in filesList if 'processed' not in s]:
        now = os.path.join(root_path, sub_file)
        for seq in os.listdir(now):
            if seq.startswith('.'):
                continue
            else:
                temp_now = os.path.join(now, seq)
            img_path = os.path.join(temp_now, 'STMap', Pic_path)

            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(temp_now, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)

            temp = cv2.imread(img_path)
            Num = temp.shape[1]
            Res = Num - frames_num - 1  # 可能是Diff数据
            Step_num = int(Res / Step)
            for i in range(Step_num):
                Step_Index = i * Step
                bvp_t = bvp[Step_Index:Step_Index + 256]

                if np.max(bvp_t) - np.min(bvp_t) == 0:
                    continue

                temp_path = sub_file + '_' + seq + '_' + str(1000 + i) + '_.mat'
                scio.savemat(os.path.join(save_path, temp_path), {'Path': temp_now, 'Step_Index': Step_Index})
                Index_path.append(temp_path)
    return Index_path



def calculate_respiration_rate(breathing_signal, sampling_rate=30):
    """
    Calculate the respiration rate from a breathing signal.

    :param breathing_signal: A 1-D numpy array of breathing signal data.
    :param sampling_rate: Sampling rate of the signal in Hz, default is 30Hz.
    :return: Respiration rate in breaths per minute.
    """
    peaks, _ = find_peaks(breathing_signal)
    num_of_breaths = len(peaks)
    duration_in_seconds = len(breathing_signal) / sampling_rate
    duration_in_minutes = duration_in_seconds / 60

    respiration_rate = num_of_breaths / duration_in_minutes
    return respiration_rate

