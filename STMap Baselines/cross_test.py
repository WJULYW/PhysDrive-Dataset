import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import MyDataset
import MyLoss
import Model
import Baseline
from Intra_Model import BVPNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from thop import profile
# from basic_module import *
import utils
from datetime import datetime
import os
from utils import Logger, time_to_str, rr_cal
from timeit import default_timer as timer
import random
from tqdm import tqdm
import pynvml
import warnings

warnings.simplefilter('ignore')

TARGET_DOMAIN = {'VIPL': ['VIPL'], \
                 'V4V': ['V4V'], \
                 'PURE': ['PURE'], \
                 'BUAA': ['BUAA'], \
                 'UBFC': ['UBFC'], \
                 'HCW': ['HCW'],
                 'VV100': ['VV100'],
                 'MMPD': ['MMPD'],
                 'On_Road_rPPG': ['PhysDrive']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap'], \
              'HCW': ['HCW', 'HCW', 'STMap_RGB'],
              'VV100': ['VV100', 'VV100', 'STMap_RGB'],
              'MMPD': ['MMPD', 'MMPD', 'STMap_RGB'],
              'PhysDrive': ['PhysDrive', 'PhysDrive',
                               'STMap_RGB']
              }

# Condition = ['time']
# 'car'       (car type),
# 'gender'    (gender),
# 'time'      (light),
# 'difficulty'(road type),
# 'speech'    (motion)

if __name__ == '__main__':
    args = utils.get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Source_domain_Names = TARGET_DOMAIN[args.tgt]
    root_file = r'/home/jywang/Data/'
    frames_num = args.frames_num
    # 参数

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_map = FILE_Name[2] + '.png'

    if args.reData == 1:
        Target_index = os.listdir(Target_fileRoot)

        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)

    group_list = MyDataset.group_samples(Target_saveRoot, ['time'])
    group_list.update(MyDataset.group_samples(Target_saveRoot, ['speech']))
    group_list.update(MyDataset.group_samples(Target_saveRoot, ['difficulty']))
    group_list['all'] = os.listdir(Target_saveRoot)

    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    rPPGNet_name = 'rPPGNet_' + Target_name + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(
        args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + Target_name + '_' + str(reTrain) + '_' + args.pt + 'cross_BUAA_fulltest' + str(
        args.test_percent) + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 运行媒介
    pynvml.nvmlInit()
    flag = 0
    max_g = []
    spaces = []
    GPU = '10'
    for gpu in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_Gpu = meminfo.free / 1024 / 1024 / 1024
        if free_Gpu > 20:
            flag = 1
            GPU = str(gpu)
            print("GPU:", GPU)
            print("free_Gpu:", free_Gpu)
            max_g = GPU
            break
        print("GPU:", gpu)
        print("free_Gpu:", free_Gpu)

    # if free_Gpu < 40:
    # GPU = max_g.index(max(max_g))
    # batch_size = 10#int(150 / (47 / max_g[GPU] / 2))
    # GPU = str(GPU)
    if args.GPU != 10 and GPU == '10':
        GPU = str(args.GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU ', GPU)
    else:
        print('on CPU')

    for key in group_list.keys():
        datalist = group_list[key]
        Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                      STMap=Target_map, frames_num=frames_num, args=args, domain_label=5,
                                      datalist=datalist)

        tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

        my_model = Baseline.BaseNet_CNN() 

        if reTrain == 1:
            pre_encoder = '/home/jywang/project/STMap_Baseline_On_Road_rPPG/pre_encoder/MMPD_10000_resnet18_full_supervised_BVPNet'
            my_model = torch.load(
                pre_encoder,
                map_location=device)
            print('load ' + pre_encoder + ' right')

        my_model.to(device=device)

        tgt_iter = iter(tgt_loader)
        tgt_iter_per_epoch = len(tgt_iter)

        max_iter = args.max_iter
        start = timer()
        loss_res = {'bvp': [], 'hr': [], 'spo': [], 'rf': [], 'resp': [], 'all': [],
                    }

        eval_bvp_hr = {'Key': [], 'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
        eval_hr = {'Key': [], 'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
        eval_rr = {'Key': [], 'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
        eval_spo = {'Key': [], 'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
        eval_resp_rr = {'Key': [], 'MAE': [], 'RMSE': [], 'MER': [], 'P': []}

        # if iter_num % 500 == 0 and iter_num > 500:
        # temp = pd.DataFrame(loss_res)
        # temp.to_csv('loss_res_w_demintor.csv')
        # 测试
        print('Test:\n')
        print('Current group:  ' + key)

        my_model.eval()
        loss_mean = []
        Label_pr = []
        Label_gt = []
        HR_pr_temp = []
        HR_rel_temp = []
        BVP_ALL = []
        BVP_PR_ALL = []
        Spo_pr_temp = []
        Spo_rel_temp = []
        RF_pr_temp = []
        RF_rel_temp = []
        Resp_ALL = []
        Resp_PR_ALL = []
        for step, (data, bvp, HR_rel, spo, resp, rf, _, _, _, _, _, _, _) in tqdm(enumerate(tgt_loader)):
            data = Variable(data).float().to(device=device)
            Wave = Variable(bvp).float().to(device=device)
            Resp = Variable(resp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            Spo_rel = Variable(spo).float().to(device=device)
            RF_rel = Variable(rf).float().to(device=device)
            Wave = Wave.unsqueeze(dim=1)
            Resp = Resp.unsqueeze(dim=1)
            rand_idx = torch.randperm(data.shape[0])
            Wave_pr, HR_pr, Spo_pr, RESP_pr, RF_pr = my_model(data)

            HR_rel_temp.extend(HR_rel.data.cpu().numpy())
            # temp, HR_pr = loss_func_SP(Wave_pr, HR_pr)
            HR_pr_temp.extend(HR_pr.data.cpu().numpy())
            RF_pr_temp.extend(RF_pr.data.cpu().numpy())
            RF_rel_temp.extend(RF_rel.data.cpu().numpy())
            BVP_ALL.extend(Wave.data.cpu().numpy())
            BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())
            Resp_ALL.extend(Resp.data.cpu().numpy())
            Resp_PR_ALL.extend(RESP_pr.data.cpu().numpy())
            Spo_pr_temp.extend(Spo_pr.data.cpu().numpy())
            Spo_rel_temp.extend(Spo_rel.data.cpu().numpy())

        # print('HR:')
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
        print(Target_name)
        log.write(
            'Test HR:'
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')
        eval_hr['Key'].append(key)
        eval_hr['MAE'].append(MAE)
        eval_hr['RMSE'].append(RMSE)
        eval_hr['MER'].append(MER)
        eval_hr['P'].append(P)

        ME, STD, MAE, RMSE, MER, P = utils.MyEval_bvp_hr(BVP_PR_ALL, BVP_ALL)
        log.write(
            'Test HR from BVP:'
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')
        eval_bvp_hr['Key'].append(key)
        eval_bvp_hr['MAE'].append(MAE)
        eval_bvp_hr['RMSE'].append(RMSE)
        eval_bvp_hr['MER'].append(MER)
        eval_bvp_hr['P'].append(P)

        ME, STD, MAE, RMSE, MER, P = utils.MyEval(Spo_pr_temp, Spo_rel_temp)
        log.write(
            'Test SPO2:'
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')
        eval_spo['Key'].append(key)
        eval_spo['MAE'].append(MAE)
        eval_spo['RMSE'].append(RMSE)
        eval_spo['MER'].append(MER)
        eval_spo['P'].append(P)

        ME, STD, MAE, RMSE, MER, P = utils.MyEval_resp_rr(Resp_PR_ALL, Resp_ALL)
        log.write(
            'Test RR from RESP:'
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')
        eval_resp_rr['Key'].append(key)
        eval_resp_rr['MAE'].append(MAE)
        eval_resp_rr['RMSE'].append(RMSE)
        eval_resp_rr['MER'].append(MER)
        eval_resp_rr['P'].append(P)

        ME, STD, MAE, RMSE, MER, P = utils.MyEval(RF_pr_temp, RF_rel_temp)
        log.write(
            'Test RR:'
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')
        eval_rr['Key'].append(key)
        eval_rr['MAE'].append(MAE)
        eval_rr['RMSE'].append(RMSE)
        eval_rr['MER'].append(MER)
        eval_rr['P'].append(P)

        # if not os.path.exists('./visuals/result_visual/'):
        #     os.makedirs('./visuals/result_visual/')
        # eval_hr_save = pd.DataFrame(eval_hr)
        # eval_hr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
        #     reTrain) + '_' + args.pt + 'cross_BUAA' + '_'.join(Condition) + '_HR.csv')
        # eval_bvp_hr_save = pd.DataFrame(eval_bvp_hr)
        # eval_bvp_hr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
        #     reTrain) + '_' + args.pt + 'cross_BUAA' + '_'.join(Condition) + '_BVP_HR.csv')
        # eval_resp_rr_save = pd.DataFrame(eval_resp_rr)
        # eval_resp_rr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
        #     reTrain) + '_' + args.pt + 'cross_BUAA' + '_'.join(Condition) + '_RESP_RR.csv')
        # eval_spo_save = pd.DataFrame(eval_spo)
        # eval_spo_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
        #     reTrain) + '_' + args.pt + 'cross_BUAA' + '_'.join(Condition) + '_SPO.csv')
        # eval_rr_save = pd.DataFrame(eval_rr)
        # eval_rr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
        #     reTrain) + '_' + args.pt + 'cross_BUAA' + '_'.join(Condition) + '_RR.csv')
