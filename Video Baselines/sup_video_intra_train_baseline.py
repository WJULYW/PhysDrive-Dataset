import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import MyDataset
import MyLoss
from Video_Baselines.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from Video_Baselines.MTTS_CAN import MTTS_CAN
from Video_Baselines.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from Video_Baselines.FactorizePhys import FactorizePhys
from Video_Baselines.RhythmFormer import RhythmFormer
from Video_Baselines.DeepPhys import DeepPhys
from Video_Baselines.iBVPNet import iBVPNet
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
from losses import loss_init
import warnings

warnings.simplefilter('ignore')


TARGET_DOMAIN = {'VIPL': ['VIPL'], \
                 'V4V': ['V4V'], \
                 'PURE': ['PURE'], \
                 'BUAA': ['BUAA'], \
                 'UBFC': ['UBFC'], \
                 'VV100': ['VV100'],
                 'MMPD': ['MMPD'],
                 'PhysDrive': ['PhysDrive']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap'], \
              'VV100': ['VV100', 'VV100', 'STMap_RGB'],
              'MMPD': ['MMPD', 'MMPD', 'STMap_RGB'],
              'PhysDrive': ['PhysDrive', 'PhysDrive',
                               'STMap_RGB']
              }

torch.backends.cudnn.benchmark = False

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
    root_file = r'xxx' #your data address

    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_map = FILE_Name[2] + '.png'

    if args.reData == 1:
        Target_index = os.listdir(Target_fileRoot)

        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)

    train_list, test_list = MyDataset.CrossValidation(Target_saveRoot, fold_num=args.fold_num,
                                                      test_percent=args.test_percent)


    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU


    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    rPPGNet_name = 'rPPGNet_' + Target_name + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(
        args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + Target_name + '_' + str(reTrain) + '_' + args.pt + '_intra_supervised' + str(
        args.test_percent) + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

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

   
    if args.GPU != 10 and GPU == '10':
        GPU = str(args.GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU ', GPU)
    else:
        print('on CPU')



    source_db_0 = MyDataset.Data_Video(root_dir=Target_saveRoot, dataName=Target_name, \
                                       STMap=Target_map, frames_num=frames_num, args=args, domain_label=0,
                                       datalist=train_list)
    Target_db = MyDataset.Data_Video(root_dir=Target_saveRoot, dataName=Target_name, \
                                     STMap=Target_map, frames_num=frames_num, args=args, domain_label=5,
                                     datalist=test_list)

    src_loader_0 = DataLoader(source_db_0, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

  
    my_model = RhythmFormer()


    if reTrain == 1:
        my_model = torch.load('./Result_Model/VV100_50_Un_pretrain',
                              map_location=device)
        print('load ' + rPPGNet_name + ' right')
    my_model.to(device=device)



    optimizer_rPPG = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    
    loss_func_NP = [MyLoss.P_loss3().to(device), nn.SmoothL1Loss().to(device)]
    loss_func_SPO = nn.SmoothL1Loss().to(device)
    loss_func_RF = nn.SmoothL1Loss().to(device)
    loss_func_L1 = nn.SmoothL1Loss().to(device)
    loss_func_RESP = nn.SmoothL1Loss().to(device)  # MyLoss.P_loss3().to(device)
    loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)

    src_iter_0 = src_loader_0.__iter__()
    src_iter_per_epoch_0 = len(src_iter_0)

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    loss_res = {'bvp': [], 'hr': [], 'spo': [], 'rf': [], 'resp': [], 'IrrelevantPowerRatio': [],
                'NegativeMaxCrossCorr': [], 'NegPearsonsCorrLoss': [], 'NegSNRLoss': [],
                'IPR_SSL': [], 'EMD_SSL': [], 'SNR_SSL': []}
    loss_res_aug = {'bvp': [], 'hr': [], 'spo': [], 'rf': [], 'resp': [], 'IrrelevantPowerRatio': [],
                    'NegativeMaxCrossCorr': [], 'NegPearsonsCorrLoss': [], 'NegSNRLoss': [],
                    'IPR_SSL': [], 'EMD_SSL': [], 'SNR_SSL': []}
    eval_bvp_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_bvp_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_spo = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_resp_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    with tqdm(range(max_iter + 1)) as it:
        for iter_num in it:
            my_model.train()
            if (iter_num % src_iter_per_epoch_0 == 0):
                src_iter_0 = src_loader_0.__iter__()

            ######### data prepare #########
            data0, bvp0, HR_rel0, spo_rel0, rf_rel0, resp_rel0, data_aug0, bvp_aug0, HR_rel_aug0, spo_rel_aug0, rf_rel_aug0, resp_rel_aug0, domain_label0 = src_iter_0.__next__()

            rf_rel0 = Variable(torch.Tensor(rf_rel0)).float().to(device=device)
            data0 = Variable(data0).float().to(device=device)
            bvp0 = Variable(bvp0).float().to(device=device).unsqueeze(dim=1)
            HR_rel0 = Variable(torch.Tensor(HR_rel0)).float().to(device=device)
            spo_rel0 = Variable(torch.Tensor(spo_rel0)).float().to(device=device)
            resp_rel0 = Variable(torch.Tensor(resp_rel0)).float().to(device=device).unsqueeze(dim=1)
            # rf_rel0 = Variable(torch.Tensor(rf_rel0)).float().to(device=device)

            rf_rel_aug0 = Variable(torch.Tensor(rf_rel_aug0)).float().to(device=device)
            data_aug0 = Variable(data_aug0).float().to(device=device)
            bvp_aug0 = Variable(bvp_aug0).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug0 = Variable(torch.Tensor(HR_rel_aug0)).float().to(device=device)
            spo_rel_aug0 = Variable(torch.Tensor(spo_rel_aug0)).float().to(device=device)
            resp_rel_aug0 = Variable(torch.Tensor(resp_rel_aug0)).float().to(device=device).unsqueeze(dim=1)
            # rf_rel_aug0 = Variable(torch.Tensor(rf_rel_aug0)).float().to(device=device)

            optimizer_rPPG.zero_grad()
            N, D, C, H, W = data0.shape
            #data0 = data0.reshape(N * D, C, H, W)
            #data_aug0 = data_aug0.reshape(N * D, C, H, W)

            bvp_pre_zip, HR_pr = my_model(data0)
            bvp_pre_zip = bvp_pre_zip.reshape(N, 1, D)

            bvp_pre_aug_zip, HR_pr_aug = my_model(data_aug0)
            bvp_pre_aug_zip = bvp_pre_aug_zip.reshape(N, 1, D)

            rppg_loss_0, loss_res = MyLoss.get_loss2(bvp_pre_zip, HR_pr, bvp0, HR_rel0, Target_name, \
                                                     loss_func_NP, loss_func_L1, args, iter_num, spo_pre=0,
                                                     spo_gt=0,
                                                     loss_spo=loss_func_SPO, rf_pre=0,
                                                     rf_gt=0,
                                                     loss_rf=loss_func_RF, resp_pre=0,
                                                     resp_gt=0, loss_resp=loss_func_RESP, loss_res=loss_res)

            rppg_loss_aug_0, _ = MyLoss.get_loss2(bvp_pre_aug_zip, HR_pr_aug, bvp_aug0, HR_rel_aug0,
                                                  Target_name, \
                                                  loss_func_NP, loss_func_L1, args, iter_num,
                                                  spo_pre=0,
                                                  spo_gt=0, loss_spo=loss_func_SPO, rf_pre=0,
                                                  rf_gt=0,
                                                  loss_rf=loss_func_RF, resp_pre=0,
                                                  resp_gt=0, loss_resp=loss_func_RESP,
                                                  loss_res=loss_res_aug)


            rppg_loss = rppg_loss_0

            # k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0
            loss_all = rppg_loss  # + 0.1 * k * loss_TA + 0.001 * k * loss_CM + 0.01 * k * loss_DM

            if torch.sum(torch.isnan(rppg_loss)) > 0:
                print('Nan')
                continue
            else:
                rppg_loss.backward()
                optimizer_rPPG.step()
                # spo_loss.backward()
                # optimizer_spo.step()
                it.set_postfix(
                    ordered_dict={
                        "Train Inter": iter_num,
                        "rppg loss": rppg_loss.data.cpu().numpy(),
                        # "spo loss": spo_loss.data.cpu().numpy()
                    },
                    refresh=False,
                )

            log.write(
                'Train Inter:' + str(iter_num) \
                + ' | Overall Loss:  ' + str(loss_all.data.cpu().numpy()) \
                + ' | rppg Loss:  ' + str(rppg_loss.data.cpu().numpy()) \
                # + ' | TA Loss:  ' + str(loss_TA.data.cpu().numpy()) \
                + ' |' + time_to_str(timer() - start, 'min'))
            log.write('\n')

            if (iter_num > 0) and (iter_num % 1000 == 0):
                print('Test:\n')

                my_model.eval()
                loss_mean = []
                Label_pr = []
                Label_gt = []
                HR_pr_temp = []
                HR_rel_temp = []
                HR_pr2_temp = []
                BVP_ALL = []
                BVP_PR_ALL = []
                Spo_pr_temp = []
                Spo_rel_temp = []
                RF_pr_temp = []
                RF_rel_temp = []
                Resp_ALL = []
                Resp_PR_ALL = []
                for step, (data, bvp, HR_rel, spo, rf, resp, _, _, _, _, _, _, _) in tqdm(enumerate(tgt_loader)):
                    data = Variable(data).float().to(device=device)
                    Wave = Variable(bvp).float().to(device=device)
                    Wave = (Wave - torch.mean(Wave, axis=-1).view(-1, 1)) / torch.std(Wave, axis=-1).view(-1, 1)
                    HR_rel = Variable(HR_rel).float().to(device=device)
                    Spo_rel = Variable(spo).float().to(device=device)
                    RF_rel = Variable(rf).float().to(device=device)
                    Resp_rel = Variable(resp).float().to(device=device)
                    Wave = Wave.unsqueeze(dim=1)
                    Resp_rel = Resp_rel.unsqueeze(dim=1)
                    rand_idx = torch.randperm(data.shape[0])
                    N, D, C, H, W = data.shape
                    #data = data.reshape(N * D, C, H, W)
                    Wave_pr, HR_pr = my_model(data)
                    Wave_pr = Wave_pr.reshape(N, 1, D)

                    HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                    HR_pr_temp.extend(HR_pr.data.cpu().numpy())

                    BVP_ALL.extend(Wave.data.cpu().numpy())
                    BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())

                ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
                print(Target_name)
                log.write(
                    'Test Inter HR:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_hr['MAE'].append(MAE)
                eval_hr['RMSE'].append(RMSE)
                eval_hr['MER'].append(MER)
                eval_hr['P'].append(P)

                ME, STD, MAE, RMSE, MER, P = utils.MyEval_bvp_hr(BVP_PR_ALL, BVP_ALL)
                log.write(
                    'Test Inter HR from BVP:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_bvp_hr['MAE'].append(MAE)
                eval_bvp_hr['RMSE'].append(RMSE)
                eval_bvp_hr['MER'].append(MER)
                eval_bvp_hr['P'].append(P)

                if not os.path.exists('./visuals/intra_visual/'):
                    os.makedirs('./visuals/intra_visual/')
                eval_hr_save = pd.DataFrame(eval_hr)
                eval_hr_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                    reTrain) + '_fulltest_RhythmFormer' + '_HR.csv')
                eval_bvp_hr_save = pd.DataFrame(eval_bvp_hr)
                eval_bvp_hr_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                    reTrain) + '_' + args.pt + '_fulltest_RhythmFormer' + '_BVP_HR.csv')


                if not os.path.exists('./Result_Model'):
                    os.makedirs('./Result_Model')
                torch.save(my_model,
                           './Result_Model/' + Target_name + '_' + str(iter_num) + '_full_supervised_RhythmFormer')
                print('saveModel As ' + Target_name + '_' + str(iter_num) + '_full_supervised_RhythmFormer')
