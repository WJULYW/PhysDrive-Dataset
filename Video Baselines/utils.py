""" Utilities """

import numpy as np
import argparse
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter, welch
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.sparse import spdiags


def _detrend(input_signal, lambda_value=100):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, c, h, w = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, c, h, w), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, c, h, w), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    #data = np.append(diffnormalized_data, data, axis=1)
    return diffnormalized_data


def Drop_HR(whole_max_idx, delNum=4):
    Row_Num, Individual_Num = whole_max_idx.shape
    HR = []
    for individual in range(Individual_Num):
        HR_sorted = np.sort(whole_max_idx[:, individual])
        HR.append(np.mean(HR_sorted[delNum:-delNum]))
    return np.array(HR)


def hr_fft(sig, fs=30, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    if len(sort_idx) == 0:
        return [0], sig_f_original, None
    elif len(sort_idx) < 2:
        peak_idx1 = peak_idx[sort_idx[0]]
        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60
        x_hr = np.arange(len(sig)) / len(sig) * fs * 60
        return hr1, sig_f_original, x_hr

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1 - 2 * hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig)) / len(sig) * fs * 60
    return hr, sig_f_original, x_hr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_args():
    parser = argparse.ArgumentParser(description='Train ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 训练参数
    parser.add_argument('-g', '--GPU', dest='GPU', type=str, default='0',
                        help='the index of GPU')
    parser.add_argument('-p', '--pp', dest='num_workers', type=int, default=4,
                        help='num_workers')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=64 * 64 * 64,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-r', '--lora-r', type=int, nargs='?', default=16,
                        help='lora gamma', dest='r')
    parser.add_argument('-alpha', '--lora-alpha', type=int, nargs='?', default=32,
                        help='lora alpha', dest='alpha')
    parser.add_argument('-lemda', '--lemda', type=float, nargs='?', default=0.0000001,
                        help='lemda', dest='lemda')
    parser.add_argument('-w1', '--w1', type=float, nargs='?', default=0.01,
                        help='w1', dest='w1')
    parser.add_argument('-w2', '--w2', type=float, nargs='?', default=0.1,
                        help='w2', dest='w2')
    parser.add_argument('-pr', '--pr', type=float, nargs='?', default=0.1,
                        help='prune_ratio', dest='prune_ratio')
    parser.add_argument('-pt', '--pretrain', dest='pt', type=str, default='resnet18',
                        help='pretrained model')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('-testper', '--test_percent', type=int, default=20,
                        help='test_percent', dest='test_percent')
    parser.add_argument('-rT', '--reTrain', dest='reTrain', type=int, default=0,
                        help='Load model')
    parser.add_argument('-rD', '--reData', dest='reData', type=int, default=0,
                        help='re Data')
    parser.add_argument('-mi', '--max_iter', dest='max_iter', type=int, default=20000,
                        help='re Data')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=0,
                        help='seed')

    # parser.add_argument('-tk', '--topk', type=float, default=20,
    # help='topk positive samples', dest='tk')

    # parser.add_argument('-si', '--standard_interval', type=float, default=2,
    # help='standard interval gap', dest='si')

    parser.add_argument('-tp', '--temperature', type=float, default=0.1,
                        help='temperature for contrastive', dest='tp')

    parser.add_argument('-tr', '--temporal_aug_rate', type=float, default=0.5,
                        help='temporal_aug_rate', dest='temporal_aug_rate')
    parser.add_argument('-sr', '--spatial_aug_rate', type=float, default=0.5,
                        help='spatial_aug_rate', dest='spatial_aug_rate')

    # 图片参数
    parser.add_argument('-f', '--form', dest='form', type=str, default='Resize',
                        help='the form of input img')
    parser.add_argument('-wg', '--weight', dest='weight', type=int, default=36,
                        help='the weight of img')
    parser.add_argument('-hg', '--height', dest='height', type=int, default=36,
                        help='the height of img')
    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=128,
                        help='the num of frames')
    parser.add_argument('-t', '--tgt', dest='tgt', type=str, default='On-Road-rPPG',
                        help='the name of target domain: PURE, UBFC, V4V, UBFC...')
    parser.add_argument('-fd', '--fd', dest='fgt', type=str, default='None',
                        help='the name of target domain: PURE, UBFC, V4V, UBFC...')
    return parser.parse_args()


def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    # HR_pr = (HR_pr - np.min(HR_pr)) / (np.max(HR_pr) - np.min(HR_pr))
    # HR_rel = (HR_rel - np.min(HR_rel)) / (np.max(HR_rel) - np.min(HR_rel))
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel))) / (
            0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


def loss_visual(loss_res, save_path, multi_dataset=None):
    if multi_dataset is not None:
        for key in loss_res.keys():
            # draw the loss curve, x-axis is the number of iterations, y-axis is the loss value
            save_path_temp = save_path.replace('key', key)
            plt.figure()
            for d in multi_dataset.keys():
                plt.plot(multi_dataset[d][key], label=d)
            plt.title('main task loss for ' + key, fontdict={'family': 'Times New Roman', 'size': 15})
            plt.xlabel('iteration', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.ylabel('loss', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.legend(prop={'size': 10, 'family': 'Times New Roman', 'weight': 'bold'})
            plt.savefig(save_path_temp)
    else:
        for key in loss_res.keys():
            # draw the loss curve, x-axis is the number of iterations, y-axis is the loss value
            save_path_temp = save_path.replace('key', key)
            plt.figure()
            plt.plot(loss_res[key])
            plt.title('main task loss for ' + key, fontdict={'family': 'Times New Roman', 'size': 15})
            plt.xlabel('iteration', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.ylabel('loss', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.savefig(save_path_temp)


def hr_cal(bvp_signal, fps=30):
    # Preprocess the signal
    b, a = signal.butter(1, 0.5 / (fps / 2), btype='high')
    preprocessed_signal = signal.filtfilt(b, a, bvp_signal)

    # Find peaks
    peaks, _ = signal.find_peaks(preprocessed_signal,
                                 distance=(fps / 140) * 60)  # Assuming a minimum heart rate of 40 bpm
    if len(peaks) < 2:  # Need at least two peaks to calculate heart rate
        return 0

    # Calculate peak intervals and heart rate
    peak_intervals = np.diff(peaks) / fps
    heart_rate = 60 / peak_intervals.mean()

    return heart_rate


def rr_cal(bvp, fps=30):
    # Calculate PSD using Welch's method and find peak frequency
    peaks, _ = signal.find_peaks(bvp)

    rr_intervals = np.diff(peaks) / 30  # 30是BVP信号的采样率

    frequencies, power_spectrum = signal.welch(rr_intervals)

    respiratory_band = (0.12, 0.4)  # 呼吸频率带的范围
    mask = np.logical_and(frequencies >= respiratory_band[0], frequencies <= respiratory_band[1])
    if np.sum(mask) == 0:
        return 18
    respiratory_power_spectrum = power_spectrum[mask]
    respiratory_frequencies = frequencies[mask]
    respiratory_frequencies = respiratory_frequencies[np.argmax(respiratory_power_spectrum)]
    rr = 60 / 1 * respiratory_frequencies
    return rr


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to the data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def fft_spectrum(data, fs):
    """
    Calculates the FFT spectrum and frequencies.
    """
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 / fs), N // 2)
    psd = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, psd


def MyEval_bvp_hr(bvp_pr, bvp_rel):
    HR_pr, HR_rel = [], []
    for i in range(len(bvp_pr)):
        bvp = np.array(bvp_pr[i]).reshape(-1)
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        bvp = _detrend(bvp)
        # bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        res = hr_cal(bvp)
        # res,_,_ = hr_fft(bvp)
        # print('HR_pr',res)
        HR_pr.append(res)
        bvp = np.array(bvp_rel[i]).reshape(-1)
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        res = hr_cal(bvp)
        # res, _, _ = hr_fft(bvp)
        # print('HR_rel',res)
        HR_rel.append(res)
    num_test_samples = len(HR_pr)
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel))) / (
            0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    # Pearson_FFT = np.corrcoef(HR_pr, HR_rel)
    # correlation_coefficient = Pearson_FFT[0][1]
    # p = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))

    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


def MyEval_bvp_rr(bvp_pr, bvp_rel):
    RR_pr, RR_rel = [], []
    for i in range(len(bvp_pr)):
        bvp = np.array(bvp_pr[i]).reshape(-1)
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        RR_pr.append(rr_cal(bvp))
        bvp = np.array(bvp_rel[i]).reshape(-1)
        bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        RR_rel.append(rr_cal(bvp))
    RR_pr = np.array(RR_pr).reshape(-1)
    RR_rel = np.array(RR_rel).reshape(-1)
    temp = RR_pr - RR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / RR_rel)
    p = np.sum((RR_pr - np.mean(RR_pr)) * (RR_rel - np.mean(RR_rel))) / (
            0.01 + np.linalg.norm(RR_pr - np.mean(RR_pr), ord=2) * np.linalg.norm(RR_rel - np.mean(RR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


def resp_rr(resp, fs=30):
    frequencies, psd = signal.welch(resp, fs, nperseg=256)

    # 找到最大PSD的频率, 限制在呼吸频率的合理范围内（0.1到0.5 Hz）
    breathing_mask = (frequencies >= 0.1) & (frequencies <= 0.5)
    dominant_frequency = frequencies[breathing_mask][np.argmax(psd[breathing_mask])]
    return dominant_frequency


def MyEval_resp_rr(resp_pr, resp_rel):
    RR_pr, RR_rel = [], []
    for i in range(len(resp_pr)):
        resp = np.array(resp_pr[i]).reshape(-1)
        resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
        resp = resp.astype('float32')
        RR_pr.append(resp_rr(resp, fs=30))
        resp = np.array(resp_rel[i]).reshape(-1)
        resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
        resp = resp.astype('float32')
        RR_rel.append(resp_rr(resp, fs=30))
    RR_pr = np.array(RR_pr).reshape(-1)
    RR_rel = np.array(RR_rel).reshape(-1)
    temp = RR_pr - RR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / RR_rel)
    p = np.sum((RR_pr - np.mean(RR_pr)) * (RR_rel - np.mean(RR_rel))) / (
            0.01 + np.linalg.norm(RR_pr - np.mean(RR_pr), ord=2) * np.linalg.norm(RR_rel - np.mean(RR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.
