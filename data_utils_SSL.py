import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
import pandas as pd
import random

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


# 完整保留原始函数
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


# 完整保留原始数据增强函数
def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                     args.minF, args.maxF, args.minBW, args.maxBW,
                                     args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                         args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                         args.minG, args.maxG, args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    else:
        feature = feature
    return feature


# 新增自定义数据集类
class Dataset_AIGC(Dataset):
    def __init__(self, args, csv_path, audio_dir, is_train=False, is_eval=False, algo=0):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.cut = 64600  # ~4秒音频
        self.algo = algo
        self.args = args
        self.is_train = is_train
        self.is_eval = is_eval

        # 创建标签映射
        self.labels = {}
        if not is_eval:
            for idx, row in self.df.iterrows():
                self.labels[row['utt']] = 1 if row['label'] == 'Bonafide' else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        utt_id = row['utt']
        wav_path = os.path.join(self.audio_dir, row['wav_path'])

        # 加载音频
        try:
            X, fs = librosa.load(wav_path, sr=16000)
        except Exception as e:
            print(f"Error loading {wav_path}: {str(e)}")
            # 返回空数据
            X = np.zeros(self.cut)
            fs = 16000

        # 训练时应用数据增强
        if self.is_train and self.algo > 0:
            Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        else:
            Y = X

        # 填充/裁剪
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)

        # 评估模式只返回ID
        if self.is_eval:
            return x_inp, utt_id

        # 训练/验证模式返回标签
        target = self.labels[utt_id]
        return x_inp, target, utt_id