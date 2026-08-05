import os

import numpy as np
import pandas as pd
import torch

from model import MultiModalDepressionClassifier


target_stamp = 125


# ==================== 分段函数 ====================

def segment_sequence(data, seg_len, stride, num_seg):
    segments = []
    for i in range(num_seg):
        start = i * stride
        segments.append(data[start:start + seg_len])
    return segments


# ==================== 加载样本 ====================

def load_sample(sample_id, data_dir, mfcc_dir, audio_dir, gender):
    # ---------- AU ----------
    au_file = os.path.join(data_dir, f"{sample_id}.csv")
    au_df = pd.read_csv(au_file)
    au_features = au_df[
        [
            ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r',
            ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r',
            ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r',
        ]
    ].values
    stamps = len(au_features) // target_stamp
    au_segments = segment_sequence(
        au_features,
        target_stamp,
        75,
        stamps,
    )

    # ---------- MFCC ----------
    mfcc_file = os.path.join(mfcc_dir, f"{sample_id}_mfcc.csv")
    mfcc_df = pd.read_csv(mfcc_file)
    mfcc_features = mfcc_df[
        [f"mfcc_sma[{i}]" for i in range(1, 13)]
    ].values
    mfcc_segments = segment_sequence(
        mfcc_features,
        4 * target_stamp,
        400,
        stamps,
    )

    # ---------- Prosody ----------
    audio_file = os.path.join(audio_dir, f"{sample_id}_eGeMAPSv02.csv")
    audio_df = pd.read_csv(audio_file)
    audio_features = audio_df[
        [
            'Loudness_sma3',
            'F0semitoneFrom27.5Hz_sma3nz',
            'jitterLocal_sma3nz',
            'shimmerLocaldB_sma3nz',
            'HNRdBACF_sma3nz',
        ]
    ].values
    prosody_segments = segment_sequence(
        audio_features,
        4 * target_stamp,
        400,
        stamps,
    )

    # ---------- 合并音频特征 ----------
    audio_segments = [
        np.concatenate([mfcc_segments[i], prosody_segments[i]], axis=1)
        for i in range(stamps)
    ]
    gender_list = [gender] * stamps
    sample_dict = {
        "audio": audio_segments,
        "au": au_segments,
        "gender": gender_list,
    }
    return sample_dict


# ==================== 加载模型 ====================

def load_model(model_path, config):
    model = MultiModalDepressionClassifier(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ==================== 预测函数 ====================

def predict_sample(model, sample):
    audio_list = sample["audio"]
    au_list = sample["au"]
    gender_list = sample["gender"]
    probs = []

    for i in range(len(audio_list)):
        audio = torch.FloatTensor(audio_list[i]).unsqueeze(0)
        au = torch.FloatTensor(au_list[i]).unsqueeze(0)
        gender = torch.FloatTensor([[gender_list[i]]])
        with torch.no_grad():
            prob = model(audio, au, gender).item()
        probs.append(prob)

    # 多段平均
    final_prob = np.mean(probs)
    label = 1 if final_prob > 0.5 else 0
    result = {
        "depression_probability": float(final_prob),
        "prediction": int(label),
        "segment_number": len(probs),
    }
    return result


# ==================== 主程序 ====================

if __name__ == "__main__":
    data_dir = 'F:/视频数据文件/0_data/'
    mfcc_dir = 'F:/视频数据文件/MFCC文件/'
    audio_dir = 'F:/视频数据文件/MFCC文件2/'
    sample_id = 1702974235206
    gender = 1

    # ---------- 加载样本 ----------
    sample = load_sample(
        sample_id,
        data_dir,
        mfcc_dir,
        audio_dir,
        gender,
    )

    # ---------- 配置 ----------
    class Config:
        hidden_dim = 128

    # ---------- 加载模型 ----------
    model = load_model("best_model.pth", Config)

    # ---------- 预测 ----------
    result = predict_sample(model, sample)

    print("预测结果：")
    print(result)
