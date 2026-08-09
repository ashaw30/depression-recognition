import os

import numpy as np
import pandas as pd


target_stamp = 125


def segment_sequence(data, seg_len, stride, num_seg):
    """Slice a sequence into fixed-length segments."""
    segments = []
    for i in range(num_seg):
        start = i * stride
        segments.append(data[start:start + seg_len])
    return segments


def load_sample(sample_id, data_dir, mfcc_dir, audio_dir, gender):
    # ================= AU =================
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
        100,
        stamps,
    )

    # ================= MFCC =================
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

    # ================= Prosody =================
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
    audio_segments = segment_sequence(
        audio_features,
        4 * target_stamp,
        400,
        stamps,
    )

    # ================= Merge audio features =================
    audio_segments = [
        np.concatenate([mfcc_segments[i], audio_segments[i]], axis=1)
        for i in range(stamps)
    ]

    gender_list = [gender] * stamps
    sample_dict = {
        "audio": audio_segments,
        "au": au_segments,
        "gender": gender_list,
    }

    return sample_dict


if __name__ == "__main__":
    data_dir = 'F:/video_data/0_data/'
    mfcc_dir = 'F:/video_data/MFCC/'
    audio_dir = 'F:/video_data/eGeMAPS/'

    sample_id = 1702974235206
    gender = 1

    sample = load_sample(
        sample_id,
        data_dir,
        mfcc_dir,
        audio_dir,
        gender,
    )

    print("Number of segments:", len(sample["audio"]))
    print("Audio shape:", sample["audio"][0].shape)
    print("AU shape:", sample["au"][0].shape)
