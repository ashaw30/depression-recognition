"""Feature dimensions used by the Baseline model."""

# OpenFace Action Unit intensities (17-D)
AU_COLUMNS = [
    " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r",
    " AU10_r", " AU12_r", " AU14_r", " AU15_r", " AU17_r", " AU20_r", " AU23_r",
    " AU25_r", " AU26_r", " AU45_r",
]

# MFCC coefficients (12-D)
MFCC_COLUMNS = [f"mfcc_sma[{i}]" for i in range(1, 13)]

# Selected eGeMAPS prosody descriptors (5-D)
PROSODY_COLUMNS = [
    "Loudness_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
]

MFCC_DIM = 12
PROSODY_DIM = 5
AU_DIM = 17
AUDIO_DIM = MFCC_DIM + PROSODY_DIM  # 17

# Time-alignment targets (stamp_size = 125)
AUDIO_SEQ_LEN = 500  # 4 * 125
AU_SEQ_LEN = 125
