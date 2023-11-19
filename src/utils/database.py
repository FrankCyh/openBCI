import os

import numpy as np

SAMPLE_RATE = 250  # Sampling rate of data in Hz
NUM_ROWS_TO_SKIP = 4  # header data in the .txt file
NUM_CHANNELS = 8  # number of EEG channels
HOME_DIR = os.path.dirname(__file__)

ELECTRODE_MONTAGE_DEFAULT = [
    {
        "name": "FP1",
        "position": np.array([-3.022797, 10.470795, 7.084885]),
    },
    {
        "name": "FP2",
        "position": np.array([2.276825, 10.519913, 7.147003]),
    },
    {
        "name": "C3",
        "position": np.array([-7.339218, -0.774994, 11.782791]),
    },
    {
        "name": "C4",
        "position": np.array([6.977783, -1.116196, 12.059814]),
    },
    {
        "name": "P7",
        "position": np.array([-7.177689, -5.466278, 3.646164]),
    },
    {
        "name": "P8",
        "position": np.array([7.306992, -5.374619, 3.843689]),
    },
    {
        "name": "O1",
        "position": np.array([-2.681717, -9.658279, 3.634674]),
    },
    {
        "name": "O2",
        "position": np.array([2.647095, -9.638092, 3.818619]),
    },
]