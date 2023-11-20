import os

import numpy as np

from lab.lab2_3.code.lab2 import NUM_CHANNELS

SAMPLE_RATE = 250  # Sampling rate of data in Hz
NUM_ROWS_TO_SKIP = 4  # header data in the .txt file
HOME_DIR = os.path.dirname(__file__)

ELECTRODE_MONTAGE_DEFAULT = [
    {
        "name": "FP1",
        "num": 0,
        "position": np.array([-3.022797, 10.470795, 7.084885]),
    },
    {
        "name": "FP2",
        "num": 1,
        "position": np.array([2.276825, 10.519913, 7.147003]),
    },
    {
        "name": "C3",
        "num": 2,
        "position": np.array([-7.339218, -0.774994, 11.782791]),
    },
    {
        "name": "C4",
        "num": 3,
        "position": np.array([6.977783, -1.116196, 12.059814]),
    },
    {
        "name": "P7",
        "num": 4,
        "position": np.array([-7.177689, -5.466278, 3.646164]),
    },
    {
        "name": "P8",
        "num": 5,
        "position": np.array([7.306992, -5.374619, 3.843689]),
    },
    {
        "name": "O1",
        "num": 6,
        "position": np.array([-2.681717, -9.658279, 3.634674]),
    },
    {
        "name": "O2",
        "num": 7,
        "position": np.array([2.647095, -9.638092, 3.818619]),
    },
]

ELECTRODE_MONTAGE_FRONTALIS = [x for x in ELECTRODE_MONTAGE_DEFAULT if x["name"] in ["FP1", "FP2"]]

ELECTRODE_MONTAGE_OCCIPITAL = [x for x in ELECTRODE_MONTAGE_DEFAULT if x["name"] in ["O1", "O2"]]

ELECTRODE_NAMES_DEFAULT = [x["name"] for x in ELECTRODE_MONTAGE_DEFAULT]

NUM_CHANNELS_DEFAULT = len(ELECTRODE_MONTAGE_DEFAULT)