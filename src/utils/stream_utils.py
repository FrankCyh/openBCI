import os
import time

import pandas as pd

from utils.data_utils import *
from utils.database import *
from utils.mne_utils import *


def mock_stream(
    txt_path: str,
    period: int = 1,
    stride: int = None,
    montage: list[dict] = ELECTRODE_MONTAGE_DEFAULT,
    mock_sleep: bool = True,
):
    """
    Mock brainflow data stream.
    Implementation: yield data for in `period` seconds, then move forward `stride` seconds, then yield data for `period` seconds, and so on.

    Arg:
        path: path to the file that contains the recorded streaming data. The file should have suffix `txt` and should be generated by openBCI device directly
        period: time interval of stream of data in seconds. If `period` = 1, then the stream will yield 250 samples each time
        stride: starting time between two period. 
            `stride` must be smaller than period, otherwise some data will be skipped
            If not specified, `stride` will be equal to period and the data stream won't overlap
        montage: a list of dicts, each dict contains the name and position of an electrode
        mock_sleep: if True, mock sleep for `stride` seconds
    """
    if stride:
        assert stride <= period, "Invalid stride time, must be smaller than period"
    else:
        stride = period

    data_df = get_eeg_from_txt_as_df(txt_path, montage)

    bottom_line_num = data_df.shape[0]
    curr_line_num = 0
    num_line_read = int(SAMPLE_RATE * period)
    num_line_stride = int(SAMPLE_RATE * stride)

    while (curr_line_num + num_line_read < bottom_line_num):
        yield data_df.iloc[curr_line_num: curr_line_num + num_line_read]
        curr_line_num += num_line_stride
        if mock_sleep:
            time.sleep(stride)