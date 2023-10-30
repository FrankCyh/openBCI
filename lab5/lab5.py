import time
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations
import numpy as np

def calculate_alpha_beta_ratio(
    port="COM4",
):
    broad_id = BoardIds.CYTON_BOARD
    broad_desc = BoardShim.get_board_descr(broad_id)
    broad_sampling_rate = int(broad_desc["sampling_rate"])
    