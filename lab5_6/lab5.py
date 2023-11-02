import enum
import time
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations
import numpy as np


def calculate_alpha_beta_ratio(
    port="COM4",
):
    """
    calculates the alpha beta ratio

    Arg:
        (port): a string containing the name of the port that will be employed
    """
    broad_id = BoardIds.CYTON_BOARD
    broad_desc = BoardShim.get_board_descr(broad_id)
    broad_sampling_rate = int(broad_desc["sampling_rate"])

    ## Set the board parameters
    board_params = BrainFlowInputParams()
    board_params.serial_port = port

    ## Create the board object
    board_obj = BoardShim(
        board_id=broad_id,
        board_params=board_params,
    )

    ## Prepare the board session and start the stream
    board_obj.prepare_session()
    board_obj.start_stream()

    ## Calculate the nearest power of two for spectral power estimation
    sampling_rate_in_pow_2 = DataFilter.get_nearest_power_of_two(broad_sampling_rate) # start by calculating the nearest power of two to the sampling rate

    ## Acquire data
    # Pause the execution of the code to allows time for data to accumulate before processing
    time.sleep(2)
    # Retrieve data from the OpenBCI board
    data = board_obj.get_board_data()

    ## Extract the EEG channel information
    eeg_channel_l = broad_desc["eeg_channels"]
    
    ## Initialize the alpha-beta ratio array
    alpha_beta_ratio = np.zeros(len(eeg_channel_l))
    
    for i, eeg_channel in enumerate(eeg_channel_l):
        data_filter_obj = DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
        power_spectrum = DataFilter.get_psd_welch(eeg_channel, sampling_rate_in_pow_2, sampling_rate_in_pow_2 // 2, sampling_rate_in_pow_2, WindowOperations.HANNING.value) # calculate the power spectrum
        

calculate_alpha_beta_ratio()