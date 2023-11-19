import datetime
import enum
import logging
import time

import numpy as np
from brainflow import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import (DataFilter, DetrendOperations,
                                   WindowOperations)


def calculate_alpha_beta_ratio(
    port="/dev/cu.usbserial-DP04VYIB",
):
    """
    Calculates the alpha beta ratio

    Arg:
        (port): a string containing the name of the port that will be employed
    """
    broad_id = BoardIds.CYTON_BOARD
    broad_desc = BoardShim.get_board_descr(broad_id)
    broad_sampling_rate = int(broad_desc["sampling_rate"])

    #$ Set the board parameters
    board_params = BrainFlowInputParams()
    board_params.serial_port = port

    #$ Create the board object
    board_obj = BoardShim(
        board_id=broad_id,
        input_params=board_params,
    )

    #$ Prepare the board session and start the stream
    board_obj.prepare_session()
    board_obj.start_stream()

    #$ Calculate the nearest power of two for spectral power estimation
    sampling_rate_in_pow_2 = DataFilter.get_nearest_power_of_two(broad_sampling_rate) # start by calculating the nearest power of two to the sampling rate

    #$ Acquire data
    # Pause the execution of the code to allows time for data to accumulate before processing
    time.sleep(3) # filter out the first 3 seconds of data collected

    timestamp = 0
    NUM_TIMESTAMP = 50
    # get current timestamp
    data = board_obj.get_board_data() # clear all data before enterting the loop
    while (timestamp < NUM_TIMESTAMP):
        time.sleep(1) # should get approximately 250 * 2 = 500 samples

        # Retrieve data from the OpenBCI board
        data = board_obj.get_board_data()

        #$ Extract the EEG channel information
        eeg_channel_l = broad_desc["eeg_channels"]

        #$ Initialize the alpha-beta ratio array
        alpha_beta_ratio = np.zeros((NUM_TIMESTAMP, len(eeg_channel_l),))

        logging.info(f"Timestamp {timestamp}".center(40, "="))
        for i, eeg_channel in enumerate(eeg_channel_l):
            data_filter_obj = DataFilter()
            data_filter_obj.detrend(data[i], DetrendOperations.LINEAR.value) # initialize a DataFilter object with the detrend() method
            power_spectrum = data_filter_obj.get_psd_welch(
                data=data[i],
                nfft=sampling_rate_in_pow_2 // 2, # sampling_rate_in_pow_2
                # using `sampling_rate_in_pow_2` sometimes results in error, dividing it by 2 solve error
                overlap=sampling_rate_in_pow_2 // 4, # sampling_rate_in_pow_2 // 2
                sampling_rate=broad_sampling_rate,
                window=WindowOperations.HANNING.value,
            ) # calculate the power spectrum

            # Extract the power of the alpha and beta bands
            alpha_power = data_filter_obj.get_band_power(
                psd=power_spectrum,
                freq_start=7,
                freq_end=13,
            )
            beta_power = data_filter_obj.get_band_power(
                psd=power_spectrum,
                freq_start=14,
                freq_end=30,
            )
            alpha_beta_ratio[timestamp][i] = alpha_power / beta_power
            logging.info(f"Alpha beta ratio for channel {eeg_channel} is {round(alpha_beta_ratio[timestamp][i], 2)}")
        logging.info(f"Average alpha beta ratio is {round(np.mean(alpha_beta_ratio[timestamp][~np.isnan(alpha_beta_ratio[timestamp])]), 2)}")
        logging.info("\n")
        timestamp += 1

    # Check the status of the session using the is_prepared() method from the board object (created in step 4.a), and stop streaming using the release_session() method.
    if board_obj.is_prepared():
        board_obj.stop_stream()
        board_obj.release_session()

    return np.mean(alpha_beta_ratio)

## Set logging configuration
logging.basicConfig(
    level=logging.INFO,
    filename=f'{datetime.datetime.now().strftime("%m-%d_%H-%M")}.log',
    filemode='w',
    datefmt='%m-%d-%H:%M:%S',
    format='[%(asctime)s] %(name)s - %(funcName)-15s - %(levelname)-7s - %(message)s'
)

#$ log to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(console)

#calculate_alpha_beta_ratio("/dev/cu.usbserial-DP04VYIB") # channel 16?
calculate_alpha_beta_ratio("/dev/cu.usbserial-DP04WG3B") # channel 12