import numpy as np

from brainflow import DataFilter, DetrendOperations, WindowOperations
from utils.database import *


def get_band_power(
    data: np.ndarray,
    nfft=SAMPLE_RATE_IN_POW2,
    overlap=SAMPLE_RATE_IN_POW2 // 2,
):
    """
    Get band power of 1d signal

    Arg:
        data: 1d numpy array
        (nfft): FFT Window size, must be even
        (overlap): overlap of FFT Windows, must be between 0 and `nfft`

    Return: band power
    """    
    data_filter_obj = DataFilter()
    data_filter_obj.detrend(data, DetrendOperations.LINEAR.value) # initialize a DataFilter object with the detrend() method
    power_spectrum = data_filter_obj.get_psd_welch(
        data=data,
        nfft=nfft, # sampling_rate_in_pow_2
        # using `sampling_rate_in_pow_2` sometimes results in error, dividing it by 2 solve error
        overlap=overlap, # sampling_rate_in_pow_2 // 2
        sampling_rate=SAMPLE_RATE,
        window=WindowOperations.HANNING.value,
    ) # calculate the power spectrum
    
    delta_power = data_filter_obj.get_band_power(
        psd=power_spectrum,
        freq_start=0.5,
        freq_end=4,
    )
    theta_power = data_filter_obj.get_band_power(
        psd=power_spectrum,
        freq_start=4,
        freq_end=8,
    )
    alpha_power = data_filter_obj.get_band_power(
        psd=power_spectrum,
        freq_start=8,
        freq_end=13,
    )
    beta_power = data_filter_obj.get_band_power(
        psd=power_spectrum,
        freq_start=13,
        freq_end=30,
    )
    return (delta_power, theta_power, alpha_power, beta_power)