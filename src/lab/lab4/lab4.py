import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

NUM_ROWS_TO_SKIP = 4  # header data in the .txt file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # directory containing this code
DATA_DIR = "data"  # directory containing recording files to be loaded and plotted
sys.path.insert(0, os.path.dirname(ROOT_DIR)) # add root folder to path
from lab2_3.code.lab2 import NUM_CHANNELS, NUM_ROWS_TO_SKIP, SAMPLE_RATE
from lab2_3.code.lab2_utils import clean_eeg_dataframe
from lab2_3.code.lab3 import (ELECTRODE_MONTAGE, ELECTRODE_NAMES,
                              filter_band_pass, filter_notch_60, show_psd)


## 1. Load the .txt file as pandas DataFrame
def load_recording_file_lab4(
    fname
):
    """
    Returns a pandas dataframe that consists of all the timeseries
    data, with associated column names, from an OpenBCI GUI recording file.
    The filename of the recording file is given by `fname`.

    Arg:
        fname: string representing the name of the recording file to load

    Return: dataframe loaded
    """
    file_path = os.path.join(ROOT_DIR, DATA_DIR, fname)

    # change this code here, you may add more lines
    data_df = pd.read_csv(
        file_path,
        skiprows=NUM_ROWS_TO_SKIP,
        index_col=False, # The “Sample Index” column, because of parcellation issues, must not be used as the index of the DataFrame. Set the optional argument as “index_col=False”
        usecols=range(1, 9) # The recording generates much more data than just EEG recordings; such as accelerometer information, analog recordings, timestamps, etc. Since the only channels of interest are located in columns 2 through 9, set the optional argument as “usecols=range(1,9)”
    )

    # Set the name of the pandas DataFrame’s columns as ['FP1', 'FP2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'] in that specific order.
    data_df.columns = ELECTRODE_NAMES

    return data_df[500:]

## 2. Convert the DataFrame to a numpy array and Create an MNE RawArray
def construct_mne_from_df(
    data_df
) -> mne.io.RawArray:
    """ Returns a numpy array of dimension (# of EEG channels) x
    (# of samples), containing only EEG channel data present in
    <data_df>. The order of the rows is in ascending numeric order
    found in the initial file ordering:

        EEG Channel 1, 2, ... (# of channels)

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2.load_recording_file
    """
    
    #$ 2.1 Before loading the data into MNE you have to make sure it is in the correct format.
    # MNE employs the convention of “channels x samples” as opposed to OpenBCI’s “samples x channels”. For this, it is necessary to transpose the data frame.
    eeg_data = np.array(data_df.values.T)

    # MNE’s voltage convention is to measure in Volts (V), whereas OpenBCI records in microVolts (µV). Therefore, you must convert to the appropriate units. Multiplying the full data frame by 1e-6 is the recommended method.
    eeg_data = eeg_data * 1e-6
    
    #$ 2.2 Now that the data is ready for MNE, you will create an MNE array and its underlying information.
    mne_info = mne.create_info(
        ch_names=ELECTRODE_NAMES,
        sfreq=SAMPLE_RATE,
        ch_types=['eeg'] * NUM_CHANNELS,
    )
    mne_raw_obj = mne.io.RawArray(eeg_data, mne_info) # https://mne.tools/stable/generated/mne.io.RawArray.html
    
    # Create a DigMontage object with electrode positions
    montage = mne.channels.make_dig_montage(
        ch_pos=ELECTRODE_MONTAGE,
    ) # https://mne.tools/stable/generated/mne.channels.make_dig_montage.html#mne-channels-make-dig-montage

    # Set the montage for the Raw object
    mne_raw_obj.set_montage(montage) # Warning: RuntimeWarning: Fiducial point nasion not found, assuming identity unknown to head transformation
    
    return mne_raw_obj

## 3. Apply bandpass filter from 0.1 to 100 Hz
def plot_mne_raw_obj(
    mne_raw_obj: mne.io.RawArray,
    duration=10,
    scalings=dict(eeg=200e-6), # plot signals that are 400µV peak-to-peak
    file_name=None,
    file_dir=None,
):
    """ Plots the data from an MNE Raw object, with each EEG channel
    plotted in a different color.

    mne_raw_obj: an MNE Raw object
    """
    mne_raw_obj.plot(
        duration=duration,
        scalings=scalings,
        clipping=None, # Some of the signals are larger than the set scaling, to see the full signal set “clipping=None”.
    ) # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot  
    
    if file_name:
        file_name_png = file_name.split('.')[0] + "_raw" + '.png'
        assert os.path.exists(file_dir)
        os.chdir(file_dir)
        plt.savefig(file_name_png)

## 4. Preliminary plotting and rejection of bad channels
def reject_bad_channels(
    mne_raw_obj: mne.io.RawArray,
    bad_channel_l: list,
) -> mne.io.RawArray:
    mne_raw_obj.info["bads"].extend(bad_channel_l)
    return mne_raw_obj

def lab4_routine(file_name):
    # 1. Load the .txt file as pandas DataFrame
    data_df = load_recording_file_lab4(os.path.join(ROOT_DIR, DATA_DIR, file_name))
    
    # 2. Convert the DataFrame to a numpy array and create an MNE RawArray
    mne_raw_obj = construct_mne_from_df(data_df)
    
    # 3. Apply bandpass filter from 0.1 to 100 Hz
    mne_raw_obj = filter_band_pass(mne_raw_obj, 0.1, 100)
    mne_raw_obj = filter_notch_60(mne_raw_obj)
    
    # 4. Preliminary plotting and rejection of bad channels
    mne_raw_obj = reject_bad_channels(mne_raw_obj, ["C3", "C4"])
    plot_mne_raw_obj(
        mne_raw_obj,
        file_name=file_name,
        file_dir=os.path.join(ROOT_DIR, DATA_DIR, "img"),
    )
    
    # 5. Plot the spectral response for each channel
    show_psd(
        mne_raw_obj,
        fmin=1,
        fmax=30,
        file_name=file_name,
        file_dir=os.path.join(ROOT_DIR, DATA_DIR, "img"),
    )
    
    # 6. Create the ICA object and fit it to the data
    mne_ica_obj = mne.preprocessing.ICA() # https://mne.tools/stable/generated/mne.preprocessing.ICA.html
    mne_ica_obj.fit(mne_raw_obj) # https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.fit

    # 7. Visual inspection of components and frequency response
    fig_l = mne_ica_obj.plot_properties(
        mne_raw_obj,
        picks=range(6),
        psd_args=dict(fmin=1, fmax=30),
    ) # https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_properties
    for i, fig in enumerate(fig_l):
        fig.savefig(os.path.join(ROOT_DIR, DATA_DIR, "img", f"ica_{file_name.split('.')[0]}_{i + 1}.png"))

if __name__ == "__main__":
    for file in ("RecA.txt", "RecB.txt"):
        lab4_routine(file)
    
    print("Done")