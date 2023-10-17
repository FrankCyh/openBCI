import os
import numpy as np
import mne
import lab2
import lab2_utils

ELECTRODE_NAMES = ['FP1', 'FP2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
ELECTRODE_MONTAGE = {
    "FP1": np.array([-3.022797, 10.470795, 7.084885]),
    "FP2": np.array([2.276825, 10.519913, 7.147003]),
    "C3": np.array([-7.339218, -0.774994, 11.782791]),
    "C4": np.array([6.977783, -1.116196, 12.059814]),
    "P7": np.array([-7.177689, -5.466278, 3.646164]),
    "P8": np.array([7.306992, -5.374619, 3.843689]),
    "O1": np.array([-2.681717, -9.658279, 3.634674]),
    "O2": np.array([2.647095, -9.638092, 3.818619])
}

BAND_START = 0.1
BAND_STOP = 120


def get_eeg_as_numpy_array(data_df):
    """ Returns a numpy array of dimension (# of EEG channels) x
    (# of samples), containing only EEG channel data present in
    <data_df>. The order of the rows is in ascending numeric order
    found in the initial file ordering:

        EEG Channel 1, 2, ... (# of channels)

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2.load_recording_file
    """
    eeg_data = data_df[[f"eeg ch{x}" for x in range(1, 9)]].values.T
    # `.reshape(len(ELECTRODE_NAMES), -1)`, not needed as I have already transposed the data
    return np.array(eeg_data)


def construct_mne(data_df):
    """ Returns an MNE Raw object, consisting of lab2.NUM_CHANNELS
    channels of EEG data.

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2.load_recording_file
    """
    
     # Convert column name from "eeg ch<num>" to dict key in `ELECTRODE_MONTAGE`
    eeg_data = data_df[[f"eeg ch{x}" for x in range(1, 9)]][500:]
    renamed_eeg_data = eeg_data.rename(columns=dict(zip(eeg_data.columns, ELECTRODE_MONTAGE.keys()))).values.T

    # Create an MNE Info object with channel names and sampling rate
    ch_types = ['eeg'] * lab2.NUM_CHANNELS
    sfreq = data_df.shape[0] / (data_df['timestamp'].iloc[-1] - data_df['timestamp'].iloc[1]).total_seconds() # convert `pd.Timedelta` to seconds
    
    info = mne.create_info(
        ch_names=ELECTRODE_NAMES,
        ch_types=ch_types,
        sfreq=sfreq
    ) # https://mne.tools/stable/generated/mne.create_info.html
    
    # Create an MNE RawArray object with the EEG data and Info object
    mne_raw_obj = mne.io.RawArray(renamed_eeg_data, info) # https://mne.tools/stable/generated/mne.io.RawArray.html
    
    # Create a DigMontage object with electrode positions
    montage = mne.channels.make_dig_montage(
        ch_pos=ELECTRODE_MONTAGE,
    ) # https://mne.tools/stable/generated/mne.channels.make_dig_montage.html#mne-channels-make-dig-montage

    # Set the montage for the Raw object
    mne_raw_obj.set_montage(montage) # Q: RuntimeWarning: Fiducial point nasion not found, assuming identity unknown to head transformation
    
    return mne_raw_obj


def show_psd(data_mne, fmin=0, fmax=np.inf, file_name=None, picks=None):
    """ Plots the power spectral density of the EEG signals in
    `data_mne`, limiting the range of the horizontal axis of the plot to
    [fmin, fmax].

    data_mne: MNE Raw object
    fmin: lower end of horizontal axis range
    fmax: upper end of horizontal axis range
    """
    # Compute the power spectral density of the EEG signals
    spectrum = data_mne.compute_psd(fmin=fmin, fmax=fmax, picks=picks) # https://mne.tools/dev/generated/mne.io.Raw.html#mne.io.Raw.compute_psd

    # Plot the power spectral density
    plt = spectrum.plot(picks=picks) # https://mne.tools/dev/generated/mne.time_frequency.Spectrum.html#mne.time_frequency.Spectrum.plot
    if file_name:
        #plt.ylim([110, 170]) # limit y-axis to [110, 170] for lab3
        # error: 'MNELineFigure' object has no attribute 'ylim'
        file_name_png = file_name.split('.')[0] + "_spectrum" + '.png'
        # set working directory to `lab2-3/code`
        os.chdir(os.path.join(lab2.ROOT_DIR, lab2.DATA_DIR, "img"))
        plt.savefig(file_name_png)


def filter_band_pass(data_mne, band_start=BAND_START, band_stop=BAND_STOP):
    """ Mutates `data_mne`, applying a band-pass filter
    with band defined by `band_start` and `band_stop`, where
    `band_start` < `band_stop`.

    data_mne: MNE Raw object
    """
    return data_mne.filter(
        l_freq=band_start,
        h_freq=band_stop,
    ) # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter


def filter_notch_60(data_mne):
    """ Mutates `data_mne`, applying a notch filter
    to remove 60 Hz electrical noise

    data_mne: MNE Raw object
    """
    return data_mne.notch_filter(
        freqs=60,
    )
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter

def plot_spectrum_from_lab2():
    for mode in ["open", "closed"]:
        for i in range(1, 4):
            file_name = f"{mode}_{i}.txt"
            data_df = lab2.load_recording_file(file_name)
            data_mne = construct_mne(data_df)
            data_mne = filter_band_pass(data_mne, band_start=0, band_stop=30)
            data_mne = filter_notch_60(data_mne)
            show_psd(
                data_mne,
                fmin=0,
                fmax=30, # only looking for 10Hz
                file_name=file_name,
                picks=["C3", "O2", "O1"] # Plotting O1 and O2 at the same time have problems
            )


if __name__ == "__main__":
    
    #data_df = lab2.load_recording_file("sample_data.txt")
    #data_mne = construct_mne(data_df)
    #data_mne = filter_band_pass(data_mne, band_start=0, band_stop=120)
    #data_mne = filter_notch_60(data_mne)
    #show_psd(data_mne)
    
    plot_spectrum_from_lab2()
    
    print("Done")