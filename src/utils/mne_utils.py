import matplotlib.pyplot as plt
import mne

from utils.database import *


def construct_mne_from_df(
    data_df,
    montage: list[dict] = ELECTRODE_MONTAGE_DEFAULT,
) -> mne.io.RawArray:
    """
    Returns a numpy array of dimension (# of EEG channels) x (# of samples), containing only EEG channel data present in `data_df`. The order of the rows is in ascending numeric order found in the initial file ordering:
        EEG Channel 1, 2, ... (# of channels)
    Arg:
        data_df: a pandas dataframe, with format as defined by the return value of lab2.load_recording_file
        montage: a list of dicts, each dict contains the name and position of an electrode
    """

    #$ Before loading the data into MNE you have to make sure it is in the correct format.
    data_df = data_df[[x for x in data_df.columns if x.startswith("channel")]] # filter our other columns such as timestamp and index
    # MNE employs the convention of “channels x samples” as opposed to OpenBCI’s “samples x channels”. For this, it is necessary to transpose the data frame.
    eeg_data = np.array(data_df.values.T)

    # MNE’s voltage convention is to measure in Volts (V), whereas OpenBCI records in microVolts (µV). Therefore, you must convert to the appropriate units. Multiplying the full data frame by 1e-6 is the recommended method.
    eeg_data = eeg_data * 1e-6

    #$ Now that the data is ready for MNE, you will create an MNE array and its underlying information.
    mne_info = mne.create_info(
        ch_names=[x["name"] for x in montage],
        sfreq=SAMPLE_RATE,
        ch_types=['eeg'] * len(montage),
    )
    mne_raw_obj = mne.io.RawArray(eeg_data, mne_info) # https://mne.tools/stable/generated/mne.io.RawArray.html

    # Create a DigMontage object with electrode positions
    montage = mne.channels.make_dig_montage(
        ch_pos={x["name"]: x["position"] for x in montage},
    ) # https://mne.tools/stable/generated/mne.channels.make_dig_montage.html#mne-channels-make-dig-montage

    # Set the montage for the Raw object
    mne_raw_obj.set_montage(montage) # Warning: RuntimeWarning: Fiducial point nasion not found, assuming identity unknown to head transformation

    return mne_raw_obj


def filter_band_pass(
    data_mne: mne.io.RawArray,
    band_start=0.1,
    band_stop=120,
) -> mne.io.RawArray:
    """
    Mutates `data_mne`, applying a band-pass filter
    with band defined by `band_start` and `band_stop`, where
    `band_start` < `band_stop`.

    Arg:
        data_mne: MNE Raw object
    """
    return data_mne.filter(
        l_freq=band_start,
        h_freq=band_stop,
    ) # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter


def filter_notch_60(
    data_mne
) -> mne.io.RawArray:
    """
    Mutates `data_mne`, applying a notch filter
    to remove 60 Hz electrical noise

    Arg:
        data_mne: MNE Raw object
    """
    return data_mne.notch_filter(
        freqs=60,
    )
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter


def show_psd(
    data_mne,
    fmin=0,
    fmax=np.inf,
    picks=None,
):
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
    plt = spectrum.plot() # https://mne.tools/dev/generated/mne.time_frequency.Spectrum.html#mne.time_frequency.Spectrum.plot


def plot_mne_raw_obj(
    mne_raw_obj: mne.io.RawArray,
    duration: int = 10,
    scalings=dict(eeg=200e-6), # plot signals that are 400µV peak-to-peak
    save_file_name: str = None,
    save_file_dir: str = None,
):
    """ Plots the data from an MNE Raw object, with each EEG channel
    plotted in a different color.

    mne_raw_obj: an MNE Raw object
    """
    plt = mne_raw_obj.plot(
        duration=duration,
        scalings=scalings,
        clipping=None, # Some of the signals are larger than the set scaling, to see the full signal set “clipping=None”.
    ) # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot
    if save_file_name and save_file_dir:
        assert os.path.exists(save_file_dir)
        os.chdir(save_file_dir)
        file_name_png = save_file_name.split('.')[0] + "_raw" + '.png'
        plt.savefig(file_name_png)
