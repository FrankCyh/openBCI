import pandas as pd
import matplotlib.pyplot as plt

import os
import lab2_utils as utils

## Defined constants used to load in the .txt EEG files

NUM_ROWS_TO_SKIP = 4  # header data in the .txt file
NUM_CHANNELS = 8  # number of EEG channels
EEG_CHANNEL_PREFIX = utils.NEW_EEG_CHANNEL_PREFIX  # label to help us identify columns with EEG data
SAMPLE_RATE = 250  # Sampling rate of data in Hz

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # directory containing this code
DATA_DIR = "data"  # directory containing recording files to be loaded and plotted
TIMESTAMP_STR = "timestamp"  # name of column with recording timestamps

# Used to plot EEG channels in the same colors as shown on the
# OpenBCI GUI (and also matching the wire colors connected to
# each electrode). Keys are the EEG channel, values are the hex
# code for the corresponding color.
EEG_CHANNEL_COLORS = {
    "1": "#878787",  # gray
    "2": "#a670db",  # purple
    "3": "#697de0",  # blue
    "4": "#6be069",  # green
    "5": "#e6de4e",  # yellow
    "6": "#e6954e",  # orange
    "7": "#eb4444",  # red
    "8": "#703d1f",  # brown
}

## Functions to load and plot EEG data

def load_recording_file(fname):
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
    data_df = pd.read_csv(file_path, skiprows=NUM_ROWS_TO_SKIP)

    utils.clean_eeg_dataframe(data_df)  # does some cleanup
    return data_df


def is_eeg(col_name):
    """ Returns True if the column given by <col_name> contains EEG data, and
    False otherwise.

    col_name: a string representing a column in the dataframe loaded using
    load_recording_file
    """
    pass  # replace this with your code


def plot_eeg_data(data_df):
    """ Plots all EEG channel data found in the pandas dataframe
    <data_df> with respect to time.

    data_df: a Pandas dataframe consisting of EEG data to be plot
    """
    # creates 8 rows and 1 column of subplots
    fig, ax = plt.subplots(
        NUM_CHANNELS, 1, sharex='all', figsize=(15, 15)
    )

    # iterates through columns in the dataframe
    plot_idx = 0
    for col_name in data_df.columns.values:
        # plot EEG channel 1 on the first subplot, and so on
        if col_name.startswith(EEG_CHANNEL_PREFIX):
            ax[plot_idx].plot(data_df[col_name], color=EEG_CHANNEL_COLORS[col_name[-1]])
            plot_idx += 1


    # Adding title, legends, and axes labels
    [ax[i].legend(loc="lower left", fontsize=18) for i in range(NUM_CHANNELS)]
    fig.suptitle("EEG data over time", fontsize=22)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    plt.xlabel("Time (HH:MM:SS)", fontsize=20)
    plt.rcParams['text.usetex'] = True
    fig.text(0.06, 0.5, 'Recorded Signal ($\mu$V)', va='center', rotation='vertical', fontsize=20)
    plt.show()
    plt.rcParams['text.usetex'] = False


if __name__ == "__main__":
    data_df = load_recording_file("sample_data.txt")

    pd.set_option('display.max_columns', None)
    print(data_df)

    plot_eeg_data(data_df)