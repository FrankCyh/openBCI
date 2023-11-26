import mne
import pandas as pd
import datetime
import time

from utils.database import *


def clean_eeg_dataframe(
    data_df: pd.DataFrame,
    montage: list[dict] = ELECTRODE_MONTAGE_DEFAULT,
) -> pd.DataFrame:
    """
    Apply renaming to panda DataFrame `data_df`, get electrode value if the electrode is in `ELECTRODE_MONTAGE_DEFAULT`. Timestamp-ify timestamps, and remove unnecessary columns.

    Arg:
        data_df: a pandas dataframe read directly from an OpenBCI recording file.
        montage: a list of dicts, each dict contains the name and position of an electrode
    """
    montage_electrode_idx_l = [x["num"] for x in montage]
    
    data_df.columns = [col.strip() for col in data_df.columns]
    rename_mapping = {
        f"EXG Channel {x}": f"channel_{x}_{ELECTRODE_MONTAGE_DEFAULT[x]['name']}" for x in montage_electrode_idx_l
    }
    rename_mapping.update({
        "Timestamp (Formatted)": "timestamp",
        "Sample Index": "index",
    }) # keeping only the formatted timestamp and discard the raw timestamp
    data_df.rename(columns=rename_mapping, inplace=True)

    if "timestamp" in data_df.columns:
        data_df["timestamp"] = pd.to_datetime(data_df["timestamp"]) # convert to datetime object
    elif "Timestamp" in data_df.columns: # timestamp in RAW format, when streaming from Cython chip
        data_df["timestamp"] = pd.to_datetime(data_df["Timestamp"], unit="s") # convert to datetime object from unix timestamp

    data_df = data_df[[col for col in data_df.columns.values if col.startswith("channel")] + ["timestamp", "index"]] # extract only 10 channels of interest

    return data_df


def get_eeg_from_txt_as_df(
    txt_path: str,
    montage: list[dict] = ELECTRODE_MONTAGE_DEFAULT,
) -> pd.DataFrame:
    """
    Returns a pandas dataframe that consists of all the timeseries data, with associated column names.

    Arg:
        txt_path: _description_
        montage: a list of dicts, each dict contains the name and position of an electrode
    """
    assert txt_path.endswith(".txt"), "Invalid file type, must be .txt file"
    assert os.path.exists(txt_path), "Invalid path to the file that contains the recorded streaming data"
    data_df = pd.read_csv(txt_path, skiprows=NUM_ROWS_TO_SKIP)
    data_df = clean_eeg_dataframe(data_df, montage)
    return data_df


def get_eeg_from_df_as_numpy_array(
    data_df: pd.DataFrame,
    montage: list[dict] = ELECTRODE_MONTAGE_DEFAULT,
):
    """
    Returns a numpy array of dimension (# of EEG channels) x (# of samples), containing only EEG channel data present in <data_df>. The order of the rows is in ascending numeric order found in the initial file ordering:
        EEG Channel 1, 2, ... (# of channels)

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2.load_recording_file
    """
    eeg_data = data_df[[f"channel_{x}_{montage[x]['name']}" for x in range(1, 9)]].values.T
    # `.reshape(len(ELECTRODE_NAMES), -1)`, not needed as I have already transposed the data
    return np.array(eeg_data)


def get_eeg_from_txt_as_numpy_array(
    txt_path: str,
) -> np.ndarray:
    """
    Returns a numpy array from file

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2.load_recording_file
    """
    data_df = get_eeg_from_txt_as_df(txt_path)
    return get_eeg_from_df_as_numpy_array(data_df)

def collect_eeg(
    insn: str,
    sec: int = 5,
):
    """
    Collect EEG data from the OpenBCI board and save it to a file.
    """
    from utils.stream_utils import prepare_session
    board_obj, board_desc = prepare_session()
    
    print(f"Think {insn} in 3")
    time.sleep(1.5)
    board_obj.start_stream()  # use this for default options
    time.sleep(1.3)
    data = board_obj.get_current_board_data(250)
    board_obj.stop_stream()
    
collect_eeg("Motor Imagery")