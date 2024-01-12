import pandas as pd

import os
import sys
current_path = os.getcwd()
base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.insert(1, os.path.join(base_path, 'src'))

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