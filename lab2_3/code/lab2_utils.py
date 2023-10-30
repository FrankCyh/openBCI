import pandas as pd

OLD_EEG_CHANNEL_PREFIX = "exg channel"  # original label for EEG columns
NEW_EEG_CHANNEL_PREFIX = "eeg ch"  # new label for EEG columns


def clean_eeg_dataframe(data_df):
    """ Mutates the pandas dataframe given by <data_df> to lowercase all column names,
    rename columns with EEG data to be more understandable, timestamp-ify timestamps,
    and remove unnecessary columns.

    data_df: a pandas dataframe read directly from an OpenBCI recording file.
    """
    data_df.rename(columns={
        old_name: old_name.strip().lower() for old_name in data_df.columns.values
    }, inplace=True)

    data_df.rename(columns={
        old_name: NEW_EEG_CHANNEL_PREFIX + \
                  str(int(old_name.split(OLD_EEG_CHANNEL_PREFIX)[-1].strip()) + 1)
        for old_name in data_df.columns.values
        if old_name.startswith(OLD_EEG_CHANNEL_PREFIX)
    }, inplace=True)

    data_df.drop(
        columns=[col for col in data_df.columns.values if col.startswith("other")] + \
                ["timestamp"], inplace=True
    )

    data_df["timestamp (formatted)"] = pd.to_datetime(
        data_df["timestamp (formatted)"].str.strip(),
    )
    data_df.rename(columns={"timestamp (formatted)": "timestamp"}, inplace=True)

