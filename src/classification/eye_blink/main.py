import os
import sys

from utils.data_utils import *
from utils.database import *
from utils.mne_utils import *
from utils.stream_utils import mock_stream

if __name__ == "__main__":
    # Frequency of eye blink is 1 second. Use 0.2 seconds stride to see the pattern
    count = 0
    last_time = [0, 0]
    delta = [0, 0]

    ## Print the change of average in each channel
    for data_df in mock_stream(
        os.path.join(
            os.getenv("ROOT_DIR"),
            "data",
            "OpenBCISession_2023-11-16_15-17-12_eye_blink",
            "OpenBCI-RAW-2023-11-16_15-19-34_cropped.txt"
        ),
        0.1,
        montage=ELECTRODE_MONTAGE_FRONTALIS,
        mock_sleep=False,
    ):
        data_df = data_df[[x for x in data_df.columns if x.startswith("channel")]]
        mean = data_df.mean()
        print(f"Average from {round(count * 0.2, 1)} to {round((count + 1) * 0.2, 1)}")
        count += 1
        delta = [round(mean.iloc[0] - last_time[0]), round(mean.iloc[1] - last_time[1])]
        last_time = [mean.iloc[0], mean.iloc[1]]
        print(f"\t{data_df.columns[0]}: {delta[0]:+}\n\t{data_df.columns[1]}: {delta[1]:+}")

    ## Plot the MNE RawArray in separate windows
    for data_df in mock_stream(
        os.path.join(
            os.getenv("ROOT_DIR"),
            "data",
            "OpenBCISession_2023-11-16_15-17-12_eye_blink",
            "OpenBCI-RAW-2023-11-16_15-19-34_cropped.txt"
        ),
        5,
        montage=ELECTRODE_MONTAGE_OCCIPITAL,
    ):
        mne_raw_obj = construct_mne_from_df(
            data_df,
            montage=ELECTRODE_MONTAGE_OCCIPITAL,
        )
        plot_mne_raw_obj(mne_raw_obj)

    ## Plot the MNE RawArray in one window
    data_df = get_eeg_from_txt_as_df(
        os.path.join(
            os.getenv("ROOT_DIR"),
            "data",
            "OpenBCISession_2023-11-16_15-17-12_eye_blink",
            "OpenBCI-RAW-2023-11-16_15-19-34_cropped.txt"
        ),
        ELECTRODE_MONTAGE_FRONTALIS,
    )
    mne_raw_obj = construct_mne_from_df(
        data_df,
        montage=ELECTRODE_MONTAGE_FRONTALIS,
    )
    plot_mne_raw_obj(
        mne_raw_obj,
        save_file_name="eye_blink.png",
        save_file_dir=os.path.dirname(__file__), # save to current file's directory
    )
    # This plot is a little different from the plot directly from openBCI GUI. I think the reason is mne applied some filters to the data.

    print("End")