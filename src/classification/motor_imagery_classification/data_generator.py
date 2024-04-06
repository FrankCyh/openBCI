import numpy as np
from scipy.signal import butter
import csv
import os
import torch


def read_and_select_columns(folder_path, columns):
    """
    Reads a CSV file and selects specified columns to produce a 2D array.
    Parse the data to discard the setup time and make all the data to same size.

    Arg:
        folder_path: Path to the CSV file.
        columns: List of column indexes to select.

    Return: A 2D array with size num_trials * 1250 * 3 （data point * length in frequency * channel）
    """

    # Create labels
    data = []
    num_files = 0
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                num_files = num_files + 1
                csv_path = os.path.join(folder_path, filename)

                with open(csv_path, 'r', newline='') as csvfile:
                    # Create a CSV reader object
                    reader = csv.reader(csvfile, delimiter='\t')

                    # Initialize the 2D array to store selected data
                    selected_data = []

                    # Iterate through the rows and select the specified columns
                    for row in reader:
                        selected_row = [row[i] for i in columns]
                        selected_data.append(selected_row)

                    data.append(selected_data)

        if ("LH" in folder_path):
            label = [0] * num_files
        elif ("RH" in folder_path):
            label = [1] * num_files

        # cast data until all data are of size 1250 * 3. first get the smallest data and discard the end data of other until
        # all have the size of the smallest data. Then, discard the start data until all size with 1250 * 3
        #data = np.array(data)
        data_casted = []
        data_final = []
        min_size = 5000
        for data_trial in data:
            min_size = min(len(data_trial), min_size)
        for data_trial in data:
            data_casted.append(data_trial[0:min_size])
        for data_trial in data_casted:
            data_final.append(data_trial[min_size - 1250:min_size])

        return np.array(data_final).astype(float), np.array(label).astype(float)

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    

def read_and_select_columns_txt(folder_path, columns):
    """
    Reads a CSV file and selects specified columns to produce a 2D array.
    Parse the data to discard the setup time and make all the data to same size.

    Arg:
        folder_path: Path to the CSV file.
        columns: List of column indexes to select.

    Return: A 2D array with size num_trials * 1250 * 3 （data point * length in frequency * channel）
    """

    # Create labels
    data = []
    num_files = 0
    label = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                num_files = num_files + 1
                txt_path = os.path.join(folder_path, filename)

                with open(txt_path, 'r', newline='') as txtfile:
                    # Create a CSV reader object
                    reader = csv.reader(txtfile, delimiter=',')

                    # Initialize the 2D array to store selected data
                    selected_data = []
                    row_counter = 0
                    sample_counter = 0

                    # Iterate through the rows and select the specified columns, notice, each data piece
                    # contains 1250 samples of three channels, we discard the first 300 samples from each trial
                    for row in reader:
                        row_counter += 1
                        if row_counter < 300:
                            continue
                        else:
                            if sample_counter < 1250:
                                selected_row = [row[i] for i in columns]
                                selected_data.append(selected_row)
                                sample_counter += 1
                            else:
                                if len(selected_data) == 1250:
                                    data.append(selected_data)
                                    if ("_l" in txt_path):
                                        label.append(0)
                                    elif ("_r" in txt_path):
                                        label.append(1)
                                selected_data = []
                                sample_counter = 0

        return np.array(data).astype(float), label

    except FileNotFoundError:
        print(f"Error: File not found at {txt_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# bandpath filter, channel selection
def load_filterbank(bandwidth, fs, max_freq=32, order=2, ftype='butter'):
    """
    Calculate Filters bank with Butterworth filter  

    Arg:
        bandwidth: (list/int) containing bandwiths ex. [2,4,8,16,32] or 4
        fs: sampling frequency
        (max_freq): max freq used in filterbanks
        (order): The order of filter used
        (ftype): Type of digital filter used

    Return: numpy array containing filters coefficients dimesnions 'butter': [N_bands,order,6] 'fir': [N_bands,order]
    """    
    f_bands = np.zeros((6, 2)).astype(float)

    band_counter = 0

    if type(bandwidth) is list:
        for bw in bandwidth:
            startfreq = 7
            while (startfreq + bw <= max_freq):
                f_bands[band_counter] = [startfreq, startfreq + bw]

                if bw == 1: # do 1Hz steps
                    startfreq = startfreq + 1
                elif bw == 2: # do 2Hz steps
                    startfreq = startfreq + 2
                else: # do 4 Hz steps if Bandwidths >= 4Hz
                    startfreq = startfreq + 4

                band_counter += 1

    if type(bandwidth) is int:
        startfreq = 7
        while (startfreq + bandwidth <= max_freq):
            f_bands[band_counter] = [startfreq, startfreq + bandwidth]
            startfreq = startfreq + bandwidth
            band_counter += 1

        # convert array to normalized frequency
    f_band_nom = 2 * f_bands[:band_counter] / fs
    n_bands = f_band_nom.shape[0]

    filter_bank = np.zeros((n_bands, order, 6))

    for band_idx in range(n_bands):
        if ftype == 'butter':
            filter_bank[band_idx] = butter(order, f_band_nom[band_idx], analog=False, btype='bandpass', output='sos')

    return filter_bank


def smooth_time_mask(X, y, mask_start_per_sample, mask_len_samples):
    """Smoothly replace a contiguous part of all channels by zeros.

    Originally proposed in [1]_ and [2]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    mask_start_per_sample : torch.tensor
        Tensor of integers containing the position (in last dimension) where to
        start masking the signal. Should have the same size as the first
        dimension of X (i.e. one start position per example in the batch).
    mask_len_samples : int
        Number of consecutive samples to zero out.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    """
    batch_size, n_channels, seq_len = X.shape
    t = torch.arange(seq_len, device=X.device).float()
    t = t.repeat(batch_size, n_channels, 1)
    mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
    s = 1000 / seq_len
    mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
            torch.sigmoid(s * (t - mask_start_per_sample - mask_len_samples))
            ).float().to(X.device)
    return X * mask, y