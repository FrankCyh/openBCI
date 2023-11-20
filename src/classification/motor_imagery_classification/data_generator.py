import numpy as np
from scipy.signal import butter
import csv
import os


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
