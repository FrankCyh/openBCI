# Used for testing trained model on each person's data
# Difference with process_train_nov25_newtrain:
# Add: data_four_person_separate
# Delete: print statements for debugging
# Delete: schuffle and split data
# Change: train to inference
# Change: calculate accuracy to print confidence score of each prediction
# Delete: plot training curve
# Delete: save model

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from model import ConvNet

from train_helper import *

import os
import sys
current_path = os.getcwd()
base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))



# ==================================================== Data Loader ====================================================

# TOTAL_TIME = 60
# IMAGE_TIME = 0.5


load_filename_pair = [
    ('OpenBCI-RAW-2023-11-25_19-24-37_yuhe.txt',   'white_time_yuhe.txt'),   # 0: yuhe
    ('OpenBCI-RAW-2023-11-25_19-34-34_xiao.txt',   'white_time_xiao.txt'),   # 1: xiao
    ('OpenBCI-RAW-2023-11-25_19-45-02_jiayi.txt',  'white_time_jiayi.txt'),  # 2: jiayi
    ('OpenBCI-RAW-2023-11-25_19-54-15_leting.txt', 'white_time_leting.txt')  # 3: leting
]

is_nov02 = False

data = []  # four person's data concatenated
data_four_person_separate = []  # four person's data each stored in index 0, 1, 2, 3

for filename_pair in load_filename_pair:
    openbci_data_filename = filename_pair[0]
    white_time_filename = filename_pair[1]

    # read CSV or TXT file (data)
    if is_nov02:
        # data collected on Nov 2 is loaded via CSV
        file_path_csv = os.path.join(base_path, 'data', 'OpenBCI_2023-11-02_p300', 'BrainFlow-RAW_2023-11-02_16-27-02_67.csv')
        df = pd.read_csv(file_path_csv, sep='\t', header=None)
    else:
        # data collected on Nov 25 is loaded via TXT
        df = get_eeg_from_txt_as_df(
            os.path.join(base_path, 'data', 'OpenBCI_2023-11-25_p300', openbci_data_filename),
            ELECTRODE_MONTAGE_P300,
        )

    # select columns of 5 electrodes
    if is_nov02:
        # electrode positions are specific for data collected on Nov 2
        Fz_column = df.iloc[:, 3]
        Cz_column = df.iloc[:, 8]
        P3_column = df.iloc[:, 7]
        Pz_column = df.iloc[:, 6]
        P4_column = df.iloc[:, 5]
    else:
        # reorder electrodes from 0-4, electrode positions are specific for data collected on Nov 25
        Fz_column = df.iloc[:, 1]
        Cz_column = df.iloc[:, 2]
        P3_column = df.iloc[:, 0]
        Pz_column = df.iloc[:, 4]
        P4_column = df.iloc[:, 3]

    Fz_array = Fz_column.to_numpy()
    Cz_array = Cz_column.to_numpy()
    P3_array = P3_column.to_numpy()
    Pz_array = Pz_column.to_numpy()
    P4_array = P4_column.to_numpy()

    x_array = np.column_stack((Fz_array, Cz_array, P3_array, Pz_array, P4_array))

    # select column of timestamp and convert unix timestamp to formatted timestamp
    timestamp_column = df.iloc[:, -2]
    timestamp_array = list(timestamp_column)
    if is_nov02:
        formatted_timestamp_array = [datetime.fromtimestamp(unix_timestamp) for unix_timestamp in timestamp_array]
        formatted_timestamp_array = [timestamp - timedelta(hours=4) for timestamp in formatted_timestamp_array]
    else:
        formatted_timestamp_array = timestamp_array

    # read TXT file (ground truth)
    with open(os.path.join(base_path, 'data', 'OpenBCI_2023-11-25_p300', white_time_filename), 'r') as file:
        lines = file.readlines()

    # initialize lists to store timestamps
    start_timestamp = None
    white_timestamps = []

    # process each line
    for line in lines:
        if "Recording start at:" in line:
            # extract and convert the absolute start timestamp
            start_timestamp_str = line.split(": ", 1)[1].strip()
            start_timestamp = datetime.fromisoformat(start_timestamp_str)
        elif "White image shown at:" in line:
            # extract and convert the relative white image timestamps
            white_timestamp_str = line.split(": ", 1)[1].strip()
            hours, minutes, seconds = map(float, white_timestamp_str.split(':'))
            white_timestamp = timedelta(hours=hours, minutes=minutes, seconds=seconds)
            white_timestamps.append(white_timestamp)

    if is_nov02:
        absolute_white_timestamps = [start_timestamp + delta for delta in white_timestamps]
    else:
        end_timestamp = start_timestamp + timedelta(minutes=3)
        absolute_white_timestamps = [start_timestamp + delta for delta in white_timestamps]
        # don't need the line below, as the 3 min constraint is applied for the formatted_timestamp_array
        #absolute_white_timestamps = [t for t in absolute_white_timestamps if t < end_timestamp]

    # Now:
    # x_array: numpy array of electrode data from headset, 5 columns -> 5 electrodes
    # formatted_timestamp_array: list of timestamp of electrode data from headset, same number of rows as x_array
    # absolute_white_timestamps: list of timestamp of white image appearance from white_image script, duration: 3 mins

    # crop data into length of 125 and combine with label
    # 125: 0.5 seconds of 250 Hz data

    white_timestamp_i = start_timestamp
    next_whitetime_index = 0
    data_index = 0

    while formatted_timestamp_array[data_index] < start_timestamp:
        data_index += 1

    data_one_person = []  # for debugging each person's data

    while formatted_timestamp_array[data_index + 125] < end_timestamp:

        # next segment is black (non-target, 0)
        if formatted_timestamp_array[data_index + 125] < absolute_white_timestamps[next_whitetime_index]:
            # combine data with label 0
            data_i = [x_array[data_index:data_index+125, :], 0]
            data_one_person.append(data_i)
            data.append(data_i)
            data_index += 125

        # next segment is white (target, 1)
        else:
            # advance data_index to the timestamp of next_whitetime_index
            while formatted_timestamp_array[data_index] < absolute_white_timestamps[next_whitetime_index]:
                data_index += 1
            # combine data with label 1
            data_i = [x_array[data_index:data_index+125, :], 1]
            data_one_person.append(data_i)
            data.append(data_i)
            data_index += 125
            # move on to the next white time
            next_whitetime_index += 1

    data_one_person = np.array(data_one_person, dtype=object)
    data_four_person_separate.append(data_one_person)

data = np.array(data, dtype=object)
data_yuhe, data_xiao, data_jiayi, data_leting = data_four_person_separate  # hard-coded order



# ================== load the saved state dictionary as pretrained model, and set to evaluation mode ==================

model = ConvNet()

pretrained_model_name = 'final_model_epoch_50000_bs_128-newtrain-4people-conf.pth'
pretrained_model_path = os.path.join('pretrained_models', 'Nov25_balanced_50000', pretrained_model_name)
model.load_state_dict(torch.load(pretrained_model_path))

model.eval()



# ================================ testing on each person's data or the whole dataset =================================

batch_size = 128  # same as training

test_pred_conf = []
test_pred = []

for b in minibatch(data_leting, batch_size):
    test_batch_pred_conf, test_batch_pred = val_batch_confidence(model, b, batch_size)
    test_pred_conf.append(test_batch_pred_conf)
    test_pred.append(test_batch_pred)

test_pred_conf = np.concatenate(test_pred_conf, axis=0)
test_pred = np.concatenate(test_pred, axis=0)
test_target = data_leting[:, 1].reshape(-1)


print(len(test_pred_conf), len(test_pred), len(test_target))

# print each prediction result and summarize
print("\nPrediction Confidence\tPredicton\tTarget")
total_count, tp, tn, fp, fn = 0, 0, 0, 0, 0
for i in range(len(test_target)):
    print(f"{test_pred_conf[i]}\t{test_pred[i]}\t\t{test_target[i]}")

    total_count += 1
    if test_pred[i] == 1 and test_target[i] == 1:
        tp += 1
    elif test_pred[i] == 0 and test_target[i] == 0:
        tn += 1
    elif test_pred[i] == 1 and test_target[i] == 0:
        fp += 1
    elif test_pred[i] == 0 and test_target[i] == 1:
        fn += 1
print(f'\nTotal: {total_count}')
print(f'True Positive: {tp}')
print(f'True Negative: {tn}')
print(f'False Positive: {fp}')
print(f'False Negative: {fn}')
print(f'Accuracy: {(tp + tn) / total_count}')