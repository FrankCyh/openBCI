# Difference with process_train_nov25_newtrain:
# After training, immediately load model and check prediction confidence

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

    # print(len(absolute_white_timestamps))
    # print(absolute_white_timestamps[0])
    # print(absolute_white_timestamps[-1])
    # print(end_timestamp)

    # print(len(x_array))
    # print(len(formatted_timestamp_array))

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

    # print(formatted_timestamp_array[data_index - 1])
    # print(start_timestamp)
    # print(formatted_timestamp_array[data_index])

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

    # print(data_one_person.shape)
    # print(data_one_person[0][0].shape)
    # print(data_one_person[15][1])
    # print(data_one_person[16][1])
    # print(data_one_person[17][1])
    # print(data_one_person[18][1])
    # print(data_one_person[19][1])

    # for i in range(20):
    #     if data_one_person[i][1] == 1:
    #         print(i)
    #         print(data_one_person[i][2])
    #         print(data_one_person[i][0][0])

data = np.array(data, dtype=object)
data_yuhe, data_xiao, data_jiayi, data_leting = data_four_person_separate  # hard-coded order



# ========================================= Balance, Schuffle and Split Data ==========================================

# By examing the combined dataset, there is 16.43% target and 83.57% non-target data.
# To balance target and non-target data, oversample the target data by duplicating each of them 5 times.
data_unbalanced = data  # save a copy of data before balancing

#print(len(data), data[0].shape, data[0][0].shape, data[0][1])
data_target_1 = [data_example for data_example in data if data_example[1] == 1]
data_target_1 = np.array(data_target_1, dtype=object)
#print(len(data_target_1), data_target_1[0].shape, data_target_1[0][0].shape, data_target_1[0][1])

for i in range(4):
    data = np.concatenate((data, data_target_1))
#print(len(data), data[0].shape, data[0][0].shape, data[0][1])


# load data
# data format: [(x, y)]
data_size = len(data)

# shuffle data
np.random.seed(1006255446)
shuffle_idx = np.random.permutation(data_size)
data = data[shuffle_idx]

# 70-20-10 split train/test
cutoff_1 = int(data_size * 70 // 100)
cutoff_2 = int(data_size * 90 // 100)
train_data = data[:cutoff_1]
val_data = data[cutoff_1:cutoff_2]
test_data = data[cutoff_2:]

# check balance label in train_data
train_data_size = len(train_data)
train_data_true_count = np.sum([x[1] for x in train_data])
train_data_false_count = train_data_size - train_data_true_count
print('train_data_size, train_data_true_count, train_data_false_count:')
print(train_data_size, train_data_true_count, train_data_false_count)

val_data_size = len(val_data)
val_data_true_count = np.sum([x[1] for x in val_data])
val_data_false_count = val_data_size - val_data_true_count
print('val_data_size, val_data_true_count, val_data_false_count:')
print(val_data_size, val_data_true_count, val_data_false_count)

test_data_size = len(test_data)
test_data_true_count = np.sum([x[1] for x in test_data])
test_data_false_count = test_data_size - test_data_true_count
print('test_data_size, test_data_true_count, test_data_false_count:')
print(test_data_size, test_data_true_count, test_data_false_count)

# DON'T NEED TO BALANCE OUR OWN DATASET
# see colab notebook for balance steps



# ======================================================= Train =======================================================

batch_size = 128

# for debug
# np.random.seed(0)

# train helper functions used to be here

epoch = 50000

# init model
model = ConvNet()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 * 2, weight_decay=1e-2)

model.train()

# for plotting
train_acc_list = []
val_acc_list = []

for curr_epoch in range(epoch):

    # train minibatch
    train_pred = []
    train_data_curr = train_data[np.random.permutation(len(train_data))]
    for b in minibatch(train_data_curr, batch_size):
        train_batch_pred = train_batch(model, criterion, optimizer, b, batch_size)
        train_pred.append(train_batch_pred)
    train_pred = np.concatenate(train_pred, axis=0)

    val_pred = []
    for b in minibatch(val_data, batch_size):
        val_batch_pred = val_batch(model, b, batch_size)
        val_pred.append(val_batch_pred)
    val_pred = np.concatenate(val_pred, axis=0)

    if (curr_epoch + 1) % 100 == 0:
        # calculate acc
        train_target = train_data_curr[:, 1].reshape(-1)
        train_acc = cal_acc(train_pred, train_target)
        val_target = val_data[:, 1].reshape(-1)
        val_acc = cal_acc(val_pred, val_target)

        # print stats
        print(f"epoch: {curr_epoch+1}, train acc: {train_acc}, val acc: {val_acc}")

        # for plotting
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    
    if (curr_epoch + 1) >= 1000 and (curr_epoch + 1) % 200 == 0:
        saved_model_name = f'epoch_{curr_epoch + 1}.pth'
        saved_model_path = os.path.join('saved_models', saved_model_name)
        torch.save(model.state_dict(), saved_model_path)

# for plotting
import matplotlib.pyplot as plt
# Plotting the training curve
plt.figure(figsize=(10, 6))
plt.plot(train_acc_list, label='Training Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
# Save the plot to a file
plt.savefig('output\\training_curve.png')
plt.close()

# test acc
model = model.eval()
test_pred = []
for b in minibatch(test_data, batch_size):
    test_batch_pred = val_batch(model, b, batch_size)
    test_pred.append(test_batch_pred)
test_pred = np.concatenate(test_pred, axis=0)
test_target = test_data[:, 1].reshape(-1)
test_acc = cal_acc(test_pred, test_target)
test_f_score, test_percision, test_recall = cal_f(test_pred, test_target)
print(f"test acc: {test_acc}")
print(f"test percision: {test_percision}, test recall: {test_recall}, test f score: {test_f_score}")

# save the last model
if True:
    # saved_model_name = f'model_epoch_{epoch}_bs_{batch_size}.pth'
    saved_model_name = f'final_model_epoch_{epoch}_bs_{batch_size}-newtrain-4people-conf.pth'
    saved_model_path = os.path.join('saved_models', saved_model_name)
    torch.save(model.state_dict(), saved_model_path)

    # export to onnx
    model.eval()
    dummy_input = torch.randn(1, 1, 125, 5)
    onnx_path = os.path.join('saved_models', 'final.onnx')
    torch.onnx.export(model, dummy_input, onnx_path)



# ================== load the saved state dictionary as pretrained model, and set to evaluation mode ==================

model = ConvNet()

pretrained_model_name = f'final_model_epoch_{epoch}_bs_{batch_size}-newtrain-4people-conf.pth'
pretrained_model_path = os.path.join('saved_models', pretrained_model_name)
model.load_state_dict(torch.load(pretrained_model_path))

model.eval()



# ================================ testing on each person's data or the whole dataset =================================

batch_size = 128  # same as training

test_pred_conf = []
test_pred = []

for b in minibatch(test_data, batch_size):
    test_batch_pred_conf, test_batch_pred = val_batch_confidence(model, b, batch_size)
    test_pred_conf.append(test_batch_pred_conf)
    test_pred.append(test_batch_pred)

test_pred_conf = np.concatenate(test_pred_conf, axis=0)
test_pred = np.concatenate(test_pred, axis=0)
test_target = test_data[:, 1].reshape(-1)

# print result
# print(len(test_pred_conf), len(test_pred), len(test_target))
# print("\nPrediction Confidence\tPredicton\tTarget")
# for i in range(len(test_target)):
#     print(f"{test_pred_conf[i]}\t{test_pred[i]}\t\t{test_target[i]}")

# store test result in test_result.txt
with open('output\\test_result.txt', 'w') as f:
    f.write(f"{len(test_pred_conf)}, {len(test_pred)}, {len(test_target)}\n")
    f.write("\nPrediction Confidence\tPredicton\tTarget\n")
    for i in range(len(test_target)):
        f.write(f"{test_pred_conf[i]}\t\t{test_pred[i]}\t\t\t{test_target[i]}\n")