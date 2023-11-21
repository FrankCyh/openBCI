# Test the pretrained model's performance on a completely new dataset 2

import os
import pandas as pd
import numpy as np
from datetime import timedelta

import torch
from model import ConvNet
from inference_helper import batch_size, minibatch, cal_acc, cal_f, val_batch



# ========================================== load new dataset 2 for testing ===========================================

TOTAL_TIME = 60
IMAGE_TIME = 0.5

import os
current_path = os.getcwd()
base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

# read CSV file (data)
file_path_csv = os.path.join(base_path, 'data', 'OpenBCI_2023-11-02_p300', 'BrainFlow-RAW_2023-11-02_16-27-02_68.csv')
df = pd.read_csv(file_path_csv, sep='\t', header=None)

# select columns of 5 electrodes
Fz_array = (df.iloc[:, 3]).to_numpy()
Cz_array = (df.iloc[:, 8]).to_numpy()
P3_array = (df.iloc[:, 7]).to_numpy()
Pz_array = (df.iloc[:, 6]).to_numpy()
P4_array = (df.iloc[:, 5]).to_numpy()

x_array = np.column_stack((Fz_array, Cz_array, P3_array, Pz_array, P4_array))

# DON'T NEED TO STORE timestamp_array AND formatted_timestamp_array\

#print(x_array.shape)
#print(x_array[0])

# read TXT file (ground truth)
with open(os.path.join(base_path, 'data', 'OpenBCI_2023-11-02_p300', 'white_time_2.txt'), 'r') as file:
    lines = file.readlines()

# initialize lists to store timestamps
# DON'T NEED TO STORE start_timestamp AND absolute_white_timestamps
white_timestamps = []

# process each line
for line in lines:
  if "White image shown at:" in line:
    # extract and convert the relative white image timestamps
    white_timestamp_str = line.split(": ", 1)[1].strip()
    hours, minutes, seconds = map(float, white_timestamp_str.split(':'))
    white_timestamp = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    white_timestamps.append(white_timestamp)

# compute relative white time in seconds
white_t = []
for i in range(len(white_timestamps)):
  white_t.append(white_timestamps[i].seconds + white_timestamps[i].microseconds / 1000000)

# label data (0: non-target, 1: target)
# Since total time is 60s and each image is 0.5s, there are 120 labels
label = [0] * int(TOTAL_TIME / IMAGE_TIME)
for t in white_t:
  if round(t / IMAGE_TIME) + 1 < len(label):
    label[round(t / IMAGE_TIME) + 1] = 1

# Segment data into 119 segments, each corresponding to one label
# dataset 2: start from line 1304, timestamp=1698959953.774875 to match start time of white-black image
data = []

index = 1303
for i in range(119):
  data_i = [x_array[index:index+125, :], label[i]]
  data.append(data_i)
  index += 125

data = np.array(data, dtype=object)
data_size = len(data)

# shuffle data
shuffle_idx = np.random.permutation(data_size)
data = data[shuffle_idx]

# all data for testing
test_data = data[:]

test_data_size = len(test_data)
test_data_true_count = np.sum([x[1] for x in test_data])
test_data_false_count = test_data_size - test_data_true_count
print('test_data_size, test_data_true_count, test_data_false_count:', \
        test_data_size, test_data_true_count, test_data_false_count)



# ================== load the saved state dictionary as pretrained model, and set to evaluation mode ==================

model = ConvNet()

pretrained_model_name = 'model_fold_10_epoch_10_bs_128.pth'
pretrained_model_path = os.path.join('pretrained_models', pretrained_model_name)
model.load_state_dict(torch.load(pretrained_model_path))

model.eval()



# ============================================= testing on new dataset 2 ==============================================

test_pred = []
for b in minibatch(test_data, batch_size):
    test_batch_pred = val_batch(model, b)
    test_pred.append(test_batch_pred)
test_pred = np.concatenate(test_pred, axis=0)

test_target = test_data[:, 1].reshape(-1)
test_acc = cal_acc(test_pred, test_target)
test_f_score, test_percision, test_recall = cal_f(test_pred, test_target)

print(f"test acc: {test_acc}")
print(f"test percision: {test_percision}, test recall: {test_recall}, test f score: {test_f_score}")