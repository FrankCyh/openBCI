import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from model import ConvNet



# ==================================================== Data Loader ====================================================

TOTAL_TIME = 60
IMAGE_TIME = 0.5

import os
current_path = os.getcwd()
base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

# read CSV file (data)
file_path_csv = os.path.join(base_path, 'data', 'OpenBCI_2023-11-02_p300', 'BrainFlow-RAW_2023-11-02_16-27-02_67.csv')
df = pd.read_csv(file_path_csv, sep='\t', header=None)

# select columns of 5 electrodes
Fz_column = df.iloc[:, 3]
Cz_column = df.iloc[:, 8]
P3_column = df.iloc[:, 7]
Pz_column = df.iloc[:, 6]
P4_column = df.iloc[:, 5]

Fz_array = Fz_column.to_numpy()
Cz_array = Cz_column.to_numpy()
P3_array = P3_column.to_numpy()
Pz_array = Pz_column.to_numpy()
P4_array = P4_column.to_numpy()

x_array = np.column_stack((Fz_array, Cz_array, P3_array, Pz_array, P4_array))

# select column of timestamp and convert unix timestamp to formatted timestamp
timestamp_column = df.iloc[:, -2]
timestamp_array = timestamp_column.to_numpy()
formatted_timestamp_array = [datetime.fromtimestamp(unix_timestamp) for unix_timestamp in timestamp_array]
formatted_timestamp_array = [timestamp - timedelta(hours=4) for timestamp in formatted_timestamp_array]

#print(x_array.shape)
#print(x_array[0])

# read TXT file (ground truth)
with open(os.path.join(base_path, 'data', 'OpenBCI_2023-11-02_p300', 'white_time_1.txt'), 'r') as file:
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

absolute_white_timestamps = [start_timestamp + delta for delta in white_timestamps]

# compute relative white time in seconds
white_t = []
for i in range(len(white_timestamps)):
  white_t.append(white_timestamps[i].seconds + white_timestamps[i].microseconds / 1000000)

#print(white_timestamps)
#print(white_t)

# label data (0: non-target, 1: target)
# Since total time is 60s and each image is 0.5s, there are 120 labels

label = [0] * int(TOTAL_TIME / IMAGE_TIME)
for t in white_t:
  label[round(t / IMAGE_TIME) + 1] = 1

#print(len(label))

# Segment data into 120 segments, each corresponding to one label
# dataset 1: start from line 1339, timestamp=1698959840.671408 to match start time of white-black image
data = []

index = 1339
for i in range(120):
  data_i = [x_array[index:index+125, :], label[i]]
  data.append(data_i)
  index += 125

data = np.array(data, dtype=object)



# ============================================== Schuffle and Split Data ==============================================

# load data
# data format: [(x, y)]
data_size = len(data)

# shuffle data
shuffle_idx = np.random.permutation(data_size)
data = data[shuffle_idx]

# 80-20 split train/test
cutoff = int(data_size * 80 // 100)
train_data = data[:cutoff]
test_data = data[cutoff:]

len(train_data)

# check balance label in train_data
train_data_size = len(train_data)
train_data_true_count = np.sum([x[1] for x in train_data])
train_data_false_count = train_data_size - train_data_true_count
print('train_data_size, train_data_true_count, train_data_false_count:')
print(train_data_size, train_data_true_count, train_data_false_count)

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

# divide data into minibatches
def minibatch(data, batch_size):
    start = 0
    while True:

        end = start + batch_size
        yield data[start:end]

        start = end
        if start >= len(data):
            break

# calculate acc
def cal_acc(pred, target):
    assert len(pred) == len(target)
    acc = np.sum(pred == target) / len(pred)
    return acc

def cal_f(pred, target):
    assert len(pred) == len(target)
    tp = 0
    for i in range(len(pred)):
        if pred[i] == target[i] and pred[i] == 1:
            tp += 1
    percision = tp / np.sum(pred == 1)
    recall = tp / np.sum(target == 1)
    f_score = (2 * percision * recall) / (percision + recall)
    return f_score, percision, recall

# train function
def train_batch(model, criterion, optimizer, batch):

    model.zero_grad()

    # forward pass
    ##x = torch.FloatTensor([i for i in batch[:, 0]]).cuda()
    ##x = torch.FloatTensor([i for i in batch[:, 0]])
    x_numpy_array = np.array([i for i in batch[:, 0]])##
    x = torch.FloatTensor(x_numpy_array)##
    _, height, width = x.size()
    x = x.view(min(batch_size, len(x)), 1, height, width)
    ##y = torch.FloatTensor([i for i in batch[:, 1]]).cuda()
    ##y = torch.FloatTensor([i for i in batch[:, 1]])
    y_numpy_array = np.array([i for i in batch[:, 1]])##
    y = torch.FloatTensor(y_numpy_array)##
    pred = model(x)

    # back proporgation
    loss = criterion(pred.view(-1), y)
    loss.backward()
    optimizer.step()

    pred = pred.cpu().detach().numpy().reshape(-1)
    pred = np.array([1 if n >= 0.5 else 0 for n in pred])
    return pred

def val_batch(model, batch):

    with torch.no_grad():

        # forward pass
        ##x = torch.FloatTensor([i for i in batch[:, 0]]).cuda()
        ##x = torch.FloatTensor([i for i in batch[:, 0]])
        x_numpy_array = np.array([i for i in batch[:, 0]])##
        x = torch.FloatTensor(x_numpy_array)##
        _, height, width = x.size()
        x = x.view(min(batch_size, len(x)), 1, height, width)
        ##y = torch.FloatTensor([i for i in batch[:, 1]]).cuda()
        ##y = torch.FloatTensor([i for i in batch[:, 1]])
        y_numpy_array = np.array([i for i in batch[:, 1]])##
        y = torch.FloatTensor(y_numpy_array)##
        pred = model(x)

        pred = pred.cpu().detach().numpy().reshape(-1)
        pred = np.array([1 if n >= 0.5 else 0 for n in pred])
        return pred

epoch = 2

# init model
model = ConvNet()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 * 2, weight_decay=1e-2)

# train loop
# use k-fold validation
k_fold = 10
fold_size = int(train_data_size // k_fold)
for i in range(k_fold):

    # split data into train/val
    val_data_curr_fold = train_data[i*fold_size:(i+1)*fold_size]
    train_data_curr_fold_head = train_data[:i*fold_size]
    train_data_curr_fold_tail = train_data[(i+1)*fold_size:]
    train_data_curr_fold = np.concatenate((train_data_curr_fold_head, train_data_curr_fold_tail))

    # epoch
    model = model.train()
    for curr_epoch in range(epoch):

        # train minibatch
        train_pred = []
        train_data_curr_fold = train_data_curr_fold[np.random.permutation(len(train_data_curr_fold))]
        for b in minibatch(train_data_curr_fold, batch_size):
            train_batch_pred = train_batch(model, criterion, optimizer, b)
            train_pred.append(train_batch_pred)
        train_pred = np.concatenate(train_pred, axis=0)

        val_pred = []
        for b in minibatch(val_data_curr_fold, batch_size):
            val_batch_pred = val_batch(model, b)
            val_pred.append(val_batch_pred)
        val_pred = np.concatenate(val_pred, axis=0)

        # calculate acc
        train_target = train_data_curr_fold[:, 1].reshape(-1)
        train_acc = cal_acc(train_pred, train_target)
        val_target = val_data_curr_fold[:, 1].reshape(-1)
        val_acc = cal_acc(val_pred, val_target)

        # print stats
        print(f"fold: {i}, epoch: {curr_epoch}, train acc: {train_acc}, val acc: {val_acc}")

    # test acc
    model = model.eval()
    test_pred = []
    for b in minibatch(test_data, batch_size):
        test_batch_pred = val_batch(model, b)
        test_pred.append(test_batch_pred)
    test_pred = np.concatenate(test_pred, axis=0)
    test_target = test_data[:, 1].reshape(-1)
    test_acc = cal_acc(test_pred, test_target)
    test_f_score, test_percision, test_recall = cal_f(test_pred, test_target)
    print(f"fold: {i}, test acc: {test_acc}")
    print(f"fold: {i}, test percision: {test_percision}, test recall: {test_recall}, test f score: {test_f_score}")

    # save the model after the last fold
    if i == k_fold - 1:
        model_name = f'model_fold_{i+1}_epoch_{epoch}_bs_{batch_size}.pth'
        model_save_path = os.path.join('saved_models', model_name)
        torch.save(model.state_dict(), model_save_path)