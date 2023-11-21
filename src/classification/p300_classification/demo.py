import sys
import os
src_path = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(src_path)

import os
import pandas as pd
import numpy as np
from datetime import timedelta

import torch
from model import ConvNet
from inference_helper import cal_acc, cal_f_demo, val_single
import time

from utils.database import *
from utils.stream_utils import mock_stream

from utils.database import *

from utils.database import *

TOTAL_TIME = 60  ## later, take 25-55s in dataset 1
IMAGE_TIME = 0.5


def test_mock_p300():

    # get a stride and test using pretrained model
    model = ConvNet()
    pretrained_model_name = 'model_fold_10_epoch_10_bs_128.pth'
    pretrained_model_path = os.path.join(SRC_DIR, "classification", "p300_classification", 'pretrained_models', pretrained_model_name)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    
    # generate label
    white_timestamps = []
    with open(os.path.join(DATA_DIR, 'OpenBCI_2023-11-02_p300', 'white_time_1.txt'), 'r') as file:
        lines = file.readlines()
    for line in lines:
        if "White image shown at:" in line:
            white_timestamp_str = line.split(": ", 1)[1].strip()
            hours, minutes, seconds = map(float, white_timestamp_str.split(':'))
            white_timestamp = timedelta(hours=hours, minutes=minutes, seconds=seconds)
            white_timestamps.append(white_timestamp)
    white_t = []
    for i in range(len(white_timestamps)):
        white_t.append(white_timestamps[i].seconds + white_timestamps[i].microseconds / 1000000)
    label = [0] * int(TOTAL_TIME / IMAGE_TIME)
    for t in white_t:
        if round(t / IMAGE_TIME) + 1 < len(label):
            label[round(t / IMAGE_TIME) + 1] = 1

    label = label[50:]  ## only take 25-60s

    count = 0  # used as index into label list
    label_1_count = 0

    test_pred = []  # used to save test predictions for all streams

    for i in mock_stream(
        os.path.join(
            DATA_DIR,
            "OpenBCI_2023-11-02_p300",
            "demo_cropped.txt"
        ),
        0.5,  # in training, each data is 0.5 seconds long, so here in testing should be compatible
        0.5,  # generate a prediction per 0.5 second
    ):
        # actually select 2-Fz, 7-Cz, 6-P3, 5-Pz, 4-P4
        selected_columns = ['channel_2_C3', 'channel_7_O2', 'channel_6_O1', 'channel_5_P8', 'channel_4_P7']

        # extract the selected columns from the DataFrame
        selected_data = i[selected_columns]

        # generate test data
        test_x = selected_data.values
        
        # concatenate test_x and label
        test_data = [test_x, label[count]]
        test_data = np.array(test_data, dtype=object)
        #print(test_data[0].shape)

        # predict
        start_inference_time = time.perf_counter()  # time

        test_single_pred, test_single_pred_conf = val_single(model, test_data)

        end_inference_time = time.perf_counter()  # time
        inference_duration = end_inference_time - start_inference_time  # time

        test_single_pred = test_single_pred[0]
        test_pred.append(test_single_pred)
        conf = float(test_single_pred_conf[0][0])

        if label[count] == 1:
            label_1_count += 1
            print("\n\n%d/%d \tTarget stimulus at %.1f seconds  \tInference time: %.3f ms/step" % (label_1_count, label_1_count, count / 2, inference_duration * 1000))
            # print("Inference time: %.3fms/step" % (inference_duration * 1000))
            print("Confidence = %.2f" % conf)
            if test_single_pred == 1:
                print("Predicted class: Target \tActual class: Target")
                print("[SUCCESSFUL] Stimulus is indentified")
            else:
                print("Predicted class: Non-target \tActual class: Target")
                print("[FAILED] Stimulus is not indentified")

        count += 1
        if count == 70:  ## only take 25-60s, 70 samples
            break
    
    test_target = np.array(label).reshape(-1)
    test_acc = cal_acc(test_pred, test_target)
    test_f_score, test_percision, test_recall = cal_f_demo(test_pred, test_target)    
    print("\n\nTotal Accuracy: %.4f" % test_acc)
    print("Total Precision: %.4f, Recall: %.4f, F1 Score: %.4f" % (test_percision, test_recall, test_f_score))
    print("\n\n\n")

test_mock_p300()