import os

import cnn
import csp_extraction
import data_generator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import torch

from utils.database import DATA_DIR

fs = 250
window_length = 7

# import data
DIR_NAME = os.path.dirname(os.path.abspath(__file__))
selected_columns = [3, 4] # C3 C4 channels are all what we need

'''X_imagery_LH, y_imagery_LH = data_generator.read_and_select_columns(os.path.join(DATA_DIR, "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery", "imagery_LH"), selected_columns)
X_imagery_RH, y_imagery_RH = data_generator.read_and_select_columns(os.path.join(DATA_DIR, "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery", "imagery_RH"), selected_columns)

X = np.concatenate((X_imagery_LH, X_imagery_RH), axis=0)
y = np.concatenate((y_imagery_LH, y_imagery_RH), axis=0)'''

X, y = data_generator.read_and_select_columns_txt(os.path.join(DATA_DIR, "OpenBCI-2023-11-25-motor-imagery", "motor"), selected_columns)
X = np.transpose(X, (0, 2, 1))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=300)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=300)


# Test Data Evalutaion
CNN_model = load_model(os.path.join(DIR_NAME, "model_init", "model_motor.h5"))
#csp_test = csp_extraction.extract_feature(X_test, csp_filter, filterbanks, time_windows)
y_test = pd.get_dummies(y_test).values
y = pd.get_dummies(y).values
#test_acc = cnn.model_evaluation(CNN_model, csp_test, y_test)
np.expand_dims(X_test, axis = -1)
test_acc = cnn.model_evaluation(CNN_model, X, y)
