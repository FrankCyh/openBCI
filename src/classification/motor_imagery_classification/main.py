import os

import cnn
import csp_extraction
import data_generator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

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
print(X.shape)
X = np.transpose(X, (0, 2, 1))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=300)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=300)

# time windows used in generating csp filters
time_windows = time_windows = np.array([
    [0, 1],
    [0.5, 1.5],
    [1, 2],
    [1.5, 2.5],
    [2, 3],
    [2.5, 3.5],
    [3, 4],
    [3.5, 4.5],
    [4, 5],
    [0, 2],
    [0.5, 2.5],
    [1, 3],
    [1.5, 3.5],
    [2, 4],
    [2.5, 4.5],
    [3, 5],
    [0, 3.5],
    [0.5, 4],
    [1, 4.5],
    [1.5, 5],
    [0, 5]
]) * 250

# CSP
filterbanks = data_generator.load_filterbank(bandwidth=4, fs=fs)

csp_filter = csp_extraction.generate_projection(X_train, y_train, filterbanks, time_windows)

csp_train = csp_extraction.extract_feature(X_train, csp_filter, filterbanks, time_windows)

csp_val = csp_extraction.extract_feature(X_val, csp_filter, filterbanks, time_windows)

y_train = pd.get_dummies(y_train).values
y_val = pd.get_dummies(y_val).values

print(csp_train.shape)
print(csp_val.shape)


# Train CNN
CNN_model = cnn.create_model()
CNN_model.save(os.path.join(DIR_NAME, "model_init", "model_motor.h5"))
CNN_model = load_model(os.path.join(DIR_NAME, "model_init", "model_motor.h5"))
cnn.train_model(csp_train, y_train, csp_val, y_val, CNN_model, 30, 300)
CNN_model.save(os.path.join(DIR_NAME, "model_init", "model_motor.h5"))

# Test Data Evalutaion
CNN_model = load_model(os.path.join(DIR_NAME, "model_init", "model_motor.h5"))
csp_test = csp_extraction.extract_feature(X_test, csp_filter, filterbanks, time_windows)
y_test = pd.get_dummies(y_test).values

test_acc = cnn.model_evaluation(CNN_model, csp_test, y_test)
