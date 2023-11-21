from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from classification.motor_imagery_classification.csp_extraction import *
from classification.motor_imagery_classification.data_generator import *
from utils.database import *
from utils.stream_utils import mock_stream, stream


def test_mock_motor():
    # get csp_filter from train data
    fs = 250
    selected_columns = [3, 4, 8] # C3 Cz C4 channels are all what we need
    X_imagery_LH, y_imagery_LH = read_and_select_columns(os.path.join(DATA_DIR, "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery", "imagery_LH"), selected_columns)
    X_imagery_RH, y_imagery_RH = read_and_select_columns(os.path.join(DATA_DIR, "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery", "imagery_RH"), selected_columns)
    X = np.concatenate((X_imagery_LH, X_imagery_RH), axis=0)
    y = np.concatenate((y_imagery_LH, y_imagery_RH), axis=0)
    X = np.transpose(X, (0, 2, 1))
    label = [0]*23 + [1]*30 + [0]*10 + [1]*20
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=300)
    filterbanks = load_filterbank(bandwidth=4, fs=fs)
    # time windows used in generating csp filters
    time_windows = time_windows = np.array([
        [1.5, 2.5],
        [2, 3],
        [2.5, 3.5],
        [3, 4],
        [3.5, 4.5],
        [4, 5],
        [1.5, 3.5],
        [2, 4],
        [2.5, 4.5],
        [3, 5],
        [1.5, 5]
    ]) * 250
    csp_filter = generate_projection(X_train, y_train, filterbanks, time_windows) # this filter will be used in test below


    # get a stride and test using pretrained model
    CNN_model = load_model(os.path.join(SRC_DIR, "classification", "motor_imagery_classification", "model_init", "model_motor.h5")) # load pretrained model
    correct = 0
    iteration = 0
    for i in stream(
        #os.path.join(
        #    DATA_DIR,
        #    "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery",
        #    "demo.txt"
        #),
        5,  # in training, each data is 5 seconds long, so here in testing should be compatible
        0.5,  # generate a prediction per 0.5 second
    ):
        selected_columns = ['channel_2_C3', 'channel_3_C4', 'channel_7_O2'] # C3, C4, Cz

        # Extract the selected columns from the DataFrame
        selected_data = i[selected_columns]

        # select columns and Convert the selected columns to a NumPy array
        X_test = selected_data.values
        X_test = np.transpose(X_test, (1, 0))

        # CSP
        X_test = np.expand_dims(X_test, axis=0)
        csp_test = extract_feature(X_test, csp_filter, filterbanks, time_windows)

        # predict
        print()
        prediction = CNN_model.predict(csp_test)
        y_pred = np.argmax(prediction, axis=1) # prediction: 0 for left hand, 1 for right hand
        print("Confidential_score Left Hand :", prediction[0][0], " Confidential_score Right Hand :", prediction[0][1])
        if y_pred == 0:
            result = "Left Hand"
        if y_pred == 1:
            result = "Right Hand"   
        if label[iteration] == 0:
            actual_result = "Left Hand"
        if label[iteration] == 1:
            actual_result = "Right Hand"
        print("Predicted class :", result, "  Actual class :", actual_result)
        
        # output acc
        if y_pred == label[iteration]:
            print("Prediction is correct")
            correct += 1
        else:
            print("Prediction is wrong")
        iteration += 1
        print("Accumulated accuracy :", correct/iteration)
        
test_mock_motor()