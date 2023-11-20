from utils.stream_utils import *
from utils.database import *

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from classification.motor_imagery_classification.csp_extraction import *
from classification.motor_imagery_classification.data_generator import *
from utils.database import DATA_DIR



def test_mock_stream():
    for i in mock_stream(
        "/Users/frank.c/Code/BCI/openBCI/data/OpenBCISession_2023-11-16_15-17-12_eye_blink/OpenBCI-RAW-2023-11-16_15-19-34.txt",
        0.020,
        0.008,
    ):
        # As each sample is takes 0.04 seconds, this code will print some like the following:
        """ 
        channel_0_FP1  channel_1_FP2  channel_2_C3  ...  channel_7_O2               timestamp  index
        0       0.000000       0.000000      0.000000  ...      0.000000 2023-11-16 15:19:34.968    0.0
        1   90250.823230   92985.335646  -3044.598167  ...  18720.211830 2023-11-16 15:19:34.984    1.0
        2   90256.433518   92983.927486  -3031.991784  ...  18733.734636 2023-11-16 15:19:34.987    2.0
        3   90256.724090   92998.523176  -3016.054990  ...  18813.016273 2023-11-16 15:19:34.991    3.0
        4   90241.837828   92998.098492  -3039.636080  ...  18813.105680 2023-11-16 15:19:34.995    4.0

        [5 rows x 10 columns]
        channel_0_FP1  channel_1_FP2  channel_2_C3  ...  channel_7_O2               timestamp  index
        2   90256.433518   92983.927486  -3031.991784  ...  18733.734636 2023-11-16 15:19:34.987    2.0
        3   90256.724090   92998.523176  -3016.054990  ...  18813.016273 2023-11-16 15:19:34.991    3.0
        4   90241.837828   92998.098492  -3039.636080  ...  18813.105680 2023-11-16 15:19:34.995    4.0
        5   90222.101238   92970.896419  -3068.939217  ...  18742.630630 2023-11-16 15:19:34.998    5.0
        """
        print(i)


def test_mock_motor():
    # get csp_filter from train data
    fs = 250
    DIR_NAME = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the repo dir
    selected_columns = [3, 4, 8] # C3 Cz C4 channels are all what we need
    X_imagery_LH, y_imagery_LH = read_and_select_columns(os.path.join(DATA_DIR, "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery", "imagery_LH"), selected_columns)
    X_imagery_RH, y_imagery_RH = read_and_select_columns(os.path.join(DATA_DIR, "OpenBCI-RAW-2023-11-02_17-02-05_motor_imagery", "imagery_RH"), selected_columns)
    X = np.concatenate((X_imagery_LH, X_imagery_RH), axis=0)
    y = np.concatenate((y_imagery_LH, y_imagery_RH), axis=0)
    X = np.transpose(X, (0, 2, 1))
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
    for i in mock_stream(
        os.path.join(DATA_DIR, "OpenBCISession_2023-11-16_15-17-12_eye_blink", "OpenBCI-RAW-2023-11-16_15-19-34.txt"),
        5,  # in training, each data is 5 seconds long, so here in testing should be compatible
        0.5,  # generate a prediction per 0.5 second
    ):
        DIR_NAME = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the repo dir
        CNN_model = load_model(os.path.join(DIR_NAME, "model_init", "model_motor.h5")) # load pretrained model
        selected_columns = ['channel_2_C3', 'channel_3_C4', 'channel_0_FP1'] # C3, C4, Cz

        # Extract the selected columns from the DataFrame
        selected_data = i[selected_columns]

        # select columns and Convert the selected columns to a NumPy array
        X_test = selected_data.values
        X_test = np.transpose(X_test, (1, 0))

        # CSP
        X_test = np.expand_dims(X_test, axis=0)
        csp_test = extract_feature(X_test, csp_filter, filterbanks, time_windows)
        
        # predict
        y_pred = np.argmax(CNN_model.predict(csp_test), axis = 1) # prediction: 0 for left hand, 1 for right hand
        print(y_pred)


test_mock_motor()
#test_mock_stream()