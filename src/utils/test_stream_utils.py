from utils.brainflow_utils import get_band_power
from utils.database import *
from utils.database import DATA_DIR
from utils.stream_utils import *


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
        6   90231.846599   92972.841021  -3055.058784  ...  18742.898851 2023-11-16 15:19:35.002    6.0


        [5 rows x 10 columns]
        channel_0_FP1  channel_1_FP2  channel_2_C3  ...  channel_7_O2               timestamp  index
        4   90241.837828   92998.098492  -3039.636080  ...  18813.105680 2023-11-16 15:19:34.995    4.0
        5   90222.101238   92970.896419  -3068.939217  ...  18742.630630 2023-11-16 15:19:34.998    5.0
        6   90231.846599   92972.841021  -3055.058784  ...  18742.898851 2023-11-16 15:19:35.002    6.0
        7   90251.940817   92993.561088  -3027.499083  ...  18802.488602 2023-11-16 15:19:35.006    7.0
        8   90239.870875   92985.693274  -3040.150170  ...  18817.441919 2023-11-16 15:19:35.011    8.0
        """
        print(i)
#test_mock_stream()



def stream_demo():
    for x in stream(
        #os.path.join(
        #    ROOT_DIR,
        #    "data",
        #    "OpenBCISession_2023-11-16_15-17-12_eye_blink",
        #    "OpenBCI-RAW-2023-11-16_15-19-34_cropped.txt"
        #),
        period=5,
        stride=2,
        stream_sec=10,
    ):
        print(f"Data streamed: {x}")
        channel_l = [xx for xx in x if xx.startswith("channel_")]
        for channel in channel_l:
            delta, theta, alpha, beta = get_band_power(x[channel].values)
            print(f"{channel}:\n\tdelta: {delta}, theta: {theta}, alpha: {alpha}, beta: {beta}")


#test_mock_stream()
stream_demo()
