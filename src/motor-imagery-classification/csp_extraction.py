import numpy as np
from scipy.signal import sosfilt
from scipy import linalg
from scipy.special import binom


def generate_projection(
    data,
    labels,
    filter_bank,
    time_windows,
    no_classes=2,
    m=3,
):
    """
    Generate spatial filters for every timewindow and frequancy band

    Arg:
        data: numpy array of size [NO_trials,channels,time_samples]
        labels: containing the class labels, numpy array of size [NO_trials]
        filter_bank: numpy array containing butter sos filter coeffitions dim  [NO_bands,order,6]
        time_windows: numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]]
        (no_classes): number of classes being predicted
        (m): no of csp that will be taken, no of csp = 2*m

    Return: _description_
    """
    no_bands = filter_bank.shape[0]
    no_time_windows = len(time_windows)
    no_channels = data.shape[1]
    no_trials = len(data)
    no_csp = int(binom(no_classes, 2)) * 2 * m

    # Initialize spatial filter:
    w = np.zeros((no_time_windows, len(filter_bank), no_channels, no_csp))

    # iterate through all time windows
    for window in range(no_time_windows):
        # get start and end point of current time window
        t_start = int(time_windows[window, 0])
        t_end = int(time_windows[window, 1])

        # iterate through all frequency bandwids
        for subband in range(no_bands):
            cov = np.zeros((no_classes, no_trials, no_channels, no_channels)) # sum of covariance depending on the class
            cov_avg = np.zeros((no_classes, no_channels, no_channels))
            cov_cntr = np.zeros(no_classes).astype(int) # counter of class occurence
            for trial in range(no_trials):
                data_filter = sosfilt(filter_bank[subband], data[trial, :, t_start:t_end])

                cur_class_idx = int(labels[trial] - 1)

                # caclulate current covariance matrix
                cov[cur_class_idx, cov_cntr[cur_class_idx], :, :] = np.dot(data_filter, np.transpose(data_filter)) / np.trace(np.dot(data_filter, np.transpose(data_filter)))

                # update covariance matrix and class counter
                cov_cntr[cur_class_idx] += 1

            # calculate average of covariance matrix
            for clas in range(0, no_classes):
                cov_avg[clas, :, :] = np.average(cov[clas, :cov_cntr[clas], :, :], axis=0)

            w[window, subband, :, :] = csp_one_one(cov_avg, no_channels, m, no_classes)
    return w


def csp_one_one(
    cov_matrix,
    no_channels,
    m,
    no_classes=4
) -> np.ndarray:
    """
    Calculate spatial filter for class all pairs of classes 

    Arg:
        cov_matrix: numpy array of size [NO_channels, NO_channels]
        no_channels: number of channels that is recorded
        m: no of csp that will be taken, no of csp = 2*m
        (no_classes): number of classes being predicted    

    Return: Spatial filter numpy array of size [22,NO_csp] 
    """
    n_comb = int(binom(no_classes, 2))

    w = np.zeros((no_channels, n_comb * 2 * m))

    kk = 0 # internal counter
    for cc1 in range(0, no_classes):
        for cc2 in range(cc1 + 1, no_classes):
            w[:, m * 2 * (kk):m * 2 * (kk + 1)] = gevd(cov_matrix[cc1], cov_matrix[cc2], m)
            kk += 1
    return w


def gevd(
    x1,
    x2,
    no_pairs,
) -> np.ndarray:
    """
    Solve generalized eigenvalue decomposition

    Arg:
        x1: numpy array of size [NO_channels, NO_samples]
        x2: numpy array of size [NO_channels, NO_samples]
        no_pairs: number of pairs of eigenvectors to be returned 

    Return: numpy array of 2*No_pairs eigenvectors 
    """
    ev, vr = linalg.eig(x1, x2, right=True)
    evAbs = np.abs(ev)
    sort_indices = np.argsort(evAbs)
    chosen_indices = np.zeros(2 * no_pairs).astype(int)
    chosen_indices[0:no_pairs] = sort_indices[0:no_pairs]
    chosen_indices[no_pairs:2 * no_pairs] = sort_indices[-no_pairs:]

    w = vr[:, chosen_indices] # ignore nan entries
    return w


def extract_feature(data, w, filter_bank, time_windows):
    """
    Calculate features using the precalculated spatial filters

    Arg:
        data: numpy array of size [NO_trials,channels,time_samples]
        w: spatial filters, numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp]
        filter_bank: numpy array containing butter sos filter coeffitions dim  [NO_bands,order,6]
        time_windows: numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 

    Return: features, numpy array of size [NO_trials,(NO_csp*NO_bands*NO_time_windows)] 
    """
    no_csp = len(w[0, 0, 0, :])
    no_time_windows = time_windows.shape[0]
    no_bands = filter_bank.shape[0]
    no_trials = data.shape[0]

    feature_mat = np.zeros((no_trials, no_time_windows, no_bands * no_csp))

    # initialize feature vector
    feat = np.zeros((no_time_windows, no_bands * no_csp))

    # go through all trials
    for trial in range(0, no_trials):
        # iterate through all time windows
        for t_wind in range(0, no_time_windows):
            # get start and end point of current time window
            t_start = int(time_windows[t_wind, 0])
            t_end = int(time_windows[t_wind, 1])

            for subband in range(0, no_bands):
                #frequency filtering
                cur_data_f_s = sosfilt(filter_bank[subband], data[trial, :, t_start:t_end])

                #Apply spatial Filter to data
                cur_data_s = np.dot(np.transpose(w[t_wind, subband]), cur_data_f_s)

                # calculate variance of all channels
                feat[t_wind, no_csp * subband:no_csp * (subband + 1)] = np.log10(np.var(cur_data_s, axis=1) / np.sum(np.var(cur_data_s, axis=1)))


        # store feature in list
        feature_mat[trial, :, :] = feat

    # return np.reshape(feature_mat,(NO_trials,-1))
    return feature_mat[:, :, :, np.newaxis]
