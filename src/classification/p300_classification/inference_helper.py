import numpy as np
import torch

batch_size = 128  # should be the same as for training

def minibatch(data, batch_size):
    start = 0
    while True:
        end = start + batch_size
        yield data[start:end]

        start = end
        if start >= len(data):
            break

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

def val_batch(model, batch):

    with torch.no_grad():

        # forward pass
        x_numpy_array = np.array([i for i in batch[:, 0]])
        x = torch.FloatTensor(x_numpy_array)
        _, height, width = x.size()
        x = x.view(min(batch_size, len(x)), 1, height, width)
        y_numpy_array = np.array([i for i in batch[:, 1]])
        y = torch.FloatTensor(y_numpy_array)
        pred = model(x)

        pred = pred.cpu().detach().numpy().reshape(-1)
        pred = np.array([1 if n >= 0.5 else 0 for n in pred])
        return pred



def cal_f_demo(pred, target):
    assert len(pred) == len(target)
    
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(pred)):
        if pred[i] == 1 and target[i] == 1:
            tp += 1
        elif pred[i] == 0 and target[i] == 1:
            fn += 1
        elif pred[i] == 1 and target[i] == 0:
            fp += 1
        elif pred[i] == 0 and target[i] == 0:
            tn += 1
    assert tp + fn + fp + tn == len(pred)

    percision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = (2 * percision * recall) / (percision + recall)
    return f_score, percision, recall

def val_single(model, single_data):

    with torch.no_grad():

        # forward pass
        x_numpy_array = np.array(single_data[0])
        x = torch.FloatTensor(x_numpy_array)
        height, width = x.size()
        x = x.view(1, 1, height, width)
        y_numpy_array = np.array(single_data[1])
        y = torch.FloatTensor(y_numpy_array)
        pred_conf = model(x)

        pred = pred_conf.cpu().detach().numpy().reshape(-1)
        pred = np.array([1 if n >= 0.5 else 0 for n in pred])
        return pred, pred_conf
