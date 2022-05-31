import torch
import numpy as np

from sklearn.metrics import confusion_matrix


def accuracy(model, test_dl, device):

    model.eval()
    corr, total = 0, 0

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=1)
        total += y.shape[0]
        corr += torch.sum(y_pred == y)

    score = corr / float(total)
    return score


def balanced_accuracy(model, test_dl, device):

    model.eval()

    Y_gold, Y_predicted = [], []

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=-1)
        y_pred = y_pred.squeeze().detach().cpu().tolist()

        Y_predicted.extend(y_pred)
        Y_gold.extend(y.detach().cpu().tolist())

    conf_matrix = confusion_matrix(Y_gold, Y_predicted)
    
    true_neg = conf_matrix[0][0]
    false_neg = conf_matrix[1][0]

    true_pos = conf_matrix[1][1]
    false_pos = conf_matrix[0][1]

    TPR = true_pos / float(true_pos + false_neg)
    TNR = true_neg / float(true_neg + false_pos)

    acc = 0.5 * (TPR + TNR)
    return acc


def spouse_consistency(model, test_dl, test_dl_flipped, device):
    
    model.eval()

    predictions_original = []
    for x, _ in test_dl:
        x = x.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=-1)
        y_pred = y_pred.squeeze().detach().cpu().tolist()
        predictions_original.extend(y_pred)
    
    predictions_flipped = []
    for x, _ in test_dl_flipped:
        x = x.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=-1)
        y_pred = y_pred.squeeze().detach().cpu().tolist()
        predictions_flipped.extend(y_pred)

    predictions_original = np.array(predictions_original)
    predictions_flipped = np.array(predictions_flipped)
    
    score = np.mean(predictions_original == predictions_flipped)
    return score
