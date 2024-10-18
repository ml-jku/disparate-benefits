import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

"""
Pytorch Modules
! These metrics should only be used on a full dataset using the evaluate method in utils/train_utils.py or similar.
! Otherwise, averaging could introduce adverse effects.
"""

class Accuracy(nn.Module):

    def forward(self, x, y, a=None):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        y = y.cpu()

        return accuracy(y_pred, y)
    
class BalancedAccuracy(nn.Module):

    def forward(self, x, y, a=None):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        y = y.cpu()

        return balanced_accuracy(y_pred, y)

class ProtectedAttributeAccuracy(nn.Module):

    def forward(self, x, y, a):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        a = a.cpu()

        return accuracy(y_pred, a)

class ProtectedAttributeBalancedAccuracy(nn.Module):

    def forward(self, x, y, a):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        a = a.cpu()

        return balanced_accuracy(y_pred, a)

class AUROC(nn.Module):

    def forward(self, x, y, a=None):

        y_probs = torch.softmax(x, dim=1)[:, 1].cpu()
        y = y.cpu()

        return auroc(y_probs, y)
    
class NLL(nn.Module):

    def forward(self, x, y, a=None):
        nll = list()
        for i in range(len(x)):
            nll.append(- torch.log(x[i, y[i]]))
        return torch.mean(torch.as_tensor(nll))
    
class EOD(nn.Module):

    def forward(self, x, y, a):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        y = y.cpu()
        a = a.cpu()

        return eod(y_pred, y, a)
    
class AOD(nn.Module):

    def forward(self, x, y, a):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        y = y.cpu()
        a = a.cpu()

        return aod(y_pred, y, a)
    
"""
Functions
"""

def accuracy(y_preds, ys):
    return torch.mean((y_preds == ys).float())

def balanced_accuracy(y_preds, ys):
    # True Positives
    TP = torch.sum((ys == 1) & (y_preds == 1)).float()
    # True Negatives
    TN = torch.sum((ys == 0) & (y_preds == 0)).float()
    # False Positives
    FP = torch.sum((ys == 0) & (y_preds == 1)).float()
    # False Negatives
    FN = torch.sum((ys == 1) & (y_preds == 0)).float()

    # Sensitivity (True Positive Rate)
    TPR = TP / (TP + FN)
    # Specificity (True Negative Rate)
    TNR = TN / (TN + FP)

    return 0.5 * (TPR + TNR)

def auroc(y_probs, ys):
    return torch.as_tensor(roc_auc_score(ys, y_probs))

def f1(y_preds, ys):
    TP = torch.sum((ys == 1) & (y_preds == 1)).float()
    FP = torch.sum((ys == 0) & (y_preds == 1)).float()
    FN = torch.sum((ys == 1) & (y_preds == 0)).float()
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return 2 * (precision * recall) / (precision + recall)

def aod(y_preds, y, a, tpr_weight=0.5, absolute=True):
    """
    Average Odds Difference
    """
    TPR_priv, TPR_unpriv = get_tprs(y_preds, y, a)
    FPR_priv, FPR_unpriv = get_fprs(y_preds, y, a)
    if absolute:
        return tpr_weight * torch.abs(TPR_priv - TPR_unpriv) + (1 - tpr_weight) * torch.abs(FPR_priv - FPR_unpriv)
    return tpr_weight * (TPR_priv - TPR_unpriv) + (1 - tpr_weight) * (FPR_priv - FPR_unpriv)
    
def eod(y_preds, y, a, absolute=False):
    """
    Equal Opportunity Difference
    """
    TPR_priv, TPR_unpriv = get_tprs(y_preds, y, a)
    if absolute:
        return torch.abs(TPR_priv - TPR_unpriv)
    return TPR_priv - TPR_unpriv

def spd(y_preds, a, absolute=False):
    """
    Statistical Parity Difference
    """
    positive_rate_priv, positive_rate_unpriv = get_prs(y_preds, a)
    if absolute:
        return torch.abs(positive_rate_priv - positive_rate_unpriv)
    return positive_rate_priv - positive_rate_unpriv

def di(y_preds, a, eps=1e-7):
    """
    Disparate Impact
    """
    positive_rate_priv, positive_rate_unpriv = get_prs(y_preds, a)
    
    return (positive_rate_priv + eps) / (positive_rate_unpriv + eps)

def ad(y_preds, y, a, eps=1e-7, absolute=False):
    """
    Accuracy Difference
    """
    acc_priv = torch.mean((y_preds[a == 1] == y[a == 1]).float())
    acc_unpriv = torch.mean((y_preds[a == 0] == y[a == 0]).float())
    
    if absolute:
        return torch.abs(acc_priv - acc_unpriv)
    return acc_priv - acc_unpriv


def get_prs(y_preds, a, privileged=1, unprivileged=0):
    positive_rate_priv = torch.mean((y_preds[a == privileged] == 1).float())
    positive_rate_unpriv = torch.mean((y_preds[a == unprivileged] == 1).float())
    return positive_rate_priv, positive_rate_unpriv

def get_tprs(y_preds, y, a):
    TPR_priv, _, TPR_unpriv, _ = _compute_rates(y_preds, y, a)
    return TPR_priv, TPR_unpriv

def get_fprs(y_preds, y, a):   
    _, FPR_priv, _, FPR_unpriv = _compute_rates(y_preds, y, a)
    return FPR_priv, FPR_unpriv

"""
Helper
"""

def _compute_rates(y_pred, y, a, privileged=1, unprivileged=0):
    mask_priv = a == privileged
    mask_unpriv = a == unprivileged
    
    # For privileged group
    TP_priv = ((y_pred == 1) & (y == 1) & mask_priv).sum().float()
    FP_priv = ((y_pred == 1) & (y == 0) & mask_priv).sum().float()
    FN_priv = ((y_pred == 0) & (y == 1) & mask_priv).sum().float()
    TN_priv = ((y_pred == 0) & (y == 0) & mask_priv).sum().float()
    
    TPR_priv = TP_priv / (TP_priv + FN_priv)
    FPR_priv = FP_priv / (FP_priv + TN_priv)

    # For unprivileged group
    TP_unpriv = ((y_pred == 1) & (y == 1) & mask_unpriv).sum().float()
    FP_unpriv = ((y_pred == 1) & (y == 0) & mask_unpriv).sum().float()
    FN_unpriv = ((y_pred == 0) & (y == 1) & mask_unpriv).sum().float()
    TN_unpriv = ((y_pred == 0) & (y == 0) & mask_unpriv).sum().float()
    
    TPR_unpriv = TP_unpriv / (TP_unpriv + FN_unpriv)
    FPR_unpriv = FP_unpriv / (FP_unpriv + TN_unpriv)

    return TPR_priv, FPR_priv, TPR_unpriv, FPR_unpriv
