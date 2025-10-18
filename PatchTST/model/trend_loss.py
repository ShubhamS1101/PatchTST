import torch
import torch.nn as nn

def trend_loss(pred, true, prev_pred, prev_true, alpha=0.2):
    """
    Combines MSE loss with a trend-aware penalty.
    - pred: predicted values (batch, seq_len, features)
    - true: ground truth values (batch, seq_len, features)
    - prev_pred: previous predicted values (batch, seq_len, features)
    - prev_true: previous ground truth values (batch, seq_len, features)
    - alpha: weight for the trend penalty
    """
    mse = nn.MSELoss()(pred, true)
    pred_trend = torch.sign(pred - prev_pred)
    true_trend = torch.sign(true - prev_true)
    trend_penalty = ((pred_trend != true_trend).float()).mean()
    return mse + alpha * trend_penalty
