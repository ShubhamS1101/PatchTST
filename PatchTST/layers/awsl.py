import torch
import torch.nn as nn

class ASWL(nn.Module):
    """Adaptive Scale-Weighted Layer"""
    def __init__(self, num_imfs: int):
        super().__init__()
        # trainable weights for each IMF
        self.weights = nn.Parameter(torch.ones(num_imfs))

    def forward(self, preds):
        """
        preds: [B, pred_len, num_imfs]
        returns: [B, pred_len, 1] (final weighted prediction)
        """
        w = torch.softmax(self.weights, dim=0)    # normalize weights
        weighted = preds * w[None, None, :]       # broadcast multiply
        return weighted.sum(dim=-1, keepdim=True)
