__all__ = ['PatchTST']

from typing import Optional
import torch
from torch import nn, Tensor

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


# ---------------- AWSL ---------------- #
class ASWL(nn.Module):
    """Adaptive Scale-Weighted Layer"""
    def __init__(self, num_imfs: int):
        super().__init__()
        # learnable weights for each IMF
        self.weights = nn.Parameter(torch.ones(num_imfs))

    def forward(self, preds: Tensor) -> Tensor:
        """
        preds: [B, pred_len, num_imfs]
        returns: [B, pred_len, 1] (final weighted prediction)
        """
        w = torch.softmax(self.weights, dim=0)          # normalize across IMFs
        weighted = preds * w[None, None, :]             # broadcast multiply
        return weighted.sum(dim=-1, keepdim=True)       # collapse IMF dimension


# ---------------- Model ---------------- #
class Model(nn.Module):
    def __init__(self, configs,
                 max_seq_len: Optional[int] = 1024,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 norm: str = 'BatchNorm',
                 attn_dropout: float = 0.,
                 act: str = "gelu",
                 key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True,
                 pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = 'zeros',
                 learn_pe: bool = True,
                 pretrain_head: bool = False,
                 head_type: str = 'flatten',
                 verbose: bool = False,
                 **kwargs):
        
        super().__init__()
        
        # backbone params
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # --- NEW: VMD + ASWL flags --- #
        self.use_vmd = getattr(configs, "use_vmd", False)
        self.num_imfs = getattr(configs, "num_imfs", 1)
        self.use_aswl = getattr(configs, "use_aswl", False)

        # --- Base backbone(s) --- #
        if decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout, act=act,
                key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout, act=act,
                key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout, act=act,
                key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )

        # --- NEW: attach ASWL if required --- #
        if self.use_vmd and self.use_aswl:
            self.aswl = ASWL(self.num_imfs)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, seq_len, C]
           if use_vmd=True -> C=num_imfs (IMFs instead of original target)
        """
        # --- VMD + ASWL path --- #
        if self.use_vmd:
            # Expect x = [B, seq_len, num_imfs]
            x = x.permute(0, 2, 1)        # -> [B, num_imfs, seq_len]
            preds = self.model(x)         # -> [B, num_imfs, pred_len]
            preds = preds.permute(0, 2, 1)  # -> [B, pred_len, num_imfs]

            if self.use_aswl:
                # Weighted sum across IMFs -> final target prediction
                preds = self.aswl(preds)   # -> [B, pred_len, 1]
            return preds

        # --- Decomposition path --- #
        elif hasattr(self, "decomp_module"):
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
            return x

        # --- Normal PatchTST --- #
        else:
            x = x.permute(0, 2, 1)  # [B, C, L]
            x = self.model(x)       # [B, C, pred_len]
            x = x.permute(0, 2, 1)  # [B, pred_len, C]
            return x
