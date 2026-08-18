"""Cognition stream -- cross-anchor attention injection (paper §3.3, gate §3.4).

A lightweight cross-attention block is inserted after the deep LLaMA layers
(24 and 28 by default):

    h_l <- h_l + g_t * RMSNorm( CrossAttn(Q=h_l, K=V=perception_anchors) )

where `perception_anchors` are the image-patch hidden states taken from the
perception layer (16), and `g_t` is the curriculum gate that ramps 0 -> 1 over
[0.3T, 0.7T] and is fixed at 1 during inference. Because the block sits on the
generation path, the model queries the anchors at *every* decoding step -- this
is guidance at generation time, not a train-only auxiliary alignment.

Two design decisions carry the whole module, and both were arrived at by
debugging a failure where the weights did not move at all:

  1. RMSNorm, not LayerNorm.  LayerNorm subtracts the mean, and that term
     scrambles gradient direction on the backward pass; q/k/v stayed locked at
     their initialisation for 25k steps. RMSNorm only divides by the RMS, so
     direction is preserved -- the same reason LLaMA itself uses RMSNorm.
  2. fp32, not bf16.  With no norm at all the forward magnitude is ~g*0.01, and
     the resulting AdamW update (~1e-7) is smaller than the bf16 quantisation
     step (~1.5e-4), so every update rounds to zero. RMSNorm normalises the
     forward magnitude to ~g*1 and amplifies the backward gradient by ~100x;
     the caller must additionally keep these parameters in fp32.

Further implementation details: multi-head attention with the standard head_dim
scaling (a single 4096-wide head collapses), q/k/v/out_proj all initialised at
std=0.02 -- the same magnitude LLaMA's own self-attention uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """LLaMA-style RMSNorm: divide by the RMS, never subtract the mean.

    The difference that matters here:
      - LayerNorm forward: (x - mean) / sqrt(var + eps) * g + b
      - RMSNorm  forward:  x / sqrt(mean(x^2) + eps) * g
    RMSNorm produces no mean-subtraction term on the backward pass, so the
    gradient direction is preserved instead of being scrambled.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # the RMS can underflow in bf16: compute it in fp32 and cast back
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(input_dtype)


class CrossAnchorAttention(nn.Module):
    # num_heads defaults to 32, matching Table I of the paper (h=32, hidden_dim=4096 => d_h=128)
    def __init__(self, hidden_dim: int, num_heads: int = 32, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # RMSNorm (LLaMA style) instead of LayerNorm: it preserves the gradient
        # direction (no mean subtraction) and amplifies the backward signal enough
        # for q/k/v to actually move under bf16.
        self.rms_norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Important: initialise through reset_parameters() rather than inline, because
        # from_pretrained() re-runs _init_weights and overwrites whatever __init__ set.
        # The training script must call it again after from_pretrained().
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise every weight. Must be called explicitly from outside,
        *after* `from_pretrained()`, which otherwise overwrites these tensors."""
        # q/k/v/out_proj all at std=0.02: LLaMA self-attention's magnitude, numerically
        # stable in bf16
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        # RMSNorm weight starts at 1 (LLaMA's convention), leaving the magnitude of
        # `out` unchanged on the first forward
        nn.init.ones_(self.rms_norm.weight)

    def forward(
        self,
        h: torch.Tensor,
        anchors: torch.Tensor,
        gamma: float = 1.0,
        anchor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h:        [B, T, D]   hidden state of the cognition stream at this layer
            anchors:  [B, N, D]   image-patch hidden states from the perception stream,
                                  already at the same hidden_dim
            gamma:    float       curriculum gate in [0, 1]: 0 early in training, ramped to 1
            anchor_mask: [B, N]   bool, True where the anchor is valid (padding mask);
                                  may be None

        Returns:
            h_new: [B, T, D]
        """
        if gamma == 0.0:
            # nothing is injected early in the curriculum: skip the compute entirely
            return h

        # fp32 island. At |w| ~ 0.02 the bf16 quantisation step is ~1.5e-4, while the
        # cross-attention backward gradient is only ~1e-6 -- every AdamW update rounds
        # to zero and the weights never move. Fix: run the whole block in fp32
        # (parameters and activations) and cast back to bf16 on the way out. The caller
        # is responsible for `.data = .data.float()` on the cross_anchor parameters in
        # the training script.
        input_dtype = h.dtype
        weight_dtype = self.q_proj.weight.dtype  # expected to be fp32
        if weight_dtype != torch.float32:
            # the training script forgot to lift the parameters to fp32; inference still
            # runs fine in bf16, it just cannot learn anything new
            pass
        h_f = h.to(weight_dtype)
        anchors_f = anchors.to(weight_dtype)

        B, T, D = h_f.shape
        N = anchors_f.size(1)

        Q = self.q_proj(h_f).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(anchors_f).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(anchors_f).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if anchor_mask is not None:
            mask = anchor_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        # RMSNorm keeps the gradient direction. Everything inside the fp32 island stays
        # fp32, including the AdamW state (which requires trainable params to be fp32),
        # so updates accumulate correctly.
        h_new_f = h_f + gamma * self.rms_norm(out)
        # cast back to bf16 so fp32 does not cascade into the downstream LLaMA layers
        return h_new_f.to(input_dtype)


def curriculum_gamma(
    global_step: int,
    warmup_start: int,
    warmup_end: int,
    final_value: float = 1.0,
) -> float:
    """
    Curriculum schedule of the gate g_t:
      - step <  warmup_start:               g = 0            (stage 1, plain SFT)
      - warmup_start <= step < warmup_end:  g ramps linearly 0 -> final_value
                                            (stage 2, cognition-perception alignment fades in)
      - step >= warmup_end:                 g = final_value  (stage 3, alignment fully on)
    """
    if global_step < warmup_start:
        return 0.0
    if global_step >= warmup_end:
        return final_value
    progress = (global_step - warmup_start) / max(1, (warmup_end - warmup_start))
    return final_value * progress
