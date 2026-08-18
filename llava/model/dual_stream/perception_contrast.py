"""Perception stream -- object-level InfoNCE loss (paper §3.2).

For every ground-truth object k in an image:

  * visual anchor  v_k : mean-pool of the layer-16 hidden states over the patch
                         tokens covered by the object's COCO bbox
  * text anchor    t_k : the object's class name through a frozen CLIP text
                         encoder

(v_k, t_k) are positives; all other objects in the batch -- including objects
from other images -- are negatives. The loss is bidirectional (v->t and t->v),
as in CLIP/SigLIP, with a learnable temperature capped at log(100):

    L = -1/K * sum_k log( exp(sim(v_k,t_k)/T) / sum_j exp(sim(v_k,t_j)/T) )

This is object-level contrast, not the whole-sentence mean-pool the earlier
version used, and it cannot short-circuit: `vis_feats` comes strictly from ROI
pooling over image patch tokens (never prompt or answer tokens) and
`text_feats` comes from an independent frozen encoder that never touches the
LLaMA embedding table.

Note the dtype guard in `forward()`. This module runs in fp32 while the
backbone runs in bf16; without the explicit up-cast, `addmm` raises a dtype
mismatch that the caller used to swallow silently -- the perception loss then
contributed nothing for an entire training run.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionContrastLoss(nn.Module):
    """
    Args:
        vis_dim:   width of the perception-layer hidden states (LLaMA-7B = 4096)
        text_dim:  output width of the CLIP text encoder (CLIP-L = 768)
        proj_dim:  width of the shared alignment space (default 512)
        temperature: InfoNCE temperature (default 0.07)
    """
    def __init__(self, vis_dim: int, text_dim: int, proj_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        # a 1-dim tensor of shape [1]: from_pretrained rejects 0-dim Parameters
        self.log_temperature = nn.Parameter(torch.tensor([math.log(1.0 / temperature)]))
        # cap the learnable temperature so the logits cannot blow up (as CLIP does)
        self.log_temperature_max = 4.6052  # log(100)
        self._init_temp = math.log(1.0 / temperature)

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise every weight. Must be called explicitly from outside,
        *after* `from_pretrained()`, which otherwise overwrites these tensors."""
        for proj in [self.vis_proj, self.text_proj]:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.log_temperature.fill_(self._init_temp)

    def forward(
        self,
        vis_feats: torch.Tensor,
        text_feats: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vis_feats:  [K, vis_dim]   visual anchors of K objects
            text_feats: [K, text_dim]  their text anchors; the positive of vis_feats[i]
                                       is text_feats[i]
            labels:     [K], optional; defaults to labels[i] = i, i.e. the two tensors
                        are row-aligned

        Returns:
            scalar contrastive loss
        """
        if vis_feats.size(0) == 0:
            # no GT object at all (rare): return zero but keep the graph alive
            return vis_feats.sum() * 0.0

        if labels is None:
            assert vis_feats.size(0) == text_feats.size(0), (
                f"row-aligned mode requires equal length, got {vis_feats.size(0)} vs {text_feats.size(0)}"
            )
            labels = torch.arange(vis_feats.size(0), device=vis_feats.device)

        # dtype alignment. This module is lifted to fp32 (DSCC's custom modules must
        # be), while the incoming vis_feats/text_feats are bf16 -- addmm then raises
        # "mat1 and mat2 must have the same dtype" and the caller's except-clause used
        # to swallow the whole perception loss: it blew up on every forward, so the
        # contrast term never contributed a single gradient step. Up-cast the inputs to
        # the projection weights' dtype and compute in fp32 throughout; adding the
        # returned scalar to the bf16 outputs.loss promotes automatically.
        _proj_dtype = self.vis_proj[0].weight.dtype
        vis_feats = vis_feats.to(_proj_dtype)
        text_feats = text_feats.to(_proj_dtype)

        v = F.normalize(self.vis_proj(vis_feats), dim=-1)   # [K, P]
        t = F.normalize(self.text_proj(text_feats), dim=-1)  # [K, P]

        # learnable temperature, clipped from above
        log_temp = torch.clamp(self.log_temperature, max=self.log_temperature_max)
        scale = log_temp.exp()

        logits = (v @ t.t()) * scale  # [K, K]
        # bidirectional contrast (v->t and t->v), as in CLIP/SigLIP
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_v2t + loss_t2v)
