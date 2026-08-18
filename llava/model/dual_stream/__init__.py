"""DSCC -- Dual-Stream Cross-Anchor Correction.

Paper: "Dual-Stream Cross-Anchor Correction: Grounding Long-Form Captions
        and the Domain Limits of Object-Level Anchors"

Two streams, coupled by a curriculum gate, are added to LLaVA-1.5 at
fine-tuning time so that evidence retrieval becomes a structural constraint at
every autoregressive step, rather than a decoding-time post-process:

  roi_utils               COCO bbox -> ViT patch grid -> ROI mean-pool   (§3.2.1)
  text_encoder            frozen CLIP text encoder = the class anchors   (§3.2.1)
  perception_contrast     bidirectional object-level InfoNCE loss        (§3.2)
  cross_anchor_attention  gated cross-attention + curriculum gate g_t    (§3.3, §3.4)
  image_utils             image preprocessing helper

Both added modules must be lifted to fp32 and re-initialised after
`from_pretrained()`; see `LlavaLlamaForCausalLM.init_dual_stream_modules()` and
the module docstrings for why. Getting either wrong makes them silently not
train at all.
"""

from .roi_utils import bbox_to_grid_indices, roi_pool_features
from .perception_contrast import PerceptionContrastLoss
from .cross_anchor_attention import CrossAnchorAttention
from .text_encoder import FrozenCLIPTextEncoder
from .image_utils import expand2square

__all__ = [
    "bbox_to_grid_indices",
    "roi_pool_features",
    "PerceptionContrastLoss",
    "CrossAnchorAttention",
    "FrozenCLIPTextEncoder",
    "expand2square",
]
