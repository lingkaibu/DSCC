#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

"""LLaVA-1.5 backbone, modified for DSCC (Dual-Stream Cross-Anchor Correction).

This file is the upstream LLaVA `llava_llama.py` with the two DSCC streams
wired in; it is the ONLY upstream file this work modifies. Changes:

  * imports the modules from `llava.model.dual_stream`
  * `init_dual_stream_modules()` -- re-initialises the added modules. It MUST be
    called after `from_pretrained()`, because HuggingFace's `_init_weights`
    overwrites whatever `__init__` set up. Training-time only: calling it at
    evaluation time wipes the trained weights.
  * cross-attention injection after the cognition layers (24, 28), gated by the
    curriculum schedule g_t
  * the perception-stream InfoNCE term, added to the SFT loss as
    `L = L_SFT + alpha * L_perc` with alpha = 0.5
  * two ablation switches read from `model.config`:
      `disable_cross_anchor`     -> force g_t = 0, no cognition injection
      `disable_perception_loss`  -> skip the perception InfoNCE entirely
    Evaluation scripts inject these from the checkpoint's
    `ablation_config.json`; without them an ablation silently evaluates as the
    full model.
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ...constants import IMAGE_TOKEN_INDEX

# ====== DSCC dual stream: the added modules ======
from ..dual_stream import (
    PerceptionContrastLoss,
    CrossAnchorAttention,
    FrozenCLIPTextEncoder,
    roi_pool_features,
)
from ..dual_stream.cross_anchor_attention import curriculum_gamma


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    # ====== dual-stream configuration ======
    PERCEPTION_LAYER_IDX = 16   # layer 16's hidden states are the perception anchors (0-indexed: after layers[15])
    CROSS_ANCHOR_LAYERS = [23, 27]  # inject after layers 24 and 28 (0-indexed)
    CROSS_ATTN_HEADS = 32       # 32 heads, matching the LLaMA backbone (head_dim=128)
    NUM_IMAGE_PATCHES = 576     # CLIP-ViT-L/14-336: 24*24 = 576
    GAMMA_WARMUP_START_RATIO = 0.30  # start of stage 2 (the first 30% is stage 1, plain SFT)
    GAMMA_WARMUP_END_RATIO = 0.70    # end of stage 2 (the last 30% runs at g = 1)

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ====== DSCC modules: perception stream (§3.2) / cognition stream (§3.3) ======
        # 1) perception stream: object-level InfoNCE loss
        self.perception_contrast = PerceptionContrastLoss(
            vis_dim=config.hidden_size,   # 4096
            text_dim=768,                 # CLIP-L text dim
            proj_dim=512,
            temperature=0.07,
        )

        # 2) cognition stream: cross-attention anchor injection, inserted twice
        self.cross_anchor_modules = nn.ModuleList([
            CrossAnchorAttention(
                hidden_dim=config.hidden_size,
                num_heads=self.CROSS_ATTN_HEADS,   # 32 heads
                dropout=0.0,
            )
            for _ in self.CROSS_ANCHOR_LAYERS
        ])

        # 3) frozen CLIP text encoder, the source of the text anchors.
        # Wrapped in a list so PyTorch does not auto-register it as a submodule: it
        # stays out of state_dict and is never saved. The CLIP weights load lazily on
        # the first forward and are re-read from the hub/cache every run.
        self._text_encoder_holder: list = [None]

        # scratch state shared between hooks and forward (never part of state_dict)
        self._captured_perception_anchors = None  # [K, N, D]
        self._captured_anchor_mask = None         # [K, N]: which anchors are real image patches
        self._anchor_gamma = 0.0
        self._image_spans = None  # List[(start, end)] per batch

        self.post_init()

        # register the forward hooks on the selected layers
        self._register_anchor_hooks()

    # ----------- re-initialise the DSCC modules (must run after from_pretrained) -----------
    def init_dual_stream_modules(self):
        """
        Must be called explicitly after `from_pretrained()`.

        Why: for keys missing from the pretrained checkpoint, HuggingFace's
        `from_pretrained` re-runs `_init_weights` (normal_ with std=0.02 by default),
        overwriting the initialisation `__init__` set up -- in particular out_proj's
        near-identity init. Without this reset, the first cross-attention injection
        (once g_t > 0) has a magnitude the design never intended.
        """
        for xa in self.cross_anchor_modules:
            xa.reset_parameters()
        self.perception_contrast.reset_parameters()
        # sanity-check the init that matters
        first_xa = self.cross_anchor_modules[0]
        out_std = first_xa.out_proj.weight.std().item()
        print(
            f"[DSCC] dual-stream modules reset | "
            f"out_proj.weight std = {out_std:.5f} (expected ~0.02) | "
            f"q_proj std = {first_xa.q_proj.weight.std().item():.4f} (expected ~0.02)"
        )

    # ----------- the hook machinery -----------
    def _get_text_encoder(self):
        if self._text_encoder_holder[0] is None:
            self._text_encoder_holder[0] = FrozenCLIPTextEncoder().to(
                next(self.parameters()).device
            )
        return self._text_encoder_holder[0]

    def _register_anchor_hooks(self):
        """Capture anchors at the perception layer, inject cross-attention at the cognition layers."""

        # perception-layer hook: capture after layer 16's forward (0-indexed layers[15])
        perception_layer_pos = self.PERCEPTION_LAYER_IDX - 1  # after forward = layers[15]

        def perception_hook(module, inputs, output):
            # output[0] is [B, L, D]
            # Re-capture whenever a valid image span exists: refreshed every training
            # step, while a decode step at inference has no image span and falls into
            # the any_valid=False branch below, keeping the anchors captured at prefill
            # available to the cross-attention.
            if self._image_spans is None:
                return output
            seq_len = output[0].size(1)
            spans = self._image_spans
            anchors = []
            any_valid = False
            for b, (s, e) in enumerate(spans):
                # guard against overruns: later generate() forwards have length 1
                if e > seq_len or e <= s:
                    anchors.append(None)
                    continue
                anchors.append(output[0][b:b+1, s:e])  # [1, N, D]
                any_valid = True
            if not any_valid:
                return output

            valid = [a for a in anchors if a is not None]
            if len(valid) == len(anchors):
                self._captured_perception_anchors = torch.cat(valid, dim=0)
                self._captured_anchor_mask = None
            else:
                N = self.NUM_IMAGE_PATCHES
                D = output[0].size(-1)
                B = len(anchors)
                padded = output[0].new_zeros((B, N, D))
                mask = torch.zeros((B, N), dtype=torch.bool, device=output[0].device)
                for b, a in enumerate(anchors):
                    if a is not None:
                        padded[b:b+1] = a
                        mask[b] = True
                self._captured_perception_anchors = padded
                self._captured_anchor_mask = mask
            return output

        self.model.layers[perception_layer_pos].register_forward_hook(perception_hook)

        # cognition-layer hooks: inject after layers 24 and 28 (0-indexed layers[23], layers[27])
        for layer_idx, xa_module in zip(self.CROSS_ANCHOR_LAYERS, self.cross_anchor_modules):

            def make_inject_hook(xa_mod):
                def inject_hook(module, inputs, output):
                    if self._captured_perception_anchors is None or self._anchor_gamma == 0.0:
                        return output
                    h = output[0]
                    h_new = xa_mod(
                        h,
                        self._captured_perception_anchors,
                        gamma=self._anchor_gamma,
                        anchor_mask=self._captured_anchor_mask,
                    )
                    # output is a tuple whose first item is hidden_states; keep the rest
                    return (h_new,) + output[1:]
                return inject_hook

            self.model.layers[layer_idx].register_forward_hook(make_inject_hook(xa_module))

    # ----------- standard interface -----------
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        # ====== DSCC training-only arguments (absent at inference, then skipped) ======
        bboxes: Optional[List[List]] = None,            # List[List[(x,y,w,h)]]
        class_names: Optional[List[List[str]]] = None,  # List[List[str]]
        img_sizes_orig: Optional[List[Tuple[int, int]]] = None,  # List[(W,H)], original image size the bboxes refer to
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ----- 1. record where image_token_index sits in input_ids, before prepare -----
        # (prepare replaces input_ids with inputs_embeds and expands the <image>
        # placeholder into N patches)
        # The key rule: only overwrite self._image_spans when THIS forward actually saw
        # an IMAGE_TOKEN_INDEX.
        #   - training forward: input_ids contains IMAGE_TOKEN_INDEX -> overwrite
        #   - prefill inside generate(): input_ids=None / inputs_embeds already built
        #     -> do not overwrite; keep the spans generate() set before super().generate()
        #   - later decode steps of generate(): input_ids=[new token], no IMAGE_TOKEN_INDEX
        #     -> do not overwrite; the anchors captured at prefill stay available
        # An earlier version assigned `self._image_spans = captured_image_spans`
        # unconditionally, which nulled the spans as soon as generation started: the
        # perception hook returned early and cross-attention injected nothing at all
        # during inference.
        captured_image_spans = None
        if input_ids is not None and inputs_embeds is None:
            spans = []
            has_image_token = False
            for b in range(input_ids.size(0)):
                pos = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    s = pos[0].item()
                    spans.append((s, s + self.NUM_IMAGE_PATCHES))
                    has_image_token = True
                else:
                    spans.append((0, 0))
            if has_image_token:
                captured_image_spans = spans
        if captured_image_spans is not None:
            self._image_spans = captured_image_spans

        # ----- 2. prepare (the standard LLaVA path) -----
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # ====== ablation switches (False by default; the main run is unaffected) ======
        # Set through model.config by train_ablation_v6.py:
        #   - disable_cross_anchor=True    -> force g_t = 0, no cognition-stream injection
        #   - disable_perception_loss=True -> skip the object-level InfoNCE entirely
        _disable_cross_anchor = getattr(self.config, 'disable_cross_anchor', False)
        _disable_perception_loss = getattr(self.config, 'disable_perception_loss', False)

        # ----- 3. hand the curriculum gate g_t to the hooks -----
        if self.training:
            global_step = getattr(self.config, 'global_step', 0)
            total_steps = getattr(self.config, 'total_steps', 1)
            warmup_start = int(total_steps * self.GAMMA_WARMUP_START_RATIO)
            warmup_end = int(total_steps * self.GAMMA_WARMUP_END_RATIO)
            _gamma_curriculum = curriculum_gamma(global_step, warmup_start, warmup_end, final_value=1.0)
        else:
            # g = 1 at inference: the trained cross-attention is fully on
            _gamma_curriculum = 1.0
        self._anchor_gamma = _gamma_curriculum

        # ablation: disabling the cognition stream forces g = 0, which only affects the
        # strength of the cross-attention injection
        if _disable_cross_anchor:
            self._anchor_gamma = 0.0

        # Whether the perception loss reaches the backbone is decided by the curriculum
        # stage alone -- detached during the first 30% (bootstrapping), coupled after --
        # and is deliberately decoupled from disable_cross_anchor.
        # NOTE: this must read _gamma_curriculum (the raw curriculum value), not
        # self._anchor_gamma. The latter is forced to 0 by disable_cross_anchor, which
        # would detach ablation A (perception only) for the whole run: L_perc would
        # never reach the backbone, A's backbone would be identical to D's, and the
        # perception stream's standalone contribution would be unmeasurable.
        self._perc_couple_backbone = (_gamma_curriculum > 0.0)

        # ----- 4. backbone forward (the hooks capture anchors and inject cross-attention) -----
        # Note: at prefill, prepare_inputs_labels_for_multimodal sets input_ids to None
        # and fills inputs_embeds; at a decode step (input_ids.shape[1] == 1) it keeps
        # input_ids and returns inputs_embeds=None. input_ids must be passed through
        # here, otherwise both are None at decode time and this raises ValueError.
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # ----- 5. perception-stream InfoNCE loss (§3.2; training only, needs bboxes + class_names) -----
        # [DUAL_STREAM_DEBUG] tell the two silent-skip paths apart: a short-circuited
        # guard versus an exception raised inside the try block
        if self.training and int(os.environ.get('DUAL_STREAM_DEBUG', '0')):
            _guard = {
                'not _disable_perception_loss': (not _disable_perception_loss),
                'bboxes is not None': bboxes is not None,
                'class_names is not None': class_names is not None,
                'img_sizes_orig is not None': img_sizes_orig is not None,
                '_captured_perception_anchors is not None': self._captured_perception_anchors is not None,
            }
            _failed = [k for k, v in _guard.items() if not v]
            if _failed:
                print(f"[DualStream WARN] perception_loss guard short-circuited (try not entered): {_failed}")
            else:
                print("[DualStream] perception_loss guards all passed -> entering try")
        if (
            self.training
            and not _disable_perception_loss   # ablation: skip when the perception stream is off
            and bboxes is not None
            and class_names is not None
            and img_sizes_orig is not None
            and self._captured_perception_anchors is not None
        ):
            try:
                # the text encoder lazy-inits inside _get_text_encoder(), loading CLIP
                # on the first call

                # ROI-pool the captured perception anchors: [B, N=576, D=4096]
                anchors = self._captured_perception_anchors

                vis_feats, batch_idx, _ = roi_pool_features(
                    image_features=anchors,
                    bboxes_per_image=bboxes,
                    img_sizes=img_sizes_orig,
                    grid_size=int(self.NUM_IMAGE_PATCHES ** 0.5),  # 24
                )

                if vis_feats.size(0) > 0:
                    # collect one class name per ROI, in batch_idx order
                    flat_class_names = []
                    for b_i in range(len(class_names)):
                        flat_class_names.extend(class_names[b_i])
                    # batch_idx should be as long as the ROIs that mapped successfully;
                    # assume every bbox mapped (roi_pool_features already dropped empty ROIs)
                    if len(flat_class_names) == vis_feats.size(0):
                        text_feats = self._get_text_encoder().encode(flat_class_names, device=vis_feats.device)
                        # match vis_feats' dtype
                        text_feats = text_feats.to(vis_feats.dtype)

                        # ===== stage 1 cuts the backward path from L_perc into the LLaMA backbone =====
                        # This fixes a "cognition-only beats full" bug: L_perc used to
                        # pour noise into layer 16 on every backward pass (the
                        # perception_contrast module was deadlocked in bf16, so L_perc
                        # never converged, stayed high, and sent a large, randomly
                        # directed gradient into the backbone). The full model therefore
                        # trained worse than the ablation that had no L_perc at all.
                        #
                        # Fix: follow the curriculum stage, via _perc_couple_backbone --
                        # which is decoupled from disable_cross_anchor, see step 3 above.
                        #   - stage 1 (first 30%): perception_contrast trains itself but
                        #     vis_feats is detached, so L_perc never reaches the LLaMA
                        #     backbone. This is the "perception bootstrapping" of §4.5.
                        #   - stage 2 onwards:     no detach; L_perc shapes layer 16 as
                        #     intended, the "cognition-perception bridging" of §4.5.
                        #   Note: ablation A (perception only) has cross-attention off
                        #   (g == 0), yet its perception stream still couples to the
                        #   backbone from stage 2 on -- that is what makes A's standalone
                        #   contribution visible (A != D).
                        if not self._perc_couple_backbone:
                            # stage 1: cut the backbone path (vis_proj/text_proj/log_temp still train)
                            vis_feats_for_loss = vis_feats.detach()
                        else:
                            vis_feats_for_loss = vis_feats

                        perception_loss = self.perception_contrast(vis_feats_for_loss, text_feats)

                        # [DUAL_STREAM_DEBUG] watch the distribution of K to catch the
                        # degenerate K=1 case:
                        #   K = 1  -> L_perc == 0 (single-class cross_entropy), no signal at all
                        #   K >= 2 -> L_perc > 0, i.e. an actual contrastive gradient
                        if self.training and int(os.environ.get('DUAL_STREAM_DEBUG', '0')):
                            print(f"[DualStream DBG] K={vis_feats.size(0)} L_perc={perception_loss.item():.4f}")

                        # fold into the total loss with weight alpha = 0.5
                        alpha = 0.5
                        if outputs.loss is not None:
                            outputs.loss = outputs.loss + alpha * perception_loss
                        else:
                            outputs.loss = alpha * perception_loss
            except Exception as e:
                # a failed perception loss must not bring down the whole training step
                if int(os.environ.get('DUAL_STREAM_DEBUG', '0')):
                    print(f"[DualStream WARN] perception_loss skipped: {e}")
                pass

        # ----- 6. clear the scratch state -----
        # Note: _captured_perception_anchors is deliberately NOT cleared here.
        #   - training:  the next forward's perception_hook overwrites it with the new batch.
        #   - inference: the decode forwards inside generate() reuse the anchors captured
        #                at prefill; generate() clears them at the end (see below).
        self._image_spans = None

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (outputs.loss,) + output if outputs.loss is not None else output
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # record the image span before prepare, for the hooks
        if inputs is not None and images is not None:
            spans = []
            for b in range(inputs.size(0)):
                pos = (inputs[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    s = pos[0].item()
                    spans.append((s, s + self.NUM_IMAGE_PATCHES))
                else:
                    spans.append((0, 0))
            self._image_spans = spans

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # g = 1 at inference: cross-attention fully on
        self._anchor_gamma = 1.0

        result = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        # clean up
        self._image_spans = None
        self._captured_perception_anchors = None
        self._captured_anchor_mask = None
        return result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
