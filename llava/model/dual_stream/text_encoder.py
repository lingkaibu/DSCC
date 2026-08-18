"""Frozen CLIP text encoder -- the class-name anchors (paper §3.2.1).

Encodes an object class name ("person", "bicycle", ...) into the text anchor
that the perception stream aligns to, using CLIP's standard prompt template
("a photo of {}") and its standard readout (the hidden state at the EOS
position).

Why CLIP text rather than the LLaMA embedding table:
  1. CLIP text and CLIP vision already share an aligned semantic space, which
     is exactly what fine-grained vision-language alignment needs.
  2. LLaMA's input embedding is a lookup table and carries no context.
  3. The encoder is small (~123M) and stays frozen, so it costs no extra
     trainable memory.
  4. It reuses the same checkpoint as the vision tower -- no second external
     dependency.

The module forces itself back to `eval()` on every `train()` call so an outer
`model.train()` cannot un-freeze it.
"""

from typing import List

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14-336", prompt_template: str = "a photo of {}"):
        """
        Args:
            model_id: HuggingFace path of the CLIP checkpoint
            prompt_template: template the class name is filled into; CLIP's standard
                             choice is "a photo of {object_name}"
        """
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModel.from_pretrained(model_id)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.eval()
        self.output_dim = self.text_model.config.hidden_size  # CLIP-L = 768
        self.prompt_template = prompt_template

    def train(self, mode: bool = True):
        # stay in eval mode no matter what an outer model.train() asks for
        super().train(mode)
        self.text_model.eval()
        return self

    @torch.no_grad()
    def encode(self, class_names: List[str], device: str = 'cuda') -> torch.Tensor:
        """
        Args:
            class_names: ["person", "bicycle", ...]
            device:      'cuda' | 'cpu'

        Returns:
            [N, output_dim] text anchors, read out at the EOS position as CLIP does
        """
        if len(class_names) == 0:
            return torch.empty((0, self.output_dim), device=device)

        prompts = [self.prompt_template.format(name) for name in class_names]
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(device)

        outputs = self.text_model(**tokens)
        # CLIP's standard readout: the hidden state at the EOS token, i.e. the last
        # position where attention_mask is 1
        eos_idx = tokens.attention_mask.sum(dim=-1) - 1
        last_hidden = outputs.last_hidden_state  # [N, L, D]
        embeds = last_hidden[torch.arange(len(prompts), device=device), eos_idx]
        return embeds  # [N, D]
