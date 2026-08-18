# DSCC — Dual-Stream Cross-Anchor Correction

Official code for the paper

> **Dual-Stream Cross-Anchor Correction: Grounding Long-Form Captions and the Domain Limits of Object-Level Anchors**
> Lingkai Bu, Qian Gao, Jun Fan, Guohui Ding, Zhenyu Yang, Yuteng Xiao, Jinyi Liang
> Qilu University of Technology (Shandong Academy of Sciences) · China Telecom Digital Intelligence Technology Co., Ltd. · Shenyang Aerospace University · University of Jinan

DSCC injects **object-level visual anchors into the language model itself during
fine-tuning**, instead of post-processing the decoding step. Two streams are
coupled by a curriculum gate:

| Stream | What it does | Paper |
|---|---|---|
| **Perception** | ROI-pooled hidden states at layer `l_p=16` are aligned to frozen CLIP text anchors by a bidirectional object-level InfoNCE loss | §3.2 |
| **Cognition** | Gated cross-attention at layers `L_c={24,28}` lets deeper layers query those anchors **at every generation step** | §3.3 |
| **CGFT gate** | `γ_t` ramps 0→1 over `[0.3T, 0.7T]`; `γ≡1` at inference | §3.4 |
| **Objective** | `L = L_SFT + α · L_perc`, `α = 0.5` | §3.5 |

> **Note on comments.** The in-code comments and docstrings record the design
> rationale in detail — in particular why `RMSNorm` (not `LayerNorm`) and why the
> added modules must run in fp32 under a bf16 backbone. Those two notes are the
> difference between the modules training and silently not training at all.

---

## Results

All numbers under one backbone (LLaVA-1.5-7B), one prompt and one scorer, on the
same 500 COCO val2014 images.

| Method | #Words ↑ | CHAIR_S ↓ |
|---|---|---|
| Vanilla LLaVA-1.5 (greedy) | 89.5 | 59.40 |
| Vanilla LLaVA-1.5 (sampling) | 104.9 | 57.80 |
| VCD | 104.0 | 58.60 |
| OPERA | 93.1 | 45.20 |
| **DSCC (ours)** | **171.5** | **38.80** |

DSCC is the only method that reaches the long-caption, low-hallucination region:
captions about **1.9×** the baseline length at **88.19 %** precision per object
mention — the highest under a density-independent criterion.

On the discriminative side (POPE, adversarial subset) DSCC reaches the highest
**precision** of the four configurations, 0.8839 vs. 0.8510 for vanilla SFT, by
answering "yes" less often (YesRatio 0.4507 vs. 0.4853). Its **F1 is a tie**
(0.8380 vs. 0.8383) — the gain is in precision, not in F1, and the paper does
not claim otherwise.

**No universal superiority is claimed.** On three out-of-domain benchmarks the
synergy is bound to the anchors' semantic domain and breaks on charts and
optical illusions. See §4.6 of the paper.

---

## Repository layout

This repository is **not** a copy of LLaVA. It ships only what is new or
modified; everything else comes from upstream LLaVA-1.5.

```
llava/model/dual_stream/           # NEW — the DSCC modules
    cross_anchor_attention.py      #   cognition stream + curriculum gate γ_t
    perception_contrast.py         #   perception stream (object-level InfoNCE)
    roi_utils.py                   #   COCO bbox → ViT patch grid → ROI mean-pool
    text_encoder.py                #   frozen CLIP text anchors
    image_utils.py
llava/model/language_model/
    llava_llama.py                 # MODIFIED — wires both streams into the backbone

prepare_v5_data.py                 # builds the ShareGPT4V x COCO caption map
train_full_v6.py                   # main experiment (v6.1)
train_ablation_v6.py               # ablations A / B / C / D
setup_ablation_configs.py          # writes ablation_config.json next to a checkpoint
verify_fixes.py                    # asserts the loaded code really is v6.1
check_v6_wandb_drift.py            # post-hoc training health check

generate_captions.py               # CHAIR — inference stage
eval_chair_official.py             # CHAIR — scoring (Rohrbach et al., 2018)
eval_chair.py                      # CHAIR — scoring (own synonym list, backup)
eval_pope.py                       # POPE (random / popular / adversarial)
eval_mirage.py  eval_mme_hallucination.py
eval_hallusion.py  eval_mmhal.py   # out-of-domain benchmarks
auto_judge_dualstream.py           # LLM-as-judge for MMHal / MIRAGE
```

See `NOTICE` for the precise list of added vs. modified files.

---

## Setup

### 1. Get LLaVA-1.5 and copy this repository over it

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA && pip install -e .

git clone https://github.com/lingkaibu/DSCC.git /tmp/DSCC
cp -r /tmp/DSCC/llava/. llava/          # adds dual_stream/, replaces llava_llama.py
cp /tmp/DSCC/*.py /tmp/DSCC/*.sh .      # training + evaluation scripts
```

### 2. Pin `huggingface_hub`

> ⚠️ **Do not run `pip install -U huggingface_hub` in this environment.**
> Version 1.x breaks `transformers==4.37.2` / `tokenizers==0.15.1`, which
> LLaVA-1.5 requires.

```bash
pip install "huggingface_hub==0.24.7"
python -c "import transformers, tokenizers, huggingface_hub"   # must not raise
```

### 3. Point the scripts at your data

Every script resolves its paths from one environment variable:

```bash
export DSCC_ROOT=/path/to/your/workspace   # default: /root/autodl-tmp
```

Expected layout under `$DSCC_ROOT`:

```
$DSCC_ROOT/
    LLaVA/                                       # the checkout from step 1
    data/
        coco/train2017/                          # training images
        coco/val2014/                            # evaluation images
        coco/annotations/instances_train2017.json
        coco/annotations/instances_val2014.json
        coco/annotations/captions_val2014.json
        pope/coco_pope_{random,popular,adversarial}.json
        sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json
        v5_caption_map.json                      # built in step 4
        MME/  HallusionBench/  MMHal-Bench/  mirage/   # optional, OOD only
    checkpoints/
    results/
```

### 4. Build the caption map

```bash
python prepare_v5_data.py
```

This joins [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)
(`sharegpt4v_instruct_gpt4-vision_cap100k.json`) with COCO `train2017`, keeping
only images that have **both** a GPT-4V long caption **and** at least one bbox
annotation, and writes `$DSCC_ROOT/data/v5_caption_map.json`
(`{image_id: caption}`). That intersection is ≈ 95 k samples, of which ≈ 49 400
are used per epoch.

> The `v5` in the filename is historical and unrelated to the model version.
> The v6.1 training scripts read exactly this filename — do not rename it.

---

## Training

```bash
# sanity check first: 3 000 samples, ~20 min, reaches stage 3 (γ=1)
QUICK_VALIDATE=1 python train_full_v6.py

# full run: 2 epochs ≈ 24 700 steps, ~7 h on one A800 80G
python train_full_v6.py
```

Output: `$DSCC_ROOT/checkpoints/dualstream_v6_1/checkpoint-final`.
Set `RUN_TAG=_something` to write to a suffixed directory instead of overwriting.

### Hyperparameters (paper Table I)

| Symbol | Meaning | Value |
|---|---|---|
| `l_p` | perception layer | 16 |
| `L_c` | cognition injection layers | {24, 28} |
| `h` | heads per cross-attention | 32 |
| `d_h` | head dimension | 128 |
| `P` | contrastive projection dim | 512 |
| `τ_0` | initial temperature | 0.07 |
| `α` | perception loss weight | 0.5 |
| — | curriculum ramp | 0.3T → 0.7T |

### Two things that will silently break training if you change them

1. **`init_dual_stream_modules()` must be called after `from_pretrained()`.**
   HuggingFace's `_init_weights` overwrites whatever `__init__` set up.
2. **The added modules must be lifted to fp32.** Under bf16 the quantisation
   step (~1.5e-4) is larger than the AdamW update these modules receive
   (~1e-7), so every update rounds to zero and the weights never move.
   `train_full_v6.py` does both (see the `p.data = p.data.float()` block).

`verify_fixes.py` asserts that the code actually imported is the v6.1 version;
`check_v6_wandb_drift.py` verifies after the fact that `q_proj.std` really drifted.

### Ablations

| Label | `ABLATION_TYPE` | Perception | Cognition |
|---|---|---|---|
| A | `perc_only` | ✅ | ❌ (γ_t ≡ 0) |
| B | `cog_only` | ❌ | ✅ |
| C | `full` | ✅ | ✅ (= DSCC) |
| D | `none` | ❌ | ❌ (≈ vanilla SFT) |

```bash
ABLATION_TYPE=perc_only python train_ablation_v6.py
```

D is the length- and density-matched control that the paper attributes gains
against: same corpus, same schedule, both streams off.

---

## Evaluation

> ⚠️ **Ablation checkpoints must be evaluated with their flags restored.**
> Every evaluation script reads `ablation_config.json` from the checkpoint
> directory and injects `disable_perception_loss` / `disable_cross_anchor` into
> `model.config`. Without it an ablation silently evaluates as the full model.
> **Never call `init_dual_stream_modules()` at evaluation time** — it is a
> training-time function and will wipe the trained weights.
> Checkpoints missing that file can be repaired with `setup_ablation_configs.py`.

### CHAIR (main result)

```bash
python generate_captions.py --model_path $DSCC_ROOT/checkpoints/dualstream_v6_1/checkpoint-final
python eval_chair_official.py --cap_file $DSCC_ROOT/results/captions/<generated>.jsonl
```

`eval_chair_official.py` is the Rohrbach et al. (2018) scorer (NLTK + WordNet)
and does not load the model. `eval_chair.py` is a backup implementation with an
own synonym list; the paper reports the official one.

### POPE

```bash
python eval_pope.py --model_path <ckpt> --pope_type all
```

### Out-of-domain

```bash
python eval_mme_hallucination.py --model_path <ckpt>
python eval_hallusion.py         --model_path <ckpt>
python eval_mmhal.py             --model_path <ckpt>
python auto_judge_dualstream.py  --jsonl_path <generated jsonl>
```

`auto_judge_dualstream.py` calls an OpenAI-compatible chat-completions endpoint
as the judge. **The key is read from the environment and is never stored in the
source:**

```bash
export AUTODL_API_KEY="your-key"
export JUDGE_API_BASE_URL="https://api.openai.com/v1"   # optional, to use another gateway
export JUDGE_MODEL="gpt-4o-mini"                        # optional
```

---

## Known gaps

Honest list of what is not (yet) in this repository:

- **Checkpoints** — not distributed here; they are too large for git.
- **`eval_mirage.py`** — line ~41 is missing `use_fast=False` when constructing
  the tokenizer; patch it before running.
- Baseline numbers for VCD / OPERA were produced in the baselines' own
  repositories, not here. Woodpecker numbers are quoted from the literature.
- The scripts were written for a single-GPU A800 workflow and are not
  distributed-training aware.

---

## Citation

```bibtex
@article{bu2026dscc,
  title   = {Dual-Stream Cross-Anchor Correction: Grounding Long-Form Captions
             and the Domain Limits of Object-Level Anchors},
  author  = {Bu, Lingkai and Gao, Qian and Fan, Jun and Ding, Guohui
             and Yang, Zhenyu and Xiao, Yuteng and Liang, Jinyi},
  year    = {2026}
}
```

## Licence

Apache License 2.0 — see `LICENSE` and `NOTICE`.
This work builds on [LLaVA](https://github.com/haotian-liu/LLaVA) (Apache-2.0).
The base model, LLaMA weights, ShareGPT4V corpus and all evaluation datasets
remain under their own licences.
