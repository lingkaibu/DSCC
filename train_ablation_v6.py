"""
DSCC ablation training script.

Every hyper-parameter, data path, prompt template, curriculum, optimizer and
scheduler here is word-for-word identical to train_full_v6.py -- including
EPOCHS=2 -- so that a score difference can only be attributed to removing a
stream, not to any other perturbation.

The only difference: two flags are injected through model.config to switch the
streams on and off.
  - disable_perception_loss=True  -> perception stream off (InfoNCE skipped)
  - disable_cross_anchor=True     -> cognition stream off (g_t forced to 0)

Usage (set ABLATION_TYPE in the environment):
    ABLATION_TYPE=perc_only python train_ablation_v6.py   # A: perception stream only
    ABLATION_TYPE=cog_only  python train_ablation_v6.py   # B: cognition stream only
    ABLATION_TYPE=none      python train_ablation_v6.py   # D: both off (~vanilla SFT baseline)
    ABLATION_TYPE=full      python train_ablation_v6.py   # C: full DSCC (control for the main run)

Output checkpoints:
    $DSCC_ROOT/checkpoints/dualstream_v6_1_ablA_perc_only/checkpoint-final
    $DSCC_ROOT/checkpoints/dualstream_v6_1_ablB_cog_only/checkpoint-final
    $DSCC_ROOT/checkpoints/dualstream_v6_1_ablD_none/checkpoint-final
    $DSCC_ROOT/checkpoints/dualstream_v6_1_ablC_full/checkpoint-final

QUICK_VALIDATE=1 runs a 3000-sample sanity check (~20 min) into a *_quick directory.
"""

import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var

# ============= 0. resolve the ablation configuration =============
ABLATION_TYPE = os.environ.get("ABLATION_TYPE", "").strip().lower()
ABLATION_CONFIG = {
    "perc_only": {
        "tag": "ablA_perc_only",
        "disable_perception_loss": False,
        "disable_cross_anchor": True,
        "wandb_name": "v6.1-ablA-perc-only",
        "desc": "Ablation A: perception stream only (InfoNCE on, cross-attention off)",
    },
    "cog_only": {
        "tag": "ablB_cog_only",
        "disable_perception_loss": True,
        "disable_cross_anchor": False,
        "wandb_name": "v6.1-ablB-cog-only",
        "desc": "Ablation B: cognition stream only (cross-attention on, InfoNCE off)",
    },
    "full": {
        "tag": "ablC_full",
        "disable_perception_loss": False,
        "disable_cross_anchor": False,
        "wandb_name": "v6.1-ablC-full",
        "desc": "Ablation C: full DSCC (control, equivalent to train_full_v6.py)",
    },
    "none": {
        "tag": "ablD_none",
        "disable_perception_loss": True,
        "disable_cross_anchor": True,
        "wandb_name": "v6.1-ablD-none",
        "desc": "Ablation D: both streams off (~vanilla LLaVA-1.5 SFT baseline, L_SFT only)",
    },
}
if ABLATION_TYPE not in ABLATION_CONFIG:
    raise SystemExit(
        f"\nSet the ABLATION_TYPE environment variable to one of: {list(ABLATION_CONFIG.keys())}\n"
        f"Example: ABLATION_TYPE=none python train_ablation_v6.py\n"
    )
abl_cfg = ABLATION_CONFIG[ABLATION_TYPE]
print(f"\n{'='*70}\n[ABLATION] {abl_cfg['desc']}\n{'='*70}\n")

# ============= 1. environment =============
os.makedirs(f"{DSCC_ROOT}/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = f"{DSCC_ROOT}/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import random
import sys
import torch
import wandb
import warnings
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

sys.path.append(f"{DSCC_ROOT}/LLaVA")
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
from PIL import Image

warnings.filterwarnings("ignore")


def expand2square_with_bboxes(pil_img, bboxes, background_color=(122, 116, 104)):
    width, height = pil_img.size
    if width == height:
        return pil_img, list(bboxes), (width, height)
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        pad = (width - height) // 2
        result.paste(pil_img, (0, pad))
        new_bboxes = [(x, y + pad, w, h) for (x, y, w, h) in bboxes]
        return result, new_bboxes, (width, width)
    result = Image.new(pil_img.mode, (height, height), background_color)
    pad = (height - width) // 2
    result.paste(pil_img, (pad, 0))
    new_bboxes = [(x + pad, y, w, h) for (x, y, w, h) in bboxes]
    return result, new_bboxes, (height, height)


# ============= 2. hyper-parameters (word-for-word identical to train_full_v6.py) =============
# Any hyper-parameter that differs from the main run makes the ablation's score
# difference unattributable to the removed stream.
_QUICK = os.environ.get("QUICK_VALIDATE", "0") == "1"
EPOCHS = 1 if _QUICK else 2
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
SAVE_STEPS = 10000
LEARNING_RATE = 2e-5
# The added modules (cross_anchor + perception_contrast) get their own higher LR
# and no weight decay: with wd=0.01 the decay dominated once the cosine LR had
# fallen to 1e-7 and slowly pulled the new weights back towards their init.
CROSS_ANCHOR_LR_MULT = 5
CROSS_ANCHOR_WEIGHT_DECAY = 0.0
COSINE_ETA_MIN = 1e-6
MAX_SAMPLES = 3000 if _QUICK else None

_quick_suffix = "_quick" if _QUICK else ""
OUTPUT_DIR = f"{DSCC_ROOT}/checkpoints/dualstream_v6_1_{abl_cfg['tag']}{_quick_suffix}"
CAPTION_MAP_JSON = f"{DSCC_ROOT}/data/v5_caption_map.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
wandb.init(
    project="DualStream-MLLM",
    name=abl_cfg["wandb_name"] + _quick_suffix,
    config={
        "version": "v6.1",
        "ablation_type": ABLATION_TYPE,
        "tag": abl_cfg["tag"],
        "disable_perception_loss": abl_cfg["disable_perception_loss"],
        "disable_cross_anchor": abl_cfg["disable_cross_anchor"],
        "lr": LEARNING_RATE,
        "cross_anchor_lr_mult": CROSS_ANCHOR_LR_MULT,
        "cross_anchor_wd": CROSS_ANCHOR_WEIGHT_DECAY,
        "cosine_eta_min": COSINE_ETA_MIN,
        "grad_accum": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS,
        "max_samples": MAX_SAMPLES,
        "quick_validate": _QUICK,
    },
)


# ============= 3. dataset =============
class DualStreamCOCODataset(Dataset):
    def __init__(self, caption_map_json, instance_anno, img_dir, max_samples=None):
        print(f"-> loading the ShareGPT4V caption map: {caption_map_json}")
        with open(caption_map_json, 'r', encoding='utf-8') as f:
            raw_map = json.load(f)
        self.img_to_caption = {int(k): v for k, v in raw_map.items()}
        print(f"   captions: {len(self.img_to_caption)} images")

        print(f"-> loading COCO instances (bboxes): {instance_anno}")
        with open(instance_anno, 'r', encoding='utf-8') as f:
            inst_data = json.load(f)

        self.img_dir = img_dir
        cat_id_to_name = {c['id']: c['name'] for c in inst_data['categories']}

        self.img_to_objs = {}
        for ann in inst_data['annotations']:
            iid = ann['image_id']
            self.img_to_objs.setdefault(iid, []).append(
                (tuple(ann['bbox']), cat_id_to_name[ann['category_id']])
            )

        self.img_meta = {img['id']: (img['width'], img['height'], img['file_name'])
                         for img in inst_data['images']}

        valid_ids = (set(self.img_to_caption.keys())
                     & set(self.img_to_objs.keys())
                     & set(self.img_meta.keys()))
        self.valid_ids = sorted(valid_ids)
        if max_samples:
            self.valid_ids = self.valid_ids[:max_samples]
        self._all_ids_for_neg = list(self.valid_ids)
        print(f"-> usable training samples: {len(self.valid_ids)} images")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        W, H, fname = self.img_meta[img_id]
        img_path = os.path.join(self.img_dir, fname)
        pos_text = self.img_to_caption[img_id]
        objs = self.img_to_objs[img_id]
        bboxes = [b for b, _ in objs]
        class_names = [n for _, n in objs]
        neg_id = img_id
        while neg_id == img_id:
            neg_id = random.choice(self._all_ids_for_neg)
        neg_text = self.img_to_caption[neg_id]
        return {
            "img_path": img_path, "pos_text": pos_text, "neg_text": neg_text,
            "bboxes": bboxes, "class_names": class_names, "img_size": (W, H),
        }


def collate_fn_no_stack(batch):
    return batch


train_dataset = DualStreamCOCODataset(
    caption_map_json=CAPTION_MAP_JSON,
    instance_anno=f"{DSCC_ROOT}/data/coco/annotations/instances_train2017.json",
    img_dir=f"{DSCC_ROOT}/data/coco/train2017",
    max_samples=MAX_SAMPLES,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, collate_fn=collate_fn_no_stack,
)


# ============= 4. load the model =============
print("-> loading LLaVA 7B (bfloat16)...")
model_id = "liuhaotian/llava-v1.5-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

model = LlavaLlamaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda",
)
model.init_dual_stream_modules()

# ============ lift the dual-stream parameters to fp32 ============
# At |w| ~ 0.02 the bf16 quantisation step is ~1.5e-4 while the cross-attention
# backward gradient is only ~1e-6, so every AdamW update rounds to zero and the
# weights never move. The fp32 island breaks that deadlock.
# Note: in D (none) and A (perc_only) the cross_anchor gradient is always 0 and
# the cast is not strictly needed, but the code path is kept identical to the
# main run so no score difference can be blamed on it.
print("\n[fp32] lifting cross_anchor + perception_contrast to fp32 (breaks the bf16 quantisation deadlock)")
_n_promoted = 0
for xa in model.cross_anchor_modules:
    for p in xa.parameters():
        p.data = p.data.float()
        _n_promoted += p.numel()
for p in model.perception_contrast.parameters():
    p.data = p.data.float()
    _n_promoted += p.numel()
print(f"  ✓ lifted {_n_promoted/1e6:.2f}M parameters to fp32 (~{_n_promoted*2/1e9:.2f}GB extra GPU memory)")

# ============ start-up sanity check ============
_xa0 = model.cross_anchor_modules[0]
_o_std = _xa0.out_proj.weight.std().item()
_q_std = _xa0.q_proj.weight.std().item()
_has_ln = hasattr(_xa0, 'layer_norm')
_has_rms = hasattr(_xa0, 'rms_norm')
_q_dtype = _xa0.q_proj.weight.dtype
_o_dtype = _xa0.out_proj.weight.dtype
print(f"\n========== start-up sanity check ==========")
print(f"  cross_anchor_modules[0].out_proj.std    = {_o_std:.5f}  (expected ~0.02)")
print(f"  cross_anchor_modules[0].q_proj.std      = {_q_std:.5f}  (expected ~0.02)")
print(f"  has self.layer_norm                     = {_has_ln}    (expected False)")
print(f"  has self.rms_norm                       = {_has_rms}     (expected True)")
print(f"  q_proj.weight.dtype                     = {_q_dtype}  (expected torch.float32)")
print(f"  out_proj.weight.dtype                   = {_o_dtype}  (expected torch.float32)")
if _o_std < 0.015 or _has_ln or not _has_rms:
    raise SystemExit(
        "\n❌ cross_anchor_attention.py is not the current version!\n"
        "   Required: no self.layer_norm, a self.rms_norm, and out_proj.std=0.02\n"
    )
if _q_dtype != torch.float32 or _o_dtype != torch.float32:
    raise SystemExit(
        f"\n❌ the cross_anchor parameters were not lifted to fp32 (q_proj.dtype={_q_dtype})!\n"
    )
print("  ✅ checks passed, starting training\n")

# ============ inject the ablation flags ============
# llava_llama.py's forward reads these two fields to decide which stream runs
model.config.disable_perception_loss = abl_cfg['disable_perception_loss']
model.config.disable_cross_anchor = abl_cfg['disable_cross_anchor']
print(
    f"[ABLATION FLAGS]"
    f" disable_perception_loss = {abl_cfg['disable_perception_loss']}"
    f" | disable_cross_anchor = {abl_cfg['disable_cross_anchor']}\n"
)

model.train()
if not hasattr(model.config, "attention_dropout"):
    model.config.attention_dropout = 0.0
if not hasattr(model.config, "rope_theta"):
    model.config.rope_theta = 10000.0
model.config.output_hidden_states = True

print("-> waking up the vision tower...")
vision_tower = model.get_vision_tower()
if not getattr(vision_tower, 'is_loaded', False):
    vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16, device='cuda')

frozen, trainable = 0, 0
for name, param in model.named_parameters():
    if 'vision_tower' in name:
        param.requires_grad = False
        frozen += param.numel()
    else:
        trainable += param.numel()
print(f"-> frozen vision_tower: {frozen/1e9:.2f}B | trainable: {trainable/1e9:.2f}B")

# When a stream is off, its module's parameters stay in the trainable set with a
# gradient of exactly 0. That costs a little memory and nothing else, and keeps
# the parameter table identical to the main run for a fair comparison.
if abl_cfg['disable_cross_anchor']:
    print("   [note] cross-anchor gradients are always 0 (g_t = 0 short-circuits the injection)")
if abl_cfg['disable_perception_loss']:
    print("   [note] perception_contrast gradients are always 0 (L_perc is skipped)")


# ============ split the optimizer parameter groups ============
new_module_params = []
backbone_params = []
new_module_numel = 0
backbone_numel = 0
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith('cross_anchor_modules.') or name.startswith('perception_contrast.'):
        new_module_params.append(p)
        new_module_numel += p.numel()
    else:
        backbone_params.append(p)
        backbone_numel += p.numel()

print(f"\n-> optimizer param groups:")
print(f"   backbone   : {backbone_numel/1e9:.3f}B params @ lr={LEARNING_RATE:.1e}, wd=0.01")
print(f"   new_module : {new_module_numel/1e6:.2f}M params "
      f"@ lr={LEARNING_RATE*CROSS_ANCHOR_LR_MULT:.1e} ({CROSS_ANCHOR_LR_MULT}x), "
      f"wd={CROSS_ANCHOR_WEIGHT_DECAY}")
if new_module_numel == 0:
    raise SystemExit("\n❌ no cross_anchor_modules / perception_contrast parameters found!\n")

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params,   "lr": LEARNING_RATE,
         "weight_decay": 0.01},
        {"params": new_module_params, "lr": LEARNING_RATE * CROSS_ANCHOR_LR_MULT,
         "weight_decay": CROSS_ANCHOR_WEIGHT_DECAY},
    ],
)

total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * EPOCHS
warmup_steps = int(total_steps * 0.03)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=COSINE_ETA_MIN
)


def warmup_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    return 1.0


warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
model.config.total_steps = total_steps
print(f"-> total_steps = {total_steps} "
      f"(Stage1: 0~{int(total_steps*0.3)}, "
      f"Stage2: {int(total_steps*0.3)}~{int(total_steps*0.7)}, "
      f"Stage3: {int(total_steps*0.7)}~{total_steps})")
print(f"-> OUTPUT_DIR = {OUTPUT_DIR}\n")


# ============= 5. cross-attention weight monitoring =============
def snapshot_cross_attn_stats(model):
    stats = {}
    with torch.no_grad():
        for i, xa in enumerate(model.cross_anchor_modules):
            stats[f"xa{i}_q_std"]   = xa.q_proj.weight.std().item()
            stats[f"xa{i}_k_std"]   = xa.k_proj.weight.std().item()
            stats[f"xa{i}_v_std"]   = xa.v_proj.weight.std().item()
            stats[f"xa{i}_o_std"]   = xa.out_proj.weight.std().item()
            stats[f"xa{i}_o_abs_max"] = xa.out_proj.weight.abs().max().item()
    return stats


_init_stats = snapshot_cross_attn_stats(model)
print("[weight snapshot at start]")
for k, v in sorted(_init_stats.items()):
    print(f"  {k}: {v:.5f}")
wandb.log({**{f"init/{k}": v for k, v in _init_stats.items()}, "global_step": 0})


# ============= 6. training loop =============
print(f"\n[ABLATION:{ABLATION_TYPE}] starting training (EPOCHS={EPOCHS})...")
global_step = 0
human_prompt = "Please describe this image in detail."
# Identical to train_full_v6.py / eval_pope.py / eval_mirage.py /
# generate_captions.py: the bare short template, without a system prefix.
text_template = f"USER: {DEFAULT_IMAGE_TOKEN}\n{human_prompt} ASSISTANT: "

_dbg_prompt_ids = tokenizer_image_token(text_template, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
print(f"-> prompt template length: {_dbg_prompt_ids.shape[0]} tokens | "
      f"EOS: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

optimizer.zero_grad()

for epoch in range(EPOCHS):
    for step, batch_list in enumerate(train_loader):
        sample = batch_list[0]

        image_raw = Image.open(sample['img_path']).convert('RGB')
        image_sq, bboxes_sq, img_size_sq = expand2square_with_bboxes(image_raw, sample['bboxes'])
        image_tensor = image_processor.preprocess(
            image_sq, return_tensors='pt'
        )['pixel_values'].to(model.dtype).to('cuda')

        prompt_ids = tokenizer_image_token(
            text_template, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).to('cuda')
        answer_ids = tokenizer(
            sample['pos_text'], return_tensors='pt', add_special_tokens=False
        )['input_ids'][0].to('cuda')
        eos_tensor = torch.tensor([tokenizer.eos_token_id], device='cuda', dtype=answer_ids.dtype)
        answer_ids = torch.cat([answer_ids, eos_tensor], dim=0)

        input_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)
        labels = input_ids.clone()
        labels[0, :prompt_ids.shape[0]] = -100

        model.config.global_step = global_step

        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            labels=labels,
            bboxes=[bboxes_sq],
            class_names=[sample['class_names']],
            img_sizes_orig=[img_size_sq],
            return_dict=True,
        )

        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(backbone_params + new_module_params, max_norm=1.0)
            optimizer.step()
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            real_loss = loss.item() * GRAD_ACCUM_STEPS
            current_lr = optimizer.param_groups[0]['lr']
            current_lr_new = optimizer.param_groups[1]['lr']
            current_gamma = getattr(model, '_anchor_gamma', 0.0)

            log_dict = {
                "train_loss": real_loss,
                "lr": current_lr,
                "lr_new_module": current_lr_new,
                "gamma_t": current_gamma,
                "epoch": epoch + 1,
                "global_step": global_step,
            }
            if global_step % 100 == 0:
                xa_stats = snapshot_cross_attn_stats(model)
                log_dict.update(xa_stats)
            wandb.log(log_dict)

            if global_step % 500 == 0:
                xa_stats = snapshot_cross_attn_stats(model)
                _q = xa_stats["xa0_q_std"]
                _o = xa_stats["xa0_o_std"]
                _drift_q = _q - _init_stats["xa0_q_std"]
                _drift_o = _o - _init_stats["xa0_o_std"]
                # gate-aware health check:
                # - disable_cross_anchor=True  (A/D): cross_anchor is not meant to move
                # - disable_cross_anchor=False (B/C): still at init during stage 1 is
                #   fine, but it must move once g > 0
                if abl_cfg['disable_cross_anchor']:
                    health = f"💤 [{abl_cfg['tag']}] cross-anchor is off, static weights are expected"
                elif current_gamma == 0.0:
                    health = "💤 stage 1 (g=0): cross-attn is not meant to train yet"
                elif abs(_drift_q) > 1e-4:
                    health = "✅ training"
                else:
                    health = "⚠️ g>0 but q has not moved (small drift early in stage 2 is normal)"
                print(
                    f"[{abl_cfg['tag']}][Step {global_step}] Loss={real_loss:.4f} g={current_gamma:.3f} | "
                    f"xa0.q.std={_q:.5f} (Δ{_drift_q:+.5f}) "
                    f"xa0.o.std={_o:.5f} (Δ{_drift_o:+.5f}) {health}"
                )
            else:
                print(
                    f"[{abl_cfg['tag']}] Epoch [{epoch+1}/{EPOCHS}] | Step [{global_step}] | "
                    f"Loss: {real_loss:.4f} | LR: {current_lr:.2e} | g_t: {current_gamma:.3f}"
                )

            if global_step % SAVE_STEPS == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"💾 intermediate checkpoint: {save_path}")


# ============= 7. final save + weight snapshot =============
final_save_path = os.path.join(OUTPUT_DIR, "checkpoint-final")
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

_final_stats = snapshot_cross_attn_stats(model)
print("\n[weight snapshot at end]")
for k, v in sorted(_final_stats.items()):
    init_v = _init_stats[k]
    drift = v - init_v
    print(f"  {k}: {v:.5f}  (init {init_v:.5f}, Δ{drift:+.5f})")

# write a training summary next to the checkpoint
_drift_q = abs(_final_stats["xa0_q_std"] - _init_stats["xa0_q_std"])
_pass_threshold = 5e-6 if _QUICK else 1e-4
# The q_proj_trained verdict only means something when cross-anchor is enabled;
# in A and D it is designed not to train, so the field is written as null rather
# than as a failure.
if abl_cfg['disable_cross_anchor']:
    q_proj_trained_field = None
    q_judge_desc = "N/A (disable_cross_anchor=True, cross-anchor is not meant to train)"
else:
    q_proj_trained_field = bool(_drift_q > _pass_threshold)
    q_judge_desc = "PASS" if q_proj_trained_field else "FAIL"

summary = {
    "version": "v6.1",
    "ablation_type": ABLATION_TYPE,
    "tag": abl_cfg["tag"],
    "disable_perception_loss": abl_cfg["disable_perception_loss"],
    "disable_cross_anchor": abl_cfg["disable_cross_anchor"],
    "mode": "QUICK_VALIDATE" if _QUICK else "FULL",
    "total_steps": total_steps,
    "drift_q_pass_threshold": _pass_threshold,
    "init_stats": _init_stats,
    "final_stats": _final_stats,
    "weight_drift": {k: _final_stats[k] - _init_stats[k] for k in _init_stats},
    "q_proj_trained": q_proj_trained_field,
    "q_judge": q_judge_desc,
}
with open(os.path.join(final_save_path, "training_summary_v6.json"), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# the ablation config, read automatically by eval_pope.py / eval_mirage.py /
# generate_captions.py at evaluation time
with open(os.path.join(final_save_path, "ablation_config.json"), 'w', encoding='utf-8') as f:
    json.dump({
        "ablation_type": ABLATION_TYPE,
        "tag": abl_cfg["tag"],
        "disable_perception_loss": abl_cfg["disable_perception_loss"],
        "disable_cross_anchor": abl_cfg["disable_cross_anchor"],
        "desc": abl_cfg["desc"],
        "version": "v6.1",
        "total_steps": total_steps,
        "data": {
            "caption_source": "ShareGPT4V (sharegpt4v_instruct_gpt4-vision_cap100k)",
            "bbox_source": "COCO instances_train2017.json",
            "num_samples": len(train_dataset),
        },
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 [{abl_cfg['tag']}] final checkpoint: {final_save_path}")
print(f"   ablation_config.json + training_summary_v6.json written")
print(f"   verdict: delta q_std = {_drift_q:.2e}, threshold = {_pass_threshold:.0e} -> {q_judge_desc}")
print(f"✅ ablation [{ABLATION_TYPE}] training complete")
