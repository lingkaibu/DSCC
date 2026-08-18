import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import sys
import argparse
import json
import torch
import warnings
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
sys.path.append(f"{DSCC_ROOT}/LLaVA")
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.dual_stream import expand2square

# =====================================================================
# MMHal-Bench inference (out-of-domain generalisation; the third and heaviest OOD set).
#
# This is stage 1 of a two-stage evaluation: inference here, GPT scoring (0-6) in a
# separate judge script. It is split -- unlike eval_hallusion.py and
# eval_mme_hallucination.py, which do everything in one pass -- because MMHal's
# official protocol scores informativeness and hallucination with an LLM judge, the
# same arrangement eval_mirage.py + auto_judge_dualstream.py use.
#
# Model loading and ablation-flag injection are word-for-word identical to
# eval_pope.py / eval_hallusion.py: the slow tokenizer (use_fast=False), bf16, the
# disable_* flags from ablation_config.json, expand2square, and the
# `USER: <image>\n{q} ASSISTANT: ` template (no system prefix, as in training).
#
# The key difference from the yes/no benchmarks: MMHal expects open-ended answers, so
# max_new_tokens is 512, there is no parse_yes_no, and model_output is written out
# verbatim for the judge.
#
# Data: MMHal-Bench (96 questions across 8 question types), laid out as
#     <data_root>/response_template.json  (the sample list)
#     <data_root>/images/                 (images, named after image_id)
#   Each sample has: question_type / question_topic / image_id / image_src /
#                    image_content (list) / question / gt_answer
#
# Output: a self-contained JSONL (each line carries gt_answer and image_content), so
# the judge stage never has to read the original data.
# =====================================================================

parser = argparse.ArgumentParser(description="MMHal-Bench inference (open-ended answers)")
parser.add_argument("--model_path", type=str, required=True, help="path to the model weights to evaluate")
parser.add_argument("--data_root", type=str, default=f"{DSCC_ROOT}/data/MMHal-Bench",
                    help="MMHal-Bench root, holding response_template.json and images/")
parser.add_argument("--json_file", type=str, default=None,
                    help="path to the sample json; defaults to <data_root>/response_template.json")
parser.add_argument("--image_dir", type=str, default=None,
                    help="image directory; defaults to <data_root>/images")
parser.add_argument("--output_dir", type=str, default=f"{DSCC_ROOT}/results/mmhal",
                    help="directory the JSONL predictions are written to")
parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="generation budget for the open-ended answers")
parser.add_argument("--limit", type=int, default=0,
                    help="only run the first N samples (smoke test; 0 = all)")
parser.add_argument("--run_tag", type=str, default=None,
                    help="suffix for the output file name; by default the most distinctive directory in the checkpoint path")
args = parser.parse_args()

MODEL_PATH = args.model_path
DATA_ROOT = args.data_root
JSON_FILE = args.json_file or os.path.join(DATA_ROOT, "response_template.json")
IMAGE_DIR = args.image_dir or os.path.join(DATA_ROOT, "images")

os.makedirs(args.output_dir, exist_ok=True)
# All four ablation checkpoints end in a directory called checkpoint-final, so using the
# last path component alone would make the output JSONLs overwrite each other. Prefer
# --run_tag; otherwise, when the last component is a generic checkpoint* name, use the
# one above it (dualstream_v6_1_percfix and friends).
_parts = MODEL_PATH.rstrip('/').replace('\\', '/').split('/')
if args.run_tag:
    model_name = args.run_tag
elif _parts[-1].startswith('checkpoint') and len(_parts) >= 2:
    model_name = _parts[-2]
else:
    model_name = _parts[-1]
output_file = os.path.join(args.output_dir, f"mmhal_{model_name}.jsonl")

print(f"\n=========================================")
print(f"🎯 MMHal-Bench inference: {model_name}")
print(f"📂 data: {JSON_FILE}")
print(f"🖼️  images: {IMAGE_DIR}")
print(f"=========================================\n")

# ========== load the model (identical to eval_pope.py / eval_hallusion.py) ==========
print("-> loading the model onto the GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()

vision_tower = model.get_vision_tower()
if not getattr(vision_tower, 'is_loaded', False):
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.bfloat16)
model.eval()

# Inject the ablation flags. Required when evaluating an ablation checkpoint: without
# them the cross-attention runs on untrained weights and contaminates the comparison.
abl_cfg_path = os.path.join(MODEL_PATH, "ablation_config.json")
if os.path.exists(abl_cfg_path):
    with open(abl_cfg_path) as f:
        abl_cfg = json.load(f)
    model.config.disable_perception_loss = abl_cfg["disable_perception_loss"]
    model.config.disable_cross_anchor    = abl_cfg["disable_cross_anchor"]
    tag = abl_cfg.get("tag") or abl_cfg.get("ablation_type", "?")
    print(f"[EVAL] injecting ablation flags: {tag} "
          f"| disable_perception_loss={abl_cfg['disable_perception_loss']} "
          f"| disable_cross_anchor={abl_cfg['disable_cross_anchor']}")
else:
    print(f"[EVAL] no ablation_config.json found; evaluating as full DSCC (both streams on)")

image_processor = vision_tower.image_processor


def resolve_image_path(item: dict) -> str:
    """In MMHal's official repository the image file name is the basename of image_src,
    e.g. 'https://.../7206072054_c53d53b97d_o.jpg' -> 'images/7206072054_c53d53b97d_o.jpg'.
    Note that image_id is a hash ('df62a56fdc1bb12b'), not a file name -- it is only an
    identifier. Look the image up by image_src basename first, then fall back to image_id."""
    src = str(item.get("image_src", "")).strip()
    src_base = os.path.basename(src.split("?")[0]) if src else ""
    image_id = str(item.get("image_id", "")).strip()
    candidates = []
    if src_base:
        candidates.append(os.path.join(IMAGE_DIR, src_base))
    if image_id:
        candidates.append(os.path.join(IMAGE_DIR, image_id))
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append(os.path.join(IMAGE_DIR, image_id + ext))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0] if candidates else ""


def generate_answer(question: str, img_path: str) -> str:
    image = Image.open(img_path).convert('RGB')
    image = expand2square(image)
    image_tensor = image_processor.preprocess(
        image, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)
    # exactly the training template: USER: <image>\n{q} ASSISTANT:  (no system prefix)
    prompt = f"USER: <image>\n{question.strip()} ASSISTANT: "
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in full:
        return full.split("ASSISTANT:")[-1].strip()
    return full.strip()


# ================= run inference =================
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
# accept both {"data": [...]} and a bare [...]
if isinstance(data, dict):
    data = data.get("data") or data.get("questions") or list(data.values())

if args.limit > 0:
    data = data[:args.limit]
    print(f"[EVAL] ⚠️ --limit={args.limit}: running only the first {len(data)} samples (smoke test)\n")

results = []
n_missing_img = 0
for item in tqdm(data, desc="MMHal-Bench"):
    img_path = resolve_image_path(item)
    if not img_path or not os.path.exists(img_path):
        n_missing_img += 1
        print(f"⚠️ image missing, skipping: image_id={item.get('image_id')} src={item.get('image_src')}")
        continue
    response = generate_answer(item["question"], img_path)
    results.append({
        "question_type": item.get("question_type", ""),
        "question_topic": item.get("question_topic", ""),
        "image_id": item.get("image_id", ""),
        "image_src": item.get("image_src", ""),
        "image_content": item.get("image_content", []),
        "question": item["question"],
        "gt_answer": item.get("gt_answer", ""),
        "model_output": response,
    })

# ================= write the results =================
with open(output_file, 'w', encoding='utf-8') as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')

print(f"\n✅ inference done: {len(results)} answers written to {output_file}")
if n_missing_img:
    print(f"⚠️ {n_missing_img} samples were skipped for a missing image -- make sure {IMAGE_DIR} is complete")
print(f"\nNext step: score the answers with the GPT judge\n  python judge_mmhal.py --jsonl_path {output_file}\n")
