import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import sys
import argparse
import json
import torch
import warnings
from collections import defaultdict
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
# HallusionBench evaluation (out-of-domain generalisation).
#
# The protocol matches eval_pope.py exactly: the slow tokenizer (use_fast=False),
# bf16, the disable_* flags injected from ablation_config.json, expand2square, the
# `USER: <image>\n{q} ASSISTANT: ` template, and the default-Yes yes/no parser.
#
# Data: https://github.com/tianyi-lab/HallusionBench
#   - HallusionBench.json (the sample list)
#   - hallusion_bench.zip extracted into <data_root>/hallusion_bench/... (the images)
#   Each sample has: category (VD/VS) / subcategory (illusion/easy/hard) /
#   visual_input (1 = needs the image, 0 = text only) / set_id / figure_id /
#   question_id / question / gt_answer ("1" = yes, "0" = no) / filename (relative path)
#
# Metrics (the official three, aggregated all-correct):
#   aAcc = per-question accuracy, ungrouped
#   fAcc = grouped by figure (category, subcategory, set_id, figure_id); the group
#          counts as correct only if every question in it is
#   qAcc = grouped by question (category, subcategory, set_id, question_id), same rule
#
# One deliberate deviation from the official protocol: the reference implementation
# uses GPT-4 to map free-form answers onto yes/no/uncertain, and scores
# "VS without an image + uncertain" as correct. This script does plain string
# matching with no API call, so it never produces an uncertain state and the
# VS-uncertain rule does not apply. Papers should state that this is the
# string-match variant (sharing POPE's parse_yes_no), not the GPT-4-judge one.
# =====================================================================

parser = argparse.ArgumentParser(description="HallusionBench Evaluation (yes/no string-match)")
parser.add_argument("--model_path", type=str, required=True, help="path to the model weights to evaluate")
parser.add_argument("--data_root", type=str, default=f"{DSCC_ROOT}/data/HallusionBench",
                    help="HallusionBench root, holding HallusionBench.json and the hallusion_bench/ images")
parser.add_argument("--json_file", type=str, default=None,
                    help="path to the sample json; defaults to <data_root>/HallusionBench.json")
parser.add_argument("--output_dir", type=str, default=f"{DSCC_ROOT}/results/hallusion",
                    help="directory the JSON results are written to")
parser.add_argument("--no_answer_suffix", action="store_true",
                    help="do not append 'Please answer Yes or No.' (appended by default, which makes answers easier to parse)")
parser.add_argument("--skip_text_only", action="store_true",
                    help="skip the text-only questions (visual_input==0); by default they run without an image")
parser.add_argument("--limit", type=int, default=0,
                    help="only run the first N samples (smoke test; 0 = all)")
args = parser.parse_args()

MODEL_PATH = args.model_path
DATA_ROOT = args.data_root
JSON_FILE = args.json_file or os.path.join(DATA_ROOT, "HallusionBench.json")
ANSWER_SUFFIX = "" if args.no_answer_suffix else " Please answer the question with Yes or No."

print(f"\n=========================================")
print(f"🎯 HallusionBench evaluation: {MODEL_PATH.rstrip('/').split('/')[-1]}")
print(f"📂 data: {JSON_FILE}")
print(f"=========================================\n")

# ================= load the model (identical to eval_pope.py) =================
print("-> loading the model onto the GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()

vision_tower = model.get_vision_tower()
if not getattr(vision_tower, 'is_loaded', False):
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.bfloat16)
model.eval()

# Inject the ablation flags. Required when evaluating an ablation checkpoint, or the
# numbers come out distorted.
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
os.makedirs(args.output_dir, exist_ok=True)
model_name = MODEL_PATH.rstrip('/').split('/')[-1]


def parse_yes_no(response: str) -> str:
    """Same as eval_pope.py and LLaVA's own script: first sentence only, No if it contains
    a negative word, otherwise Yes."""
    text = response.strip()
    if text.find('.') != -1:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split()
    if any(w in {'No', 'no', 'NO'} for w in words) \
            or 'not' in words \
            or any("n't" in w for w in words):
        return "no"
    return "yes"


def resolve_image_path(filename: str) -> str:
    """HallusionBench filenames look like './VD/illusion/0_0_0.png' and land under
    <data_root>/hallusion_bench/ once extracted; this also accepts them directly under
    data_root."""
    if not filename:
        return ""
    rel = filename.lstrip("./").lstrip("/")
    candidates = [
        os.path.join(DATA_ROOT, rel),
        os.path.join(DATA_ROOT, "hallusion_bench", rel),
        os.path.join(DATA_ROOT, "hallusion_bench", os.path.basename(filename)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # return the preferred path so a missing image is reported, not swallowed


def generate_answer(question: str, img_path: str) -> str:
    prompt_q = question.strip() + ANSWER_SUFFIX
    if img_path and os.path.exists(img_path):
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image)
        image_tensor = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)
        prompt = f"USER: <image>\n{prompt_q} ASSISTANT: "
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        gen_kwargs = dict(images=image_tensor)
    else:
        # text-only question (visual_input==0): no <image> token, no images argument,
        # so this degenerates to plain LLaMA
        prompt = f"USER: {prompt_q} ASSISTANT: "
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        gen_kwargs = {}

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in full:
        return full.split("ASSISTANT:")[-1].strip().lower()
    return full.strip().lower()


# ================= run the evaluation =================
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

if args.limit > 0:
    data = data[:args.limit]
    print(f"[EVAL] ⚠️ --limit={args.limit}: running only the first {len(data)} samples (smoke test)\n")

predictions = []
n_text_only = 0
n_missing_img = 0

for item in tqdm(data, desc="HallusionBench"):
    visual_input = str(item.get("visual_input", "1"))
    filename = item.get("filename", "") or ""
    img_path = resolve_image_path(filename)

    if visual_input == "0" or not filename:
        if args.skip_text_only:
            continue
        n_text_only += 1
        img_path = ""
    elif not os.path.exists(img_path):
        n_missing_img += 1

    response = generate_answer(item["question"], img_path)
    pred = parse_yes_no(response)
    gt = "yes" if str(item["gt_answer"]) == "1" else "no"
    correct = int(pred == gt)

    predictions.append({
        "category": item.get("category", ""),
        "subcategory": item.get("subcategory", ""),
        "set_id": item.get("set_id", ""),
        "figure_id": item.get("figure_id", ""),
        "question_id": item.get("question_id", ""),
        "visual_input": visual_input,
        "question": item["question"],
        "filename": filename,
        "gt": gt,
        "pred": pred,
        "raw_response": response,
        "correct": correct,
    })


# ================= metrics (all-correct aggregation) =================
def grouped_acc(preds, keys):
    groups = defaultdict(list)
    for p in preds:
        groups[tuple(p[k] for k in keys)].append(p["correct"])
    n_all_correct = sum(1 for v in groups.values() if all(v))
    return n_all_correct / len(groups) if groups else 0.0, len(groups)


total = len(predictions)
n_correct = sum(p["correct"] for p in predictions)
aAcc = n_correct / total if total else 0.0
fAcc, n_fig = grouped_acc(predictions, ["category", "subcategory", "set_id", "figure_id"])
qAcc, n_qst = grouped_acc(predictions, ["category", "subcategory", "set_id", "question_id"])

# break down by category (VD = visual dependent, VS = visual supplement)
cat_acc = {}
for cat in sorted({p["category"] for p in predictions}):
    sub = [p for p in predictions if p["category"] == cat]
    cat_acc[cat] = sum(p["correct"] for p in sub) / len(sub) if sub else 0.0

yes_ratio = sum(1 for p in predictions if p["pred"] == "yes") / total if total else 0.0

print("\n" + "=" * 44)
print(f"🏆 HallusionBench scores ({model_name})")
print(f"samples: {total}  (text-only: {n_text_only}, missing images: {n_missing_img})")
print(f"aAcc (per-question accuracy) : {aAcc:.4f}   ({n_correct}/{total})")
print(f"fAcc (per-figure consistency): {fAcc:.4f}   (figures={n_fig})")
print(f"qAcc (per-pair consistency)  : {qAcc:.4f}   (pairs={n_qst})")
for cat, a in cat_acc.items():
    print(f"  - {cat} acc: {a:.4f}")
print(f"Yes Ratio: {yes_ratio:.4f}  (health check; around 50% is normal)")
print("=" * 44 + "\n")

if n_missing_img:
    print(f"⚠️ {n_missing_img} images were not found, so these numbers may be off -- "
          f"check that {DATA_ROOT}/hallusion_bench/ is fully extracted.\n")

output_file = os.path.join(args.output_dir, f"hallusion_{model_name}.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model_path": MODEL_PATH,
        "answer_suffix": ANSWER_SUFFIX,
        "metrics": {
            "aAcc": aAcc, "fAcc": fAcc, "qAcc": qAcc,
            "total": total, "correct": n_correct,
            "n_figures": n_fig, "n_question_pairs": n_qst,
            "n_text_only": n_text_only, "n_missing_img": n_missing_img,
            "yes_ratio": yes_ratio, "category_acc": cat_acc,
        },
        "predictions": predictions,
    }, f, ensure_ascii=False, indent=2)
print(f"📁 full results saved to: {output_file}\n")
