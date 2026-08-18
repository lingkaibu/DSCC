import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import sys
import argparse
import json
import glob
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
# MME-Hallucination evaluation (out-of-domain generalisation).
#
# The protocol is identical to eval_pope.py / eval_hallusion.py: the slow
# tokenizer (use_fast=False), bf16, the disable_* flags injected from
# ablation_config.json, expand2square, the `USER: <image>\n{q} ASSISTANT: `
# template, and the default-Yes yes/no parser.
#
# Data: the official MME_Benchmark_release_version. Both directory layouts work:
#   (a) flat:  <subtask>/<name>.jpg + <subtask>/<name>.txt
#   (b) split: <subtask>/images/<name>.jpg + <subtask>/questions_answers_YN/<name>.txt
#   Each .txt holds two lines of `question<TAB>Yes|No` -- one positive and one
#   negative question about the same image.
#
# Hallucination subset (default): existence / count / position / color, the four
# object-level tasks that the VCD and OPERA papers report. Change with --subtasks.
#
# Official metrics:
#   acc   = per-question accuracy
#   acc+  = per-image: both questions must be right (stricter)
#   score(subtask) = (acc + acc+) * 100
#   total_score    = sum over subtasks (the number papers usually report)
# =====================================================================

parser = argparse.ArgumentParser(description="MME-Hallucination Evaluation (yes/no)")
parser.add_argument("--model_path", type=str, required=True, help="path to the model weights to evaluate")
parser.add_argument("--mme_root", type=str, default=f"{DSCC_ROOT}/data/MME/MME_Benchmark_release_version",
                    help="MME data root, holding one directory per subtask")
parser.add_argument("--subtasks", type=str, default="existence,count,position,color",
                    help="comma-separated subtask names (default: the four object-level hallucination tasks)")
parser.add_argument("--output_dir", type=str, default=f"{DSCC_ROOT}/results/mme_hallucination",
                    help="directory the JSON results are written to")
parser.add_argument("--no_answer_suffix", action="store_true",
                    help="do not append 'Please answer Yes or No.' (appended by default)")
parser.add_argument("--limit", type=int, default=0, help="only run the first N images per subtask (smoke test; 0 = all)")
args = parser.parse_args()

MODEL_PATH = args.model_path
MME_ROOT = args.mme_root
SUBTASKS = [s.strip() for s in args.subtasks.split(",") if s.strip()]
ANSWER_SUFFIX = "" if args.no_answer_suffix else " Please answer the question with Yes or No."

print(f"\n=========================================")
print(f"🎯 MME-Hallucination evaluation: {MODEL_PATH.rstrip('/').split('/')[-1]}")
print(f"📂 data root: {MME_ROOT}")
print(f"📋 subtasks: {SUBTASKS}")
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
    """Same as eval_pope.py / eval_hallusion.py: first sentence only, No if it contains a
    negative word, otherwise Yes."""
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


def find_image_and_txt(subtask_dir):
    """Return [(image_path, txt_path, basename), ...], handling both the flat layout and
    the images + questions_answers_YN one."""
    img_dir = os.path.join(subtask_dir, "images")
    qa_dir = os.path.join(subtask_dir, "questions_answers_YN")
    layout_split = os.path.isdir(img_dir) and os.path.isdir(qa_dir)
    search_dir = img_dir if layout_split else subtask_dir

    images = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"):
        images.extend(glob.glob(os.path.join(search_dir, ext)))
    images = sorted(set(images))

    items = []
    for img in images:
        base = os.path.splitext(os.path.basename(img))[0]
        txt = os.path.join(qa_dir if layout_split else subtask_dir, base + ".txt")
        if os.path.exists(txt):
            items.append((img, txt, base))
    return items, layout_split


def read_qa(txt_path):
    """Read one MME txt file -- lines of `question<TAB>Yes|No` -- into
    [(question, gt_yes_no), ...]."""
    qa = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # the official files are TAB-separated; fall back to whitespace for the odd one
            if '\t' in line:
                q, a = line.rsplit('\t', 1)
            else:
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                q, a = parts
            qa.append((q.strip(), a.strip().lower()))
    return qa


@torch.inference_mode()
def answer(question, image_tensor):
    prompt = f"USER: <image>\n{question.strip()}{ANSWER_SUFFIX} ASSISTANT: "
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    output_ids = model.generate(
        input_ids, images=image_tensor, max_new_tokens=10,
        do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in full:
        return full.split("ASSISTANT:")[-1].strip().lower()
    return full.strip().lower()


# ================= run the evaluation =================
all_predictions = []
subtask_scores = {}

for sub in SUBTASKS:
    sub_dir = os.path.join(MME_ROOT, sub)
    if not os.path.isdir(sub_dir):
        print(f"⚠️ subtask directory not found, skipping: {sub_dir}")
        continue
    items, layout_split = find_image_and_txt(sub_dir)
    if args.limit > 0:
        items = items[:args.limit]
    print(f"\n[{sub}] images={len(items)} layout={'images+QA subdirs' if layout_split else 'flat'}"
          + ("  ⚠️ smoke-test limit" if args.limit > 0 else ""))

    per_q_correct = 0
    per_q_total = 0
    img_all_correct = 0  # acc+ counter: every question about the image is right
    img_total = 0

    for img_path, txt_path, base in tqdm(items, desc=f"MME-{sub}"):
        qa = read_qa(txt_path)
        if not qa:
            continue
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image)
        image_tensor = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)

        this_img_correct = True
        for q, gt in qa:
            raw = answer(q, image_tensor)
            pred = parse_yes_no(raw)
            ok = (pred == gt)
            per_q_total += 1
            if ok:
                per_q_correct += 1
            else:
                this_img_correct = False
            all_predictions.append({
                "subtask": sub, "image": base, "question": q,
                "gt": gt, "pred": pred, "raw_response": raw, "correct": int(ok),
            })
        img_total += 1
        if this_img_correct:
            img_all_correct += 1

    acc = per_q_correct / per_q_total if per_q_total else 0.0
    acc_plus = img_all_correct / img_total if img_total else 0.0
    score = (acc + acc_plus) * 100
    subtask_scores[sub] = {
        "acc": acc, "acc_plus": acc_plus, "score": score,
        "n_questions": per_q_total, "n_images": img_total,
    }
    print(f"  [{sub}] acc={acc:.4f}  acc+={acc_plus:.4f}  score={score:.2f}")

# ================= summary =================
total_score = sum(v["score"] for v in subtask_scores.values())
print("\n" + "=" * 50)
print(f"🏆 MME-Hallucination scores ({model_name})")
print(f"{'subtask':<12}{'acc':>10}{'acc+':>10}{'score':>10}")
for sub, v in subtask_scores.items():
    print(f"{sub:<12}{v['acc']:>10.4f}{v['acc_plus']:>10.4f}{v['score']:>10.2f}")
print("-" * 50)
print(f"{'TOTAL':<12}{'':>10}{'':>10}{total_score:>10.2f}   <-- the number papers usually report")
print("=" * 50 + "\n")

output_file = os.path.join(args.output_dir, f"mme_hallucination_{model_name}.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model_path": MODEL_PATH,
        "subtasks": SUBTASKS,
        "answer_suffix": ANSWER_SUFFIX,
        "total_score": total_score,
        "subtask_scores": subtask_scores,
        "predictions": all_predictions,
    }, f, ensure_ascii=False, indent=2)
print(f"📁 full results saved to: {output_file}\n")
