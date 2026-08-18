import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import sys
import argparse
import json
import torch
import warnings
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

warnings.filterwarnings("ignore")
sys.path.append(f"{DSCC_ROOT}/LLaVA")
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.dual_stream import expand2square

# ================= 0. command-line arguments =================
parser = argparse.ArgumentParser(description="POPE Evaluation Script")
parser.add_argument("--model_path", type=str, required=True, help="path to the model weights to evaluate")
parser.add_argument("--pope_type", type=str, default="all",
                    choices=["random", "popular", "adversarial", "all"],
                    help="which POPE split to run; 'all' runs random, popular and adversarial in turn")
parser.add_argument("--output_dir", type=str, default=f"{DSCC_ROOT}/results/pope",
                    help="directory the JSON results are written to")
args = parser.parse_args()

# ================= 1. paths =================
MODEL_PATH = args.model_path
IMAGE_DIR = f"{DSCC_ROOT}/data/coco/val2014"
POPE_TYPES = ["random", "popular", "adversarial"] if args.pope_type == "all" else [args.pope_type]

print(f"\n=========================================")
print(f"🎯 evaluating model: {MODEL_PATH.split('/')[-1]}")
print(f"📂 splits: {', '.join(POPE_TYPES)}")
print(f"=========================================\n")

# ================= 2. load the model (once) =================
print("-> loading the model onto the GPU...")
# use_fast=False is required: training used LLaMA's slow (sentencepiece) tokenizer, and
# the fast one differs slightly on special tokens, whitespace and BOS. That shifts the
# evaluation input away from the training distribution, and a binary task like POPE is
# sensitive enough to drift into answering Yes to everything.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()

vision_tower = model.get_vision_tower()
if not getattr(vision_tower, 'is_loaded', False):
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.bfloat16)

model.eval()

# Read the ablation flags from ablation_config.json and inject them into model.config.
# Without this, ablation A (perception only) would run its cross-attention at g = 1 with
# untrained near-identity weights, adding noise of magnitude ~0.082 to the residual
# stream and contaminating the comparison (an abnormally low Yes ratio and inflated precision).
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
    """Same parsing rule as LLaVA's official eval_pope.py: look at the first sentence and
    answer No if it contains a negative word, otherwise Yes.

    Why the default is Yes rather than No:
        All three reference POPE scripts (VCD, OPERA and LLaVA's own) default to Yes and
        only answer No when a negative word appears. Defaulting to No systematically
        depresses the Yes ratio by misreading samples that are verbose without answering
        yes/no outright, which penalises long-output models asymmetrically: the full
        model (trained cross-attention, more assertive phrasing) gets pushed back to No
        more often than ablation A (perception trained alone, shorter answers), turning
        a real difference into the illusion that A beats the full model.

    Extension: on top of LLaVA's {No, no, not}, OPERA's contraction match for "n't" is
    included (covering "doesn't", "isn't", "can't"), which fits how POPE answers are
    actually phrased.
    """
    text = response.strip()
    if text.find('.') != -1:
        text = text.split('.')[0]      # first sentence only, so later explanations cannot interfere
    text = text.replace(',', '')
    words = text.split()
    if any(w in {'No', 'no', 'NO'} for w in words) \
            or 'not' in words \
            or any("n't" in w for w in words):
        return "no"
    return "yes"


def run_pope(pope_type: str):
    pope_data_path = f"{DSCC_ROOT}/data/pope/coco_pope_{pope_type}.json"

    print(f"\n========== ▶ evaluating POPE {pope_type.capitalize()} ==========")

    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    yes_pred = 0  # feeds the yes_ratio health check
    predictions = []

    with open(pope_data_path, 'r') as f:
        pope_data = [json.loads(line) for line in f]

    for item in tqdm(pope_data, desc=f"POPE-{pope_type}"):
        img_path = os.path.join(IMAGE_DIR, item['image'])
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)

        # Must match the training template exactly:
        #   text_template = f"USER: {DEFAULT_IMAGE_TOKEN}\n{human_prompt} ASSISTANT: "
        # Training uses NO system prefix.
        # Adding one at evaluation time creates a train/eval mismatch: the first token's
        # probability drifts and Yes/No answers flip.
        prompt = f"USER: <image>\n{item['text']} ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "ASSISTANT:" in full_response:
            response = full_response.split("ASSISTANT:")[-1].strip().lower()
        else:
            response = full_response.strip().lower()

        pred_ans = parse_yes_no(response)
        gt_ans = item['label'].lower()

        total += 1
        is_correct = (pred_ans == gt_ans)
        if is_correct:
            correct += 1
        if pred_ans == "yes":
            yes_pred += 1

        if pred_ans == "yes" and gt_ans == "yes":
            true_positives += 1
        elif pred_ans == "yes" and gt_ans == "no":
            false_positives += 1
        elif pred_ans == "no" and gt_ans == "yes":
            false_negatives += 1
        elif pred_ans == "no" and gt_ans == "no":
            true_negatives += 1

        predictions.append({
            "question": item['text'],
            "image": item['image'],
            "gt": gt_ans,
            "pred": pred_ans,
            "raw_response": response,
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = yes_pred / total if total > 0 else 0

    print("\n" + "=" * 40)
    print(f"🏆 POPE {pope_type.capitalize()} final scores:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  <-- the key hallucination metric")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1_score:.4f}")
    print(f"Yes Ratio: {yes_ratio:.4f}  <-- health check: POPE's ground truth is 50%; "
          f"below 30% the model is gaming the task by answering No, and precision is inflated")
    print("=" * 40 + "\n")

    output_file = os.path.join(args.output_dir, f"pope_{model_name}_{pope_type}.json")
    result_payload = {
        "model_path": MODEL_PATH,
        "pope_type": pope_type,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "yes_ratio": yes_ratio,
            "total": total,
            "correct": correct,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
        },
        "predictions": predictions
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"📁 full results saved to: {output_file}\n")
    return {"pope_type": pope_type, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1_score": f1_score, "yes_ratio": yes_ratio}


# ================= 3. run every requested split =================
summary = [run_pope(t) for t in POPE_TYPES]

# ================= 4. overview =================
if len(summary) > 1:
    print("\n" + "#" * 50)
    print("📊 summary over all POPE splits:")
    print(f"{'Subset':<12}{'Acc':>10}{'Prec':>10}{'Recall':>10}{'F1':>10}{'YesRatio':>10}")
    for s in summary:
        print(f"{s['pope_type']:<12}{s['accuracy']:>10.4f}{s['precision']:>10.4f}"
              f"{s['recall']:>10.4f}{s['f1_score']:>10.4f}{s['yes_ratio']:>10.4f}")
    print("#" * 50 + "\n")
