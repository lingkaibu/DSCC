import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import sys
import argparse
import pandas as pd
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
parser = argparse.ArgumentParser(description="MIRAGE inference script")
parser.add_argument("--model_path", type=str, required=True, help="path to the model weights to evaluate")
parser.add_argument("--output_dir", type=str, default=f"{DSCC_ROOT}/results/mirage",
                    help="directory the JSONL predictions are written to")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
model_name = args.model_path.rstrip('/').split('/')[-1]
output_file = os.path.join(args.output_dir, f"mirage_{model_name}.jsonl")

# ================= 1. MIRAGE paths =================
MIRAGE_DATA_PATH = f"{DSCC_ROOT}/data/mirage/eval_code/data/mirage.tsv"
IMAGE_DIR = f"{DSCC_ROOT}/data/mirage/eval_code/data"  # where the images are expected to live

print(f"\n=========================================")
print(f"🎯 running the MIRAGE benchmark")
print(f"📦 model: {args.model_path.split('/')[-1]}")
print(f"=========================================\n")

# ================= 2. load the model =================
print("-> loading the model onto the GPU...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = LlavaLlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda()

vision_tower = model.get_vision_tower()
if not getattr(vision_tower, 'is_loaded', False):
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.bfloat16)

model.eval()

# Read the ablation flags from ablation_config.json and inject them into model.config.
# Without this, ablation A (perception only) would run its cross-attention at g = 1 with
# untrained near-identity weights, adding noise of magnitude ~0.082 to the residual
# stream and distorting the A-vs-full comparison.
abl_cfg_path = os.path.join(args.model_path, "ablation_config.json")
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

# use the vision tower's own image_processor, exactly as configured during training
image_processor = vision_tower.image_processor

# ================= 3. inference =================
print("-> generating reasoning chains over the MIRAGE dataset...")

# read the TSV with pandas
try:
    df = pd.read_csv(MIRAGE_DATA_PATH, sep='\t')
except FileNotFoundError:
    print(f"❌ dataset file not found, check the path: {MIRAGE_DATA_PATH}")
    sys.exit(1)

results_to_save = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_filename = row['image']
    # prefer the 'prompt' column, fall back to 'question'
    question = row.get('prompt', row.get('question', ''))

    # skip items without an image
    if pd.isna(img_filename):
        continue

    img_path = os.path.join(IMAGE_DIR, str(img_filename))

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ could not read the image: {img_path}, skipping this item.")
        continue

    # expand2square first, to match the training distribution, then the image_processor
    image = expand2square(image)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)

    # Must match the training template exactly:
    #   text_template = f"USER: {DEFAULT_IMAGE_TOKEN}\n{human_prompt} ASSISTANT: "
    # Training uses NO system prefix.
    # Adding one at evaluation time creates a train/eval mismatch: the first token of the
    # reasoning chain drifts and the whole answer goes off course.
    # The space after ASSISTANT: is literal, matching training; split("ASSISTANT:") still works.
    prompt = f"USER: <image>\n{question} ASSISTANT: "
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            use_cache=True
        )

    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "ASSISTANT:" in full_response:
        response = full_response.split("ASSISTANT:")[-1].strip()
    else:
        response = full_response.strip()

    results_to_save.append({
        "question": question,  # the question text doubles as the identifier
        "image": img_filename,
        "model_output": response
    })

# ================= 4. write the JSONL =================
print(f"-> inference done, writing results to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for res in results_to_save:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')

print(f"✅ all predictions saved to {output_file}")
