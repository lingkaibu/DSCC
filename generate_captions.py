import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
from PIL import Image
import json
import random
from tqdm import tqdm
import sys
from datetime import datetime

# LLaVA components
sys.path.append(f"{DSCC_ROOT}/LLaVA")
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.dual_stream import expand2square

# ================= 0. argparse =================
# The checkpoint path is never hard-coded: pass it with --model_path, the same way
# eval_pope.py and eval_mirage.py do.
parser = argparse.ArgumentParser(description="DSCC caption generation (the inference half of CHAIR)")
parser.add_argument("--model_path", type=str, required=True,
                    help="path to the model checkpoint to evaluate")
parser.add_argument("--output_dir", type=str, default=f"{DSCC_ROOT}/results/captions",
                    help="directory the captions are written to")
args = parser.parse_args()

# ================= 1. paths =================
MODEL_PATH = args.model_path
IMAGE_FOLDER = f"{DSCC_ROOT}/data/coco/val2014"  # the COCO validation images
OUTPUT_DIR = args.output_dir
LOG_DIR = f"{DSCC_ROOT}/results/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# timestamp every run's file names, so a re-run never overwrites the previous results or log
_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"coco_generated_captions_{_TS}.jsonl")
LOG_FILE = os.path.join(LOG_DIR, f"generate_captions_{_TS}.log")


class Tee:
    """Write stdout to the terminal and the log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_fp = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_fp)
sys.stderr = Tee(sys.__stderr__, _log_fp)
print(f"📝 log file: {LOG_FILE}")
print(f"📦 caption output file: {OUTPUT_FILE}")

# This is the exact prompt used by VCD, OPERA, Woodpecker and Rohrbach et al. 2018 --
# word for word. Set VERBOSE_PROMPT=1 for the extended version, which elicits longer
# captions; note that CHAIR numbers produced that way are not strictly comparable to the
# tables in those papers and the difference must be stated.
if os.environ.get("VERBOSE_PROMPT", "0") == "1":
    QUESTION = "Please describe this image in detail, covering all the objects, people, and the surroundings you can observe."
else:
    QUESTION = "Please describe this image in detail."

# ================= 2. load the model =================
print("-> loading the dual-stream model...")
# use_fast=False: training used LLaMA's slow (sentencepiece) tokenizer, and the fast one
# differs slightly on special tokens, whitespace and BOS, which shifts the evaluation
# input away from the training distribution. Same choice as eval_pope.py.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.bfloat16)

model.eval()
# use the vision tower's own image_processor, exactly as configured during training
image_processor = vision_tower.image_processor

# Read the ablation flags from ablation_config.json and inject them into model.config.
# Without this, ablation A (perception only) would run its cross-attention at g = 1 with
# untrained near-identity weights, adding noise of magnitude ~0.082 to the residual
# stream and contaminating the comparison. Same handling as eval_pope.py / eval_mirage.py.
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

# Vanilla-baseline mode -- the shared base that VCD and OPERA compare against.
#   Force both streams off. When an off-the-shelf LLaVA-1.5-7B is loaded through this
#   class, the dual-stream modules are randomly initialised (std=0.02); running it in
#   full DSCC mode would inject that random cross_anchor noise and the result would not
#   be vanilla at all. disable_cross_anchor=True sets g = 0 so nothing is injected; the
#   perception term is a training-time loss and does not apply here.
#   This takes priority over the ablation_config.json injection above.
if os.environ.get("VANILLA_BASELINE", "0") == "1":
    model.config.disable_perception_loss = True
    model.config.disable_cross_anchor = True
    print("[EVAL] ⚠️ VANILLA_BASELINE=1: both streams forced off, running as stock LLaVA-1.5")

# ================= 3. select the test images =================
# Four modes, highest priority first:
#   SMOKE_TEST=1   -> smoke test: 10 images, printed only, nothing written
#   CHAIR_500=1    -> the standard CHAIR setting: 500 images drawn from val2014 with
#                     random.seed(42) + shuffle, reproducing the OPERA / VCD /
#                     Woodpecker protocol so the numbers are directly comparable
#   NUM_IMAGES=N   -> the first N images by file name, for a quick sanity check
#   none of them   -> every image in IMAGE_FOLDER
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
CHAIR_500 = os.environ.get("CHAIR_500", "0") == "1"
_num_env = os.environ.get("NUM_IMAGES", "").strip()

image_files = os.listdir(IMAGE_FOLDER)
image_files = sorted([f for f in image_files if f.endswith('.jpg')])  # sort first, so the selection is machine-independent

if SMOKE_TEST:
    image_files = image_files[:10]
elif CHAIR_500:
    # OPERA's default: seed 42, shuffle, take the first 500
    rng = random.Random(42)
    rng.shuffle(image_files)
    image_files = image_files[:500]
elif _num_env:
    image_files = image_files[:int(_num_env)]

# ================= 4. run inference =================
if SMOKE_TEST:
    mode_str = "🔥 smoke test (nothing written to disk)"
elif CHAIR_500:
    mode_str = "📊 CHAIR-500 standard protocol (seed=42 shuffle)"
else:
    mode_str = "full run"
print(f"-> [{mode_str}] captioning {len(image_files)} images...")

# In CHAIR-500 mode the output file carries a chair500 marker so it is not confused with
# a full run; it still matches eval_chair.py's glob.
if CHAIR_500:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"coco_generated_captions_chair500_{_TS}.jsonl")
    print(f"📦 CHAIR-500 output file: {OUTPUT_FILE}")

# the smoke test writes nothing; the other modes write the JSONL
out_fp = None if SMOKE_TEST else open(OUTPUT_FILE, 'w', encoding='utf-8')

# caption length statistics
_word_counts = []

try:
    for idx, img_name in enumerate(tqdm(image_files)):
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        # expand2square first, to match the training distribution, then the image_processor
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16)

        # Exactly the training template, with no system prefix, matching eval_pope.py and
        # eval_mirage.py. The full template with a system prefix shifts the first token's
        # probability away from the training distribution and the CHAIR numbers drift.
        prompt = f"USER: <image>\n{QUESTION} ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Generation: max_new_tokens=512 (as in VCD and OPERA) with greedy decoding, plus
        # the anti-repetition pair repetition_penalty=1.1 and no_repeat_ngram_size=4.
        #
        # Both settings are load-bearing and were tuned the hard way. Greedy decoding on
        # this data gets stuck in fixed points -- an early validation run produced
        # "one sentence, then the same sentence four more times" -- so the anti-repetition
        # pair is on by default.
        #
        # They are not free, though. no_repeat_ngram_size=4 bans every repeated 4-gram,
        # and enumerative captions open sentences with phrases like "On the table" or
        # "There is a"; once such a 4-gram appears it is banned for the rest of the
        # caption, the model is forced to rephrase, and when it cannot it emits EOS early
        # (captions collapsing to ~35 words). repetition_penalty=1.1 compounds this under
        # greedy decoding by penalising tokens already used, and COCO class names and
        # function words necessarily repeat.
        #
        # If a run shows captions cut short, NO_ANTI_REPEAT=1 turns both off.
        gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )
        # escape hatch: NO_ANTI_REPEAT=1 disables both anti-repetition settings
        if os.environ.get("NO_ANTI_REPEAT", "0") == "1":
            gen_kwargs.pop("repetition_penalty", None)
            gen_kwargs.pop("no_repeat_ngram_size", None)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                **gen_kwargs,
            )

        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_response.split("ASSISTANT:")[
            -1].strip() if "ASSISTANT:" in full_response else full_response.strip()

        # pull the image_id out of the file name (COCO_val2014_000000xxxxxx.jpg)
        try:
            image_id = int(img_name.split('_')[-1].split('.')[0])
        except:
            image_id = img_name

        _word_counts.append(len(response.split()))

        # smoke test prints everything; a real run samples the first 5
        if SMOKE_TEST or idx < 5:
            print(f"\n[Sample {idx}] {img_name} ({_word_counts[-1]} words)\n  -> {response}")

        if out_fp is not None:
            result = {"image_id": image_id, "caption": response}
            out_fp.write(json.dumps(result) + "\n")
finally:
    if out_fp is not None:
        out_fp.close()

if _word_counts:
    _avg = sum(_word_counts) / len(_word_counts)
    _mn, _mx = min(_word_counts), max(_word_counts)
    print(f"\n📏 caption length: mean {_avg:.1f} words / min {_mn} / max {_mx} / {len(_word_counts)} captions")

if SMOKE_TEST:
    print(f"\n🔥 smoke test finished. Check the 10 captions above to see the model behaves.")
else:
    print(f"\n✅ done, all captions saved to {OUTPUT_FILE}")