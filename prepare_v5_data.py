"""
prepare_v5_data.py -- training-data preparation.

(The "v5" in this file's name and in the output v5_caption_map.json is historical
 and unrelated to the model version. The v6.1 training scripts read exactly these
 names, so do not rename them.)

Turns ShareGPT4V's long GPT-4V captions into an {image_id: long_caption} lookup
table for train_full_v6.py / train_ablation_v6.py. The bboxes and class names
still come from instances_train2017.json and are not handled here.

Inputs:
  $DSCC_ROOT/data/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json
  $DSCC_ROOT/data/coco/annotations/instances_train2017.json

Output:
  $DSCC_ROOT/data/v5_caption_map.json   (key=image_id<str>, value=caption<str>)
"""
import json
import os
from pathlib import Path

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var

SHAREGPT4V_JSON = f"{DSCC_ROOT}/data/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json"
INSTANCES_JSON  = f"{DSCC_ROOT}/data/coco/annotations/instances_train2017.json"
IMAGE_DIR       = f"{DSCC_ROOT}/data/coco/train2017"
OUTPUT_JSON     = f"{DSCC_ROOT}/data/v5_caption_map.json"


# ================= 1. load ShareGPT4V =================
print(f"-> Loading ShareGPT4V from {SHAREGPT4V_JSON}")
with open(SHAREGPT4V_JSON, 'r', encoding='utf-8') as f:
    sgpt4v = json.load(f)
print(f"   total ShareGPT4V samples: {len(sgpt4v)}")

# ================= 2. keep only the COCO train2017 subset =================
coco_only = []
for item in sgpt4v:
    img_path = str(item.get("image", ""))
    if not img_path.startswith("coco/train2017/"):
        continue
    # read the image_id off the file name, e.g. coco/train2017/000000000009.jpg -> 9
    try:
        img_id = int(Path(img_path).stem)
    except ValueError:
        continue
    coco_only.append((img_id, item))
print(f"-> COCO train2017 subset in ShareGPT4V: {len(coco_only)}")

# ================= 3. load COCO instances to find the image_ids that have bboxes =================
print(f"-> Loading COCO instances from {INSTANCES_JSON}")
with open(INSTANCES_JSON, 'r', encoding='utf-8') as f:
    coco_inst = json.load(f)
ids_with_bbox = {ann['image_id'] for ann in coco_inst['annotations']}
print(f"   COCO train2017 has {len(coco_inst['images'])} images, "
      f"{len(ids_with_bbox)} have >=1 bbox annotation")

# ============ 4. join: an image needs BOTH a ShareGPT4V caption and a bbox ============
caption_map = {}
length_stats = []
skipped_no_bbox = 0
skipped_bad_conv = 0
skipped_no_image = 0

for img_id, item in coco_only:
    if img_id not in ids_with_bbox:
        skipped_no_bbox += 1
        continue
    # make sure the image file is really there, so training cannot crash on it
    img_full_path = os.path.join(IMAGE_DIR, f"{str(img_id).zfill(12)}.jpg")
    if not os.path.exists(img_full_path):
        skipped_no_image += 1
        continue
    # take GPT's turn as the caption, matching from == 'gpt' defensively
    convs = item.get("conversations", [])
    gpt_response = None
    for c in convs:
        if c.get("from") == "gpt":
            gpt_response = c.get("value", "").strip()
            break
    if not gpt_response:
        skipped_bad_conv += 1
        continue
    caption_map[img_id] = gpt_response
    length_stats.append(len(gpt_response.split()))

# ================= 5. write it out =================
print(f"\n-> valid samples: {len(caption_map)}")
print(f"   skipped, no bbox:          {skipped_no_bbox}")
print(f"   skipped, bad conversation: {skipped_bad_conv}")
print(f"   skipped, image not found:  {skipped_no_image}")

if length_stats:
    avg = sum(length_stats) / len(length_stats)
    srt = sorted(length_stats)
    med = srt[len(srt) // 2]
    print(f"\n   caption length: avg={avg:.1f}, median={med}, min={min(length_stats)}, max={max(length_stats)}")

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
# keys are written as strings (plain JSON) and cast back with int(k) on load
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump({str(k): v for k, v in caption_map.items()}, f, ensure_ascii=False)

print(f"\n✅ caption map saved: {OUTPUT_JSON}")
sample_id = list(caption_map.keys())[0]
print(f"\nsample preview (image_id={sample_id}):")
print(f"  {caption_map[sample_id][:300]}{'...' if len(caption_map[sample_id]) > 300 else ''}")
