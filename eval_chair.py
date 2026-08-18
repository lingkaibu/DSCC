import json
import re
import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import sys
import glob
from datetime import datetime
from collections import defaultdict

# ================= 1. paths =================
CAPTION_DIR = f"{DSCC_ROOT}/results/captions"
INSTANCE_ANNO_FILE = f"{DSCC_ROOT}/data/coco/annotations/instances_val2014.json"
CAPTION_ANNO_FILE = f"{DSCC_ROOT}/data/coco/annotations/captions_val2014.json"

# Choosing which caption file to evaluate:
#   1) an absolute path in the GEN_FILE environment variable wins, so any historical
#      file can be picked by hand
#   2) otherwise the newest timestamped output in CAPTION_DIR, by modification time.
#      The underscore in the glob coco_generated_captions_*.jsonl is deliberate: an old
#      un-timestamped coco_generated_captions.jsonl can never be selected automatically.
#   3) if no timestamped file exists this raises, rather than silently falling back to
#      an old one.
def _pick_gen_file():
    env_path = os.environ.get("GEN_FILE", "").strip()
    if env_path:
        return env_path
    candidates = glob.glob(os.path.join(CAPTION_DIR, "coco_generated_captions_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped coco_generated_captions_*.jsonl found in {CAPTION_DIR}. "
            f"Run generate_captions.py first, or set GEN_FILE=<path> to point at the file to evaluate."
        )
    return max(candidates, key=os.path.getmtime)

GEN_FILE = _pick_gen_file()

# ================= logging =================
LOG_DIR = f"{DSCC_ROOT}/results/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"eval_chair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


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
print(f"📦 caption file under evaluation: {GEN_FILE}")

# ============ 2. the 80 COCO classes and their synonyms (standard CHAIR) ============
# Keys are the official COCO class names; values list every synonym or variant that may
# appear in a caption.
SYNONYMS = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
               "kid", "kids", "child", "children", "baby", "babies", "lady", "ladies", "guy", "guys",
               "gentleman", "gentlemen", "adult", "adults", "passenger", "passengers", "rider", "riders",
               "skier", "skiers", "snowboarder", "snowboarders", "surfer", "surfers", "player", "players",
               "athlete", "athletes", "biker", "bikers", "worker", "workers", "officer", "officers",
               "policeman", "chef", "cook", "tourist", "tourists", "spectator", "spectators",
               "pedestrian", "pedestrians", "crowd", "crowds", "couple", "family", "group"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "car": ["car", "cars", "vehicle", "vehicles", "automobile", "automobiles", "sedan", "suv", "taxi", "taxis"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "moped", "scooter", "scooters"],
    "airplane": ["airplane", "airplanes", "plane", "planes", "aircraft", "jet", "jets", "airliner"],
    "bus": ["bus", "buses"],
    "train": ["train", "trains", "locomotive", "subway"],
    "truck": ["truck", "trucks", "pickup", "lorry"],
    "boat": ["boat", "boats", "ship", "ships", "canoe", "kayak", "yacht", "sailboat", "rowboat", "raft"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights", "traffic signal", "traffic signals"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
    "parking meter": ["parking meter", "parking meters"],
    "bench": ["bench", "benches"],
    "bird": ["bird", "birds", "duck", "ducks", "goose", "geese", "seagull", "seagulls", "pigeon",
             "pigeons", "parrot", "parrots", "chicken", "chickens", "owl", "owls", "ostrich", "swan", "swans"],
    "cat": ["cat", "cats", "kitten", "kittens", "kitty"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "horse": ["horse", "horses", "pony", "ponies", "foal"],
    "sheep": ["sheep", "lamb", "lambs", "ram"],
    "cow": ["cow", "cows", "cattle", "bull", "bulls", "calf", "calves", "ox"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "backpacks", "knapsack"],
    "umbrella": ["umbrella", "umbrellas", "parasol"],
    "handbag": ["handbag", "handbags", "purse", "purses"],
    "tie": ["tie", "ties", "necktie", "neckties", "bowtie"],
    "suitcase": ["suitcase", "suitcases", "luggage", "briefcase"],
    "frisbee": ["frisbee", "frisbees"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "ball", "balls", "soccer ball", "tennis ball", "basketball",
                    "baseball", "football", "volleyball"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "baseball bats", "bat", "bats"],
    "baseball glove": ["baseball glove", "baseball gloves", "glove", "gloves", "mitt", "mitts"],
    "skateboard": ["skateboard", "skateboards"],
    "surfboard": ["surfboard", "surfboards"],
    "tennis racket": ["tennis racket", "tennis rackets", "racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottle", "bottles"],
    "wine glass": ["wine glass", "wine glasses", "wineglass", "wineglasses", "goblet", "goblets"],
    "cup": ["cup", "cups", "mug", "mugs"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches", "burger", "burgers", "hamburger", "hamburgers", "sub", "subs"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes", "cupcake", "cupcakes"],
    "chair": ["chair", "chairs", "stool", "stools", "armchair", "armchairs"],
    "couch": ["couch", "couches", "sofa", "sofas"],
    "potted plant": ["potted plant", "potted plants", "houseplant", "houseplants", "plant", "plants"],
    "bed": ["bed", "beds"],
    "dining table": ["dining table", "dining tables", "table", "tables", "desk", "desks"],
    "toilet": ["toilet", "toilets"],
    "tv": ["tv", "tvs", "television", "televisions", "monitor", "monitors", "screen", "screens"],
    "laptop": ["laptop", "laptops", "computer", "computers", "notebook computer"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remotes", "remote control", "remote controls"],
    "keyboard": ["keyboard", "keyboards"],
    "cell phone": ["cell phone", "cell phones", "cellphone", "cellphones", "phone", "phones",
                   "smartphone", "smartphones", "mobile phone", "iphone"],
    "microwave": ["microwave", "microwaves"],
    "oven": ["oven", "ovens", "stove", "stoves"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges"],
    "book": ["book", "books"],
    "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy bears", "stuffed animal", "stuffed animals", "stuffed bear", "plush"],
    "hair drier": ["hair drier", "hair driers", "hair dryer", "hair dryers", "blow dryer"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}

# reverse index: every synonym back to the COCO class it belongs to
SYNONYM_TO_CAT = {}
for cat, syns in SYNONYMS.items():
    for s in syns:
        SYNONYM_TO_CAT[s] = cat

# longest first, so "table" cannot steal a match from "dining table"
ALL_SYNONYM_TERMS = sorted(SYNONYM_TO_CAT.keys(), key=lambda x: -len(x))

# pre-compiled, each term matched on full word boundaries
SYNONYM_PATTERNS = [(re.compile(r'\b' + re.escape(term) + r'\b'), term) for term in ALL_SYNONYM_TERMS]

# A stricter synonym table, used only when expanding the ground truth. It drops the
# generic terms, hypernyms and collective nouns that would inflate the GT. Detection in
# the model's own captions still uses the permissive table -- counting "vehicle" as car
# is fair -- but when reference captions expand the GT, "family" or "group" must not
# quietly add person to it, or nothing the model says would ever count as a hallucination.
STRICT_BLOCKLIST = {
    "crowd", "crowds", "couple", "family", "group",     # collective nouns -> person
    "vehicle", "vehicles",                              # hypernym of car (could be a truck or bus)
    "table", "tables", "desk", "desks",                 # generic furniture -> dining table
    "monitor", "monitors", "screen", "screens",         # generic displays -> tv
    "plant", "plants",                                  # generic plants -> potted plant (usually outdoors)
}
STRICT_SYNONYM_PATTERNS = [(pat, term) for pat, term in SYNONYM_PATTERNS if term not in STRICT_BLOCKLIST]


def _extract_with_patterns(text: str, patterns) -> set:
    text = text.lower()
    found = set()
    masked = text
    for pat, term in patterns:
        if pat.search(masked):
            found.add(SYNONYM_TO_CAT[term])
            masked = pat.sub(" __MASK__ ", masked)
    return found


def extract_objects(text: str) -> set:
    """Extract every COCO class mentioned in a caption, using the permissive synonym
    table. This is what the model's own captions are checked with."""
    return _extract_with_patterns(text, SYNONYM_PATTERNS)


def extract_objects_strict(text: str) -> set:
    """Like extract_objects, but without the generic terms and collective nouns. Used
    only when expanding the ground truth, so it cannot be inflated."""
    return _extract_with_patterns(text, STRICT_SYNONYM_PATTERNS)


# ====== 3. load the GT from instances and captions: standard CHAIR + two diagnostics ======
# Standard CHAIR (Rohrbach et al., 2018) takes the GT to be the instance annotations
# UNION the objects the synonym table extracts from the reference captions. That is the
# convention behind every number in the VCD / OPERA / Woodpecker / HALC tables, and it is
# STANDARD below. INST_ONLY and STRICT are robustness diagnostics only -- appendix
# material, not the reported figure.
print("-> loading the COCO instance annotations...")
with open(INSTANCE_ANNO_FILE, 'r') as f:
    coco_inst = json.load(f)

cat_id_to_name = {cat['id']: cat['name'] for cat in coco_inst['categories']}

# GT from the instance segmentation alone: stricter than standard CHAIR, diagnostic only
gt_instance = defaultdict(set)
for ann in coco_inst['annotations']:
    gt_instance[ann['image_id']].add(cat_id_to_name[ann['category_id']])

# objects extracted from the five reference captions, in two variants: the standard one
# (Rohrbach 2018) and the strict one with generic terms removed, used for diagnostics
loose_expand = defaultdict(set)
strict_expand = defaultdict(set)
if os.path.exists(CAPTION_ANNO_FILE):
    print("-> loading the COCO reference captions (building the STANDARD and STRICT GT expansions)...")
    with open(CAPTION_ANNO_FILE, 'r') as f:
        coco_cap = json.load(f)
    for ann in coco_cap['annotations']:
        img_id = ann['image_id']
        loose_expand[img_id].update(extract_objects(ann['caption']))
        strict_expand[img_id].update(extract_objects_strict(ann['caption']))
else:
    print(f"⚠️  {CAPTION_ANNO_FILE} not found, using the instance annotations only")

# ============ 4. scoring: all three GT configurations in one pass ============
print("-> scoring the generated captions (STANDARD, INST_ONLY and STRICT together)...")

# STANDARD comes first: it is directly comparable to the VCD / OPERA / Rohrbach 2018
# tables and is the number to report.
CONFIG_NAMES = ["STANDARD", "INST_ONLY", "STRICT"]
# per configuration: [total_mentioned, total_hallucinated, total_sentences, hallucinated_sentences]
stats = {name: [0, 0, 0, 0] for name in CONFIG_NAMES}

# Caption-level statistics, independent of the GT configuration. These give the paper its
# "Avg #words" and "#obj/cap" columns, which answer the obvious reviewer objection that a
# low CHAIR score is just the product of short captions.
total_captions = 0       # includes captions that mention no COCO object at all
captions_no_mention = 0  # captions matching no COCO object (pure scene/attribute description)
total_words = 0
word_counts = []
mention_counts = []      # distinct COCO classes mentioned per caption

with open(GEN_FILE, 'r') as f:
    for line in f:
        data = json.loads(line)
        img_id = data['image_id']
        caption = data['caption']

        # length and mention statistics, always counted even when nothing is mentioned
        total_captions += 1
        n_words = len(caption.split())
        word_counts.append(n_words)
        total_words += n_words

        mentioned = extract_objects(caption)
        mention_counts.append(len(mentioned))
        if not mentioned:
            captions_no_mention += 1
            continue

        inst_gt = gt_instance.get(img_id, set())
        gt_by_config = {
            "STANDARD":  inst_gt | loose_expand.get(img_id, set()),  # the Rohrbach 2018 protocol
            "INST_ONLY": inst_gt,                                     # diagnostic: no caption expansion
            "STRICT":    inst_gt | strict_expand.get(img_id, set()),  # diagnostic: expansion without generic terms
        }

        for name, gt in gt_by_config.items():
            hallu = mentioned - gt
            s = stats[name]
            s[0] += len(mentioned)
            s[2] += 1
            if hallu:
                s[1] += len(hallu)
                s[3] += 1

# ================= 5. report =================
print("\n" + "🏆 " * 15)
print("     CHAIR evaluation report")
print("🏆 " * 15)
print("Legend:")
print("  - STANDARD  : the Rohrbach 2018 CHAIR protocol (instance UNION caption-derived) -- report this row")
print("                every VCD / OPERA / Woodpecker / HALC table uses this convention")
print("  - INST_ONLY : instance segmentation only (stricter; diagnostic, appendix material)")
print("  - STRICT    : caption expansion without family/group/desk/table/screen/vehicle (diagnostic only)")
print("-" * 70)
for name in CONFIG_NAMES:
    total_mentioned, total_hallucinated, total_sentences, hallucinated_sentences = stats[name]
    chair_i = (total_hallucinated / total_mentioned) * 100 if total_mentioned else 0
    chair_s = (hallucinated_sentences / total_sentences) * 100 if total_sentences else 0
    marker = "  ⭐ the row to report" if name == "STANDARD" else ""
    print(f"\n--- {name} ---{marker}")
    print(f"📄 samples: {total_sentences} images")
    print(f"🎯 CHAIR_I (instance level): {chair_i:.2f}%   🎯 CHAIR_S (sentence level): {chair_s:.2f}%")
    print(f"📊 mentioned: {total_mentioned} | hallucinated: {total_hallucinated} | hallu_sents: {hallucinated_sentences}")

# ============ caption-level descriptiveness (two columns the paper needs) ============
# The most common objection to a CHAIR result is that the captions are simply shorter.
# These two numbers settle it: as long as avg_words and obj_per_cap stay in the
# LLaVA-1.5 baseline range (roughly 100 words and 3-4 objects per caption), the drop in
# CHAIR cannot be explained by shorter captions.
print("\n" + "-" * 70)
print("📏 caption descriptiveness (the two columns the paper's table needs)")
print("-" * 70)
if total_captions > 0:
    avg_words = total_words / total_captions
    avg_obj = sum(mention_counts) / total_captions
    sorted_w = sorted(word_counts)
    med_w = sorted_w[len(sorted_w) // 2]
    print(f"  Total captions     : {total_captions} ({captions_no_mention} of them mention no COCO class)")
    print(f"  Avg #words / cap   : {avg_words:.1f}   (median: {med_w}, min: {min(word_counts)}, max: {max(word_counts)})")
    print(f"  Avg #obj  / cap    : {avg_obj:.2f}    (median: {sorted(mention_counts)[len(mention_counts)//2]}, max: {max(mention_counts)})")
    print()
    print("  📋 the row to put in the paper (STANDARD convention):")
    s = stats["STANDARD"]
    chair_i = (s[1] / s[0]) * 100 if s[0] else 0
    chair_s = (s[3] / s[2]) * 100 if s[2] else 0
    print(f"     | Method | #Words | #Obj/cap | CHAIR_S↓ | CHAIR_I↓ |")
    print(f"     | DSCC   | {avg_words:>6.1f} | {avg_obj:>8.2f} | {chair_s:>7.2f}% | {chair_i:>7.2f}% |")
print("\n" + "✨ " * 15 + "\n")
