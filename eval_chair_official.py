"""
The official CHAIR evaluator, ported from the LisaAnne/Hallucination repository
(with Maxlinn's revisions).

Differences from the simpler eval_chair.py in this repository:
  - NLTK WordNet lemmatisation (cars -> car, running -> run), so matching is accurate
  - the ground-truth object set is COCO segments UNION COCO captions -- more permissive
    than bboxes alone, and the setting VCD, OPERA and other papers report against
  - double-word handling, so "traffic light", "stop sign", "hot dog" are never split
  - "toilet seat" no longer matches chair; "baby bird" maps to bird; "passenger jet"
    maps to jet
  - the evaluator object is pickled, so the second run returns in seconds

Usage (finds the most recent CHAIR-500 output automatically):
  python eval_chair_official.py

With an explicit caption file:
  python eval_chair_official.py --cap_file $DSCC_ROOT/results/captions/xxx.jsonl

With an explicit COCO annotation directory (defaults to $DSCC_ROOT/data/coco/annotations):
  python eval_chair_official.py --coco_path /path/to/coco/annotations
"""

import argparse
import glob
import json
import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import pickle
import sys
from collections import defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import tqdm


# ============= NLTK resources (downloaded on the first run) =============
def ensure_nltk_resources():
    for pkg, resource in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"-> NLTK resource {pkg} is missing, downloading...")
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"   could not download {pkg}: {e} (may not matter)")


# ============= the synonym table from the CHAIR paper =============
SYNONYMS_TXT = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""


# ====== load the COCO annotations (the train split is optional; val alone works) ======
def _safe_load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def combine_coco_captions(annotation_path):
    """Merge captions_val2014 and captions_train2014; the train split is optional."""
    val = _safe_load_json(os.path.join(annotation_path, 'captions_val2014.json'))
    train = _safe_load_json(os.path.join(annotation_path, 'captions_train2014.json'))
    if val is None:
        raise FileNotFoundError(
            f"❌ captions_val2014.json is required (expected at {annotation_path}/captions_val2014.json).\n"
            f"   Download: wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        )
    if train is None:
        print(f"⚠️  captions_train2014.json not found, using val only (fine for CHAIR-500, which runs on val2014)")
        return {'annotations': val['annotations'], 'images': val['images']}
    return {
        'annotations': val['annotations'] + train['annotations'],
        'images': val['images'] + train['images'],
    }


def combine_coco_instances(annotation_path):
    """Merge instances_val2014 and instances_train2014; the train split is optional."""
    val = _safe_load_json(os.path.join(annotation_path, 'instances_val2014.json'))
    train = _safe_load_json(os.path.join(annotation_path, 'instances_train2014.json'))
    if val is None:
        raise FileNotFoundError(
            f"❌ instances_val2014.json is required (expected at {annotation_path}/instances_val2014.json)."
        )
    if train is None:
        print(f"⚠️  instances_train2014.json not found, using val only (fine for CHAIR-500, which runs on val2014)")
        return {
            'categories': val['categories'],
            'annotations': val['annotations'],
            'images': val['images'],
        }
    return {
        'categories': val['categories'],
        'annotations': val['annotations'] + train['annotations'],
        'images': val['images'] + train['images'],
    }


class CHAIR(object):
    def __init__(self, coco_path):
        self.imid_to_objects = defaultdict(list)
        self.coco_path = coco_path

        synonyms = [s.strip().split(', ') for s in SYNONYMS_TXT.splitlines() if s.strip()]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
            'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
            'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
            'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear',
            'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven',
            'hot dog', 'teddy bear', 'home plate', 'train track',
        ]
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                        'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {dw: dw for dw in coco_double_words}
        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        self._build_gt_objects()

    # ----- build the ground-truth object sets -----
    def _build_gt_objects(self):
        print("-> collecting GT objects from the segments...")
        seg = combine_coco_instances(self.coco_path)
        id_to_name = {c['id']: c['name'] for c in seg['categories']}
        for ann in seg['annotations']:
            node = self.inverse_synonym_dict[id_to_name[ann['category_id']]]
            self.imid_to_objects[ann['image_id']].append(node)

        print("-> extending them from the GT captions (lemmatised, synonyms normalised)...")
        cap = combine_coco_captions(self.coco_path)
        for ann in tqdm.tqdm(cap['annotations'], desc="parse GT caps"):
            _, node_words, _, _ = self.caption_to_words(ann['caption'])
            self.imid_to_objects[ann['image_id']].extend(node_words)

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])
        print(f"   done: GT sets built for {len(self.imid_to_objects)} images.")

    # ----- sentence -> MSCOCO object nodes -----
    @staticmethod
    def _wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('N'):
            return wordnet.NOUN
        if tag.startswith('R'):
            return wordnet.ADV
        return None

    def caption_to_words(self, caption):
        words = nltk.word_tokenize(caption.lower())
        tagged = nltk.pos_tag(words)
        wnl = WordNetLemmatizer()
        lemmas = [wnl.lemmatize(t[0], pos=(self._wordnet_pos(t[1]) or wordnet.NOUN))
                  for t in tagged]
        words = lemmas

        # merge the double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            dw = ' '.join(words[i:i + 2])
            if dw in self.double_word_dict:
                double_words.append(self.double_word_dict[dw])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        if ('toilet' in words) and ('seat' in words):
            words = [w for w in words if w != 'seat']

        idxs = [idxs[idx] for idx, w in enumerate(words) if w in set(self.mscoco_objects)]
        words = [w for w in words if w in set(self.mscoco_objects)]
        node_words = [self.inverse_synonym_dict[w] for w in words]
        return words, node_words, idxs, double_words

    # ----- the evaluation itself -----
    def compute_chair(self, cap_file, image_id_key='image_id', caption_key='caption'):
        caps, eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(caps) == len(eval_imids)

        num_caps = 0
        num_hallucinated_caps = 0
        hallucinated_word_count = 0
        coco_word_count = 0
        len_caps = 0
        num_recall_gt_objects = 0
        num_gt_objects = 0
        output = {'sentences': []}

        for i in tqdm.trange(len(caps), desc="eval CHAIR"):
            cap = caps[i]
            imid = eval_imids[i]
            words, node_words, idxs, raw_words = self.caption_to_words(cap)
            gt_objects = self.imid_to_objects.get(imid, set())

            cap_dict = {
                'image_id': imid,
                'caption': cap,
                'mscoco_hallucinated_words': [],
                'mscoco_gt_words': list(gt_objects),
                'mscoco_generated_words': list(node_words),
                'hallucination_idxs': [],
                'words': raw_words,
                'metrics': {'CHAIRs': 0, 'CHAIRi': 0, 'Recall': 0, 'Len': 0},
            }

            coco_word_count += len(node_words)
            hallucinated = False
            recall_gt_objects = set()

            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)

            num_caps += 1
            len_caps += len(raw_words)
            if hallucinated:
                num_hallucinated_caps += 1
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words']) / float(len(words))
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)
            output['sentences'].append(cap_dict)

        chair_s = num_hallucinated_caps / num_caps
        chair_i = hallucinated_word_count / max(coco_word_count, 1)
        recall = num_recall_gt_objects / max(num_gt_objects, 1)
        # Note: the official code multiplies by 0.01 here and by 100 again when printing,
        # so print_metrics can apply the same "* 100" to every metric. The convention is
        # kept as-is so the numbers match the published ones exactly.
        avg_len = 0.01 * len_caps / num_caps
        obj_per_cap = 0.01 * coco_word_count / num_caps

        output['overall_metrics'] = {
            'CHAIRs': chair_s,        # sentence-level hallucination rate (as a percentage)
            'CHAIRi': chair_i,        # instance-level hallucination rate (as a percentage)
            'Recall': recall,          # fraction of GT objects mentioned (as a percentage)
            'Len': avg_len,            # mean word count (real value)
            'Obj/Cap': obj_per_cap,    # mean COCO objects mentioned per caption (real value)
        }
        return output


def load_generated_captions(cap_file, image_id_key='image_id', caption_key='caption'):
    ext = os.path.splitext(cap_file)[-1].lower()
    if ext == '.json':
        caps = json.load(open(cap_file))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f"Unsupported ext: {ext}")
    imids = [obj[image_id_key] for obj in caps]
    texts = [obj[caption_key] for obj in caps]
    return texts, imids


def print_metrics(result):
    print("\n" + "=" * 60)
    print("  Official CHAIR Metrics (LisaAnne / Rohrbach 2018)")
    print("=" * 60)
    for k, v in result['overall_metrics'].items():
        print(f"  {k:<10}: {v * 100:.2f}")
    print("=" * 60)
    print("\n📋 the row to put in the paper's table (standard reporting convention):")
    m = result['overall_metrics']
    print(f"  | Method | #Words | #Obj/cap | Recall↑ | CHAIR_S↓ | CHAIR_I↓ |")
    print(f"  | DSCC   | {m['Len']*100:>6.1f} | {m['Obj/Cap']*100:>8.2f} | "
          f"{m['Recall']*100:>6.2f}% | {m['CHAIRs']*100:>7.2f}% | {m['CHAIRi']*100:>7.2f}% |")


def find_latest_caption_file(captions_dir):
    """Find the most recent chair500 output."""
    patterns = [
        os.path.join(captions_dir, "coco_generated_captions_chair500_*.jsonl"),
        os.path.join(captions_dir, "coco_generated_captions_*.jsonl"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default="",
                        help="path to the caption jsonl/json; empty means find the newest chair500 file")
    parser.add_argument("--coco_path", type=str,
                        default=f"{DSCC_ROOT}/data/coco/annotations",
                        help="COCO annotation directory (captions_val2014.json / instances_val2014.json)")
    parser.add_argument("--captions_dir", type=str,
                        default=f"{DSCC_ROOT}/results/captions",
                        help="caption directory searched for the newest jsonl")
    parser.add_argument("--cache", type=str,
                        default=f"{DSCC_ROOT}/cache/chair_official.pkl",
                        help="where the pickled evaluator is cached")
    parser.add_argument("--save_path", type=str, default="",
                        help="write the detailed json, including each sample's hallucinated words")
    parser.add_argument("--image_id_key", type=str, default="image_id")
    parser.add_argument("--caption_key", type=str, default="caption")
    args = parser.parse_args()

    # ---- locate the caption file ----
    if not args.cap_file:
        args.cap_file = find_latest_caption_file(args.captions_dir)
        if not args.cap_file:
            raise SystemExit(f"❌ no chair500 jsonl found in {args.captions_dir}; pass --cap_file explicitly")
        print(f"-> using the newest caption file: {args.cap_file}")

    # ---- NLTK resources ----
    ensure_nltk_resources()

    # ---- load or build the evaluator ----
    if args.cache and os.path.exists(args.cache):
        evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"-> evaluator loaded from cache: {args.cache}")
    else:
        print(f"-> no cache, building the evaluator from scratch (~2-3 minutes the first time)...")
        evaluator = CHAIR(args.coco_path)
        if args.cache:
            os.makedirs(os.path.dirname(args.cache), exist_ok=True)
            pickle.dump(evaluator, open(args.cache, 'wb'))
            print(f"-> evaluator cached to: {args.cache}")

    # ---- evaluate ----
    result = evaluator.compute_chair(args.cap_file, args.image_id_key, args.caption_key)
    print_metrics(result)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ detailed results saved: {args.save_path}")
