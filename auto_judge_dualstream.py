import argparse
import json
import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import re
import string
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

# 1. the judge endpoint
# The base URL can be overridden through the environment: OpenAI itself, or any
# OpenAI-compatible gateway, works here.
API_BASE_URL = os.environ.get("JUDGE_API_BASE_URL", "https://www.autodl.art/api/v1")
# 🔑 The key is read from the environment only and must never be written into the source
#    (a key committed here is leaked the moment it reaches GitHub).
#    Usage:  export AUTODL_API_KEY="your-key"     # Linux / macOS
#            $env:AUTODL_API_KEY="your-key"       # Windows PowerShell
API_KEY = os.environ.get("AUTODL_API_KEY")
if not API_KEY:
    sys.exit(
        "Error: the AUTODL_API_KEY environment variable is not set.\n"
        "  Linux/macOS:  export AUTODL_API_KEY=\"your-key\"\n"
        "  PowerShell :  $env:AUTODL_API_KEY=\"your-key\"\n"
        "(The judge API key is not distributed with this code; use your own.)"
    )
# The model name can be overridden through JUDGE_MODEL or the --judge_model flag.
# If the provider retires or renames a model the request fails with a "channel not
# available" style error; changing the name is enough, no code change needed.
JUDGE_MODEL_DEFAULT = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
# The Chat Completions endpoint, in the standard OpenAI format.
CHAT_COMPLETIONS_ENDPOINT = f"{API_BASE_URL}/chat/completions"

# Plain HTTP, equivalent to the provider's own curl example:
#   curl -X POST $API_BASE_URL/chat/completions \
#        -H "Authorization: Bearer $API_KEY" \
#        -H "Content-Type: application/json" \
#        --data-raw '{"model": "gpt-5.4-mini",
#                     "messages": [{"role": "user", "content": "..."}]}'
# This keeps the script independent of any SDK version.
# stream=False: the judge replies with a single word (Correct/Incorrect), so streaming
# buys nothing.
print(f"-> calling the Chat Completions API over plain HTTP ({CHAT_COMPLETIONS_ENDPOINT})")

TSV_PATH = f"{DSCC_ROOT}/data/mirage/eval_code/data/mirage.tsv"

# When a student answer runs long -- rare, but it happens when the model repeats itself --
# keep only the head and tail, so the judge's context is not blown out.
MAX_MODEL_OUTPUT_CHARS = 3000
HEAD_TAIL_KEEP = 1500

# 2. command-line arguments
parser = argparse.ArgumentParser(description="MIRAGE LLM-as-a-Judge scoring script")
parser.add_argument("--jsonl_path", type=str,
                    default=f"{DSCC_ROOT}/results/mirage/mirage_checkpoint-final_v5.jsonl",
                    help="path to the JSONL produced by eval_mirage.py")
parser.add_argument("--output_dir", type=str,
                    default=f"{DSCC_ROOT}/results/mirage",
                    help="directory the scoring results are written to")
parser.add_argument("--workers", type=int, default=8,
                    help="number of concurrent requests (mind the API rate limit)")
parser.add_argument("--max_retries", type=int, default=2,
                    help="retries per question after a failure")
parser.add_argument("--timeout", type=int, default=60,
                    help="per-request timeout, in seconds")
parser.add_argument("--resume", action="store_true", default=True,
                    help="read the partial file at start-up and skip questions already scored (on by default)")
parser.add_argument("--no_resume", dest="resume", action="store_false",
                    help="score everything again, ignoring the partial file")
parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL_DEFAULT,
                    help="name of the judge model as the provider exposes it. "
                         "JUDGE_MODEL sets the default; change this when the provider reports the model is unavailable")
args = parser.parse_args()

JUDGE_MODEL = args.judge_model
print(f"-> judge model: {JUDGE_MODEL}")

os.makedirs(args.output_dir, exist_ok=True)
# The report file is timestamped so a re-run never overwrites the previous scores, and
# carries the evaluated JSONL's basename so it is obvious which inference run it scores.
_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
_jsonl_tag = os.path.splitext(os.path.basename(args.jsonl_path))[0]
REPORT_FILE = os.path.join(args.output_dir, f"judge_report_{_jsonl_tag}_{_TS}.json")
# The partial file is NOT timestamped: every run over the same JSONL shares one, which is
# what makes resuming possible.
PARTIAL_FILE = os.path.join(args.output_dir, f"judge_partial_{_jsonl_tag}.jsonl")

# guards writes to the partial file from the worker threads
_partial_lock = threading.Lock()
# Print a full traceback for the first real exception and stay quiet afterwards: the
# console does not get flooded, and the diagnostic is not lost either.
_first_error_lock = threading.Lock()
_first_error_logged = {"v": False}

# reuse the TCP connection, saving a great many TLS handshakes under concurrency
_http = requests.Session()
_http.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
})


def _extract_text(data: dict) -> str:
    """Pull the final text out of a Chat Completions response.

    The standard shape is {"choices": [{"message": {"role": "assistant", "content": "..."}}, ...]}.
    A few models return content as a list of parts (for multimodal or tool-call replies),
    which is handled as a fallback.
    """
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


# 3. call the judge API over plain HTTP, with retries
def call_judge(prompt: str):
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    last_err = ""
    for attempt in range(args.max_retries + 1):
        try:
            resp = _http.post(CHAT_COMPLETIONS_ENDPOINT, json=payload, timeout=args.timeout)
            if resp.status_code != 200:
                # include the upstream body: its message field is usually very informative
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            else:
                data = resp.json()
                text = _extract_text(data).strip()
                if text:
                    return text, ""
                last_err = f"empty reply, raw response: {json.dumps(data, ensure_ascii=False)[:300]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            with _first_error_lock:
                if not _first_error_logged["v"]:
                    _first_error_logged["v"] = True
                    print(f"\n⚠️ first judge API failure, full traceback:", file=sys.stderr)
                    traceback.print_exc()
        if attempt < args.max_retries:
            time.sleep(1.5 * (attempt + 1))
    return "", last_err


def smoke_test():
    """Make one minimal call before starting, to confirm the API answers. If it does not,
    exit immediately rather than waste the run."""
    print(f"-> smoke test: one call to {JUDGE_MODEL} to confirm the endpoint responds...")
    text, err = call_judge("Reply with exactly one word: OK")
    if err:
        print(
            f"\n❌ the judge API is unreachable: {err}\n"
            f"   Things to check:\n"
            f"     1) is API_KEY still valid (balance and expiry in the provider's console)\n"
            f"     2) can this machine reach {CHAT_COMPLETIONS_ENDPOINT}\n"
            f"        try it with curl:\n"
            f'        curl -X POST {CHAT_COMPLETIONS_ENDPOINT} \\\n'
            f'             -H "Authorization: Bearer $API_KEY" \\\n'
            f'             -H "Content-Type: application/json" \\\n'
            f'             --data-raw \'{{"model":"{JUDGE_MODEL}","messages":[{{"role":"user","content":"hi"}}]}}\'\n'
            f"     3) is the model name {JUDGE_MODEL!r} still on the provider's list\n"
            f"        rename it: JUDGE_MODEL=<new name> python {os.path.basename(__file__)}\n"
            f"        or:        python {os.path.basename(__file__)} --judge_model <new name>\n"
        )
        sys.exit(1)
    print(f"✓ smoke test passed, reply was: {text[:80]!r}")


# 4. Verdict parsing. The judge may answer in English or Chinese, so both are matched;
# word boundaries keep "incorrect" from being read as "correct".
_RE_INCORRECT = re.compile(r'\bincorrect\b', re.IGNORECASE)
_RE_CORRECT = re.compile(r'\bcorrect\b', re.IGNORECASE)
# The Chinese patterns below are functional, not comments: they match a Chinese-language
# verdict and must stay as they are. Negatives are tested first, so a phrase meaning
# "not correct" is never matched by the pattern for "correct".
_RE_CN_NEG = re.compile(r'(不正确|不一致|错误|不对)')
_RE_CN_POS = re.compile(r'(正确|一致|对的)')


def parse_verdict(text: str):
    if _RE_INCORRECT.search(text) or _RE_CN_NEG.search(text):
        return False
    if _RE_CORRECT.search(text) or _RE_CN_POS.search(text):
        return True
    return None


def truncate_output(s: str) -> str:
    if len(s) <= MAX_MODEL_OUTPUT_CHARS:
        return s
    return s[:HEAD_TAIL_KEEP] + "\n... [truncated for judge context] ...\n" + s[-HEAD_TAIL_KEEP:]


# 5. Fuzzy question matching -- strip punctuation, lowercase, collapse whitespace -- so a
# trivial textual difference does not fall through to the fallback.
# The literal below lists CJK punctuation alongside string.punctuation; it is data the
# translation table is built from, not text, and must stay as it is.
_PUNCT_TABLE = str.maketrans('', '', string.punctuation + "，。！？、：；""''（）《》")
_WS_RE = re.compile(r'\s+')


def normalize_q(s: str) -> str:
    if not s:
        return ""
    s = s.lower().translate(_PUNCT_TABLE)
    return _WS_RE.sub(' ', s).strip()


def judge_one(item):
    student_output = truncate_output(item['model_output'])
    prompt = f"""You are a strict logic-reasoning grader. Compare the student's reasoning with the reference answer and decide whether the student's FINAL conclusion matches the reference.

Question: {item['question']}
Reference answer: {item['gt']}
Student's reasoning: {student_output}

Reply with exactly one word: Correct or Incorrect. No other text."""

    judge_text, err = call_judge(prompt)

    if err:
        verdict = None
        raw = f"[ERROR] {err}"
    else:
        verdict = parse_verdict(judge_text)
        raw = judge_text

    return {
        "image": item["image"],
        "question": item["question"],
        "gt": item["gt"],
        "model_output": item["model_output"],
        "judge_raw": raw,
        # tri-state: True = correct, False = incorrect, None = unparsable or API error
        "is_correct": verdict,
        "errored": err != "",
        "ambiguous": err == "" and verdict is None,
    }


def _record_key(image: str, question: str) -> str:
    """De-duplication key for resuming: the image plus the normalised question."""
    return f"{image}||{normalize_q(question)}"


# 6. pair up questions with reference answers; one image may carry several questions
print("-> pairing questions with reference answers...")
df = pd.read_csv(TSV_PATH, sep='\t')

gt_lookup = {}  # image -> [(question, answer), ...]
for _, row in df.iterrows():
    if pd.notna(row['image']):
        key = str(row['image'])
        q = str(row.get('prompt', row.get('question', '')))
        a = str(row['answer'])
        gt_lookup.setdefault(key, []).append((q, a))

todo = []
skipped_bad_lines = 0
with open(args.jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            skipped_bad_lines += 1
            continue
        img_key = data.get('image')
        if not img_key:
            skipped_bad_lines += 1
            continue
        candidates = gt_lookup.get(img_key, [])
        if not candidates:
            continue
        pred_q = (data.get('question') or '').strip()
        match = None
        if pred_q and len(candidates) > 1:
            pred_norm = normalize_q(pred_q)
            # 1) exact match
            for q, a in candidates:
                if q.strip() == pred_q:
                    match = (q, a)
                    break
            # 2) match after normalisation
            if match is None:
                for q, a in candidates:
                    if normalize_q(q) == pred_norm:
                        match = (q, a)
                        break
        if match is None:
            match = candidates[0]
        todo.append({
            "image": img_key,
            "question": match[0],
            "gt": match[1],
            "model_output": (data.get('model_output') or '').strip(),
        })

if skipped_bad_lines:
    print(f"⚠️ skipped {skipped_bad_lines} jsonl records that were unparsable or missing fields")

print(f"-> {len(todo)} questions to score, taken from {args.jsonl_path}")
if len(todo) == 0:
    print(
        f"\n❌ the todo list is empty. Likely causes:\n"
        f"   1) {args.jsonl_path} is empty or the path is wrong (the real jsonl basename may differ)\n"
        f"   2) none of the image fields in the jsonl appear in mirage.tsv's image column\n"
        f"   Compare the image fields in the jsonl and in mirage.tsv first.\n"
    )
    sys.exit(1)

# 7. Resume: read the partial file and drop the questions already scored successfully.
already_done = {}  # key -> previous record (dict)
if args.resume and os.path.exists(PARTIAL_FILE):
    with open(PARTIAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # only count the ones that scored successfully; errored ones are retried
            if rec.get("errored"):
                continue
            key = _record_key(rec.get("image", ""), rec.get("question", ""))
            already_done[key] = rec
    print(f"-> resuming: {len(already_done)} questions already scored in the partial file, skipping them")

todo_remaining = [it for it in todo if _record_key(it["image"], it["question"]) not in already_done]
print(f"-> {len(todo_remaining)} questions to score this round (of {len(todo)} total, "
      f"{len(already_done)} already done), asking {JUDGE_MODEL} with {args.workers} workers...")

if len(todo_remaining) == 0:
    print(
        f"\n⚠️ nothing left to score: the partial file already covers every question.\n"
        f"   To score everything again: python {os.path.basename(__file__)} --no_resume\n"
        f"   Or delete the partial file: rm {PARTIAL_FILE}\n"
    )
    # Deliberately not exiting: the code below re-aggregates the existing results into a
    # fresh report.

# smoke-test the API before starting, and bail out immediately if it is unreachable
if len(todo_remaining) > 0:
    smoke_test()

# 8. score concurrently, appending as results arrive
details = list(already_done.values())  # start from the results already on disk
# append to the partial file when resuming; truncate it under --no_resume
_open_mode = 'a' if (args.resume and os.path.exists(PARTIAL_FILE)) else 'w'
partial_fh = open(PARTIAL_FILE, _open_mode, encoding='utf-8')


def _append_partial(rec):
    with _partial_lock:
        partial_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        partial_fh.flush()


try:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge_one, item) for item in todo_remaining]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            details.append(rec)
            _append_partial(rec)
finally:
    partial_fh.close()

# 9. tri-state statistics
total = len(details)
correct = sum(1 for r in details if r["is_correct"] is True)
incorrect = sum(1 for r in details if r["is_correct"] is False and not r["errored"])
errors = sum(1 for r in details if r["errored"])
ambiguous = sum(1 for r in details if r["ambiguous"])
accuracy = correct / total if total > 0 else 0.0
# "Clean" accuracy excludes errored and ambiguous items, and is closer to the model's
# actual ability.
clean_total = correct + incorrect
clean_acc = (correct / clean_total) if clean_total > 0 else 0.0

# 10. print the scores and write the report
print("\n" + "=" * 40)
print("MIRAGE logical reasoning, final scores (LLM-as-a-Judge)")
print(f"valid questions:        {total}")
print(f"answered correctly:     {correct}")
print(f"answered incorrectly:   {incorrect}")
print(f"network/API errors:     {errors}")
print(f"ambiguous verdicts:     {ambiguous}")
print(f"Accuracy (errors and ambiguous count in the denominator): {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Clean Accuracy (correct + incorrect only): {clean_acc:.4f} ({clean_acc * 100:.2f}%)")
print("=" * 40 + "\n")

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    json.dump({
        "jsonl_path": args.jsonl_path,
        "judge_model": JUDGE_MODEL,
        "metrics": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "errors": errors,
            "ambiguous": ambiguous,
            "accuracy": accuracy,
            "clean_accuracy": clean_acc,
        },
        "details": details,
    }, f, ensure_ascii=False, indent=2)

print(f"detailed scores written to: {REPORT_FILE}")
print(f"partial file kept at: {PARTIAL_FILE} (a re-run over the same jsonl skips what is already scored)")
