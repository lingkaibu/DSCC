#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write the missing ablation_config.json files and verify the flag matrix of the
four ablation data points.

Background: at start-up the evaluation scripts (eval_pope.py, generate_captions.py)
read checkpoint-final/ablation_config.json and inject disable_perception_loss /
disable_cross_anchor from it.
- C (percfix):  trained by train_full_v6.py, which writes no config -> add "full"
- B (fromfull): renamed from the old dead-Full run, also without a config -> add "cog_only"
A and D already have configs written by train_ablation_v6.py and are left alone.

Usage:  python setup_ablation_configs.py
"""
import json
import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var

CKPT_ROOT = f"{DSCC_ROOT}/checkpoints"

# the two that need writing (A and D already have theirs)
TO_WRITE = {
    "dualstream_v6_1_percfix": {
        "ablation_type": "full",
        "tag": "C_full_percfix",
        "disable_perception_loss": False,
        "disable_cross_anchor": False,
        "desc": "C, the full dual stream (retrained after the perception-dtype and A-coupling fixes)",
        "version": "v6.1",
    },
    "dualstream_v6_1_ablB_cog_only_fromfull": {
        "ablation_type": "cog_only",
        "tag": "B_cog_only_fromfull",
        "disable_perception_loss": True,
        "disable_cross_anchor": False,
        "desc": "B, cognition stream only (from the dead-Full run: its perception gradient was always 0, i.e. cog_only)",
        "version": "v6.1",
    },
}

# the expected 2x2 matrix, as (perc_off, cross_off)
EXPECTED = {
    "dualstream_v6_1_percfix":                (False, False),  # C, both streams on
    "dualstream_v6_1_ablA_perc_only":         (False, True),   # A, perception only
    "dualstream_v6_1_ablB_cog_only_fromfull": (True,  False),  # B, cognition only
    "dualstream_v6_1_ablD_none":              (True,  True),   # D, both streams off
}


def cfg_path(name):
    return os.path.join(CKPT_ROOT, name, "checkpoint-final", "ablation_config.json")


def main():
    print("==== 1) writing the missing configs ====")
    for name, cfg in TO_WRITE.items():
        p = cfg_path(name)
        ck = os.path.dirname(p)
        if not os.path.isdir(ck):
            print(f"  [skip] {name}: {ck} not found")
            continue
        if os.path.exists(p):
            print(f"  [exists] {name}: leaving {p} alone")
            continue
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"  [write] {name} -> perc_off={cfg['disable_perception_loss']} "
              f"cross_off={cfg['disable_cross_anchor']}")

    print("\n==== 2) verifying the flag matrix of the four data points ====")
    all_ok = True
    for name, (exp_perc, exp_cross) in EXPECTED.items():
        p = cfg_path(name)
        if not os.path.exists(p):
            print(f"  [missing] {name}: no {p}")
            all_ok = False
            continue
        c = json.load(open(p, encoding="utf-8"))
        perc = c["disable_perception_loss"]
        cross = c["disable_cross_anchor"]
        ok = (perc == exp_perc and cross == exp_cross)
        all_ok = all_ok and ok
        mark = "OK " if ok else "!! "
        print(f"  {mark}{name:42s} perc_off={perc!s:5s} cross_off={cross!s:5s} "
              f"tag={c.get('tag')}  (expected {exp_perc},{exp_cross})")

    print("\n==== result ====")
    print("  all aligned, ready to evaluate." if all_ok else "  mismatches found: investigate before evaluating!")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
