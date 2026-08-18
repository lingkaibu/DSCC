"""
Pre-training self-check: confirm the code Python actually imports on the training
machine contains both critical fixes.

Usage:
    cd $DSCC_ROOT && python verify_fixes.py

Checks:
  1. where llava_llama.py is really loaded from, and whether it carries the
     ablation-A coupling fix (_perc_couple_backbone)
  2. where perception_contrast.py is really loaded from, and whether it carries
     the dtype fix (_proj_dtype)

Both PASS => ablation A will not train into a copy of D, and the perception
stream will not silently die again.
"""

import os

DSCC_ROOT = os.environ.get("DSCC_ROOT", "/root/autodl-tmp")  # root for data / weights / results; override with the DSCC_ROOT env var
import inspect
import sys

LLAVA_PATH = f"{DSCC_ROOT}/LLaVA"
sys.path.append(LLAVA_PATH)


def check(label, module_import, needle, why):
    print(f"\n[{label}]")
    try:
        mod = module_import()
    except Exception as e:
        print(f"  ❌ import failed: {type(e).__name__}: {e}")
        return False
    path = getattr(mod, "__file__", "<unknown>")
    print(f"  loaded from: {path}")
    in_expected = path.startswith(LLAVA_PATH)
    if not in_expected:
        print(f"  ⚠️ not under {LLAVA_PATH}! the import may resolve elsewhere (a pip-installed llava?)")
    try:
        has = needle in inspect.getsource(mod)
    except Exception as e:
        print(f"  ❌ could not read the source: {e}")
        return False
    mark = "✅" if has else "❌"
    print(f"  {mark} contains '{needle}': {has}   ({why})")
    return has and in_expected


def _imp_llama():
    import llava.model.language_model.llava_llama as m
    return m


def _imp_perc():
    import llava.model.dual_stream.perception_contrast as p
    return p


def main():
    print("=" * 70)
    print("  pre-training self-check: ablation-A coupling fix + perception dtype fix")
    print("=" * 70)

    ok_llama = check(
        "1/2 llava_llama.py",
        _imp_llama,
        "_perc_couple_backbone",
        "in A (perception only) the perception gradient reaching the backbone must be "
        "decoupled from disable_cross_anchor, otherwise A is identical to D",
    )
    ok_perc = check(
        "2/2 perception_contrast.py",
        _imp_perc,
        "_proj_dtype",
        "the perception inputs must be up-cast to fp32, otherwise the addmm dtype "
        "clash is swallowed silently and the perception stream is dead all run",
    )

    print("\n" + "=" * 70)
    if ok_llama and ok_perc:
        print("  ✅✅ both fixes are in place and imported from the expected path")
        sys.exit(0)
    else:
        print("  🚨 a fix is missing. Do not evaluate; sync these files to the training machine first:")
        if not ok_llama:
            print("     - LLaVA/llava/model/language_model/llava_llama.py")
        if not ok_perc:
            print("     - LLaVA/llava/model/dual_stream/perception_contrast.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
