"""
Pull the cross-attention weight history of a run from wandb and check whether
xa0_q_std / xa0_o_std / xa0_o_abs_max really drift over the 100-step samples.

Defaults to the full run; pass a run name to inspect a quick validation instead:
    python check_v6_wandb_drift.py                            # full run (default)
    python check_v6_wandb_drift.py v6-dualstream-full_quick    # quick validation
    VERBOSE=1 python check_v6_wandb_drift.py                   # every sample point
                                                               # (summarised per stage otherwise)

The pass threshold follows the mode:
  - full run:          delta q_std >= 1e-4 counts as really training
  - quick validation:  delta q_std >= 1e-6 is enough to show the gradient flows
"""

import os
import sys

try:
    import wandb
except ImportError:
    print("❌ wandb is required: pip install wandb")
    sys.exit(1)

PROJECT = "DualStream-MLLM"
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "v6.1-dualstream-full"
IS_QUICK = RUN_NAME.endswith("_quick")
DRIFT_THRESHOLD = 1e-6 if IS_QUICK else 1e-4
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

api = wandb.Api()
runs = api.runs(PROJECT, filters={"display_name": RUN_NAME})
if not runs:
    # the display_name filter may not have matched; list the 10 most recent runs
    print(f"⚠️ no run with display_name == '{RUN_NAME}'; the 10 most recent runs are:\n")
    recent = list(api.runs(PROJECT, order="-created_at", per_page=10))
    for r in recent:
        print(f"  - {r.name}  (state={r.state}, created={r.created_at})")
    print(f"\nRe-run with one of those names as the argument.")
    sys.exit(1)

run = runs[0]
print(f"✓ Found run: {run.name} | state={run.state} | url={run.url}\n")

# fetch the metric history
keys = [
    "global_step", "gamma_t", "lr", "train_loss",
    "xa0_q_std", "xa0_k_std", "xa0_v_std", "xa0_o_std", "xa0_o_abs_max",
    "xa1_q_std", "xa1_o_std",
]
history = list(run.history(keys=keys, pandas=False, samples=10000))

# keep the rows that carry xa0_q_std (logged every 100 steps)
xa_rows = [h for h in history if h.get("xa0_q_std") is not None]
if not xa_rows:
    print("⚠️ no xa0_q_std in the wandb history; it may still be syncing")
    print("   wait a minute or two and run this again")
    sys.exit(1)

xa_rows.sort(key=lambda h: h.get("global_step") or 0)

mode_tag = "QUICK_VALIDATE" if IS_QUICK else "FULL"
print(f"=== {mode_tag} cross-attn weight trajectory ({len(xa_rows)} sample points) ===\n")

# split by stage using gamma_t, so this does not depend on total_steps
g0_rows = [h for h in xa_rows if (h.get("gamma_t") or 0) < 0.01]
g1_rows = [h for h in xa_rows if (h.get("gamma_t") or 0) > 0.99]
g2_rows = [h for h in xa_rows
           if 0.01 <= (h.get("gamma_t") or 0) <= 0.99]

def _print_row(h):
    step = h.get("global_step", -1)
    g = h.get("gamma_t", 0) or 0
    lr = h.get("lr", 0) or 0
    loss = h.get("train_loss", 0) or 0
    print(f"{step:>6} {g:>5.3f} {lr:>10.2e} {loss:>7.4f} "
          f"{h['xa0_q_std']:>12.7f} {h['xa0_o_std']:>12.7f} {h['xa0_o_abs_max']:>14.7f}")

def _sample_section(rows, name, k=5):
    """Show the first and last k rows of a section, eliding the middle beyond 2k."""
    if not rows:
        return
    print(f"--- {name} ({len(rows)} sample points) ---")
    if VERBOSE or len(rows) <= 2 * k:
        for h in rows:
            _print_row(h)
    else:
        for h in rows[:k]:
            _print_row(h)
        print(f"  ... ({len(rows) - 2 * k} sample points omitted; VERBOSE=1 shows them all) ...")
        for h in rows[-k:]:
            _print_row(h)
    print()

print(f"{'step':>6} {'g':>5} {'LR':>10} {'loss':>7} "
      f"{'xa0_q_std':>12} {'xa0_o_std':>12} {'xa0_o_abs_max':>14}")
print("-" * 84)
_sample_section(g0_rows, "stage 1 (g=0, cross-attn dormant)")
_sample_section(g2_rows, "stage 2 (g ramping)")
_sample_section(g1_rows, "stage 3 (g=1, cross-attn fully active)")

# the verdict: compare the end of stage 1 (g=0) with the end of stage 3 (g=1)
if g0_rows and g1_rows:
    g0_last_q = g0_rows[-1]["xa0_q_std"]
    g0_last_o = g0_rows[-1]["xa0_o_abs_max"]
    g1_last_q = g1_rows[-1]["xa0_q_std"]
    g1_last_o = g1_rows[-1]["xa0_o_abs_max"]
    drift_q = g1_last_q - g0_last_q
    drift_o = g1_last_o - g0_last_o
    print(f"comparison: end of stage 1 (g=0, step={g0_rows[-1].get('global_step')}) "
          f"-> end of stage 3 (g=1, step={g1_rows[-1].get('global_step')})")
    print(f"  xa0_q_std    : {g0_last_q:.7f} -> {g1_last_q:.7f}  (delta {drift_q:+.7f})")
    print(f"  xa0_o_abs_max: {g0_last_o:.7f} -> {g1_last_o:.7f}  (delta {drift_o:+.7f})")
    print(f"  threshold    : |delta| >= {DRIFT_THRESHOLD:.0e}  (mode={mode_tag})")
    print()
    if abs(drift_q) >= DRIFT_THRESHOLD or abs(drift_o) >= DRIFT_THRESHOLD:
        if IS_QUICK:
            print("✅ quick validation: the cross-attn weights do drift between stage 1 and 3")
            print("   A full run (24700 steps, high LR sustained longer) should reach delta q_std >= 1e-4")
            print("   ==> safe to launch `python train_full_v6.py` without QUICK_VALIDATE")
        else:
            print("✅ full run: AdamW really updated the cross-attn weights")
            print("   The setup is stable at the 24700-step scale, q_proj_trained=True")
            print("   ==> ready for the 4x3 evaluation matrix (C/A/B/D x POPE-{random,popular,adv})")
    else:
        if IS_QUICK:
            print("❌ quick validation: the cross-attn weights are exactly at init throughout")
            print("   Do not launch a full run; debug the gradient path first:")
            print("     1) print(p.grad.abs().max()) for the cross_anchor params in train_full_v6.py")
            print("     2) confirm the cross_anchor_attention.py on the training machine is current")
            print("        (no self.layer_norm, out_proj std=0.02)")
        else:
            print(f"❌ full run: {g1_rows[-1].get('global_step')} steps in, delta q is only {drift_q:+.1e}")
            print(f"   Short of the {DRIFT_THRESHOLD:.0e} threshold by ~{DRIFT_THRESHOLD/max(abs(drift_q),1e-12):.0f}x")
            print("   The cross-attention barely trained. Check:")
            print("     1) whether the AdamW state is really fp32 (p.data.float() after init_dual_stream_modules)")
            print("     2) whether weight decay is out-weighing the gradient (is xa0_o_abs_max going down?)")
            print("     3) whether the LR is too small (cosine decaying too fast in stage 3)")
else:
    print("⚠️ incomplete data: no g=0 or no g=1 snapshot. Check the wandb UI by hand:")
    print(f"     {run.url}")
