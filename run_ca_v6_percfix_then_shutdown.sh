#!/bin/bash

DSCC_ROOT="${DSCC_ROOT:-/root/autodl-tmp}"
# ============================================================
# run_ca_v6_percfix_then_shutdown.sh
#
# The reduced re-run after the perception dtype bug was fixed: train only C (the full
# dual stream) and A (perception only), then shut the machine down.
#
#   Why B is not re-trained: the earlier "dead full" run is, in gradient terms, already
#     cog_only -- the cognition stream's cross_anchor was on while the perception
#     gradient was identically zero (the old dtype bug swallowed L_perc for the whole
#     run; the checkpoints show max|Full-D| = 0). It IS B, and it has been evaluated.
#
#   The two fixes this script checks for on the remote machine:
#     1) perception_contrast.py casts vis/text feats to the projection dtype on entry to
#        forward, breaking the "perception addmm dtype clash swallowed silently"
#        deadlock (marker: _proj_dtype).
#     2) llava_llama.py decouples "does the perception loss reach the backbone" from
#        disable_cross_anchor. Without it, A (perception only, g == 0) detaches for the
#        whole run, A's backbone becomes identical to D's, and the perception stream's
#        standalone contribution cannot be measured (marker: _perc_couple_backbone).
#
#   What gets trained:
#     C (full):         RUN_TAG=_percfix -> dualstream_v6_1_percfix (does not overwrite
#                       the old dualstream_v6_1, which is B)
#     A (perc_only):    dualstream_v6_1_ablA_perc_only
#
# Usage, on the training server:
#   chmod +x run_ca_v6_percfix_then_shutdown.sh
#   nohup bash run_ca_v6_percfix_then_shutdown.sh \
#         > ${DSCC_ROOT}/ca_percfix_runner.log 2>&1 &
#   disown
#
# Monitoring:
#   tail -f ${DSCC_ROOT}/ca_percfix_runner.log                 # overall progress
#   tail -f ${DSCC_ROOT}/ablation_logs/full_percfix_v6_*.log   # C in detail
#   tail -f ${DSCC_ROOT}/ablation_logs/ablA_perc_only_v6_*.log # A in detail
#
# To cancel the shutdown: ssh in within the 60-second window and run `shutdown -c`
# To stop early: `pkill -f run_ca_v6_percfix_then_shutdown.sh`, then
#                `pkill -f train_full_v6` / `pkill -f train_ablation_v6`
# ============================================================

set -u

WORK_DIR="${DSCC_ROOT}"
FULL_SCRIPT="${WORK_DIR}/train_full_v6.py"
ABL_SCRIPT="${WORK_DIR}/train_ablation_v6.py"
PERC_FILE="${WORK_DIR}/LLaVA/llava/model/dual_stream/perception_contrast.py"
LLAMA_FILE="${WORK_DIR}/LLaVA/llava/model/language_model/llava_llama.py"
LOG_DIR="${WORK_DIR}/ablation_logs"
SHUTDOWN_DELAY_SEC=60

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "re-training runner for C (full) + A (perc_only) started (PID=$$)"
log "  full script:      $FULL_SCRIPT (RUN_TAG=_percfix)"
log "  ablation script:  $ABL_SCRIPT (ABLATION_TYPE=perc_only)"
log "  log directory:    $LOG_DIR"
log "  shutdown delay:   ${SHUTDOWN_DELAY_SEC}s"
log "  expected runtime: ~14h (A800 80G, roughly 7h each)"
log "=========================================="

# ---------- pre-flight: the files exist ----------
for f in "$FULL_SCRIPT" "$ABL_SCRIPT" "$PERC_FILE" "$LLAMA_FILE"; do
    if [ ! -f "$f" ]; then
        log "[FATAL] file not found: $f -- sync the latest code to the server first"
        exit 1
    fi
done
if [ ! -f "${WORK_DIR}/data/v5_caption_map.json" ]; then
    log "[FATAL] caption map not found: ${WORK_DIR}/data/v5_caption_map.json"
    exit 1
fi
if ! command -v python >/dev/null 2>&1; then
    log "[FATAL] the python command is not available"
    exit 1
fi

# ★★★ guard 1: the perception dtype fix is present in the remote code ★★★
# Without this check, a forgotten sync means another 14h run with a dead perception stream.
if ! grep -q "_proj_dtype" "$PERC_FILE"; then
    log "[FATAL] no dtype fix in $PERC_FILE (marker: _proj_dtype)!"
    log "        The remote code is out of date and the perception stream would die silently again."
    log "        Sync the fixed perception_contrast.py to the server before running this."
    exit 1
fi
log "  ✓ guard 1 passed: perception_contrast.py carries the dtype fix (_proj_dtype)"

# ★★★ guard 2: the A-coupling fix is present in the remote code ★★★
# Without it, A trains into a copy of D: the perception gradient into the backbone stays
# detached for the whole run because g == 0.
if ! grep -q "_perc_couple_backbone" "$LLAMA_FILE"; then
    log "[FATAL] no A-coupling fix in $LLAMA_FILE (marker: _perc_couple_backbone)!"
    log "        The remote code is out of date; A (perc_only) would detach throughout,"
    log "        its backbone would equal D's, and the run would be wasted."
    log "        Sync the fixed llava_llama.py to the server before running this."
    exit 1
fi
log "  ✓ guard 2 passed: llava_llama.py carries the A-coupling fix (_perc_couple_backbone)"

# ★★★ guard 3: train_full_v6.py honours RUN_TAG (or it overwrites the old checkpoint) ★★★
if ! grep -q "RUN_TAG" "$FULL_SCRIPT"; then
    log "[FATAL] $FULL_SCRIPT does not support RUN_TAG -- this run would overwrite dualstream_v6_1 (= B)!"
    log "        Sync the train_full_v6.py that honours RUN_TAG before running this."
    exit 1
fi
log "  ✓ guard 3 passed: train_full_v6.py honours RUN_TAG (full run writes to dualstream_v6_1_percfix)"

# ★★★ guard 4: the B data point (the dead-full run) still exists, under either name ★★★
# B is the old main-experiment checkpoint. It may still be called dualstream_v6_1, or it
# may have been renamed to dualstream_v6_1_ablB* (dualstream_v6_1_ablB_cog_only_fromfull
# is the recommended name). Both are accepted.
# Note: the new full run writes to dualstream_v6_1_percfix, which matches neither pattern
# below, so there is no false positive.
_B_CKPT=""
for d in "${WORK_DIR}/checkpoints/dualstream_v6_1/checkpoint-final" \
         "${WORK_DIR}"/checkpoints/dualstream_v6_1_ablB*/checkpoint-final; do
    [ -d "$d" ] && _B_CKPT="$d" && break
done
if [ -n "$_B_CKPT" ]; then
    log "  ✓ guard 4 passed: the B (cog_only) data point is in place: $_B_CKPT"
else
    log "  [WARN] no B data point found (dualstream_v6_1 or dualstream_v6_1_ablB*) -- check it was not lost"
fi

cd "$WORK_DIR" || { log "[FATAL] cd $WORK_DIR failed"; exit 1; }

log "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1 | sed 's/^/  /' | head -5

T_GLOBAL_START=$(date +%s)

run_one() {
    # $1 = label, $2 = log name, $3.. = the training command
    local LABEL="$1"; shift
    local LOGNAME="$1"; shift
    local LOGFILE="${LOG_DIR}/${LOGNAME}_${TS}.log"
    log ""
    log ">>> starting ${LABEL} (expected ~7h)"
    local TST=$(date +%s)
    "$@" > "$LOGFILE" 2>&1
    local RET=$?
    local TEN=$(date +%s); local DUR=$(( TEN - TST ))
    log "<<< ${LABEL} finished | exit=$RET | took $(( DUR/3600 ))h $(( (DUR%3600)/60 ))m | log: $LOGFILE"
    if [ $RET -ne 0 ]; then
        log "    [WARN] ${LABEL} exited non-zero; tail of its log:"
        tail -30 "$LOGFILE" 2>&1 | sed 's/^/    | /'
        log "    [WARN] continuing anyway -- the runs are independent"
    fi
    return $RET
}

# ---------- [1/2] C: the full dual stream, after the perception fix ----------
run_one "[1/2] C full dual stream (RUN_TAG=_percfix)" "full_percfix_v6" \
    env RUN_TAG=_percfix python "$FULL_SCRIPT"
RET_FULL=$?

# ---------- [2/2] A: perc_only ----------
run_one "[2/2] A Ablation perc_only" "ablA_perc_only_v6" \
    env ABLATION_TYPE=perc_only python "$ABL_SCRIPT"
RET_A=$?

# ---------- summary ----------
T_GLOBAL_END=$(date +%s)
DUR_TOTAL=$(( T_GLOBAL_END - T_GLOBAL_START ))
log ""
log "=========================================="
log "re-training of C (full) + A (perc_only) finished"
log "  C (full): exit=$RET_FULL"
log "  A:        exit=$RET_A"
log "  total:    $(( DUR_TOTAL/3600 ))h $(( (DUR_TOTAL%3600)/60 ))m"
log "=========================================="

# ---------- checkpoint check ----------
CKPT_FULL="${WORK_DIR}/checkpoints/dualstream_v6_1_percfix/checkpoint-final"
CKPT_A="${WORK_DIR}/checkpoints/dualstream_v6_1_ablA_perc_only/checkpoint-final"
CKPT_D="${WORK_DIR}/checkpoints/dualstream_v6_1_ablD_none/checkpoint-final"
log "checkpoint check:"
for CKPT in "$CKPT_FULL" "$CKPT_A"; do
    NAME=$(basename "$(dirname "$CKPT")")
    if [ -d "$CKPT" ]; then
        log "  ✅ $NAME written"
    else
        log "  ❌ $NAME not found (check its log)"
    fi
done

# ---------- the point of the re-run: did the perception stream actually train? ----------
# D serves as the init ground truth. The deterministic probe (log_temp + bias) should now
# show max|. - D| >> 1e-7.
log ""
log "verifying the perception stream really trained -- the whole point of this re-run:"
if [ -d "$CKPT_FULL" ] && [ -d "$CKPT_D" ]; then
    log "  >>> checking C (full) perception weights:"
    python check_ablA_perception_weights.py --a_ckpt "$CKPT_FULL" --d_ckpt "$CKPT_D" 2>&1 \
        | sed 's/^/    /' | tail -25
fi
if [ -d "$CKPT_A" ] && [ -d "$CKPT_D" ]; then
    log "  >>> checking A (perc_only) perception weights:"
    python check_ablA_perception_weights.py --a_ckpt "$CKPT_A" --d_ckpt "$CKPT_D" 2>&1 \
        | sed 's/^/    /' | tail -25
fi
log "  Reading it: max|. - D| >> 1e-7 on the deterministic probe, with log_temp clearly"
log "  away from 2.659, means the perception stream really trained this time."
log "  If it is still ~0 the fix did not take effect: investigate before evaluating."

# ---------- shutdown countdown ----------
log ""
log "shutting down in ${SHUTDOWN_DELAY_SEC}s. To cancel, run: shutdown -c"
sleep "$SHUTDOWN_DELAY_SEC"
log "running /usr/bin/shutdown -h now ..."
/usr/bin/shutdown -h now || { shutdown -h now || poweroff || log "[FATAL] shutdown failed, power the machine off by hand"; }
