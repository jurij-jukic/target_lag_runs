#!/bin/bash
# 16-run target-lag matrix. 12 steps. Steady-state metrics are steps 3-11.
# Run from /workspace/prime-rl with .venv on PATH and UV_NO_CACHE unset.
set -u
cd /workspace/prime-rl
export PATH=/workspace/prime-rl/.venv/bin:$PATH
unset UV_NO_CACHE
LOG=/workspace/matrix.log
OUT=/workspace/matrix_summaries
mkdir -p "$OUT"
: > "$LOG"

wait_gpu() {
  local i used
  for i in $(seq 1 20); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    if [ "$used" -lt 800 ]; then
      return 0
    fi
    sleep 3
  done
  echo "GPU still busy ($used MiB)" | tee -a "$LOG"
}

cleanup_procs() {
  python3 - << 'PY'
import os, signal
me = os.getpid()
ppid = os.getppid()
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in (me, ppid):
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode()
        parent = int(open(f"/proc/{pid}/stat").read().split()[3])
    except Exception:
        continue
    if parent in (me, ppid):
        continue
    if any(s in cmd for s in (
        "/.venv/bin/rl", "/.venv/bin/inference", "/.venv/bin/trainer",
        "/.venv/bin/orchestrator", "vllm", "torchrun", "EngineCore",
    )):
        os.kill(pid, signal.SIGKILL)
PY
  sleep 2
}

summarize() {
  local name="$1"
  local rc="$2"
  python3 - "$name" "$rc" << 'PY'
import json, sys
from pathlib import Path
name, rc = sys.argv[1], int(sys.argv[2])
root = Path("/workspace/prime-rl/outputs") / name
out = Path("/workspace/matrix_summaries") / f"{name}.json"
rows = []
metrics = root / "monitors" / "file" / "metrics.jsonl"
if metrics.exists():
    for line in metrics.read_text().splitlines():
        row = json.loads(line)
        if "time/wait_for_policy" not in row:
            continue
        def g(key, nd=None):
            value = row.get(key)
            if not isinstance(value, (int, float)):
                return None
            return round(value, nd) if nd is not None else value
        rows.append({
            "step": row.get("step"),
            "t": g("time/step", 2),
            "wait": g("time/wait_for_policy", 2),
            "mean": g("off_policy/mean", 3),
            "max": g("off_policy/max", 3),
            "drop": g("off_policy/dropped", 1),
            "inflight_max": g("off_policy/in_flight/max", 3),
            "trunc": g("train/reverse-text/effective/agent/is_truncated/mean", 3),
            "out_mean": g("train/reverse-text/effective/num_output_tokens/mean", 1),
            "out_p90": g("train/reverse-text/effective/num_output_tokens/p90", 1),
            "out_max": g("train/reverse-text/effective/num_output_tokens/max", 1),
            "admit": g("curriculum/reverse-text/admission_rate", 3),
            "cancelled": g("train/reverse-text/all/cancelled/mean", 3),
        })
trainer = []
tlog = root / "logs" / "attempt_1" / "trainer.log"
if tlog.exists():
    import re
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for line in ansi.sub("", tlog.read_text(errors="replace")).splitlines():
        if "Step " in line and "Loss" in line:
            trainer.append(line.split("Step ", 1)[-1][:80])
steady = [r for r in rows if isinstance(r.get("step"), int) and 3 <= r["step"] <= 11]
def avg(key):
    vals = [r[key] for r in steady if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None
sha = Path("/tmp/matrix_sha.txt").read_text().strip() if Path("/tmp/matrix_sha.txt").exists() else ""
cmd = ""
ct = root / "configs" / "attempt_1" / "command.txt"
if ct.exists():
    cmd = ct.read_text()
payload = {
    "name": name,
    "exit": rc,
    "git": sha,
    "command": cmd,
    "steps": rows,
    "trainer": trainer,
    "steady_3_11": {
        "n": len(steady),
        "t": avg("t"),
        "wait": avg("wait"),
        "mean": avg("mean"),
        "max": avg("max"),
        "drop": avg("drop"),
        "trunc": avg("trunc"),
        "out_mean": avg("out_mean"),
        "out_p90": avg("out_p90"),
        "admit": avg("admit"),
        "cancelled": avg("cancelled"),
    },
}
out.write_text(json.dumps(payload, indent=2))
print(name, "exit", rc, "steady", json.dumps(payload["steady_3_11"]))
PY
}

run() {
  local name="$1"
  shift
  wait_gpu
  echo "===== START $name $(date -u +%H:%M:%S) =====" | tee -a "$LOG"
  git rev-parse HEAD > /tmp/matrix_sha.txt
  rl @ configs/ci/integration/reverse-text/start.toml "$@" --max_steps 12 --clean --run.name "$name" >> "$LOG" 2>&1
  local rc=$?
  echo "===== END $name exit:$rc $(date -u +%H:%M:%S) =====" | tee -a "$LOG"
  summarize "$name" "$rc" | tee -a "$LOG"
  rm -rf "outputs/$name/broadcasts" "outputs/$name/checkpoints"
  cleanup_procs
}

checkout() {
  local ref="$1"
  git checkout --detach "$ref" >> "$LOG" 2>&1 || git checkout "$ref" >> "$LOG" 2>&1
  echo "CHECKOUT $(git rev-parse --short HEAD) $(git log -1 --format=%s)" | tee -a "$LOG"
}

BASE_FAST=(
  --ckpt None
  --orchestrator.batch_size 64
  --orchestrator.group_size 8
  --orchestrator.concurrency.max_inflight 64
  --orchestrator.train.sampling.max_completion_tokens 512
  --orchestrator.train.sampling.temperature 0.6
  --inference.vllm.max_model_len 2048
)
BASE_FAT=(
  --ckpt None
  --seq_len 8192
  --orchestrator.batch_size 16
  --orchestrator.group_size 16
  --orchestrator.concurrency.max_inflight 32
  --orchestrator.train.sampling.max_completion_tokens 4096
  --orchestrator.train.sampling.temperature 1.4
  --inference.vllm.max_model_len 8192
)
BASE_BURST=(
  --orchestrator.batch_size 64
  --orchestrator.group_size 8
  --orchestrator.concurrency.max_inflight 64
  --orchestrator.train.sampling.max_completion_tokens 512
  --orchestrator.train.sampling.temperature 0.6
  --inference.vllm.max_model_len 2048
  --ckpt.interval 3
  --ckpt.keep_last 1
)

one_branch() {
  local tag="$1"
  run "fast_${tag}_off1" "${BASE_FAST[@]}" --orchestrator.max_off_policy_steps 1
  run "fast_${tag}_off8" "${BASE_FAST[@]}" --orchestrator.max_off_policy_steps 8
  run "fat_${tag}_off1" "${BASE_FAT[@]}" --orchestrator.max_off_policy_steps 1
  run "fat_${tag}_off8" "${BASE_FAT[@]}" --orchestrator.max_off_policy_steps 8
  run "pass_${tag}_off1" "${BASE_FAST[@]}" @ configs/ci/integration/reverse-text/pass_overlay.toml --orchestrator.max_off_policy_steps 1
  run "pass_${tag}_off4" "${BASE_FAST[@]}" @ configs/ci/integration/reverse-text/pass_overlay.toml --orchestrator.max_off_policy_steps 4
  run "burst_${tag}_off1" "${BASE_BURST[@]}" --orchestrator.max_off_policy_steps 1
  run "burst_${tag}_off4" "${BASE_BURST[@]}" --orchestrator.max_off_policy_steps 4
}

checkout feat/no-target-lag
one_branch new
checkout origin/main
one_branch old
git checkout feat/no-target-lag >> "$LOG" 2>&1
echo "CHECKOUT $(git rev-parse --short HEAD) $(git branch --show-current)" | tee -a "$LOG"
echo ALL_MATRIX_DONE | tee -a "$LOG"
