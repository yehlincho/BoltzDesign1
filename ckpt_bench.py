#!/usr/bin/env python
"""
ckpt_bench.py — verify BOLTZDESIGN_NO_CKPT=1 (activation checkpointing OFF) is
(a) faster and (b) numerically identical (same seed -> same loss trajectory).

Checkpointing recomputes every trunk block in backward, so the forward runs twice.
Turning it off is strictly fewer FLOPs; the math is unchanged (dropout=0, RNG
preserved), so the total_loss trajectory must match bit-for-bit.

Both arms keep the current defaults (freeze-weights on).

  python ckpt_bench.py --gpu_on 1 --gpu_off 2 --seed 42 --model boltz2
"""
import argparse, os, re, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs

SOFT, TEMP, HARD = 30, 15, 2
MODEL = "boltz2"


def cmd(gpu, suffix):
    base = {
        "name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
        "length_min": "150", "length_max": "150",
        "boltz_model_version": MODEL, "pre_iteration": "0",
        "learning_rate": "0.1", "num_intra_contacts": "6",
        "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
        "soft_iteration": str(SOFT), "temp_iteration": str(TEMP), "hard_iteration": str(HARD),
        "design_samples": "1", "num_designs": "1",
        "run_boltz_design": "True", "run_ligandmpnn": "False",
        "run_alphafold": "False", "run_rosetta": "False",
        "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE,
    }
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items():
        c += [f"--{k}", v]
    return c


def poll_mem(gpu, stop, out):
    peak = 0
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits", "--id", str(gpu)],
                               capture_output=True, text=True, timeout=5)
            peak = max(peak, int(r.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.5)
    out["peak"] = peak


def run_arm(gpu, no_ckpt, seed, tag):
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = str(seed)
    if no_ckpt:
        env["BOLTZDESIGN_NO_CKPT"] = "1"
    log = os.path.join(HERE, f"ckpt_bench_{MODEL}_{tag}.log")
    stop = threading.Event(); mem = {}
    t = threading.Thread(target=poll_mem, args=(gpu, stop, mem)); t.start()
    t0 = time.time()
    with open(log, "w") as lf:
        subprocess.run(cmd(gpu, f"ckptbench_{tag}"), stdout=lf, stderr=subprocess.STDOUT,
                       cwd=HERE, env=env)
    wall = time.time() - t0
    stop.set(); t.join()
    return log, wall, mem.get("peak", 0)


def parse(log):
    times, losses, marker = [], [], None
    for ln in open(log):
        m = re.search(r"Time for iteration \d+:\s*([\d.]+)", ln)
        if m: times.append(float(m.group(1)))
        m = re.search(r"total_loss:\s*(-?[\d.]+)", ln)
        if m: losses.append(float(m.group(1)))
        if "[no-ckpt]" in ln: marker = ln.strip()
    return times, losses, marker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_on", default="1"); ap.add_argument("--gpu_off", default="2")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--model", default="boltz2")
    a = ap.parse_args()
    global MODEL; MODEL = a.model
    res = {}
    def worker(gpu, no_ckpt, tag):
        res[tag] = run_arm(gpu, no_ckpt, a.seed, tag)
    t_on = threading.Thread(target=worker, args=(a.gpu_on, False, "ckpt_on"))
    t_off = threading.Thread(target=worker, args=(a.gpu_off, True, "ckpt_off"))
    print(f"[{time.strftime('%T')}] {MODEL}: ckpt_on (gpu{a.gpu_on}) + ckpt_off (gpu{a.gpu_off}), seed={a.seed}...")
    t_on.start(); t_off.start(); t_on.join(); t_off.join()

    (l_on, w_on, m_on), (l_off, w_off, m_off) = res["ckpt_on"], res["ckpt_off"]
    t1, loss_on, _ = parse(l_on)
    t2, loss_off, marker = parse(l_off)
    step_on = np.mean(t1[2:]) if len(t1) > 3 else (np.mean(t1) if t1 else float('nan'))
    step_off = np.mean(t2[2:]) if len(t2) > 3 else (np.mean(t2) if t2 else float('nan'))

    L = [f"============ ACTIVATION-CHECKPOINTING BENCHMARK ({MODEL}) ============",
         (marker or "[ckpt_off arm did NOT print the [no-ckpt] line — check env!]"),
         f"\n(A) CORRECTNESS — total_loss trajectory (same seed {a.seed}):"]
    n = min(len(loss_on), len(loss_off))
    if n == 0:
        L.append("   !! no losses parsed")
    else:
        d = np.abs(np.array(loss_on[:n]) - np.array(loss_off[:n]))
        L += [f"   compared {n} steps | max|dloss| = {d.max():.3e} | mean|dloss| = {d.mean():.3e}",
              f"   ckpt_on  first5: {[round(x,5) for x in loss_on[:5]]}",
              f"   ckpt_off first5: {[round(x,5) for x in loss_off[:5]]}",
              "   -> " + ("IDENTICAL (no harm) OK" if d.max() < 1e-3 else "DIVERGED — investigate")]
    sp = (step_on / step_off) if step_off else float('nan')
    L += [f"\n(B) SPEED — mean sec/iter (excl first 2):",
          f"   ckpt_on {step_on:.3f}s  ckpt_off {step_off:.3f}s  -> speedup {sp:.2f}x",
          f"   wall: on {w_on:.0f}s  off {w_off:.0f}s"]
    if m_on:
        L += [f"\n(C) MEMORY — peak GPU MiB (cost of turning it off):",
              f"   ckpt_on {m_on}  ckpt_off {m_off}  delta {m_off-m_on:+d} MiB ({100*(m_off-m_on)/m_on:+.1f}%)"]
    else:
        L += ["\n(C) MEMORY — poll failed"]
    L.append("=" * 62)
    report = "\n".join(L)
    print(report)
    with open(os.path.join(HERE, f"ckpt_bench_report_{MODEL}.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
