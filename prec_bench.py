#!/usr/bin/env python
"""
prec_bench.py — three-way precision comparison for the design loop.

  fp32 : today's default (no autocast, matmul precision "highest")
  tf32 : BOLTZDESIGN_MATMUL_PREC=high     (fp32 tensors, TF32 matmul internals)
  bf16 : BOLTZDESIGN_AUTOCAST=bf16        (trunk forward in bf16, losses in fp32)

Same seed and same design in every arm, one idle GPU each, run in parallel.
Unlike freeze-weights/no-ckpt this is NOT expected to be bit-identical: the point
is how far the trajectory drifts and whether design quality survives.

  python prec_bench.py --gpus 1,2,4 --seed 42 --model boltz2
"""
import argparse, os, re, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs

SOFT, TEMP, HARD = 30, 15, 2
MODEL = "boltz2"
ARMS = {
    "fp32": {},
    "tf32": {"BOLTZDESIGN_MATMUL_PREC": "high"},
    "bf16": {"BOLTZDESIGN_AUTOCAST": "bf16"},
}


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


def run_arm(gpu, arm, seed):
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = str(seed)
    env.update(ARMS[arm])
    log = os.path.join(HERE, f"prec_bench_{MODEL}_{arm}.log")
    stop = threading.Event(); mem = {}
    t = threading.Thread(target=poll_mem, args=(gpu, stop, mem)); t.start()
    t0 = time.time()
    with open(log, "w") as lf:
        subprocess.run(cmd(gpu, f"precbench_{arm}"), stdout=lf, stderr=subprocess.STDOUT,
                       cwd=HERE, env=env)
    wall = time.time() - t0
    stop.set(); t.join()
    return log, wall, mem.get("peak", 0)


def parse(log):
    d = {"times": [], "losses": [], "plddt": None, "seq": None, "marker": []}
    for ln in open(log):
        m = re.search(r"Time for iteration \d+:\s*([\d.]+)", ln)
        if m: d["times"].append(float(m.group(1)))
        m = re.search(r"total_loss:\s*(-?[\d.]+)", ln)
        if m: d["losses"].append(float(m.group(1)))
        m = re.match(r"Holo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["plddt"] = float(m.group(1))
        m = re.match(r"Best sequence:\s*([A-Z]+)", ln)
        if m: d["seq"] = m.group(1)
        if "[autocast]" in ln or "TF32" in ln: d["marker"].append(ln.strip())
    return d


def ident(a, b):
    if not a or not b or len(a) != len(b):
        return float("nan")
    return 100.0 * sum(x == y for x, y in zip(a, b)) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="boltz2")
    a = ap.parse_args()
    global MODEL; MODEL = a.model
    gpus = [g.strip() for g in a.gpus.split(",")]
    assert len(gpus) >= len(ARMS), f"need {len(ARMS)} gpus"

    res = {}
    def worker(gpu, arm):
        res[arm] = run_arm(gpu, arm, a.seed)
    ths = [threading.Thread(target=worker, args=(g, arm)) for g, arm in zip(gpus, ARMS)]
    print(f"[{time.strftime('%T')}] {MODEL}: " +
          ", ".join(f"{arm}=gpu{g}" for g, arm in zip(gpus, ARMS)) + f", seed={a.seed}")
    [t.start() for t in ths]; [t.join() for t in ths]

    P = {arm: parse(res[arm][0]) for arm in ARMS}
    step = {}
    for arm in ARMS:
        t = P[arm]["times"]
        step[arm] = np.mean(t[2:]) if len(t) > 3 else (np.mean(t) if t else float("nan"))

    base = step["fp32"]
    L = [f"================ PRECISION BENCHMARK ({MODEL}) ================",
         f"seed {a.seed} | FAD 150aa | {SOFT}/{TEMP}/{HARD} iters | freeze-weights on (default)",
         "", f"{'arm':6} {'s/iter':>8} {'speedup':>8} {'peakMiB':>9} {'iters':>6} "
         f"{'holo pLDDT':>11} {'seq id vs fp32':>15} {'max|dloss|':>11}"]
    for arm in ARMS:
        _, wall, peak = res[arm]
        d = P[arm]
        n = min(len(d["losses"]), len(P["fp32"]["losses"]))
        dl = (np.abs(np.array(d["losses"][:n]) - np.array(P["fp32"]["losses"][:n])).max()
              if n else float("nan"))
        sp = base / step[arm] if step[arm] else float("nan")
        L.append(f"{arm:6} {step[arm]:8.3f} {sp:8.2f}x {peak:9d} {len(d['times']):6d} "
                 f"{(d['plddt'] if d['plddt'] is not None else float('nan')):11.3f} "
                 f"{ident(P['fp32']['seq'], d['seq']):15.1f} {dl:11.3e}")
    L.append("")
    for arm in ARMS:
        if P[arm]["marker"]:
            L.append(f"  [{arm}] " + " | ".join(P[arm]["marker"][:2]))
        L.append(f"  [{arm}] wall {res[arm][1]:.0f}s  seq {(P[arm]['seq'] or '(none)')[:60]}")
    L.append("=" * 62)
    report = "\n".join(L)
    print(report)
    with open(os.path.join(HERE, f"prec_bench_report_{MODEL}.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
