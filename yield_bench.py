#!/usr/bin/env python
"""
yield_bench.py — distributional precision comparison: N designs per arm, one idle GPU
each, default iteration schedule. Boltz-level metrics only (no AF3), so this is a
smoke test for catastrophic degradation, NOT the AF3 yield decision.

  fp32 : today's default
  tf32 : BOLTZDESIGN_MATMUL_PREC=high
  bf16 : BOLTZDESIGN_AUTOCAST=bf16

  python yield_bench.py --gpus 1,2,4 --num_designs 10 --seed 42
"""
import argparse, os, re, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs

MODEL = "boltz2"
ARMS = {
    "fp32": {},
    "tf32": {"BOLTZDESIGN_MATMUL_PREC": "high"},
    "bf16": {"BOLTZDESIGN_AUTOCAST": "bf16"},
}


def cmd(gpu, suffix, n):
    base = {
        "name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
        "length_min": "150", "length_max": "150",
        "boltz_model_version": MODEL, "pre_iteration": "0",
        "design_samples": str(n), "num_designs": "1",
        "run_boltz_design": "True", "run_ligandmpnn": "False",
        "run_alphafold": "False", "run_rosetta": "False",
        "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE,
    }
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items():
        c += [f"--{k}", v]
    return c


def run_arm(gpu, arm, seed, n, deadline):
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = str(seed)
    env.update(ARMS[arm])
    log = os.path.join(HERE, f"yield_bench_{MODEL}_{arm}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd(gpu, f"yieldbench_{arm}", n), stdout=lf,
                             stderr=subprocess.STDOUT, cwd=HERE, env=env)
        while p.poll() is None:
            if time.time() > deadline:
                p.terminate()
                try:
                    p.wait(60)
                except Exception:
                    p.kill()
                break
            time.sleep(5)
    return log, time.time() - t0


def parse(log):
    d = {"times": [], "holo": [], "apo": [], "rmsd": [], "seqs": []}
    for ln in open(log):
        m = re.search(r"Time for iteration \d+:\s*([\d.]+)", ln)
        if m: d["times"].append(float(m.group(1)))
        m = re.match(r"Holo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["holo"].append(float(m.group(1)))
        m = re.match(r"Apo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["apo"].append(float(m.group(1)))
        m = re.match(r"RMSD:\s*([\d.]+)", ln)
        if m: d["rmsd"].append(float(m.group(1)))
        m = re.match(r"Best sequence:\s*([A-Z]+)", ln)
        if m: d["seqs"].append(m.group(1))
    return d


def ms(v):
    if not v:
        return "n/a"
    a = np.array(v, dtype=float)
    return f"{a.mean():.3f}+-{a.std(ddof=1) if len(a) > 1 else 0:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,4")
    ap.add_argument("--num_designs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--minutes", type=float, default=55.0)
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpus.split(",")]
    deadline = time.time() + a.minutes * 60

    res = {}
    def worker(gpu, arm):
        res[arm] = run_arm(gpu, arm, a.seed, a.num_designs, deadline)
    ths = [threading.Thread(target=worker, args=(g, arm)) for g, arm in zip(gpus, ARMS)]
    print(f"[{time.strftime('%T')}] " + ", ".join(f"{arm}=gpu{g}" for g, arm in zip(gpus, ARMS))
          + f" | {a.num_designs} designs/arm | budget {a.minutes:.0f} min")
    [t.start() for t in ths]; [t.join() for t in ths]

    P = {arm: parse(res[arm][0]) for arm in ARMS}
    step = {}
    for arm in ARMS:
        t = P[arm]["times"]
        step[arm] = np.mean(t[2:]) if len(t) > 3 else float("nan")
    L = [f"============ PRECISION YIELD SMOKE TEST ({MODEL}) ============",
         f"FAD 150aa | default schedule | seed {a.seed} | Boltz metrics only (no AF3)",
         "",
         f"{'arm':6} {'designs':>8} {'s/iter':>8} {'holo pLDDT':>18} {'apo pLDDT':>18} {'RMSD':>16}"]
    for arm in ARMS:
        d = P[arm]
        L.append(f"{arm:6} {len(d['holo']):8d} {step[arm]:8.3f} {ms(d['holo']):>18} "
                 f"{ms(d['apo']):>18} {ms(d['rmsd']):>16}")
    L += ["", "per-design holo pLDDT:"]
    for arm in ARMS:
        L.append(f"  {arm:5} {[round(x,3) for x in P[arm]['holo']]}  wall {res[arm][1]:.0f}s")
    L.append("=" * 62)
    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(HERE, f"yield_bench_report_{MODEL}.txt"), "w") as fh:
        fh.write(rep + "\n")


if __name__ == "__main__":
    main()
