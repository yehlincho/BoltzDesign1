#!/usr/bin/env python
"""
samp125_bench.py — the real test: designs run at the FULL 125-iteration schedule with
confidence mode on, at 200 / 100 / 50 design-loop sampling steps, each scored at 200.
Does cutting the in-loop sampler cost final structure quality?

init_seed is deliberately NOT pinned: pinning makes every design in an arm identical.
So this is a small distributional comparison (n per arm), not a paired one.

  python samp125_bench.py --gpus 1,2,4 --designs 3 --minutes 75
"""
import argparse, os, re, subprocess, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
ARMS = {"steps200": "200", "steps100": "100", "steps50": "50"}

def cmd(gpu, arm, n):
    base = {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
            "length_min": "150", "length_max": "150", "boltz_model_version": "boltz2",
            "pre_iteration": "0", "distogram_only": "False",
            "soft_iteration": "75", "temp_iteration": "45", "hard_iteration": "5",
            "num_intra_contacts": "6", "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
            "semi_greedy_steps": "0", "design_samples": str(n), "num_designs": "1",
            "run_boltz_design": "True", "run_ligandmpnn": "False",
            "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": str(gpu), "suffix": f"s125_{arm}", "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    return c

def run(gpu, arm, n, deadline):
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = "42"
    env["BOLTZDESIGN_DESIGN_SAMPLING_STEPS"] = ARMS[arm]
    log = os.path.join(HERE, f"samp125_{arm}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd(gpu, arm, n), stdout=lf, stderr=subprocess.STDOUT,
                             cwd=HERE, env=env)
        while p.poll() is None:
            if time.time() > deadline:
                p.terminate()
                try: p.wait(120)
                except Exception: p.kill()
                break
            time.sleep(10)
    return log, time.time() - t0

def parse(log):
    t = open(log).read()
    it = [float(x) for x in re.findall(r"Time for iteration \d+:\s*([\d.]+)", t)]
    return {"iters": it,
            "s_iter": float(np.mean(it[2:])) if len(it) > 3 else float("nan"),
            "holo": [float(x) for x in re.findall(r"^Holo Complex PLDDT:\s*([\d.]+)", t, re.M)],
            "apo": [float(x) for x in re.findall(r"^Apo Complex PLDDT:\s*([\d.]+)", t, re.M)],
            "rmsd": [float(x) for x in re.findall(r"^RMSD:\s*([\d.]+)", t, re.M)],
            "n": len(re.findall(r"^Best sequence", t, re.M)), "err": "Traceback" in t}

def ms(v):
    if not v: return "n/a"
    a = np.array(v, float)
    return f"{a.mean():.3f}+-{a.std(ddof=1) if len(a) > 1 else 0:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,4"); ap.add_argument("--designs", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=75)
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpus.split(",")]
    deadline = time.time() + a.minutes * 60
    res = {}
    def w(gpu, arm): res[arm] = run(gpu, arm, a.designs, deadline)
    ths = [threading.Thread(target=w, args=(g, arm)) for g, arm in zip(gpus, ARMS)]
    print(f"[{time.strftime('%T')}] " + ", ".join(f"{k}=gpu{g}" for g, k in zip(gpus, ARMS))
          + f" | {a.designs} designs/arm | 125 iters | cap {a.minutes:.0f} min", flush=True)
    [t.start() for t in ths]; [t.join() for t in ths]
    P = {arm: parse(res[arm][0]) for arm in ARMS}
    L = ["==== IN-LOOP SAMPLING STEPS, FULL 125-ITER SCHEDULE (confidence mode) ====",
         "FAD 150aa | final scoring at 200 steps in every arm | n designs per arm", "",
         f"{'arm':10} {'n':>2} {'s/iter':>8} {'holo pLDDT':>18} {'apo pLDDT':>18} {'RMSD':>16}"]
    for arm in ARMS:
        d = P[arm]
        L.append(f"{arm:10} {d['n']:2d} {d['s_iter']:8.2f} {ms(d['holo']):>18} "
                 f"{ms(d['apo']):>18} {ms(d['rmsd']):>16}" + ("  ERR" if d["err"] else ""))
    L.append("")
    for arm in ARMS:
        L.append(f"  {arm}: holo={[round(x,3) for x in P[arm]['holo']]} "
                 f"rmsd={[round(x,2) for x in P[arm]['rmsd']]} wall={res[arm][1]:.0f}s")
    L.append("=" * 74)
    rep = "\n".join(L); print(rep)
    open(os.path.join(HERE, "samp125_report.txt"), "w").write(rep + "\n")

if __name__ == "__main__":
    main()
