#!/usr/bin/env python
"""
score_bench.py — does bf16 on the scoring calls (matching upstream's bf16-mixed)
speed up predict_step, and does it move the scores?

Design loop stays fp32 in BOTH arms and the seed is fixed, so the design being
scored is identical -> a paired comparison of the scoring step alone.
Run sequentially on one GPU so the timings aren't contended.

  python score_bench.py --gpu 0 --seed 42
"""
import argparse, os, re, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs

SOFT, TEMP, HARD = 30, 15, 2
ARMS = {"score_fp32": {}, "score_bf16": {"BOLTZDESIGN_AUTOCAST": "bf16_score"}}


def cmd(gpu, suffix):
    base = {
        "name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
        "length_min": "150", "length_max": "150",
        "boltz_model_version": "boltz2", "pre_iteration": "0",
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


def run(gpu, arm, seed):
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = str(seed)
    env.update(ARMS[arm])
    log = os.path.join(HERE, f"score_bench_{arm}.log")
    with open(log, "w") as lf:
        subprocess.run(cmd(gpu, f"scorebench_{arm}"), stdout=lf,
                       stderr=subprocess.STDOUT, cwd=HERE, env=env)
    return log


def parse(log):
    d = {"score_s": [], "holo": None, "apo": None, "rmsd": None, "iters": [], "seq": None}
    for ln in open(log):
        m = re.search(r"\[score\] predict_step took ([\d.]+)s", ln)
        if m: d["score_s"].append(float(m.group(1)))
        m = re.search(r"Time for iteration \d+:\s*([\d.]+)", ln)
        if m: d["iters"].append(float(m.group(1)))
        m = re.match(r"Holo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["holo"] = float(m.group(1))
        m = re.match(r"Apo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["apo"] = float(m.group(1))
        m = re.match(r"RMSD:\s*([\d.]+)", ln)
        if m: d["rmsd"] = float(m.group(1))
        m = re.match(r"Best sequence:\s*([A-Z]+)", ln)
        if m: d["seq"] = m.group(1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    P = {}
    for arm in ARMS:
        print(f"[{time.strftime('%T')}] running {arm} on gpu{a.gpu} ...", flush=True)
        P[arm] = parse(run(a.gpu, arm, a.seed))

    L = ["============ SCORING-PRECISION BENCHMARK (boltz2) ============",
         f"FAD 150aa | design loop fp32 in both arms | seed {a.seed} | gpu {a.gpu}", ""]
    for arm in ARMS:
        d = P[arm]
        tot = sum(d["score_s"])
        L.append(f"{arm:11} predict_step calls {len(d['score_s'])} | "
                 f"total {tot:7.2f}s | each {[round(x,2) for x in d['score_s']]}")
    f, b = P["score_fp32"], P["score_bf16"]
    tf, tb = sum(f["score_s"]), sum(b["score_s"])
    L += ["",
          f"speedup on scoring: {tf/tb:.2f}x  ({tf:.2f}s -> {tb:.2f}s, saved {tf-tb:.2f}s/design)",
          "",
          "did the scores move? (identical design, so this is scoring precision alone)",
          f"  holo complex pLDDT : {f['holo']} -> {b['holo']}",
          f"  apo  complex pLDDT : {f['apo']} -> {b['apo']}",
          f"  RMSD               : {f['rmsd']} -> {b['rmsd']}",
          f"  same design?         {'YES' if f['seq'] == b['seq'] else 'NO — designs differ, comparison invalid'}",
          "",
          f"design loop s/iter: fp32 arm {np.mean(f['iters'][2:]):.3f}  bf16-score arm "
          f"{np.mean(b['iters'][2:]):.3f}  (should match; loop is fp32 in both)",
          "=" * 62]
    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(HERE, "score_bench_report.txt"), "w") as fh:
        fh.write(rep + "\n")


if __name__ == "__main__":
    main()
