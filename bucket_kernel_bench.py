#!/usr/bin/env python
"""
bucket_kernel_bench.py — try BindCraft2's trick here: do repeating shapes let the
fused scoring kernels amortise their Triton JIT across designs?

  random_len : kernels on,  lengths random (current behaviour) -> recompile per design
  bucketed   : kernels on,  --length_bucket 32                 -> shapes repeat
  baseline   : kernels off, lengths random                     -> today's default

3 designs per arm, TRITON_CACHE_DIR set so compilation also persists across processes.
The signal is the per-design scoring time: design 1 pays the JIT, designs 2-3 should not
if the cache is being hit.

  python bucket_kernel_bench.py --gpu 1
"""
import os, re, subprocess, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs

CACHE = os.path.join(HERE, ".triton_cache_bench")
ARMS = {
    "baseline":   {"bucket": "0",  "env": {}},
    "random_len": {"bucket": "0",  "env": {"BOLTZDESIGN_SCORE_KERNELS": "1"}},
    "bucketed":   {"bucket": "32", "env": {"BOLTZDESIGN_SCORE_KERNELS": "1"}},
}

def cmd(gpu, suffix, bucket):
    base = {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
            "length_min": "130", "length_max": "180", "length_bucket": bucket,
            "boltz_model_version": "boltz2", "pre_iteration": "0",
            "soft_iteration": "10", "temp_iteration": "5", "hard_iteration": "2",
            "num_intra_contacts": "6", "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
            "design_samples": "3", "num_designs": "1",
            "run_boltz_design": "True", "run_ligandmpnn": "False",
            "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    return c

def run(gpu, arm):
    os.makedirs(CACHE, exist_ok=True)
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = "42"
    env["TRITON_CACHE_DIR"] = CACHE
    env.update(ARMS[arm]["env"])
    log = os.path.join(HERE, f"bucket_kernel_{arm}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        subprocess.run(cmd(gpu, f"bk_{arm}", ARMS[arm]["bucket"]), stdout=lf,
                       stderr=subprocess.STDOUT, cwd=HERE, env=env)
    t = open(log).read()
    return {"wall": time.time() - t0,
            "score_s": [float(x) for x in re.findall(r"\[score\] predict_step took ([\d.]+)s", t)],
            "lengths": re.findall(r"\[length-bucket\] (\d+) -> (\d+)", t),
            "designs": len(re.findall(r"^Best sequence", t, re.M)),
            "seqlens": [len(x) for x in re.findall(r"^Best sequence:\s*([A-Z]+)", t, re.M)]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="1"); a = ap.parse_args()
    P = {}
    for arm in ARMS:
        print(f"[{time.strftime('%T')}] {arm}...", flush=True); P[arm] = run(a.gpu, arm)
    L = ["===== LENGTH BUCKETING + KERNEL CACHE (BindCraft2's trick, tried here) =====",
         "FAD | lengths 130-180 | 3 designs/arm | 17-iter schedule | shared TRITON_CACHE_DIR", ""]
    for arm in ARMS:
        d = P[arm]
        L.append(f"{arm:11} designs={d['designs']} binder_lens={d['seqlens']} wall={d['wall']:.0f}s")
        L.append(f"{'':11} scoring calls (s): {[round(x,2) for x in d['score_s']]}")
        if d["lengths"]:
            L.append(f"{'':11} bucket snaps: {d['lengths']}")
    L += ["", "read: if bucketing works, 'bucketed' scoring times should drop after the",
          "first design, while 'random_len' keeps paying JIT on every new shape.", "=" * 76]
    rep = "\n".join(L); print(rep)
    open(os.path.join(HERE, "bucket_kernel_report.txt"), "w").write(rep + "\n")

if __name__ == "__main__":
    main()
