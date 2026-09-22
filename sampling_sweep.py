#!/usr/bin/env python
"""
sampling_sweep.py — with the confidence module ON (--distogram_only False), how few
diffusion sampling steps does the design loop need?

The sampler runs inside torch.no_grad(), so fewer steps cannot change the gradient --
only the coordinates the confidence head reads. So the question is purely whether
plddt/pae stay informative. Sequential on one GPU, same seed and init_seed.

  python sampling_sweep.py --gpu 1
"""
import os, re, subprocess, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs

STEPS = [200, 100, 50, 20]

def run(gpu, n):
    base = {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
            "length_min": "150", "length_max": "150", "boltz_model_version": "boltz2",
            "pre_iteration": "0", "distogram_only": "False",
            "soft_iteration": "8", "temp_iteration": "4", "hard_iteration": "1",
            "num_intra_contacts": "6", "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
            "semi_greedy_steps": "0", "init_seed": "42",
            "design_samples": "1", "num_designs": "1", "run_boltz_design": "True",
            "run_ligandmpnn": "False", "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": str(gpu), "suffix": f"sampsweep_{n}", "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = "42"
    env["BOLTZDESIGN_DESIGN_SAMPLING_STEPS"] = str(n)
    log = os.path.join(HERE, f"sampsweep_{n}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    t = open(log).read()
    it = [float(x) for x in re.findall(r"Time for iteration \d+:\s*([\d.]+)", t)]
    def grab(name):
        v = [float(x) for x in re.findall(rf"'{name}:(-?[\d.]+)'", t)]
        return v
    return {"wall": time.time() - t0, "iters": it,
            "s_iter": (np.mean(it[1:]) if len(it) > 2 else (np.mean(it) if it else float('nan'))),
            "plddt": grab("plddt_loss"), "pae": grab("pae_loss"), "ipae": grab("i_pae_loss"),
            "err": "Traceback" in t}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="1"); a = ap.parse_args()
    R = {}
    for n in STEPS:
        print(f"[{time.strftime('%T')}] sampling_steps={n} ...", flush=True)
        R[n] = run(a.gpu, n)
    L = ["===== DESIGN-LOOP DIFFUSION SAMPLING STEPS (confidence module ON) =====",
         "FAD 150aa | 13-iter schedule | distogram_only False | seed+init_seed 42", "",
         f"{'steps':>6} {'s/iter':>8} {'speedup':>8} {'plddt_loss (first 3)':>26} {'pae_loss':>22}"]
    base = R[200]["s_iter"]
    for n in STEPS:
        d = R[n]
        sp = base / d["s_iter"] if d["s_iter"] else float("nan")
        L.append(f"{n:6d} {d['s_iter']:8.2f} {sp:8.2f}x "
                 f"{str([round(x,3) for x in d['plddt'][:3]]):>26} "
                 f"{str([round(x,3) for x in d['pae'][:3]]):>22}"
                 + ("  ERROR" if d["err"] else ""))
    L += ["", "gradient is unaffected by this knob (sampler is inside no_grad); the question",
          "is only whether plddt/pae stay informative at fewer steps.", "=" * 74]
    rep = "\n".join(L); print(rep)
    open(os.path.join(HERE, "sampling_sweep_report.txt"), "w").write(rep + "\n")

if __name__ == "__main__":
    main()
