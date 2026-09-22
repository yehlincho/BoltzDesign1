#!/usr/bin/env python
"""
determinism_test.py — is the design loop reproducible run-to-run at a fixed seed?

Runs the SAME configuration twice per precision, sequentially on one GPU, and compares
the total_loss trajectories. fp32 is the control: freeze_bench proved max|dloss| = 0
there, so fp32 must come back identical. If bf16 does not, we have traded away
bit-identical A/B testing.

  python determinism_test.py --gpu 2
"""
import os, re, subprocess, sys, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs

def cmd(gpu, suffix):
    base = {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
            "length_min": "150", "length_max": "150", "boltz_model_version": "boltz2",
            "pre_iteration": "0", "learning_rate": "0.1", "num_intra_contacts": "6",
            "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
            "soft_iteration": "12", "temp_iteration": "6", "hard_iteration": "2",
            "design_samples": "1", "num_designs": "1",
            "run_boltz_design": "True", "run_ligandmpnn": "False",
            "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items():
        c += [f"--{k}", v]
    return c

def run(gpu, prec, rep):
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = "42"
    env["BOLTZDESIGN_AUTOCAST"] = prec
    log = os.path.join(HERE, f"determinism_{prec}_{rep}.log")
    with open(log, "w") as lf:
        subprocess.run(cmd(gpu, f"det_{prec}_{rep}"), stdout=lf,
                       stderr=subprocess.STDOUT, cwd=HERE, env=env)
    return [float(x) for x in re.findall(r"^total_loss:\s*(-?[\d.]+)", open(log).read(), re.M)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="2")
    a = ap.parse_args()
    out = ["============ DETERMINISM TEST (same seed, same GPU, 2 runs each) ============"]
    for prec in ("fp32", "bf16"):
        print(f"[{time.strftime('%T')}] {prec} run 1...", flush=True); r1 = run(a.gpu, prec, 1)
        print(f"[{time.strftime('%T')}] {prec} run 2...", flush=True); r2 = run(a.gpu, prec, 2)
        n = min(len(r1), len(r2))
        if n == 0:
            out.append(f"{prec}: no losses parsed"); continue
        d = np.abs(np.array(r1[:n]) - np.array(r2[:n]))
        out += [f"{prec}: {n} steps compared | max|dloss| = {d.max():.3e} | step0 delta = {d[0]:.3e}",
                f"       run1 first4: {[round(x,6) for x in r1[:4]]}",
                f"       run2 first4: {[round(x,6) for x in r2[:4]]}",
                f"       -> {'REPRODUCIBLE' if d.max() < 1e-6 else 'NOT reproducible'}"]
    out.append("=" * 76)
    rep = "\n".join(out); print(rep)
    open(os.path.join(HERE, "determinism_report.txt"), "w").write(rep + "\n")

if __name__ == "__main__":
    main()
