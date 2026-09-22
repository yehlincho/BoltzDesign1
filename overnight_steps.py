#!/usr/bin/env python
"""
overnight_steps.py — does cutting the design-loop confidence sampling steps (200 -> 50)
preserve the pipeline's AF3 success rate?

4 targets x 2 arms, confidence mode ON, full 125-iteration schedule, LigandMPNN and AF3
validation ON. The FINAL prediction always uses 200 steps; only the in-loop sampler changes.

  targets : FAD, SAM (small molecule) | PDL1, BHRF1 (protein, cached MSAs)
  arms    : BOLTZDESIGN_DESIGN_SAMPLING_STEPS = 200 (control) and 50

Jobs are queued across GPU slots; each job is an independent process so one failure
cannot take down the night. Writes overnight_steps_report.txt at the end, and after
every job so partial results survive.
"""
import argparse, glob, os, re, shutil, subprocess, sys, threading, time, queue
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
from calibrate import PDL1_SEQ, BHRF1_SEQ

AF3_PY = os.path.expanduser("~/ProteinHunter/.conda/envs/af3/bin/python")
MSA_CACHE = {
    "PDL1": "inputs/protein_PDL1_grid_b1_pi0_lr0.1_ni6_hx-0.3/MSA/PDL1_B_env/msa.npz",
    "BHRF1": "inputs/protein_BHRF1_grid_b1_pi30_lr0.1_ni2_hx-0.3/MSA/BHRF1_B_env/msa.npz",
}
TARGETS = {
    "FAD":   {"target_type": "small_molecule", "target_seq": "FAD",
              "length_min": "130", "length_max": "180"},
    "SAM":   {"target_type": "small_molecule", "target_seq": "SAM",
              "length_min": "130", "length_max": "180"},
    "PDL1":  {"target_type": "protein", "pdb_target_ids": "A", "target_seq": PDL1_SEQ,
              "use_msa": "True", "msa_max_seqs": "4096",
              "length_min": "90", "length_max": "120"},
    "BHRF1": {"target_type": "protein", "pdb_target_ids": "A", "target_seq": BHRF1_SEQ,
              "use_msa": "True", "msa_max_seqs": "4096",
              "length_min": "90", "length_max": "120"},
}
STEPS = ["200", "50"]

def seed_msa(target, suffix):
    src = MSA_CACHE.get(target)
    if not src or not os.path.exists(os.path.join(HERE, src)):
        return
    main = os.path.join(HERE, f"inputs/{TARGETS[target]['target_type']}_{target}_{suffix}")
    for ch in ("A", "B"):
        d = os.path.join(main, "MSA", f"{target}_{ch}_env")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "msa.npz")
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(HERE, src), dst)

def cmd(target, steps, gpu, n, suffix):
    base = {"name": target, "boltz_model_version": "boltz2", "pre_iteration": "0",
            "distogram_only": "False",
            "soft_iteration": "75", "temp_iteration": "45", "hard_iteration": "5",
            "semi_greedy_steps": "0", "design_samples": str(n), "num_designs": "2",
            "run_boltz_design": "True", "run_ligandmpnn": "True",
            "run_alphafold": "True", "run_rosetta": "False",
            "af3_env_python": AF3_PY,
            "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE}
    base.update(TARGETS[target])
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    return c

def outdir(target, suffix):
    return os.path.join(HERE, "outputs", f"{TARGETS[target]['target_type']}_{target}_{suffix}")

def tally(target, suffix):
    d = outdir(target, suffix)
    designs = len(glob.glob(os.path.join(d, "results_yaml", "*.yaml")))
    succ = glob.glob(os.path.join(d, "**", "03_af_pdb_success", "high_iptm_confidence_scores.csv"),
                     recursive=True)
    n_succ = 0
    for f in succ:
        try:
            n_succ += max(0, sum(1 for _ in open(f)) - 1)
        except Exception:
            pass
    af3 = glob.glob(os.path.join(d, "**", "af3_validation_results.csv"), recursive=True)
    n_af3 = 0
    for f in af3:
        try:
            n_af3 += max(0, sum(1 for _ in open(f)) - 1)
        except Exception:
            pass
    return designs, n_af3, n_succ

def report(state, path):
    L = ["==== OVERNIGHT: in-loop confidence sampling steps vs AF3 success ====",
         "confidence mode ON | 125 iters | final prediction always 200 steps",
         "LigandMPNN + AF3 validation ON", "",
         f"{'target':7} {'steps':>5} {'designs':>8} {'af3_rows':>9} {'af3_success':>12} {'rate':>7} {'status':>10}"]
    for (t, s), v in sorted(state.items()):
        d, a, ok = tally(t, v["suffix"])
        rate = f"{100.0*ok/d:.0f}%" if d else "-"
        L.append(f"{t:7} {s:>5} {d:8d} {a:9d} {ok:12d} {rate:>7} {v['status']:>10}")
    L += ["", "designs = result yamls written; af3_success = rows in "
          "03_af_pdb_success/high_iptm_confidence_scores.csv", "=" * 70]
    open(path, "w").write("\n".join(L) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,4")
    ap.add_argument("--designs", type=int, default=5)
    ap.add_argument("--hours", type=float, default=9.0)
    a = ap.parse_args()
    deadline = time.time() + a.hours * 3600
    rpt = os.path.join(HERE, "overnight_steps_report.txt")
    jobs = queue.Queue()
    state = {}
    # Order matters for an unattended run: each target's two arms are queued adjacently
    # so a target either gets both arms (comparable) or none, never a control-less arm.
    # Proteins first -- they were the explicitly requested targets.
    for t in ["PDL1", "BHRF1", "FAD", "SAM"]:
        for s in STEPS:
            suffix = f"ovn_s{s}"
            state[(t, s)] = {"suffix": suffix, "status": "queued"}
            jobs.put((t, s, suffix))
    report(state, rpt)

    def worker(gpu):
        while time.time() < deadline:
            try:
                t, s, suffix = jobs.get_nowait()
            except queue.Empty:
                return
            seed_msa(t, suffix)
            state[(t, s)]["status"] = f"gpu{gpu}"
            report(state, rpt)
            log = os.path.join(HERE, f"overnight_{t}_s{s}.log")
            env = bs._env_with_conda()
            env["BOLTZDESIGN_SEED"] = "42"
            env["BOLTZDESIGN_DESIGN_SAMPLING_STEPS"] = s
            env["AF3_ENV_PYTHON"] = AF3_PY
            print(f"[{time.strftime('%T')}] start {t} steps={s} on gpu{gpu}", flush=True)
            with open(log, "w") as lf:
                p = subprocess.Popen(cmd(t, s, gpu, a.designs, suffix), stdout=lf,
                                     stderr=subprocess.STDOUT, cwd=HERE, env=env)
                while p.poll() is None:
                    if time.time() > deadline:
                        p.terminate()
                        try: p.wait(180)
                        except Exception: p.kill()
                        state[(t, s)]["status"] = "timeout"
                        break
                    time.sleep(15)
            if state[(t, s)]["status"] != "timeout":
                state[(t, s)]["status"] = "done" if p.returncode == 0 else f"rc{p.returncode}"
            print(f"[{time.strftime('%T')}] end   {t} steps={s} -> {state[(t,s)]['status']}", flush=True)
            report(state, rpt)

    ths = [threading.Thread(target=worker, args=(g.strip(),)) for g in a.gpus.split(",")]
    print(f"[{time.strftime('%T')}] 8 jobs, {a.designs} designs each, gpus {a.gpus}, "
          f"cap {a.hours}h", flush=True)
    [t.start() for t in ths]; [t.join() for t in ths]
    report(state, rpt)
    print(open(rpt).read())

if __name__ == "__main__":
    main()
