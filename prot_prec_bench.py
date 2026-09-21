#!/usr/bin/env python
"""
prot_prec_bench.py — precision comparison on a PROTEIN target (PDL1), the case the
FAD/small-molecule runs did NOT cover: the MSA module actually runs (1024 subsampled
rows per iteration instead of depth 1) and N is larger (target + binder).

  fp32 / tf32 (MATMUL_PREC=high) / bf16 (AUTOCAST=bf16), one idle GPU each.
MSA is pre-seeded from a cached run so nothing hits the MSA server.

  python prot_prec_bench.py --gpus 1,2,4 --designs 3 --minutes 50
"""
import argparse, os, re, shutil, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs
from calibrate import PDL1_SEQ

ARMS = {
    "fp32": {},
    "msa_nockpt": {"BOLTZDESIGN_NO_MSA_CKPT": "1"},
    "bf16_msa_nockpt": {"BOLTZDESIGN_AUTOCAST": "bf16", "BOLTZDESIGN_NO_MSA_CKPT": "1"},
}
MSA_SRC = os.path.join(HERE, "inputs/protein_PDL1_grid_b1_pi0_lr0.1_ni6_hx-0.3/MSA/PDL1_B_env/msa.npz")


def seed_msa(suffix):
    """pre-place the cached target MSA so the run doesn't call the MSA server"""
    main = os.path.join(HERE, f"inputs/protein_PDL1_{suffix}")
    for chain in ("A", "B"):
        d = os.path.join(main, "MSA", f"PDL1_{chain}_env")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "msa.npz")
        if not os.path.exists(dst):
            shutil.copy2(MSA_SRC, dst)
    return main


def cmd(gpu, suffix, n):
    base = {
        "name": "PDL1", "target_type": "protein", "pdb_target_ids": "A",
        "target_seq": PDL1_SEQ, "use_msa": "True", "msa_max_seqs": "4096",
        "length_min": "90", "length_max": "120",
        "boltz_model_version": "boltz2", "pre_iteration": "0",
        "soft_iteration": "30", "temp_iteration": "15", "hard_iteration": "2",
        "semi_greedy_steps": "0",
        "design_samples": str(n), "num_designs": "1",
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
        time.sleep(1.0)
    out["peak"] = peak


def run_arm(gpu, arm, n, deadline):
    suffix = f"precprot_{arm}"
    seed_msa(suffix)
    env = bs._env_with_conda()
    env["BOLTZDESIGN_SEED"] = "42"
    env.update(ARMS[arm])
    log = os.path.join(HERE, f"prot_prec_{arm}.log")
    stop = threading.Event(); mem = {}
    t = threading.Thread(target=poll_mem, args=(gpu, stop, mem)); t.start()
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd(gpu, suffix, n), stdout=lf, stderr=subprocess.STDOUT,
                             cwd=HERE, env=env)
        while p.poll() is None:
            if time.time() > deadline:
                p.terminate()
                try: p.wait(90)
                except Exception: p.kill()
                break
            time.sleep(5)
    wall = time.time() - t0
    stop.set(); t.join()
    return log, wall, mem.get("peak", 0)


def parse(log):
    d = {"iters": [], "holo": [], "apo": [], "rmsd": [], "iptm": [], "designs": 0, "err": None}
    for ln in open(log):
        m = re.search(r"Time for iteration \d+:\s*([\d.]+)", ln)
        if m: d["iters"].append(float(m.group(1)))
        m = re.match(r"Holo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["holo"].append(float(m.group(1)))
        m = re.match(r"Apo Complex PLDDT:\s*([\d.]+)", ln)
        if m: d["apo"].append(float(m.group(1)))
        m = re.match(r"RMSD:\s*([\d.]+)", ln)
        if m: d["rmsd"].append(float(m.group(1)))
        m = re.search(r"i_?ptm[\"']?[:=]\s*([\d.]+)", ln, re.I)
        if m: d["iptm"].append(float(m.group(1)))
        if ln.startswith("Best sequence"): d["designs"] += 1
        if "Traceback" in ln or "Error" in ln: d["err"] = d["err"] or ln.strip()[:120]
    return d


def ms(v):
    if not v: return "n/a"
    a = np.array(v, float)
    return f"{a.mean():.3f}+-{a.std(ddof=1) if len(a) > 1 else 0:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,4"); ap.add_argument("--designs", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=50)
    a = ap.parse_args()
    gpus = [g.strip() for g in a.gpus.split(",")]
    deadline = time.time() + a.minutes * 60
    res = {}
    def worker(gpu, arm): res[arm] = run_arm(gpu, arm, a.designs, deadline)
    ths = [threading.Thread(target=worker, args=(g, arm)) for g, arm in zip(gpus, ARMS)]
    print(f"[{time.strftime('%T')}] PDL1 protein target | " +
          ", ".join(f"{arm}=gpu{g}" for g, arm in zip(gpus, ARMS)) +
          f" | {a.designs} designs/arm | cap {a.minutes:.0f} min", flush=True)
    [t.start() for t in ths]; [t.join() for t in ths]

    P = {arm: parse(res[arm][0]) for arm in ARMS}
    L = ["========= PRECISION BENCHMARK, PROTEIN TARGET (PDL1/boltz2) =========",
         "binder 90-120 aa | MSA on (4096 max, 1024 subsampled) | 75/45/5 | seed 42", "",
         f"{'arm':6} {'designs':>8} {'s/iter':>8} {'speedup':>8} {'peakMiB':>9} "
         f"{'holo pLDDT':>18} {'RMSD':>16}"]
    base = None
    for arm in ARMS:
        d, (_, wall, peak) = P[arm], res[arm]
        st = np.mean(d["iters"][2:]) if len(d["iters"]) > 3 else float("nan")
        if arm == "fp32": base = st
        sp = base / st if st and base else float("nan")
        L.append(f"{arm:6} {d['designs']:8d} {st:8.3f} {sp:8.2f}x {peak:9d} "
                 f"{ms(d['holo']):>18} {ms(d['rmsd']):>16}")
    L.append("")
    for arm in ARMS:
        d = P[arm]
        L.append(f"  {arm}: iters={len(d['iters'])} wall={res[arm][1]:.0f}s "
                 f"holo={[round(x,3) for x in d['holo']]} rmsd={[round(x,2) for x in d['rmsd']]}"
                 + (f" ERR={d['err']}" if d["err"] else ""))
    L.append("=" * 68)
    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(HERE, "prot_prec_report.txt"), "w") as fh:
        fh.write(rep + "\n")


if __name__ == "__main__":
    main()
