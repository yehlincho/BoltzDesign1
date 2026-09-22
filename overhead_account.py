import os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
base = {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
        "length_min": "150", "length_max": "150", "boltz_model_version": "boltz2",
        "pre_iteration": "0", "soft_iteration": "20", "temp_iteration": "8", "hard_iteration": "2",
        "num_intra_contacts": "6", "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
        "semi_greedy_steps": "0", "init_seed": "42",
        "design_samples": "2", "num_designs": "1", "run_boltz_design": "True",
        "run_ligandmpnn": "False", "run_alphafold": "False", "run_rosetta": "False",
        "gpu_id": "2", "suffix": "ovhd", "work_dir": HERE}
c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
for k, v in base.items(): c += [f"--{k}", v]
env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"; env["BOLTZDESIGN_TIME_STAGES"] = "1"
t0 = time.time()
with open(os.path.join(HERE, "ovhd.log"), "w") as lf:
    subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
wall = time.time() - t0
t = open(os.path.join(HERE, "ovhd.log")).read()
it = [float(x) for x in re.findall(r"Time for iteration \d+:\s*([\d.]+)", t)]
sc = [float(x) for x in re.findall(r"\[score\] predict_step took ([\d.]+)s", t)]
st = {}
for lab, v in re.findall(r"\[stage\] +([^:]+): ([\d.]+)s", t):
    st.setdefault(lab.strip(), []).append(float(v))
n = len(re.findall(r"^Best sequence", t, re.M))
LOAD = 33.3
prep = sum(sum(v) for k, v in st.items() if not k.startswith("get_batch."))
acct = LOAD + sum(it) + sum(sc) + prep
print(f"[acct] designs completed      : {n}")
print(f"[acct] wall                   : {wall:7.1f}s")
print(f"[acct] model load (measured)  : {LOAD:7.1f}s")
print(f"[acct] iterations ({len(it):3d})      : {sum(it):7.1f}s")
print(f"[acct] scoring ({len(sc)} calls)     : {sum(sc):7.1f}s")
print(f"[acct] prep stages            : {prep:7.1f}s   {({k: [round(x,2) for x in v] for k, v in st.items()})}")
print(f"[acct] UNEXPLAINED            : {wall - acct:7.1f}s  -> {(wall-acct)/max(n,1):6.1f}s per design")
