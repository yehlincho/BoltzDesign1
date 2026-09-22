import os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
base = {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
        "length_min": "150", "length_max": "150", "boltz_model_version": "boltz2",
        "pre_iteration": "0", "soft_iteration": "6", "temp_iteration": "3", "hard_iteration": "1",
        "num_intra_contacts": "6", "helix_loss_min": "-0.3", "helix_loss_max": "-0.3",
        "semi_greedy_steps": "0", "init_seed": "42",
        "design_samples": "2", "num_designs": "1", "run_boltz_design": "True",
        "run_ligandmpnn": "False", "run_alphafold": "False", "run_rosetta": "False",
        "gpu_id": "4", "suffix": "posttime", "work_dir": HERE}
c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
for k, v in base.items(): c += [f"--{k}", v]
env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"; env["BOLTZDESIGN_TIME_STAGES"] = "1"
t0 = time.time()
with open(os.path.join(HERE, "posttime.log"), "w") as lf:
    subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
t = open(os.path.join(HERE, "posttime.log")).read()
print("[post] wall %.1fs" % (time.time() - t0))
for lab, v in re.findall(r"\[stage\] +([^:]+): ([\d.]+)s", t):
    if not lab.strip().startswith("get_batch."):
        print(f"[post] {lab.strip():28} {float(v):7.2f}s")
it = [float(x) for x in re.findall(r"Time for iteration \d+:\s*([\d.]+)", t)]
sc = [float(x) for x in re.findall(r"\[score\] predict_step took ([\d.]+)s", t)]
print(f"[post] iterations {len(it)} = {sum(it):.1f}s | scoring {len(sc)} = {sum(sc):.1f}s")
print(f"[post] designs {len(re.findall(r'^Best sequence', t, re.M))} err={'YES' if 'Traceback' in t else 'no'}")
