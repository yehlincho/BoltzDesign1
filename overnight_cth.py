#!/usr/bin/env python
"""
overnight_cth.py — third arm: confidence module only in the temp+hard stages (50 in-loop
sampling steps), directly comparable to the `ovn_s50` arm of overnight_steps.py, which
runs confidence in every stage at 50 steps.

  soft stage (75 iters) : distogram only   <- the saving
  temp+hard (50 iters)  : distogram + confidence

Sequential on one GPU, 4 targets, LigandMPNN + AF3 on. Writes overnight_cth_report.txt
after every job.
"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
import overnight_steps as ovn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0"); ap.add_argument("--designs", type=int, default=4)
    ap.add_argument("--hours", type=float, default=8.5)
    a = ap.parse_args()
    deadline = time.time() + a.hours * 3600
    rpt = os.path.join(HERE, "overnight_cth_report.txt")
    state = {}
    def report():
        L = ["==== OVERNIGHT arm 3: confidence in temp+hard only, 50 in-loop steps ====",
             "compare against the ovn_s50 rows of overnight_steps_report.txt",
             "(same 50 steps, but confidence in EVERY stage there)", "",
             f"{'target':7} {'designs':>8} {'af3_rows':>9} {'af3_success':>12} {'rate':>7} {'status':>10}"]
        for t, v in state.items():
            d, af3, ok = ovn.tally(t, v["suffix"])
            rate = f"{100.0*ok/d:.0f}%" if d else "-"
            L.append(f"{t:7} {d:8d} {af3:9d} {ok:12d} {rate:>7} {v['status']:>10}")
        L.append("=" * 62)
        open(rpt, "w").write("\n".join(L) + "\n")

    for t in ["PDL1", "BHRF1", "FAD", "SAM"]:
        state[t] = {"suffix": "ovn_cth50", "status": "queued"}
    report()

    for t in list(state):
        if time.time() > deadline:
            state[t]["status"] = "skipped"; report(); continue
        ovn.seed_msa(t, "ovn_cth50")
        env = bs._env_with_conda()
        env["BOLTZDESIGN_SEED"] = "42"
        env["BOLTZDESIGN_DESIGN_SAMPLING_STEPS"] = "50"
        env["BOLTZDESIGN_CONFIDENCE_FROM"] = "temp"      # the whole point of this arm
        env["AF3_ENV_PYTHON"] = ovn.AF3_PY
        state[t]["status"] = f"gpu{a.gpu}"; report()
        log = os.path.join(HERE, f"overnight_cth_{t}.log")
        print(f"[{time.strftime('%T')}] start {t} (conf temp+hard, 50 steps)", flush=True)
        with open(log, "w") as lf:
            p = subprocess.Popen(ovn.cmd(t, "50", a.gpu, a.designs, "ovn_cth50"),
                                 stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
            while p.poll() is None:
                if time.time() > deadline:
                    p.terminate()
                    try: p.wait(180)
                    except Exception: p.kill()
                    state[t]["status"] = "timeout"; break
                time.sleep(15)
        if state[t]["status"] != "timeout":
            state[t]["status"] = "done" if p.returncode == 0 else f"rc{p.returncode}"
        print(f"[{time.strftime('%T')}] end {t} -> {state[t]['status']}", flush=True)
        report()
    print(open(rpt).read())

if __name__ == "__main__":
    main()
