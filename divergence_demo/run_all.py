"""Run every experiment, save plots under results/."""
import subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "exp1_single_run.py",
    "exp4_phase_portrait.py",
    "exp3_target_interval.py",
    "exp5_gamma_sweep.py",
    "exp6_phase3d.py",
    "exp2_divergence_sweep.py",   # slowest, last
]

for s in SCRIPTS:
    t0 = time.time()
    print(f"\n=== {s} ===")
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, s)])
    if r.returncode != 0:
        print(f"{s} failed with {r.returncode}")
        sys.exit(r.returncode)
    print(f"{s} took {time.time()-t0:.1f}s")
