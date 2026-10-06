import subprocess
import sys
import os
import time

MODE = "full"  # change to "top4"

scripts = [
    "prep_data_final.py",
    "test_prep.py",
    "dummy_method.py",
    "feature_extraction.py",
    "soft_labels_create.py",
    "perceptron.py",
    "bert_train.py",
    "soft_bert.py",
    "evaluation.py"
]

SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

env = os.environ.copy()
env["DATA_MODE"] = MODE

t0 = time.time()
for script in scripts:
    print(f"\n{'='*60}\nRunning {script} [{MODE}]...\n{'='*60}")
    t = time.time()
    subprocess.run([sys.executable, os.path.join(SRC, script)], check=True, env=env, cwd=ROOT)
    print(f"Done ({time.time() - t:.0f}s).")

print(f"\nAll done [{MODE}] in {(time.time() - t0)/60:.1f} min.")
