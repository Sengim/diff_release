import subprocess
import sys

# Configurations
target_script = "main.py"
param_name = "--noise_schedule"
param_values = ["sig"]#, "quad", "lin", "cos"]

for value in param_values:
    print(f"Running '{target_script}' with {param_name}={value}...")
    result = subprocess.run(
        [sys.executable, target_script, param_name, str(value), "--dataset" , "ascadv2", "--dataset_dim", "2000"],
        capture_output=False, text=False
    )
    result = subprocess.run(
        [sys.executable, target_script, param_name, str(value)],
        capture_output=False, text=False
    )

print("All iterations complete.")