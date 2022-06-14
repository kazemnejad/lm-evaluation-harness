import random
from pathlib import Path

random.seed(42)
SEEDS = [random.randint(1, 1e6) for _ in range(5)]
print("Seeds:", SEEDS)
DATASETS = ["mrpc", "rte", "cola", "boolq", "cb"]
MODELS = [
    # ("gpt2", "EleutherAI/gpt-neo-125M"),
    ("opt", "facebook/opt-125m"),
    # ("gpt2", "EleutherAI/gpt-neo-1.3B"),
    ("opt", "facebook/opt-350m"),
    # ("gpt2", "EleutherAI/gpt-neo-2.7B"),
    ("opt", "facebook/opt-2.7b"),
    ("opt", "facebook/opt-13b"),
    ("opt", "facebook/opt-30b"),
]
PHASE_SHIFTS = list(range(0, 1001, 100))
FEW_SHOTS = [0, 5]

import os
import time

if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for ds in DATASETS:
            for model, model_name in MODELS:
                for shot in FEW_SHOTS:
                    for ps in PHASE_SHIFTS:
                        exp_name = f"s-{seed}___ds-{ds}___m-{model_name}___f-{shot}___ps-{ps}"
                        exp_name = exp_name.replace("EleutherAI/", "").replace(
                            "facebook/", ""
                        )
                        output_path = results_dir / f"{exp_name}.json"
                        print(f"Running results for {output_path}")
                        if output_path.exists():
                            print("Skipped.")
                            continue

                        batch_size = 1
                        python_run_script = (
                            f"python main.py "
                            f"--model {model}-ps "
                            f"--model_args pretrained={model_name},phase_shift={ps} "
                            f"--tasks {ds} "
                            f"--num_fewshot {shot} "
                            f"--batch_size {batch_size} "
                            f"--output_path {output_path} "
                            f"--seed {seed} "
                            f"--no_cache "
                        )
                        print(python_run_script)
                        try:
                            os.system(python_run_script)
                            print("\n"*5)
                            time.sleep(5)
                        except Exception as exp:
                            print(exp)
                            exit(1)
