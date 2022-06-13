import os
from pathlib import Path

DATASETS = ["mrpc", "rte", "cola", "boolq", "cb"]
MODELS = [
    ("gpt2", "gpt2-base"),
    ("gpt2", "gpt2-medium"),
    ("gpt2", "gpt2-large"),
    ("gpt2", "EleutherAI/gpt-neo-2.7B"),
    ("opt", "facebook/opt-13b"),
    ("opt", "facebook/opt-30b"),
]
PHASE_SHIFTS = list(range(0, 600, 100))
FEW_SHOTS = [0, 5]

if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        for model, model_name in MODELS:
            for shot in FEW_SHOTS:
                for ps in PHASE_SHIFTS:
                    output_path = (
                        results_dir / f"{ds}__{model}__{shot}__{model_name}__{ps}.json"
                    )
                    print(f"Running results for {output_path}")
                    if output_path.exists():
                        print("Skipped.")
                        continue

                    batch_size = (
                        8
                        if model_name in ["gpt2-base", "gpt2-medium", "gpt2-large"]
                        else 1
                    )
                    python_run_script = (
                        f"python main.py "
                        f"--model {model}-ps "
                        f"--model_args pretrained={model_name},phase_shift={ps} "
                        f"--tasks {ds} "
                        f"--num_fewshot {shot} "
                        f"--batch_size {batch_size} "
                        f"--output_path {output_path} "
                        f"--no_cache "
                    )
                    print(python_run_script)
                    os.system(python_run_script)