"""Run the existing real-data training pipeline and retain logs/progress."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "Receipt_Budget_Book"
RUNTIME = ROOT / "runtime"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--retrain", action="store_true", help="Recompute existing fold/final results")
    args = parser.parse_args()
    if args.refresh_data:
        args.retrain = True
    RUNTIME.mkdir(exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["RECEIPT_IMAGE_CACHE"] = str(ROOT / ".cache/receipt-images")
    status_path = RUNTIME / "training_status.json"

    def status(state, stage, **extra):
        value = {"status": state, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    steps = []
    if args.refresh_data or not (LEGACY / "data/layoutlm/dataset.json").exists():
        steps.append(("dataset", ["create_dataset.py"]))
    steps += [
        ("validate", ["validate_dataset.py"]),
        ("split", ["split_dataset.py"]),
        ("images", [str(ROOT / "tools/cache_images.py")]),
        ("train", ["train_layoutlmv3.py", "--device", "cpu", "--mixed-precision", "none", "--epochs", str(args.epochs)]),
    ]
    if args.retrain:
        steps[-1][1].append("--retrain-completed-folds")
    try:
        with (RUNTIME / "training.log").open("a", encoding="utf-8") as log:
            log.write("\n=== Training run " + datetime.now(timezone.utc).isoformat() + " ===\n")
            for stage, command in steps:
                status("running", stage)
                print("STAGE", stage, flush=True)
                process = subprocess.Popen([sys.executable, "-u", *command], cwd=LEGACY, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                while True:
                    try:
                        returncode = process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        status("running", stage)
                log.flush()
                if returncode:
                    raise RuntimeError(f"{stage} failed (exit {returncode}); see runtime/training.log")
        status("completed", "complete", weights="Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/final_model/linear_head.safetensors")
    except Exception as error:
        status("failed", stage, error=str(error))
        raise


if __name__ == "__main__":
    main()
