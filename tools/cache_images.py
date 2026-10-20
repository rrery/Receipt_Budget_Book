"""Prefetch existing training images concurrently without changing their contents."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Receipt_Budget_Book"))
from database import SupabaseManager
from image_cache import download_image


def main():
    samples = json.loads((ROOT / "Receipt_Budget_Book/data/layoutlm/dataset.json").read_text(encoding="utf-8"))
    local = threading.local()
    def fetch(sample):
        for attempt in range(4):
            if not hasattr(local, "db"):
                local.db = SupabaseManager()
            try:
                return download_image(local.db.supabase, sample["image_path"], "images")
            except Exception:
                del local.db
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, sample) for sample in samples]
        for number, future in enumerate(as_completed(futures), 1):
            future.result().close()
            if number % 20 == 0 or number == len(samples):
                print(f"Image cache: {number}/{len(samples)}", flush=True)


if __name__ == "__main__":
    main()
