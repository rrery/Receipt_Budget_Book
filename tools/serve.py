"""One-command local preview. Keep the terminal open while testing."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

if __name__ == "__main__":
    import uvicorn
    print("Receipt calendar: http://127.0.0.1:8000", flush=True)
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, workers=1)
