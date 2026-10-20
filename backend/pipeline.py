"""Adapters around the existing OCR, parser and LayoutLMv3 prediction modules."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "Receipt_Budget_Book"
MODEL = Path(os.getenv("BUDGET_MODEL_DIR", LEGACY / "outputs" / "layoutlmv3_linear_probe"))
CACHE = Path(os.getenv("BUDGET_CACHE_DIR", ROOT / ".cache"))
sys.path.insert(0, str(LEGACY))


def ocr_python():
    configured = os.getenv("OCR_PYTHON")
    if configured:
        return Path(configured)
    candidate = ROOT / ".ocr-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.exists() else Path(sys.executable)


def readiness():
    isolated = ocr_python() != Path(sys.executable)
    return {
        "ocr_environment": ocr_python().exists() and (isolated or importlib.util.find_spec("paddleocr") is not None),
        "model_weights": (MODEL / "final_model" / "linear_head.safetensors").exists() and (MODEL / "model_info.json").exists(),
        "mode": "local_single_user",
        "database": "SQLite",
    }


class LocalOCRAdapter:
    """Provide only the read interfaces expected by existing parsing/prediction."""
    def __init__(self, image: Path, raw: dict):
        self.image, self.raw = image, raw
        self.supabase = SimpleNamespace(storage=self)

    def get_ocr_items(self, _):
        return SimpleNamespace(data=self.raw["ocr_result"])

    def get_ocr_raw(self, _):
        return SimpleNamespace(data={"image_name": self.image.name})

    def from_(self, _):
        return self

    def download(self, _):
        return self.image.read_bytes()


def process_receipt(image: Path):
    output = image.with_suffix(".ocr.json")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    environment["HF_HOME"] = os.getenv("HF_HOME",str(CACHE / "huggingface"))
    # Paddle's Windows native loader cannot open Korean absolute model paths.
    environment["PADDLE_PDX_CACHE_HOME"] = os.path.relpath(CACHE / "paddlex",ROOT) if os.name=="nt" else str(CACHE / "paddlex")
    # A separate process keeps Paddle and Torch DLLs out of the same interpreter.
    result = subprocess.run(
        [str(ocr_python()), str(LEGACY / "ysz_ocr.py"), os.path.relpath(image, ROOT), "-o", os.path.relpath(output, ROOT)],
        cwd=ROOT, env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=900,
    )
    if result.returncode:
        # Full diagnostic is retained locally; API exposes a short actionable error.
        image.with_suffix(".error.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        raise RuntimeError("OCR 실행 실패. PaddleOCR 환경과 runtime/uploads의 error.log를 확인한 뒤 재시도하세요.")
    raw = json.loads(output.read_text(encoding="utf-8"))
    if not raw.get("ocr_result"):
        raise ValueError("이미지에서 글자를 찾지 못했습니다. 더 선명한 영수증 사진을 올려주세요.")
    return parse_raw(image, raw)


def parse_raw(image: Path, raw: dict):
    from receipt_parser import run_receipt_parser, _find_valid_date, DATE_PATTERNS
    adapter = LocalOCRAdapter(image, raw)
    parsed = run_receipt_parser(0, adapter)
    predictions = None
    engine = "paddleocr+rules"
    warnings = []
    if readiness()["model_weights"]:
        try:
            os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
            import torch
            torch.set_num_threads(4)
            from predict_layoutlmv3 import predict_ocr_raw_id
            predictions = predict_ocr_raw_id(0, MODEL, mixed_precision="none", db_manager=adapter)
            entities = predictions["entities"]
            engine += "+layoutlmv3"
            for label, target in (("STORE_NAME", "store_name"), ("DATE", "purchase_date"), ("TOTAL_AMOUNT", "total_amount")):
                candidates = sorted(entities.get(label, []), key=lambda e: e["confidence"], reverse=True)
                if not candidates or candidates[0]["confidence"] < 0.8:
                    continue
                text = candidates[0]["text"]
                value = text
                if label == "DATE":
                    value = _find_valid_date(text, DATE_PATTERNS)
                elif label == "TOTAL_AMOUNT":
                    numbers = re.findall(r"\d[\d,]*", text)
                    value = int(numbers[-1].replace(",", "")) if numbers else None
                if value is not None:
                    parsed[target] = value
        except Exception as error:
            warnings.append("모델 추론 실패: " + type(error).__name__ + ". 규칙 파싱 결과를 검토하세요.")
    else:
        warnings.append("학습 가중치가 없어 규칙 파서만 사용했습니다.")
    if not parsed.get("purchase_date"):
        warnings.append("구매 날짜를 찾지 못했습니다. 사진을 확인해 입력해주세요.")
    if parsed.get("total_amount") is None:
        warnings.append("총액을 찾지 못했습니다. 사진을 확인해 입력해주세요.")
    totals = [item.get("line_total") for item in parsed.get("items", [])]
    if totals and all(isinstance(value, int) for value in totals) and sum(totals) != parsed.get("total_amount"):
        warnings.append("품목 금액의 합과 총액이 다릅니다. 할인 또는 인식 누락을 확인해주세요.")
    parsed["warnings"] = warnings
    return {"parsed": parsed, "raw": raw, "predictions": predictions, "engine": engine}
