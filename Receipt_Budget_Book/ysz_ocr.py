

import argparse
import json
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
from paddleocr import PaddleOCR


DEFAULT_IMAGE_PATH = "0minysz/1000030506.jpg"  # 사진 파일명으로 바꿀 것
DEFAULT_OUTPUT_PATH = "receipt_ocr_raw.json"


def _to_json_serializable_box(box):
    """numpy array 형태의 box를 JSON 저장 가능한 list 형태로 바꾼다."""
    return box.tolist() if hasattr(box, "tolist") else box


def run_ocr(image_path: str, output_path: str = DEFAULT_OUTPUT_PATH, ocr_model=None) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    # PaddleOCR의 기본 한국어 기능 사용
    if ocr_model is None:
        ocr_model = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=1536,
            text_det_limit_type="max",
            enable_mkldnn=False,
        )

    result = ocr_model.predict(image_path)

    boxes = []
    texts = []
    scores = []

    for page in result:
        if "rec_polys" in page:
            boxes.extend(page["rec_polys"])  # 인식 필터를 통과한 텍스트와 같은 순서
        elif "dt_polys" in page:
            boxes.extend(page["dt_polys"])
        if "rec_texts" in page:
            texts.extend(page["rec_texts"])  # 추출한 글씨
        if "rec_scores" in page:
            scores.extend(page["rec_scores"])  # 추출한 글씨에 대한 신뢰도

    if not (len(boxes) == len(texts) == len(scores)):
        raise ValueError("OCR 인식 박스/텍스트/점수 개수가 다릅니다.")
    ocr_items = []

    for i, (box, text, score) in enumerate(zip(boxes, texts, scores), start=1):
        ocr_items.append(
            {
                "id": i,
                "text": text,
                "score": float(score),
                "box": _to_json_serializable_box(box),
            }
        )

    raw_result = {
        "image_path": image_path,
        "ocr_result": ocr_items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path}")
    return raw_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR 한국어 OCR 결과를 JSON으로 저장합니다.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=DEFAULT_IMAGE_PATH,
        help=f"OCR을 수행할 이미지 경로. 기본값: {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"저장할 JSON 파일 경로. 기본값: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ocr(args.image_path, args.output)
