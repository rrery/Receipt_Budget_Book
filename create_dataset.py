"""Supabase의 검수 완료 OCR 행을 LayoutLMv3용 JSON 데이터셋으로 변환한다."""

import argparse
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from database import SupabaseManager
from layoutlmv3_labels import ALLOWED_LABELS, normalize_label

OCR_ITEMS_TABLE = "ocr_raw_items"
OCR_RAW_TABLE = "ocr_raw"
DEFAULT_BUCKET = "images"
DEFAULT_OUTPUT = Path("data/layoutlm/dataset.json")


def polygon_to_bbox(polygon: list[list[float]]) -> list[float]:
    """PaddleOCR 4점 polygon을 [x_min, y_min, x_max, y_max]로 변환한다."""
    if not isinstance(polygon, list) or len(polygon) < 4:
        raise ValueError(f"올바르지 않은 polygon: {polygon!r}")
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox(
    bbox: list[float], image_width: int, image_height: int
) -> list[int]:
    """픽셀 bbox를 LayoutLMv3의 0~1000 좌표로 변환한다.

    OCR bbox와 이미지 좌표계가 크게 다르면 조용히 잘라내지 않고 해당 영수증을
    제외한다. 잘못된 좌표를 억지로 학습시키는 것보다 안전하다.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("이미지 크기가 0입니다.")

    x0, y0, x1, y1 = bbox
    if x0 < -0.05 * image_width or x1 > 1.05 * image_width:
        raise ValueError(
            f"bbox x좌표가 이미지와 맞지 않습니다: {bbox}, width={image_width}"
        )
    if y0 < -0.05 * image_height or y1 > 1.05 * image_height:
        raise ValueError(
            f"bbox y좌표가 이미지와 맞지 않습니다: {bbox}, height={image_height}"
        )

    values = [
        int(1000 * max(0.0, x0) / image_width),
        int(1000 * max(0.0, y0) / image_height),
        int(1000 * min(float(image_width), x1) / image_width),
        int(1000 * min(float(image_height), y1) / image_height),
    ]
    x0n, y0n, x1n, y1n = [max(0, min(1000, value)) for value in values]

    if x1n <= x0n:
        x0n, x1n = (999, 1000) if x0n >= 1000 else (x0n, x0n + 1)
    if y1n <= y0n:
        y0n, y1n = (999, 1000) if y0n >= 1000 else (y0n, y0n + 1)
    return [x0n, y0n, x1n, y1n]


def fetch_all_rows(query_builder: Any, page_size: int = 1000) -> list[dict]:
    """Supabase의 기본 반환 제한을 넘겨 모든 행을 페이지 단위로 조회한다."""
    rows: list[dict] = []
    start = 0
    while True:
        response = query_builder.range(start, start + page_size - 1).execute()
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        start += page_size


def get_verified_ocr_items(supabase: Any) -> list[dict]:
    query = (
        supabase.table(OCR_ITEMS_TABLE)
        .select("id, ocr_raw_id, text, box, label")
        .eq("verified", True)
        .not_.is_("label", "null")
        .order("ocr_raw_id")
        .order("id")
    )
    return fetch_all_rows(query)


def get_ocr_raw_map(supabase: Any) -> dict[int, str]:
    query = supabase.table(OCR_RAW_TABLE).select("id, image_name").order("id")
    rows = fetch_all_rows(query)
    return {
        int(row["id"]): row["image_name"]
        for row in rows
        if row.get("image_name")
    }


def download_image(supabase: Any, image_path: str, bucket: str) -> Image.Image:
    image_bytes = supabase.storage.from_(bucket).download(image_path)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def group_items_by_receipt(items: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        grouped[int(item["ocr_raw_id"])].append(item)
    return grouped


def build_receipt_sample(
    ocr_raw_id: int,
    items: list[dict],
    image_path: str,
    image: Image.Image,
) -> tuple[dict, int]:
    """영수증 하나를 학습 샘플로 만들고, 비어 있어 제외한 행 수를 반환한다."""
    image_width, image_height = image.size
    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []
    skipped_rows = 0

    for item in items:
        text = str(item.get("text") or "").strip()
        polygon = item.get("box")
        raw_label = item.get("label")
        if isinstance(polygon, str):
            polygon = json.loads(polygon)
        if not text or not polygon or not raw_label:
            skipped_rows += 1
            continue

        label = normalize_label(str(raw_label))
        if label not in ALLOWED_LABELS:
            raise ValueError(
                f"허용되지 않은 label={label!r}, ocr_raw_items.id={item.get('id')}"
            )

        words.append(text)
        boxes.append(
            normalize_bbox(
                polygon_to_bbox(polygon),
                image_width,
                image_height,
            )
        )
        labels.append(label)

    if not words:
        raise ValueError("사용 가능한 OCR 행이 없습니다.")
    if not (len(words) == len(boxes) == len(labels)):
        raise RuntimeError("words, boxes, labels 길이가 서로 다릅니다.")

    return {
        "ocr_raw_id": ocr_raw_id,
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "words": words,
        "boxes": boxes,
        "labels": labels,
    }, skipped_rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, separators=(",", ":"))


def create_dataset(output_path: Path, bucket: str) -> None:
    db = SupabaseManager()
    supabase = db.supabase
    items = get_verified_ocr_items(supabase)
    grouped_items = group_items_by_receipt(items)
    raw_map = get_ocr_raw_map(supabase)

    dataset: list[dict] = []
    excluded: list[dict] = []
    blank_row_count = 0

    for number, (ocr_raw_id, receipt_items) in enumerate(
        sorted(grouped_items.items()), start=1
    ):
        image_path = raw_map.get(ocr_raw_id)
        if not image_path:
            excluded.append({"ocr_raw_id": ocr_raw_id, "reason": "이미지 경로 없음"})
            continue

        try:
            image = download_image(supabase, image_path, bucket)
            sample, skipped = build_receipt_sample(
                ocr_raw_id, receipt_items, image_path, image
            )
            dataset.append(sample)
            blank_row_count += skipped
        except Exception as error:
            excluded.append({
                "ocr_raw_id": ocr_raw_id,
                "image_path": image_path,
                "reason": str(error),
            })
            print(f"[제외] ocr_raw_id={ocr_raw_id}: {error}")

        if number % 20 == 0 or number == len(grouped_items):
            print(f"데이터셋 변환: {number}/{len(grouped_items)}")

    if not dataset:
        raise RuntimeError("학습 가능한 영수증이 한 장도 없습니다.")

    write_json(output_path, dataset)
    label_counts = Counter(
        label for sample in dataset for label in sample["labels"]
    )
    report_path = output_path.with_name("dataset_build_report.json")
    write_json(report_path, {
        "receipt_count": len(dataset),
        "excluded_receipt_count": len(excluded),
        "blank_row_count": blank_row_count,
        "label_counts": dict(sorted(label_counts.items())),
        "excluded_receipts": excluded,
    })

    print(f"\nDataset 생성 완료: {len(dataset)}장")
    print(f"제외 영수증: {len(excluded)}장")
    print(f"저장 위치: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supabase 학습 데이터셋 생성")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args.output, args.bucket)
