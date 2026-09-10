import os
import json
from collections import defaultdict
from io import BytesIO

from PIL import Image
from supabase import create_client


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

OCR_ITEMS_TABLE = "ocr_raw_items"
OCR_RAW_TABLE = "ocr_raw"
IMAGE_BUCKET = "images"

OUTPUT_PATH = "data/layoutlm/dataset.json"


def polygon_to_bbox(polygon):
    """PaddleOCR의 4개 꼭짓점 좌표를 [x_min, y_min, x_max, y_max]로 변환."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


def normalize_bbox(bbox, image_width, image_height):
    """bbox 좌표를 LayoutLM 계열에서 사용하는 0~1000 범위로 정규화."""
    x_min, y_min, x_max, y_max = bbox

    normalized = [
        int(1000 * x_min / image_width),
        int(1000 * y_min / image_height),
        int(1000 * x_max / image_width),
        int(1000 * y_max / image_height),
    ]

    # 좌표가 이미지 범위를 조금 벗어난 경우를 대비
    return [max(0, min(1000, value)) for value in normalized]


def get_verified_ocr_items(supabase):
    """검수가 완료된 OCR item만 가져온다."""
    response = (
        supabase.table(OCR_ITEMS_TABLE)
        .select("id, ocr_raw_id, text, box, label")
        .eq("verified", True)
        .order("ocr_raw_id")
        .order("id")
        .execute()
    )

    return response.data


def get_ocr_raw_map(supabase):
    """
    ocr_raw_id와 이미지 파일을 연결한다.

    현재는 ocr_raw 테이블에 id, image_path 컬럼이 있다고 가정한다.
    실제 컬럼명이 다르면 select 부분만 수정하면 된다.
    """
    response = (
        supabase.table(OCR_RAW_TABLE)
        .select("id, image_path")
        .execute()
    )

    return {
        row["id"]: row["image_path"]
        for row in response.data
        if row.get("image_path")
    }


def download_image(supabase, image_path):
    """Supabase Storage에서 이미지를 내려받고 PIL Image로 읽는다."""
    image_bytes = supabase.storage.from_(IMAGE_BUCKET).download(image_path)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def group_items_by_receipt(items):
    """ocr_raw_id 기준으로 OCR token들을 한 영수증 단위로 묶는다."""
    grouped = defaultdict(list)

    for item in items:
        grouped[item["ocr_raw_id"]].append(item)

    return grouped


def build_receipt_sample(ocr_raw_id, items, image_path, image):
    """영수증 한 장을 하나의 학습 샘플 형태로 만든다."""
    image_width, image_height = image.size

    words = []
    boxes = []
    labels = []

    for item in items:
        text = item.get("text")
        polygon = item.get("box")
        label = item.get("label")

        if not text or not polygon or not label:
            continue

        bbox = polygon_to_bbox(polygon)
        normalized_bbox = normalize_bbox(
            bbox,
            image_width,
            image_height,
        )

        words.append(text)
        boxes.append(normalized_bbox)
        labels.append(label)

    if not (len(words) == len(boxes) == len(labels)):
        raise ValueError(
            f"ocr_raw_id={ocr_raw_id}: words, boxes, labels 길이가 서로 다릅니다."
        )

    return {
        "ocr_raw_id": ocr_raw_id,
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "words": words,
        "boxes": boxes,
        "labels": labels,
    }


def create_dataset():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "환경변수 SUPABASE_URL, SUPABASE_KEY를 설정해주세요."
        )

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    items = get_verified_ocr_items(supabase)
    grouped_items = group_items_by_receipt(items)
    raw_map = get_ocr_raw_map(supabase)

    dataset = []

    for ocr_raw_id, receipt_items in grouped_items.items():
        image_path = raw_map.get(ocr_raw_id)

        if not image_path:
            print(f"[SKIP] ocr_raw_id={ocr_raw_id}: 이미지 경로 없음")
            continue

        try:
            image = download_image(supabase, image_path)

            sample = build_receipt_sample(
                ocr_raw_id=ocr_raw_id,
                items=receipt_items,
                image_path=image_path,
                image=image,
            )

            if sample["words"]:
                dataset.append(sample)

        except Exception as error:
            print(f"[ERROR] ocr_raw_id={ocr_raw_id}: {error}")

    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Dataset 생성 완료: {len(dataset)}개 영수증")
    print(f"저장 위치: {output_file}")


if __name__ == "__main__":
    create_dataset()
