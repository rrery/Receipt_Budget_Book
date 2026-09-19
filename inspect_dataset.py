"""정규화 bbox와 BIO 라벨을 이미지 위에 표시하여 눈으로 검사한다."""

import argparse
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from database import SupabaseManager

DEFAULT_DATASET = Path("data/layoutlm/dataset.json")


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_sample_by_id(dataset: list[dict], ocr_raw_id: int) -> dict | None:
    return next(
        (sample for sample in dataset if sample["ocr_raw_id"] == ocr_raw_id),
        None,
    )


def download_image(supabase, image_path: str, bucket: str) -> Image.Image:
    image_bytes = supabase.storage.from_(bucket).download(image_path)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def denormalize_bbox(box: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        int(x0 * width / 1000),
        int(y0 * height / 1000),
        int(x1 * width / 1000),
        int(y1 * height / 1000),
    ]


def draw_labeled_boxes(image: Image.Image, sample: dict) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box, label in zip(sample["boxes"], sample["labels"]):
        if label == "O":
            continue
        original_box = denormalize_bbox(box, width, height)
        draw.rectangle(original_box, outline="red", width=2)
        draw.text((original_box[0], original_box[1]), label, fill="red")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset bbox/라벨 시각 검사")
    parser.add_argument("ocr_raw_id", type=int)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bucket", default="images")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    samples = load_dataset(args.dataset)
    sample = get_sample_by_id(samples, args.ocr_raw_id)
    if sample is None:
        raise SystemExit(f"ocr_raw_id={args.ocr_raw_id}를 찾을 수 없습니다.")

    db = SupabaseManager()
    source_image = download_image(db.supabase, sample["image_path"], args.bucket)
    inspected = draw_labeled_boxes(source_image.copy(), sample)
    output_path = args.output or Path("outputs/inspection") / f"receipt_{args.ocr_raw_id}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inspected.save(output_path, quality=90)

    print(f"ocr_raw_id: {sample['ocr_raw_id']}")
    print(f"OCR token 수: {len(sample['words'])}")
    print(f"라벨 표시 이미지: {output_path}")
