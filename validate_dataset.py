"""LayoutLMv3 JSON 데이터셋의 구조와 라벨을 학습 전에 검증한다."""

import argparse
import json
from collections import Counter
from pathlib import Path

from layoutlmv3_labels import ALLOWED_LABELS

DEFAULT_DATASET_PATH = Path("data/layoutlm/dataset.json")
REQUIRED_KEYS = {"ocr_raw_id", "image_path", "words", "boxes", "labels"}


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        raise ValueError("dataset 최상위 값은 list여야 합니다.")
    return value


def validate_dataset(dataset: list[dict]) -> list[str]:
    """오류 메시지 목록을 반환한다. 빈 목록이면 학습 가능한 구조다."""
    errors: list[str] = []
    seen_ids: set[int] = set()

    for sample_index, sample in enumerate(dataset):
        missing = REQUIRED_KEYS - set(sample)
        if missing:
            errors.append(f"sample[{sample_index}] 필수 키 누락: {sorted(missing)}")
            continue

        receipt_id = sample["ocr_raw_id"]
        if receipt_id in seen_ids:
            errors.append(f"ocr_raw_id 중복: {receipt_id}")
        seen_ids.add(receipt_id)

        words = sample["words"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        if not words:
            errors.append(f"ocr_raw_id={receipt_id}: OCR token이 비어 있음")
        if not (len(words) == len(boxes) == len(labels)):
            errors.append(
                f"ocr_raw_id={receipt_id}: 길이 불일치 "
                f"words={len(words)}, boxes={len(boxes)}, labels={len(labels)}"
            )
            continue

        for index, (word, box, label) in enumerate(zip(words, boxes, labels)):
            if not isinstance(word, str) or not word.strip():
                errors.append(f"ocr_raw_id={receipt_id}, token={index}: 빈 text")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(value, int) for value in box)
                or not all(0 <= value <= 1000 for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                errors.append(
                    f"ocr_raw_id={receipt_id}, token={index}: 잘못된 bbox={box!r}"
                )
            if label not in ALLOWED_LABELS:
                errors.append(
                    f"ocr_raw_id={receipt_id}, token={index}: 허용되지 않은 label={label!r}"
                )

    return errors


def print_summary(dataset: list[dict]) -> None:
    label_counts = Counter(
        label for sample in dataset for label in sample.get("labels", [])
    )
    entity_receipts = Counter()
    for sample in dataset:
        entities = {
            label.split("-", 1)[1]
            for label in sample.get("labels", [])
            if label != "O" and "-" in label
        }
        entity_receipts.update(entities)

    print(f"Dataset 영수증 수: {len(dataset)}")
    print(f"OCR token 수: {sum(label_counts.values())}")
    print("\n라벨별 token 수")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<20} {count:>6}")
    print("\nEntity가 등장하는 영수증 수(O 제외)")
    for entity, count in sorted(entity_receipts.items()):
        print(f"  {entity:<20} {count:>6}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LayoutLMv3 데이터셋 검증")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    samples = load_dataset(args.dataset)
    print_summary(samples)
    problems = validate_dataset(samples)
    if problems:
        print(f"\n[실패] 데이터 오류 {len(problems)}개")
        for problem in problems[:50]:
            print(f"- {problem}")
        if len(problems) > 50:
            print(f"... 나머지 {len(problems) - 50}개 생략")
        raise SystemExit(1)
    print("\n[성공] 데이터셋 검증 통과")
