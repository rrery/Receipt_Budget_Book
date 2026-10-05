"""영수증 단위 데이터 누수 없이 Train/Validation/Test를 균형 분할한다."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path("data/layoutlm/dataset.json")
DEFAULT_OUTPUT_DIR = Path("data/layoutlm")


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def entity_set(sample: dict) -> set[str]:
    """층화 계산에 사용할 entity 종류. 과다한 O 라벨은 제외한다."""
    return {
        label.split("-", 1)[1]
        for label in sample["labels"]
        if label != "O" and "-" in label
    }


def balanced_split(
    dataset: list[dict], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[dict]]:
    """희소 entity를 먼저 배치하는 결정적 multilabel 균형 분할."""
    if not dataset:
        raise ValueError("Dataset이 비어 있습니다.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train/validation 비율과 남은 test 비율은 모두 0보다 커야 합니다.")

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    target_sizes = {
        "train": train_size,
        "validation": val_size,
        "test": total - train_size - val_size,
    }
    total_entities = Counter(entity for sample in dataset for entity in entity_set(sample))
    target_entities = {
        name: {
            entity: count * size / total
            for entity, count in total_entities.items()
        }
        for name, size in target_sizes.items()
    }

    rng = random.Random(seed)
    ordered = list(dataset)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda sample: (
            sum(1.0 / total_entities[entity] for entity in entity_set(sample)),
            len(entity_set(sample)),
        ),
        reverse=True,
    )

    result: dict[str, list[dict]] = {name: [] for name in target_sizes}
    entity_counts = {name: Counter() for name in target_sizes}

    for sample in ordered:
        entities = entity_set(sample)
        candidates = [
            name for name in target_sizes if len(result[name]) < target_sizes[name]
        ]

        def placement_cost(name: str) -> tuple[float, float, str]:
            entity_cost = 0.0
            for entity in entities:
                current = entity_counts[name][entity]
                target = target_entities[name][entity]
                entity_cost += (current + 1 - target) ** 2 - (current - target) ** 2
            size_fill = len(result[name]) / max(1, target_sizes[name])
            return entity_cost, size_fill, name

        selected = min(candidates, key=placement_cost)
        result[selected].append(sample)
        entity_counts[selected].update(entities)

    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, separators=(",", ":"))


def validate_split(dataset: list[dict], splits: dict[str, list[dict]]) -> None:
    all_ids = [sample["ocr_raw_id"] for sample in dataset]
    split_ids = {
        name: {sample["ocr_raw_id"] for sample in samples}
        for name, samples in splits.items()
    }
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("원본 Dataset에 중복 ocr_raw_id가 있습니다.")
    if split_ids["train"] & split_ids["validation"]:
        raise RuntimeError("Train/Validation 데이터 누수가 있습니다.")
    if split_ids["train"] & split_ids["test"]:
        raise RuntimeError("Train/Test 데이터 누수가 있습니다.")
    if split_ids["validation"] & split_ids["test"]:
        raise RuntimeError("Validation/Test 데이터 누수가 있습니다.")
    if set(all_ids) != set().union(*split_ids.values()):
        raise RuntimeError("분할 과정에서 영수증이 누락됐습니다.")


def summarize(splits: dict[str, list[dict]]) -> dict:
    manifest: dict[str, Any] = {}
    print("\n===== Dataset 분할 =====")
    for name, samples in splits.items():
        counts = Counter(entity for sample in samples for entity in entity_set(sample))
        manifest[name] = {
            "receipt_count": len(samples),
            "receipt_ids": [sample["ocr_raw_id"] for sample in samples],
            "entity_receipt_counts": dict(sorted(counts.items())),
        }
        entity_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"{name:<10}: {len(samples):>3}장 | {entity_text}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LayoutLMv3 Dataset 균형 분할")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    samples = load_dataset(args.dataset)
    split_result = balanced_split(
        samples, args.train_ratio, args.validation_ratio, args.seed
    )
    validate_split(samples, split_result)
    manifest = summarize(split_result)

    write_json(args.output_dir / "train.json", split_result["train"])
    write_json(args.output_dir / "validation.json", split_result["validation"])
    write_json(args.output_dir / "test.json", split_result["test"])
    write_json(args.output_dir / "split_manifest.json", {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": 1 - args.train_ratio - args.validation_ratio,
        "splits": manifest,
    })
    print("=========================\n분할 파일 저장 완료")
