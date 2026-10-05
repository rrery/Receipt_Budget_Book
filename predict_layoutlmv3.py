"""학습된 최소 Linear head로 Supabase OCR 행의 BIO 라벨을 예측한다."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor

from create_dataset import normalize_bbox, polygon_to_bbox
from database import SupabaseManager
from train_layoutlmv3 import (
    LinearTokenClassifier,
    autocast_context,
    resolve_precision,
    select_device,
)


def download_image(supabase: Any, image_path: str, bucket: str) -> Image.Image:
    image_bytes = supabase.storage.from_(bucket).download(image_path)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def group_entities(predictions: list[dict]) -> dict[str, list[dict]]:
    """연속된 B/I 예측을 entity 문자열로 묶는다."""
    grouped: dict[str, list[dict]] = {}
    current: dict | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        current["text"] = " ".join(current.pop("words"))
        current["confidence"] = sum(current.pop("confidences")) / len(current["indices"])
        grouped.setdefault(current.pop("entity"), []).append(current)
        current = None

    for item in predictions:
        label = item["label"]
        if label == "O" or "-" not in label:
            finish()
            continue
        prefix, entity = label.split("-", 1)
        if prefix == "B" or current is None or current["entity"] != entity:
            finish()
            current = {
                "entity": entity,
                "words": [item["text"]],
                "indices": [item["word_index"]],
                "confidences": [item["confidence"]],
            }
        else:
            current["words"].append(item["text"])
            current["indices"].append(item["word_index"])
            current["confidences"].append(item["confidence"])
    finish()
    return grouped


@torch.no_grad()
def predict_ocr_raw_id(
    ocr_raw_id: int,
    model_dir: Path,
    bucket: str = "images",
    mixed_precision: str = "auto",
) -> dict[str, Any]:
    info_path = model_dir / "model_info.json"
    head_path = model_dir / "final_model" / "linear_head.safetensors"
    if not info_path.exists() or not head_path.exists():
        raise FileNotFoundError(
            "학습 결과가 없습니다. model_info.json과 final_model/linear_head.safetensors를 확인하세요."
        )
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)

    db = SupabaseManager()
    raw_response = db.get_ocr_raw(ocr_raw_id)
    raw = raw_response.data
    item_response = db.get_ocr_items(ocr_raw_id)
    items = sorted(item_response.data or [], key=lambda row: int(row["id"]))
    if not raw or not items:
        raise ValueError(f"ocr_raw_id={ocr_raw_id}의 OCR 데이터가 없습니다.")

    image_path = raw.get("image_name")
    if not image_path:
        raise ValueError(f"ocr_raw_id={ocr_raw_id}에 image_name이 없습니다.")
    image = download_image(db.supabase, image_path, bucket)
    width, height = image.size

    words: list[str] = []
    boxes: list[list[int]] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        polygon = item.get("box")
        if isinstance(polygon, str):
            polygon = json.loads(polygon)
        if not text or not polygon:
            continue
        words.append(text)
        boxes.append(normalize_bbox(polygon_to_bbox(polygon), width, height))
    if not words:
        raise ValueError("예측할 OCR text가 없습니다.")

    model_name = info["model_name"]
    labels = info["labels"]
    max_length = int(info["max_length"])
    stride = int(info["stride"])
    device = select_device()
    precision = resolve_precision(mixed_precision, device)

    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    backbone = AutoModel.from_pretrained(model_name).to(device).eval()
    head = LinearTokenClassifier(
        int(info["hidden_size"]), len(labels), dropout=0.0
    ).to(device).eval()
    head.load_state_dict(load_file(str(head_path)))

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    pixel_values = encoding["pixel_values"]
    if isinstance(pixel_values, (list, tuple)):
        pixel_values = torch.stack([torch.as_tensor(value) for value in pixel_values])

    best_by_word: dict[int, tuple[str, float]] = {}
    chunk_count = int(encoding["input_ids"].shape[0])
    for chunk_index in range(chunk_count):
        pixel_index = 0 if pixel_values.shape[0] == 1 else chunk_index
        input_ids = encoding["input_ids"][chunk_index : chunk_index + 1].to(device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][chunk_index : chunk_index + 1].to(device),
            "bbox": encoding["bbox"][chunk_index : chunk_index + 1].to(device),
            "pixel_values": pixel_values[pixel_index : pixel_index + 1].to(device),
        }
        with autocast_context(device, precision):
            features = backbone(**batch).last_hidden_state[:, : input_ids.shape[1], :]
            probabilities = torch.softmax(head(features)["logits"], dim=-1)[0]

        seen_in_chunk: set[int] = set()
        for token_index, word_id in enumerate(encoding.word_ids(batch_index=chunk_index)):
            if word_id is None or word_id in seen_in_chunk:
                continue
            seen_in_chunk.add(word_id)
            confidence, label_id = probabilities[token_index].max(dim=-1)
            candidate = (labels[int(label_id)], float(confidence))
            if word_id not in best_by_word or candidate[1] > best_by_word[word_id][1]:
                best_by_word[word_id] = candidate

    predictions = []
    for index, (word, box) in enumerate(zip(words, boxes)):
        label, confidence = best_by_word.get(index, ("O", 0.0))
        predictions.append({
            "word_index": index,
            "text": word,
            "bbox": box,
            "label": label,
            "confidence": round(confidence, 6),
        })

    return {
        "ocr_raw_id": ocr_raw_id,
        "image_path": image_path,
        "entities": group_entities(predictions),
        "token_predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supabase 영수증 BIO 라벨 예측")
    parser.add_argument("ocr_raw_id", type=int)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("outputs/layoutlmv3_linear_probe")
    )
    parser.add_argument("--bucket", default="images")
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "none", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_ocr_raw_id(
        args.ocr_raw_id,
        args.model_dir,
        args.bucket,
        args.mixed_precision,
    )
    output_path = args.output or Path("outputs/predictions") / f"ocr_raw_{args.ocr_raw_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result["entities"], ensure_ascii=False, indent=2))
    print(f"전체 예측 저장: {output_path}")
