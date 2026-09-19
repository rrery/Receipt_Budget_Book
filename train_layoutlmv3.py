"""LayoutLMv3 최소 파라미터 Linear Probe + 5-Fold 학습.

동결한 LayoutLMv3 backbone은 문서 특징을 한 번만 계산한다. 이후 5-Fold와
최종 학습에서는 hidden_size -> BIO label 수의 단일 Linear 층만 업데이트한다.
Fold별 대용량 모델은 저장하지 않고 지표 JSON만 남긴다.
"""

import argparse
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

from database import SupabaseManager
from layoutlmv3_labels import LABEL_LIST
from validate_dataset import load_dataset, validate_dataset

ENCODED_KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "labels")


@dataclass
class EncodedData:
    chunks: list[dict[str, Any]]
    indices_by_receipt: dict[int, list[int]]
    receipt_count: int
    over_length_count: int


@dataclass
class FeatureData:
    chunks: list[dict[str, Any]]
    indices_by_receipt: dict[int, list[int]]
    hidden_size: int


class DictDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class LinearTokenClassifier(nn.Module):
    """학습되는 유일한 층: LayoutLMv3 hidden feature -> BIO label."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        logits = self.classifier(self.dropout(features))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="동결 LayoutLMv3 + 최소 Linear head 5-Fold 학습"
    )
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
        help="실행 장치. Intel XPU가 불안정하면 cpu를 명시하세요.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/layoutlm"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/layoutlmv3_linear_probe")
    )
    parser.add_argument("--bucket", default="images")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "none", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--restart-active-run",
        action="store_true",
        help="현재 Fold/final의 작은 resume_state.pt만 무시",
    )
    parser.add_argument(
        "--retrain-completed-folds",
        action="store_true",
        help="기존 fold_result.json도 무시하고 모든 Fold 재학습",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU를 사용할 수 없습니다.")
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        if requested == "fp16" and device.type != "cuda":
            raise ValueError("fp16은 이 코드에서 CUDA 장치에만 사용합니다.")
        if requested == "bf16" and device.type not in {"cuda", "xpu"}:
            raise ValueError("bf16은 CUDA/XPU 장치에서만 사용합니다.")
        return requested
    if device.type == "cuda":
        return "fp16"
    if device.type == "xpu":
        return "bf16"
    return "none"


def autocast_context(device: torch.device, precision: str):
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in module.state_dict().items()
    }


def download_image(supabase: Any, image_path: str, bucket: str) -> Image.Image:
    image_bytes = supabase.storage.from_(bucket).download(image_path)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def encode_samples(
    samples: list[dict],
    processor: Any,
    label2id: dict[str, int],
    supabase: Any,
    bucket: str,
    max_length: int,
    stride: int,
) -> EncodedData:
    chunks: list[dict[str, Any]] = []
    indices_by_receipt: dict[int, list[int]] = defaultdict(list)
    over_length_count = 0

    for number, sample in enumerate(samples, start=1):
        receipt_id = int(sample["ocr_raw_id"])
        image = download_image(supabase, sample["image_path"], bucket)
        word_labels = [label2id[label] for label in sample["labels"]]

        tokenized = processor.tokenizer(
            sample["words"],
            boxes=sample["boxes"],
            add_special_tokens=True,
            truncation=False,
        )
        if len(tokenized["input_ids"]) > max_length:
            over_length_count += 1

        encoding = processor(
            images=image,
            text=sample["words"],
            boxes=sample["boxes"],
            word_labels=word_labels,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        chunk_count = int(encoding["input_ids"].shape[0])
        pixel_values = encoding["pixel_values"]
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.stack([torch.as_tensor(value) for value in pixel_values])
        if pixel_values.shape[0] not in (1, chunk_count):
            raise ValueError(f"ocr_raw_id={receipt_id}: image/text chunk 수 불일치")

        seen_word_ids: set[int] = set()
        for chunk_index in range(chunk_count):
            labels = encoding["labels"][chunk_index].clone()
            word_ids = encoding.word_ids(batch_index=chunk_index)
            for token_index, word_id in enumerate(word_ids):
                if word_id is not None and word_id in seen_word_ids:
                    labels[token_index] = -100
            seen_word_ids.update(word_id for word_id in word_ids if word_id is not None)

            pixel_index = 0 if pixel_values.shape[0] == 1 else chunk_index
            row = {
                "receipt_id": receipt_id,
                "input_ids": encoding["input_ids"][chunk_index],
                "attention_mask": encoding["attention_mask"][chunk_index],
                "bbox": encoding["bbox"][chunk_index],
                "pixel_values": pixel_values[pixel_index],
                "labels": labels,
            }
            global_index = len(chunks)
            chunks.append(row)
            indices_by_receipt[receipt_id].append(global_index)

        if number % 10 == 0 or number == len(samples):
            print(f"  Processor 변환: {number}/{len(samples)}")

    return EncodedData(
        chunks=chunks,
        indices_by_receipt=dict(indices_by_receipt),
        receipt_count=len(samples),
        over_length_count=over_length_count,
    )


def collate_encoded(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in batch]) for key in ENCODED_KEYS}


def collate_features(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([row["features"] for row in batch]),
        "labels": torch.stack([row["labels"] for row in batch]),
    }


@torch.no_grad()
def extract_frozen_features(
    encoded: EncodedData,
    model_name: str,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> FeatureData:
    """동결 backbone을 단 한 번 통과시켜 text 특징을 CPU RAM에 저장한다."""
    print("\n동결 LayoutLMv3 특징 추출 중... (이 단계만 backbone 사용)")
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    loader = DataLoader(
        DictDataset(encoded.chunks),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_encoded,
    )
    feature_chunks: list[dict[str, Any]] = []
    # autocast를 끈 가속기 실행에서는 Linear weight와 dtype을 맞추기 위해 fp32 유지.
    if device.type == "cpu":
        storage_dtype = torch.float32
    elif precision == "fp16":
        storage_dtype = torch.float16
    elif precision == "bf16":
        storage_dtype = torch.bfloat16
    else:
        storage_dtype = torch.float32

    for batch_number, batch in enumerate(loader, start=1):
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast_context(device, precision):
            outputs = backbone(**batch)
        text_length = batch["input_ids"].shape[1]
        features = outputs.last_hidden_state[:, :text_length, :]

        for index in range(features.shape[0]):
            feature_chunks.append({
                "features": features[index].to(device="cpu", dtype=storage_dtype),
                "labels": labels[index].cpu(),
            })
        if batch_number % 10 == 0 or batch_number == len(loader):
            print(f"  특징 batch: {batch_number}/{len(loader)}")

    hidden_size = int(backbone.config.hidden_size)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()

    return FeatureData(
        chunks=feature_chunks,
        indices_by_receipt=encoded.indices_by_receipt,
        hidden_size=hidden_size,
    )


def build_folds(samples: list[dict], n_splits: int, seed: int) -> list[dict[str, list[int]]]:
    """O를 제외한 entity 종류를 층화하고 ocr_raw_id 단위로 5-Fold를 만든다."""
    if len(samples) < n_splits:
        raise ValueError(f"영수증 {len(samples)}장으로 {n_splits}-Fold를 만들 수 없습니다.")

    y: list[str] = []
    groups: list[int] = []
    for sample in samples:
        entities = sorted({
            label.split("-", 1)[1]
            for label in sample["labels"]
            if label != "O" and "-" in label
        }) or ["NO_ENTITY"]
        for entity in entities:
            y.append(entity)
            groups.append(int(sample["ocr_raw_id"]))

    group_array = np.asarray(groups)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds: list[dict[str, list[int]]] = []
    for train_indices, validation_indices in splitter.split(
        np.zeros((len(y), 1), dtype=np.int8),
        np.asarray(y),
        group_array,
    ):
        folds.append({
            "train_ids": sorted(set(group_array[train_indices].tolist())),
            "validation_ids": sorted(set(group_array[validation_indices].tolist())),
        })
    return folds


def print_fold_distribution(
    samples: list[dict], folds: list[dict[str, list[int]]]
) -> list[dict[str, Any]]:
    by_id = {int(sample["ocr_raw_id"]): sample for sample in samples}
    rows: list[dict[str, Any]] = []
    print("\n===== 5-Fold 구성 =====")
    for number, fold in enumerate(folds, start=1):
        counts = Counter()
        for receipt_id in fold["validation_ids"]:
            entities = {
                label.split("-", 1)[1]
                for label in by_id[receipt_id]["labels"]
                if label != "O" and "-" in label
            }
            counts.update(entities)
        row = {
            "fold": number,
            "train_ids": fold["train_ids"],
            "validation_ids": fold["validation_ids"],
            "entity_receipt_counts": dict(sorted(counts.items())),
        }
        rows.append(row)
        text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"Fold {number}: validation {len(fold['validation_ids'])}장 | {text}")
    print("========================")
    return rows


def feature_rows_for_receipts(
    receipt_ids: Iterable[int], feature_data: FeatureData
) -> list[dict[str, Any]]:
    return [
        feature_data.chunks[index]
        for receipt_id in receipt_ids
        for index in feature_data.indices_by_receipt[int(receipt_id)]
    ]


def bio_entities(labels: list[str]) -> set[tuple[str, int, int]]:
    entities: set[tuple[str, int, int]] = set()
    current_type: str | None = None
    start = -1
    for index, label in enumerate(labels + ["O"]):
        if label == "O" or "-" not in label:
            prefix, entity_type = "O", None
        else:
            prefix, entity_type = label.split("-", 1)
        if current_type is not None and (
            prefix in {"O", "B"} or entity_type != current_type
        ):
            entities.add((current_type, start, index - 1))
            current_type = None
        if prefix == "B" or (prefix == "I" and current_type is None):
            current_type = entity_type
            start = index
    return entities


def calculate_metrics(
    true_rows: list[list[str]], predicted_rows: list[list[str]]
) -> dict[str, Any]:
    tp = fp = fn = correct = total = 0
    per_entity: dict[str, dict[str, int]] = {}
    for true_labels, predicted_labels in zip(true_rows, predicted_rows):
        correct += sum(t == p for t, p in zip(true_labels, predicted_labels))
        total += len(true_labels)
        true_entities = bio_entities(true_labels)
        predicted_entities = bio_entities(predicted_labels)
        tp += len(true_entities & predicted_entities)
        fp += len(predicted_entities - true_entities)
        fn += len(true_entities - predicted_entities)

        for entity_type in {item[0] for item in true_entities | predicted_entities}:
            values = per_entity.setdefault(entity_type, {"tp": 0, "fp": 0, "fn": 0})
            true_type = {item for item in true_entities if item[0] == entity_type}
            pred_type = {item for item in predicted_entities if item[0] == entity_type}
            values["tp"] += len(true_type & pred_type)
            values["fp"] += len(pred_type - true_type)
            values["fn"] += len(true_type - pred_type)

    def scores(counts: dict[str, int]) -> dict[str, float]:
        local_tp, local_fp, local_fn = counts["tp"], counts["fp"], counts["fn"]
        precision = local_tp / (local_tp + local_fp) if local_tp + local_fp else 0.0
        recall = local_tp / (local_tp + local_fn) if local_tp + local_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    result: dict[str, Any] = scores({"tp": tp, "fp": fp, "fn": fn})
    result["accuracy"] = correct / total if total else 0.0
    result["per_entity"] = {
        name: scores(counts) for name, counts in sorted(per_entity.items())
    }
    return result


@torch.no_grad()
def evaluate_head(
    model: LinearTokenClassifier,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    id2label: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    true_rows: list[list[str]] = []
    predicted_rows: list[list[str]] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast_context(device, precision):
            outputs = model(**batch)
        losses.append(float(outputs["loss"].detach().cpu()))
        predicted_ids = outputs["logits"].argmax(dim=-1)
        for true_ids, pred_ids in zip(batch["labels"], predicted_ids):
            true_labels: list[str] = []
            pred_labels: list[str] = []
            for true_id, pred_id in zip(true_ids.tolist(), pred_ids.tolist()):
                if true_id == -100:
                    continue
                true_labels.append(id2label[true_id])
                pred_labels.append(id2label[pred_id])
            true_rows.append(true_labels)
            predicted_rows.append(pred_labels)

    metrics = calculate_metrics(true_rows, predicted_rows)
    metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
    return metrics


def make_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    return LambdaLR(optimizer, lr_lambda)


def train_one_fold(
    fold_number: int,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_data: FeatureData,
    args: argparse.Namespace,
    device: torch.device,
    precision: str,
    id2label: dict[int, str],
) -> dict[str, Any]:
    """Fold 모델 파일은 남기지 않고 결과 JSON만 저장한다."""
    fold_dir = args.output_dir / "cv" / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "fold_result.json"
    resume_path = fold_dir / "resume_state.pt"

    if result_path.exists() and not args.retrain_completed_folds:
        print(f"Fold {fold_number}: 완료 결과 재사용")
        return read_json(result_path)

    set_seed(args.seed + fold_number)
    model = LinearTokenClassifier(
        feature_data.hidden_size, len(LABEL_LIST), args.dropout
    ).to(device)
    train_loader = DataLoader(
        DictDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_features,
        generator=torch.Generator().manual_seed(args.seed + fold_number),
    )
    validation_loader = DataLoader(
        DictDataset(validation_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_features,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    scheduler = make_scheduler(
        optimizer, max(1, updates_per_epoch * args.epochs), args.warmup_ratio
    )

    scaler = None
    if precision == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()

    baseline = evaluate_head(model, validation_loader, device, precision, id2label)
    best_f1 = -1.0
    best_epoch = 0
    best_state = cpu_state_dict(model)
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if resume_path.exists() and not args.restart_active_run:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_state = state["best_state"]
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1
        print(f"Fold {fold_number}: epoch {start_epoch}부터 재개")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, precision):
                output = model(**batch)
                loss = output["loss"] / args.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)

            if (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            ):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate_head(model, validation_loader, device, precision, id2label)
        history.append({
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "validation": metrics,
        })
        print(
            f"Fold {fold_number} | Epoch {epoch:>2} | "
            f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f}"
        )

        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 약 수백 KB 이하의 Linear head/optimizer 상태만 저장한다.
        atomic_torch_save({
            "epoch": epoch,
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_state": best_state,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }, resume_path)

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    best_metrics = evaluate_head(model, validation_loader, device, precision, id2label)
    result = {
        "fold": fold_number,
        "train_chunk_count": len(train_rows),
        "validation_chunk_count": len(validation_rows),
        "baseline_validation": baseline,
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
    }
    write_json(result_path, result)
    if resume_path.exists():
        resume_path.unlink()
    del model
    return result


def train_final_head(
    development_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_data: FeatureData,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    precision: str,
    id2label: dict[int, str],
) -> dict[str, Any]:
    final_dir = args.output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    head_path = final_dir / "linear_head.safetensors"
    result_path = final_dir / "final_result.json"
    resume_path = final_dir / "resume_state.pt"

    model = LinearTokenClassifier(
        feature_data.hidden_size, len(LABEL_LIST), args.dropout
    ).to(device)
    test_loader = DataLoader(
        DictDataset(test_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_features,
    )

    if head_path.exists() and result_path.exists() and not args.retrain_completed_folds:
        model.load_state_dict(load_file(str(head_path)))
        result = read_json(result_path)
        print("최종 head 학습 완료 결과 재사용")
        return result

    train_loader = DataLoader(
        DictDataset(development_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_features,
        generator=torch.Generator().manual_seed(args.seed + 1000),
    )
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    scheduler = make_scheduler(
        optimizer, max(1, updates_per_epoch * epochs), args.warmup_ratio
    )
    start_epoch = 1
    if resume_path.exists() and not args.restart_active_run:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1

    scaler = None
    if precision == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, precision):
                loss = model(**batch)["loss"] / args.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            ):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        atomic_torch_save({
            "epoch": epoch,
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, resume_path)
        print(f"최종 head 학습: epoch {epoch}/{epochs}")

    test_metrics = evaluate_head(model, test_loader, device, precision, id2label)
    metadata = {
        "model_name": args.model_name,
        "labels": json.dumps(LABEL_LIST, ensure_ascii=False),
        "architecture": "frozen_layoutlmv3_linear_probe",
        "hidden_size": str(feature_data.hidden_size),
    }
    save_file(cpu_state_dict(model), str(head_path), metadata=metadata)
    result = {"epochs": epochs, "test": test_metrics}
    write_json(result_path, result)
    if resume_path.exists():
        resume_path.unlink()
    return result


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    precision = resolve_precision(args.mixed_precision, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_dataset(args.data_dir / "train.json")
    validation_samples = load_dataset(args.data_dir / "validation.json")
    test_samples = load_dataset(args.data_dir / "test.json")
    for name, samples in (
        ("train", train_samples),
        ("validation", validation_samples),
        ("test", test_samples),
    ):
        problems = validate_dataset(samples)
        if problems:
            raise ValueError(f"{name}.json 검증 실패: {problems[0]}")

    development_samples = train_samples + validation_samples
    all_samples = development_samples + test_samples
    development_ids = {int(sample["ocr_raw_id"]) for sample in development_samples}
    test_ids = {int(sample["ocr_raw_id"]) for sample in test_samples}
    if development_ids & test_ids:
        raise RuntimeError("개발 데이터와 Test 데이터 사이에 영수증 ID 누수가 있습니다.")

    label2id = {label: index for index, label in enumerate(LABEL_LIST)}
    id2label = {index: label for label, index in label2id.items()}
    print(f"학습 장치: {device} / mixed precision: {precision}")
    print("학습 범위: LayoutLMv3 100% 동결, 단일 Linear 출력층만 학습")

    processor = AutoProcessor.from_pretrained(args.model_name, apply_ocr=False)
    db = SupabaseManager()
    encoded = encode_samples(
        all_samples,
        processor,
        label2id,
        db.supabase,
        args.bucket,
        args.max_length,
        args.stride,
    )
    features = extract_frozen_features(
        encoded,
        args.model_name,
        device,
        precision,
        args.feature_batch_size,
    )

    trainable_count = features.hidden_size * len(LABEL_LIST) + len(LABEL_LIST)
    print(f"실제 학습 파라미터 수: {trainable_count:,}")
    folds = build_folds(development_samples, args.folds, args.seed)
    fold_manifest = print_fold_distribution(development_samples, folds)
    write_json(args.output_dir / "fold_manifest.json", fold_manifest)

    fold_results: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        print(f"\n[Fold {fold_number}/{args.folds}]")
        fold_results.append(train_one_fold(
            fold_number,
            feature_rows_for_receipts(fold["train_ids"], features),
            feature_rows_for_receipts(fold["validation_ids"], features),
            features,
            args,
            device,
            precision,
            id2label,
        ))

    baseline_f1 = [
        float(result["baseline_validation"]["f1"]) for result in fold_results
    ]
    fold_f1 = [float(result["best_validation"]["f1"]) for result in fold_results]
    fold_delta_f1 = [
        trained - baseline for baseline, trained in zip(baseline_f1, fold_f1)
    ]
    best_epochs = [int(result["best_epoch"]) for result in fold_results]
    cv_summary = {
        "folds": args.folds,
        "baseline_fold_f1": baseline_f1,
        "fold_f1": fold_f1,
        "fold_delta_f1": fold_delta_f1,
        "baseline_mean_f1": statistics.mean(baseline_f1),
        "mean_f1": statistics.mean(fold_f1),
        "mean_delta_f1": statistics.mean(fold_delta_f1),
        "std_f1": statistics.pstdev(fold_f1),
        "best_epochs": best_epochs,
        "recommended_final_epochs": max(1, int(round(statistics.median(best_epochs)))),
    }
    write_json(args.output_dir / "cross_validation_summary.json", cv_summary)

    development_rows = feature_rows_for_receipts(development_ids, features)
    test_rows = feature_rows_for_receipts(test_ids, features)
    final_result = train_final_head(
        development_rows,
        test_rows,
        features,
        cv_summary["recommended_final_epochs"],
        args,
        device,
        precision,
        id2label,
    )

    model_info = {
        "architecture": "frozen_layoutlmv3_linear_probe",
        "model_name": args.model_name,
        "labels": LABEL_LIST,
        "hidden_size": features.hidden_size,
        "trainable_parameter_count": trainable_count,
        "folds": args.folds,
        "max_length": args.max_length,
        "stride": args.stride,
        "cv": cv_summary,
        "final": final_result,
    }
    write_json(args.output_dir / "model_info.json", model_info)

    output_bytes = sum(
        path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file()
    )
    print("\n===== 전체 완료 =====")
    print(
        f"학습 전 평균 F1: {cv_summary['baseline_mean_f1']:.4f}"
        f" → 학습 후: {cv_summary['mean_f1']:.4f}"
        f" | 개선: {cv_summary['mean_delta_f1']:+.4f}"
    )
    print(
        f"5-Fold F1: {cv_summary['mean_f1']:.4f} "
        f"± {cv_summary['std_f1']:.4f}"
    )
    print(f"Test F1: {final_result['test']['f1']:.4f}")
    print(f"최종 출력 용량: {output_bytes / 1024 / 1024:.2f} MB")
    print(f"최종 head: {args.output_dir / 'final_model' / 'linear_head.safetensors'}")


if __name__ == "__main__":
    main(parse_args())
