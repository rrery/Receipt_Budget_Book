"""완료된 5-Fold 결과에서 학습 전후 성능 개선량을 출력한다.

모델을 다시 불러오거나 재학습하지 않고 fold_result.json만 읽는다.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("precision", "recall", "f1", "accuracy", "loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-Fold 학습 전후 성능 비교")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/layoutlmv3_linear_probe"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.mean(float(row.get(metric, 0.0)) for row in rows)


def format_change(before: float, after: float, *, lower_is_better: bool = False) -> str:
    raw_delta = after - before
    improvement = -raw_delta if lower_is_better else raw_delta
    return f"{before:.4f} → {after:.4f} | 개선 {improvement:+.4f}"


def main() -> None:
    args = parse_args()
    result_paths = sorted((args.output_dir / "cv").glob("fold_*/fold_result.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"fold_result.json을 찾지 못했습니다: {args.output_dir / 'cv'}"
        )

    fold_results = [read_json(path) for path in result_paths]
    missing = [
        result.get("fold", "?")
        for result in fold_results
        if "baseline_validation" not in result or "best_validation" not in result
    ]
    if missing:
        raise ValueError(f"학습 전후 지표가 없는 Fold가 있습니다: {missing}")

    print("\n===== Fold별 학습 전후 F1 =====")
    fold_rows: list[dict[str, Any]] = []
    for result in fold_results:
        before = result["baseline_validation"]
        after = result["best_validation"]
        delta = float(after["f1"]) - float(before["f1"])
        print(
            f"Fold {int(result['fold'])}: "
            f"{float(before['f1']):.4f} → {float(after['f1']):.4f} "
            f"| ΔF1={delta:+.4f} | best epoch={int(result['best_epoch'])}"
        )
        fold_rows.append({
            "fold": int(result["fold"]),
            "best_epoch": int(result["best_epoch"]),
            "baseline": before,
            "trained": after,
            "delta_f1": delta,
        })

    baselines = [row["baseline"] for row in fold_rows]
    trained = [row["trained"] for row in fold_rows]
    averages: dict[str, dict[str, float]] = {}

    print("\n===== 5-Fold 평균 개선 =====")
    for metric in METRICS:
        before = mean_metric(baselines, metric)
        after = mean_metric(trained, metric)
        lower_is_better = metric == "loss"
        print(
            f"{metric.upper():<9} "
            f"{format_change(before, after, lower_is_better=lower_is_better)}"
        )
        averages[metric] = {
            "baseline": before,
            "trained": after,
            "improvement": before - after if lower_is_better else after - before,
        }

    entity_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"baseline": [], "trained": []}
    )
    for row in fold_rows:
        before_entities = row["baseline"].get("per_entity", {})
        after_entities = row["trained"].get("per_entity", {})
        for entity in set(before_entities) | set(after_entities):
            entity_values[entity]["baseline"].append(
                float(before_entities.get(entity, {}).get("f1", 0.0))
            )
            entity_values[entity]["trained"].append(
                float(after_entities.get(entity, {}).get("f1", 0.0))
            )

    entity_summary: dict[str, dict[str, float]] = {}
    if entity_values:
        print("\n===== Entity별 평균 F1 개선 =====")
        for entity in sorted(entity_values):
            before = statistics.mean(entity_values[entity]["baseline"])
            after = statistics.mean(entity_values[entity]["trained"])
            print(f"{entity:<14} {before:.4f} → {after:.4f} | Δ={after-before:+.4f}")
            entity_summary[entity] = {
                "baseline_f1": before,
                "trained_f1": after,
                "delta_f1": after - before,
            }

    summary = {
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "average": averages,
        "per_entity": entity_summary,
        "comparison_note": "baseline은 무작위 초기화 Linear head의 학습 전 성능입니다.",
    }
    output_path = args.output_dir / "improvement_summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"\n비교 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
