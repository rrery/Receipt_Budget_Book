import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _box_center(box: List[List[float]]) -> tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _box_height(box: List[List[float]]) -> float:
    ys = [p[1] for p in box]
    return max(ys) - min(ys)


def _median(values: List[float], default: float = 30.0) -> float:
    if not values:
        return default
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def group_ocr_items_into_lines(
    ocr_result: List[Dict[str, Any]],
    y_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    PaddleOCR raw_result["ocr_result"]를 line 단위로 묶는다.

    Parameters
    ----------
    ocr_result : list[dict]
        ysz_ocr.py가 만든 raw_result["ocr_result"]
    y_threshold : float | None
        같은 줄로 볼 y축 허용 오차. None이면 box 높이 기준 자동 계산.

    Returns
    -------
    list[dict]
        각 줄 정보를 담은 리스트
    """
    items: List[Dict[str, Any]] = []

    for item in ocr_result:
        box = item["box"]
        cx, cy = _box_center(box)
        h = _box_height(box)

        items.append(
            {
                "id": item["id"],
                "text": item["text"],
                "score": item["score"],
                "box": box,
                "cx": cx,
                "cy": cy,
                "h": h,
            }
        )

    # 위 -> 아래, 같은 줄이면 왼쪽 -> 오른쪽
    items.sort(key=lambda x: (x["cy"], x["cx"]))

    # 자동 threshold 계산
    if y_threshold is None:
        heights = [it["h"] for it in items if it["h"] > 0]
        median_h = _median(heights, default=30.0)
        y_threshold = median_h * 0.45

    lines: List[Dict[str, Any]] = []

    for item in items:
        placed = False

        for line in lines:
            line_y = sum(it["cy"] for it in line["items"]) / len(line["items"])
            if abs(item["cy"] - line_y) <= y_threshold:
                line["items"].append(item)
                placed = True
                break

        if not placed:
            lines.append({"items": [item]})

    # line 후처리
    for idx, line in enumerate(lines, start=1):
        line["items"].sort(key=lambda x: x["cx"])

        line["line_no"] = idx
        line["y_center"] = sum(it["cy"] for it in line["items"]) / len(line["items"])
        line["texts"] = [it["text"] for it in line["items"]]
        line["joined_text"] = " ".join(line["texts"])

    return lines


def load_raw_result(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_line_result(line_result: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(line_result, f, ensure_ascii=False, indent=2)

def build_line_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    ysz_ocr.py의 raw_result 전체를 받아 line 단위 결과 JSON 구조로 변환한다.
    """
    lines = group_ocr_items_into_lines(raw_result["ocr_result"])

    return {
        "image_path": raw_result.get("image_path"),
        "line_result": [
            {
                "line_no": line["line_no"],
                "y_center": line["y_center"],
                "texts": line["texts"],
                "joined_text": line["joined_text"],
                "items": [
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "score": item["score"],
                        "box": item["box"],
                        "cx": item["cx"],
                        "cy": item["cy"],
                        "h": item["h"],
                    }
                    for item in line["items"]
                ],
            }
            for line in lines
        ],
    }

def run_line_grouping(ocr_raw_id: int, db_manager) -> Dict[str, Any]:
    result = db_manager.get_ocr_items(ocr_raw_id)
    items = result.data

    for item in items:
        if isinstance(item["box"], str):
            item["box"] = json.loads(item["box"])

    raw_result = {"image_path": None, "ocr_result": items}
    return build_line_result(raw_result)


if __name__ == "__main__":
    run_line_grouping()