import json
from pathlib import Path


DATASET_PATH = Path("data/layoutlm/dataset.json")

ALLOWED_LABELS = {
    "O",

    "B-STORE_NAME",
    "I-STORE_NAME",

    "B-ADDRESS",
    "I-ADDRESS",

    "B-DATE",
    "I-DATE",

    "B-ITEM_NAME",
    "I-ITEM_NAME",

    "B-QUANTITY",
    "I-QUANTITY",

    "B-UNIT_PRICE",
    "I-UNIT_PRICE",

    "B-ITEM_TOTAL",
    "I-ITEM_TOTAL",

    "B-TOTAL_AMOUNT",
    "I-TOTAL_AMOUNT",
}

def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset

def check_duplicate_ids(dataset):
    """
    Dataset 안에서 ocr_raw_id가 중복되어 있는지 확인한다.

    하나의 영수증은 하나의 sample로만 존재해야 하므로,
    같은 ocr_raw_id가 두 번 이상 있으면 중복 데이터로 판단한다.
    """
    ids = [sample["ocr_raw_id"] for sample in dataset]

    duplicate_ids = {
        ocr_raw_id
        for ocr_raw_id in ids
        if ids.count(ocr_raw_id) > 1
    }

    if duplicate_ids:
        print(f"[ERROR] 중복된 ocr_raw_id: {sorted(duplicate_ids)}")
    else:
        print("ocr_raw_id 중복 없음")

def check_sample_lengths(dataset):
    """
    각 영수증 sample에서 words, boxes, labels의 개수가
    모두 동일한지 확인한다.

    각 OCR token은 하나의 bbox와 하나의 label을 가져야 하므로
    세 리스트의 길이가 다르면 Dataset 구조 오류로 판단한다.
    """
    invalid_samples = []

    for sample in dataset:
        words_len = len(sample["words"])
        boxes_len = len(sample["boxes"])
        labels_len = len(sample["labels"])

        if not (words_len == boxes_len == labels_len):
            invalid_samples.append({
                "ocr_raw_id": sample["ocr_raw_id"],
                "words": words_len,
                "boxes": boxes_len,
                "labels": labels_len,
            })

    if invalid_samples:
        print("[ERROR] words / boxes / labels 길이가 다른 sample이 있습니다.")

        for sample in invalid_samples:
            print(sample)
    else:
        print("words / boxes / labels 길이 정상")

def check_bbox_range(dataset):
    """
    모든 bbox 좌표가 LayoutLM 입력 범위인 0~1000 안에 있는지 확인한다.

    bbox는 [x_min, y_min, x_max, y_max] 형태이며,
    하나라도 범위를 벗어나면 잘못된 좌표로 판단한다.
    """
    invalid_boxes = []

    for sample in dataset:
        ocr_raw_id = sample["ocr_raw_id"]

        for index, box in enumerate(sample["boxes"]):
            if len(box) != 4:
                invalid_boxes.append({
                    "ocr_raw_id": ocr_raw_id,
                    "index": index,
                    "box": box,
                    "reason": "bbox 길이가 4가 아님",
                })
                continue

            if not all(0 <= value <= 1000 for value in box):
                invalid_boxes.append({
                    "ocr_raw_id": ocr_raw_id,
                    "index": index,
                    "box": box,
                    "reason": "0~1000 범위를 벗어남",
                })

    if invalid_boxes:
        print("[ERROR] 잘못된 bbox가 있습니다.")

        for box_info in invalid_boxes:
            print(box_info)
    else:
        print("bbox 범위 정상")

def check_labels(dataset):
    """
    Dataset에 존재하는 모든 label 종류를 확인하고,
    앞뒤 공백이나 개행 문자가 포함된 이상 label이 있는지 검사한다.
    """
    labels = set()
    suspicious_labels = set()

    for sample in dataset:
        for label in sample["labels"]:
            labels.add(label)

            if label != label.strip():
                suspicious_labels.add(label)

    print("\n사용 중인 label 목록:")
    for label in sorted(labels):
        print(f"- {repr(label)}")

    if suspicious_labels:
        print("\n[ERROR] 공백 또는 개행이 포함된 label이 있습니다.")

        for label in sorted(suspicious_labels):
            print(f"- {repr(label)}")
    else:
        print("\nlabel 문자열 정상")

def check_allowed_labels(dataset):
    """
    Dataset의 모든 label이 미리 정의한 허용 Label 목록에 포함되는지 확인한다.

    BIO 형식이 빠졌거나 오타가 있는 label이 존재하면
    잘못된 label로 판단하여 출력한다.
    """
    invalid_labels = set()

    for sample in dataset:
        for label in sample["labels"]:
            if label not in ALLOWED_LABELS:
                invalid_labels.add(label)

    if invalid_labels:
        print("\n[ERROR] 허용되지 않은 label이 있습니다.")

        for label in sorted(invalid_labels):
            print(f"- {repr(label)}")
    else:
        print("\n모든 label이 허용된 목록에 포함되어 있습니다.")

if __name__ == "__main__":
    dataset = load_dataset()

    print(f"Dataset sample 수: {len(dataset)}")

    check_duplicate_ids(dataset)
    check_sample_lengths(dataset)
    check_bbox_range(dataset)
    check_labels(dataset)
    check_allowed_labels(dataset)