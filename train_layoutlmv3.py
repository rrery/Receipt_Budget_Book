import json
from pathlib import Path
from transformers import AutoProcessor
from PIL import Image


MODEL_NAME = "microsoft/layoutlmv3-base"

TRAIN_PATH = Path("data/layoutlm/train.json")
VAL_PATH = Path("data/layoutlm/validation.json")
TEST_PATH = Path("data/layoutlm/test.json")

IMAGE_DIR = Path("receipt")

LABEL_LIST = [
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
]


label2id = {
    label: index
    for index, label in enumerate(LABEL_LIST)
}

id2label = {
    index: label
    for index, label in enumerate(LABEL_LIST)
}

def load_json_dataset(path):
    """
    지정한 JSON Dataset 파일을 읽어서
    영수증 sample 목록으로 반환한다.
    """
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset

def load_processor():
    """
    사전학습된 LayoutLMv3 Processor를 불러온다.

    OCR 결과(words, bbox)를 이미 PaddleOCR로 생성했으므로
    Processor 내부 OCR 기능은 사용하지 않는다.
    """
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        apply_ocr=False
    )

    return processor

def convert_labels_to_ids(labels):
    """
    Dataset의 문자열 BIO label을
    모델 학습에 사용할 숫자 ID로 변환한다.
    """
    return [label2id[label] for label in labels]

def load_local_image(image_path):
    """
    프로젝트의 receipt 폴더에서 영수증 이미지를 불러와
    PIL RGB 이미지로 반환한다.
    """
    full_path = IMAGE_DIR / image_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {full_path}"
        )

    return Image.open(full_path).convert("RGB")

if __name__ == "__main__":
    print(f"Label 수: {len(LABEL_LIST)}")
    print(label2id)

    train_data = load_json_dataset(TRAIN_PATH)
    val_data = load_json_dataset(VAL_PATH)
    test_data = load_json_dataset(TEST_PATH)

    print(f"Train: {len(train_data)}개")
    print(f"Validation: {len(val_data)}개")
    print(f"Test: {len(test_data)}개")

    processor = load_processor()

    print("LayoutLMv3 Processor 로드 완료")

    sample = train_data[0]

    image = load_local_image(
        sample["image_path"]
    )

    word_labels = convert_labels_to_ids(
        sample["labels"]
    )

    encoding = processor(
        image,
        sample["words"],
        boxes=sample["boxes"],
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    print("\nProcessor 출력:")
    for key, value in encoding.items():
        print(key, value.shape)