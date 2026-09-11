import json
from pathlib import Path
from transformers import AutoProcessor

from io import BytesIO

from PIL import Image

from database import SupabaseManager

MODEL_NAME = "microsoft/layoutlmv3-base"

TRAIN_PATH = Path("data/layoutlm/train.json")
VAL_PATH = Path("data/layoutlm/validation.json")
TEST_PATH = Path("data/layoutlm/test.json")

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

def download_image(supabase, image_path):
    """
    Supabase Storage의 images 버킷에서 영수증 이미지를 다운로드하고
    PIL Image 객체로 변환한다.
    """
    image_bytes = (
        supabase.storage
        .from_("images")
        .download(image_path)
    )

    return Image.open(BytesIO(image_bytes)).convert("RGB")

def convert_labels_to_ids(labels):
    """
    Dataset의 문자열 BIO label을
    모델 학습에 사용할 숫자 ID로 변환한다.
    """
    return [label2id[label] for label in labels]



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

    # 여기부터 추가
    db = SupabaseManager()
    supabase = db.supabase

    sample = train_data[0]

    image = download_image(
        supabase,
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