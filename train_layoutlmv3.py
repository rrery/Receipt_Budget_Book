import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import torch
from transformers import AutoProcessor, AutoModelForTokenClassification

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

def load_model():
    """
    사전학습된 LayoutLMv3 모델을 Token Classification 용도로 불러온다.

    영수증 OCR token을 BIO label로 분류할 수 있도록
    프로젝트의 label 개수와 label-ID 매핑 정보를 설정한다.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    return model

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

class ReceiptDataset(Dataset):
    """
    영수증 sample 목록을 PyTorch Dataset 형태로 변환한다.

    각 sample의 이미지, OCR text, bbox, label을 불러와
    LayoutLMv3 Processor를 통해 모델 입력 Tensor로 변환한다.
    """

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = load_local_image(
            sample["image_path"]
        )

        word_labels = convert_labels_to_ids(
            sample["labels"]
        )

        encoding = self.processor(
            image,
            sample["words"],
            boxes=sample["boxes"],
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        return {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

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

    train_dataset = ReceiptDataset(
        train_data,
        processor
    )

    val_dataset = ReceiptDataset(
        val_data,
        processor
    )

    test_dataset = ReceiptDataset(
        test_data,
        processor
    )

    print(f"\nPyTorch Train Dataset: {len(train_dataset)}개")
    print(f"PyTorch Validation Dataset: {len(val_dataset)}개")
    print(f"PyTorch Test Dataset: {len(test_dataset)}개")

    sample_encoding = train_dataset[0]

    print("\nTrain Dataset 첫 sample:")
    for key, value in sample_encoding.items():
        print(key, value.shape)

        # 사전학습된 LayoutLMv3 모델을 불러온다.
    model = load_model()

    print("\nLayoutLMv3 모델 로드 완료")

    # Dataset에서 가져온 단일 sample에 batch 차원을 추가한다.
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample_encoding.items()
    }

    # 실제 학습은 하지 않고 Forward만 실행하여
    # 모델 출력과 Loss가 정상적으로 계산되는지 확인한다.
    with torch.no_grad():
        outputs = model(**batch)

    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss.item()}")

    # 실제 학습이 가능하도록 모델을 학습 모드로 전환한다.
    model.train()

    # 모델의 weight를 업데이트할 optimizer를 생성한다.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5
    )

    print("\n간단한 학습 Loop 시작")

    for step in range(3):
        # Train Dataset에서 sample 하나를 가져온다.
        sample_encoding = train_dataset[step]

        # 모델 입력을 위해 batch 차원을 추가한다.
        batch = {
            key: value.unsqueeze(0)
            for key, value in sample_encoding.items()
        }

        # 이전 step에서 계산된 gradient를 초기화한다.
        optimizer.zero_grad()

        # Forward
        outputs = model(**batch)

        # 현재 예측과 정답 사이의 Loss
        loss = outputs.loss

        # Backward
        loss.backward()

        # 계산된 gradient를 이용해 모델 weight 수정
        optimizer.step()

        print(
            f"Step {step + 1} | "
            f"Loss: {loss.item():.4f}"
        )