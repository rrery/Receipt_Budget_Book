import json
import random
from pathlib import Path


DATASET_PATH = Path("data/layoutlm/dataset.json")

RANDOM_SEED = 42

TRAIN_PATH = Path("data/layoutlm/train.json")
VAL_PATH = Path("data/layoutlm/validation.json")
TEST_PATH = Path("data/layoutlm/test.json")

def load_dataset():
    """
    전체 영수증 Dataset을 JSON 파일에서 읽어온다.
    각 원소는 영수증 한 장에 해당하는 sample이다.
    """
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def shuffle_dataset(dataset):
    """
    영수증 sample의 순서를 무작위로 섞는다.

    RANDOM_SEED를 고정하여 코드를 다시 실행해도
    동일한 Train / Validation / Test 분할 결과를 얻도록 한다.
    """
    shuffled_dataset = dataset.copy()

    random.seed(RANDOM_SEED)
    random.shuffle(shuffled_dataset)

    return shuffled_dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    영수증 단위 Dataset을 Train / Validation / Test로 분리한다.

    train_ratio와 val_ratio를 기준으로 Train과 Validation 크기를 정하고,
    나머지 sample은 Test에 배정한다.
    """
    total_count = len(dataset)

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    train_data = dataset[:train_count]
    val_data = dataset[train_count:train_count + val_count]
    test_data = dataset[train_count + val_count:]

    return train_data, val_data, test_data

def save_split_dataset(data, output_path):
    """
    분리된 Dataset을 JSON 파일로 저장한다.

    output_path의 상위 폴더가 없으면 자동으로 생성한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    dataset = load_dataset()

    print(f"전체 Dataset: {len(dataset)}개")

    dataset = shuffle_dataset(dataset)

    train_data, val_data, test_data = split_dataset(dataset)

    print(f"Train: {len(train_data)}개")
    print(f"Validation: {len(val_data)}개")
    print(f"Test: {len(test_data)}개")

    save_split_dataset(train_data, TRAIN_PATH)
    save_split_dataset(val_data, VAL_PATH)
    save_split_dataset(test_data, TEST_PATH)

    print("Train / Validation / Test 저장 완료")