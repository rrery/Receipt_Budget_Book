import json
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageDraw

from database import SupabaseManager

DATASET_PATH = Path("data/layoutlm/dataset.json")


def load_dataset():
    """
    생성된 dataset.json 파일을 읽어서
    영수증 sample 목록을 반환한다.
    """
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def get_sample_by_id(dataset, ocr_raw_id):
    """
    Dataset에서 특정 ocr_raw_id를 가진
    영수증 sample 하나를 찾아 반환한다.
    """
    for sample in dataset:
        if sample["ocr_raw_id"] == ocr_raw_id:
            return sample

    return None

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

def denormalize_bbox(box, image_width, image_height):
    """
    LayoutLM용 0~1000 범위의 bbox를
    실제 이미지의 픽셀 좌표로 변환한다.

    입력:
        box = [x_min, y_min, x_max, y_max]

    출력:
        실제 이미지 크기에 대응하는 픽셀 좌표
    """
    x_min, y_min, x_max, y_max = box

    return [
        int(x_min * image_width / 1000),
        int(y_min * image_height / 1000),
        int(x_max * image_width / 1000),
        int(y_max * image_height / 1000),
    ]

def draw_labeled_boxes(image, sample):
    """
    Dataset의 bbox를 실제 이미지 좌표로 복원한 뒤,
    O가 아닌 학습 대상 token의 bbox와 label을 이미지 위에 표시한다.
    """
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    for word, box, label in zip(
        sample["words"],
        sample["boxes"],
        sample["labels"]
    ):
        if label == "O":
            continue

        original_box = denormalize_bbox(
            box,
            image_width,
            image_height
        )

        draw.rectangle(
            original_box,
            outline="red",
            width=2
        )

        draw.text(
            (original_box[0], original_box[1]),
            label,
            fill="red"
        )

    return image

if __name__ == "__main__":
    dataset = load_dataset()

    sample = get_sample_by_id(dataset, 15)

    if sample:
        print(f"ocr_raw_id: {sample['ocr_raw_id']}")
        print(f"image_path: {sample['image_path']}")
        print(f"token 수: {len(sample['words'])}")

        db = SupabaseManager()
        supabase = db.supabase

        image = download_image(
            supabase,
            sample["image_path"]
        )

        print(f"이미지 크기: {image.size}")

        image_width, image_height = image.size

        for index in range(min(3, len(sample["boxes"]))):
            normalized_box = sample["boxes"][index]

            original_box = denormalize_bbox(
                normalized_box,
                image_width,
                image_height
            )

            print(
                f"{index}: "
                f"{sample['words'][index]} / "
                f"{sample['labels'][index]} / "
                f"{normalized_box} -> {original_box}"
            )

        inspected_image = draw_labeled_boxes(
            image.copy(),
            sample
        )

        inspected_image.show()

    else:
        print("해당 ocr_raw_id를 찾을 수 없습니다.")