"""LayoutLMv3 학습·검증·추론에서 공통으로 사용하는 BIO 라벨 정의."""

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

ALLOWED_LABELS = set(LABEL_LIST)

# 과거 라벨링 과정에서 사용된 명칭을 현재 표준 라벨로 통일한다.
LABEL_ALIASES = {
    "TOTAL_AMOUNT": "B-TOTAL_AMOUNT",
    "B-ITEM_PRICE": "B-UNIT_PRICE",
    "I-ITEM_PRICE": "I-UNIT_PRICE",
}


def normalize_label(label: str) -> str:
    """공백과 과거 별칭을 제거하여 표준 BIO 라벨을 반환한다."""
    normalized = label.strip()
    return LABEL_ALIASES.get(normalized, normalized)
