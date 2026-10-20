import re
import json
from datetime import date
from typing import Any, Dict, List, Optional

from ocr_linebyline import build_line_result

# --------------------------날짜-------------------------------------------

DATE_PATTERNS = [
    r"\b(\d{4})[-./](\d{1,2})[-./](\d{1,2})\b",
    r"\b(\d{2})[-./](\d{1,2})[-./](\d{1,2})\b",
]

def _is_valid_date_parts(year: str, month: str, day: str) -> bool:
    y = int(year) + (2000 if len(year) == 2 else 0)
    m = int(month)
    d = int(day)

    # 연도는 2자리 또는 4자리만 허용
    if len(year) not in (2, 4):
        return False

    # 월/일 범위 체크
    if not (1 <= m <= 12):
        return False
    if not (1 <= d <= 31):
        return False

    try:
        date(y, m, d)
        return True
    except ValueError:
        return False

def _find_valid_date(text: str, patterns: List[str]) -> Optional[str]:
    # 1) 날짜 바로 뒤에 시간이 붙은 경우 띄워주기
    # 예: 2026-03-1913:58  ->  2026-03-19 13:58
    normalized = re.sub(
        r"(\d{2,4}[-./]\d{1,2}[-./]\d{1,2})(\d{1,2}:\d{2}(?::\d{2})?)",
        r"\1 \2",
        text
    )

    # 2) 정규식으로 날짜 후보 찾기
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            year, month, day = match.groups()
            if _is_valid_date_parts(year, month, day):
                normalized_year = int(year) + (2000 if len(year) == 2 else 0)
                return f"{normalized_year:04d}-{month.zfill(2)}-{day.zfill(2)}"

    return None

def extract_date(line_result):
    for line in line_result["line_result"]:
        found = _find_valid_date(line["joined_text"], DATE_PATTERNS)
        if found:
            return found
    return None



#-------------------------------------합계 금액--------------------------------------

TOTAL_KEYWORDS = [
    r"합\s*계(?:\s*금액)?",
    r"총\s*계|총\s*액",
    r"(?:받(?:은|을)?|밖(?:은|을)?)\s*(?:금|큼)\s*액",
    r"청구\s*(?:금|큼)\s*액",
    r"결제\s*(?:금|큼)\s*액",
    r"판매\s*(?:금|큼)\s*액",
    r"승인\s*(?:금|큼)\s*액",
    r"카드\s*(?:금|큼)\s*액",
    r"총\s*결제\s*(?:금|큼)\s*액",
    r"총\s*구매\s*(?:금|큼)\s*액",
]

def extract_total_amount(line_result: Dict[str, Any]) -> Optional[int]:
    lines = line_result["line_result"]
    if not lines:
        return None

    max_y = max(line["y_center"] for line in lines)
    bottom_limit = max_y * 0.5

    candidates = []

    for line in lines:
        if line["y_center"] < bottom_limit:
            continue

        text = line["joined_text"]

        # 받은 현금과 거스름돈은 지출 합계가 아니다.
        if re.search(r"받\s*은|거\s*스\s*름|잔\s*돈", text):
            continue

        # 정규식 패턴으로 키워드 검사
        if any(re.search(pattern, text) for pattern in TOTAL_KEYWORDS):
            nums = re.findall(r"\d[\d,\.]*", text)
            if nums:
                raw = nums[-1].replace(",", "").replace(".", "")
                if raw.isdigit():
                    candidates.append((line["y_center"], int(raw), text))

    if not candidates:
        return None

    # 가장 아래쪽 합계 관련 줄을 우선
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


#---------------------------------상호명---------------------------------------------

STORE_BAD_KEYWORDS = [  # 상호명 제외 정보 필터링
    "사업자", "대표자", "TEL", "전화", "주소",
    "가맹점주소", "여신금융", "EasyCheck", "KICC",
    "영수증", "발행일"
]

def extract_store_name(line_result: Dict[str, Any]) -> Optional[str]:
    lines = line_result["line_result"]
    if not lines:
        return None

    # 상품 표의 헤더/품목/하단 안내문이 상호명보다 길어도 상호로 선택하지 않는다.
    header_idx, _ = find_item_header_index(line_result)
    if header_idx is not None and header_idx > 0:
        lines = lines[:header_idx]

    max_y = max(line["y_center"] for line in lines)
    # top_limit = max_y * 0.30 #일단 상호명이 어디서 나올지 모르니 top_limit은 주석 처리
    candidates = []

    for line in lines:
        #if line["y_center"] > top_limit:
        #    continue

        text = line["joined_text"].strip()
        if not text:
            continue

        if any(bad in text for bad in STORE_BAD_KEYWORDS):
            continue

        if _find_valid_date(text, DATE_PATTERNS):
            continue

        hangul_count = len(re.findall(r"[가-힣A-Za-z]", text))
        digit_count = len(re.findall(r"\d", text))

        if hangul_count == 0:
            continue

        score = 0
        score += hangul_count * 2
        score -= digit_count
        score -= int(line["y_center"] * 0.01)  # 위쪽일수록 유리

        candidates.append((score, text))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    best = candidates[0][1]

    # [매장명]주꾸미 잘하는집 같은 케이스 정리
    best = re.sub(r"^\[(매장명|상호명|가맹점명)\]\s*", "", best).strip()
    return best.strip()


#---------------------------------품목 별 정보-------------------------------------------------


def find_item_header_index(line_result: Dict[str, Any]) -> tuple[Optional[int], Optional[Dict[str, bool]]]:
    header_keywords = ["상품명", "상품", "품명", "메뉴", "메뉴명", "제품", "제품명", "PDS"]
    price_keywords = ["단가", "가격", "판매가"]
    qty_keywords = ["수량", "수", "개수", "수량(개)"]
    amount_keywords = ["금액", "합계", "판매금액"]

    best_idx = None
    best_score = 0
    best_header_info = None

    for i, line in enumerate(line_result["line_result"]):
        text = line["joined_text"].replace(" ", "")

        has_name = any(k in text for k in header_keywords)
        has_price = any(k in text for k in price_keywords)
        has_qty = any(k in text for k in qty_keywords)
        has_amount = any(k in text for k in amount_keywords)

        score = 0
        if has_name:
            score += 2
        if has_price:
            score += 1
        if has_qty:
            score += 1
        if has_amount:
            score += 1

        if score >= 3 and score > best_score:
            best_score = score
            best_idx = i
            best_header_info = {
                "has_name": has_name,
                "has_price": has_price,
                "has_qty": has_qty,
                "has_amount": has_amount,
            }

    return best_idx, best_header_info

def _clean_number_token(token: str) -> Optional[int]:
    cleaned = token.replace(",", "").replace(".", "").strip()
    if re.fullmatch(r"\d+", cleaned):
        return int(cleaned)
    return None


def parse_item_line(line: Dict[str, Any], header_info: Dict[str, bool]) -> Optional[Dict[str, Any]]:
    texts = line["texts"]

    numeric_tokens = []
    text_tokens = []

    for t in texts:
        value = _clean_number_token(t)
        if value is not None:
            numeric_tokens.append(value)
        else:
            text_tokens.append(t)

    # 상품명 후보가 없으면 품목 줄 아님
    if not text_tokens:
        return None

    item_name = " ".join(text_tokens).strip()
    nums = numeric_tokens

    unit_price = None
    quantity = None
    line_total = None

    has_price = header_info.get("has_price", False)
    has_qty = header_info.get("has_qty", False)
    has_amount = header_info.get("has_amount", False)

    # 1) 단가 / 수량 / 금액 다 있는 헤더
    if has_price and has_qty and has_amount:
        if len(nums) >= 3:
            unit_price, quantity, line_total = nums[0], nums[1], nums[2]
        else:
            return None

    # 2) 수량 / 금액만 있는 헤더 (예: 메뉴 수량 금액)
    elif (not has_price) and has_qty and has_amount:
        if len(nums) >= 2:
            quantity, line_total = nums[0], nums[1]
        else:
            return None

    # 3) 단가 / 금액만 있는 헤더
    elif has_price and (not has_qty) and has_amount:
        if len(nums) >= 2:
            unit_price, line_total = nums[0], nums[1]
        else:
            return None

    # 4) 금액만 있는 헤더
    elif (not has_price) and (not has_qty) and has_amount:
        if len(nums) >= 1:
            line_total = nums[0]
        else:
            return None

    # 5) 단가 / 수량만 있고 금액은 없는 경우
    elif has_price and has_qty and (not has_amount):
        if len(nums) >= 2:
            unit_price, quantity = nums[0], nums[1]
            line_total = unit_price * quantity
        else:
            return None

    # 6) 그 외 애매한 경우는 일단 보수적으로 탈락
    else:
        return None

    return {
        "name": item_name,
        "quantity": quantity,
        "unit_price": unit_price,
        "line_total": line_total,
        "raw_line": line["joined_text"]
    }


def extract_items(line_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines = line_result["line_result"]
    header_idx, header_info = find_item_header_index(line_result)

    if header_idx is None or header_info is None:
        return []

    items = []

    for line in lines[header_idx + 1:]:
        text = line["joined_text"]

        if any(k in text for k in [
            "합계", "합계금액", "총액", "받을금액",
            "부가세", "신용카드", "카드", "승인", "결제"
        ]):
            break

        parsed = parse_item_line(line, header_info)
        if parsed:
            items.append(parsed)

    return items

#--------------------------------------------최종 실행 파트--------------------------------------------------

def parse_receipt(line_result: Dict[str, Any]) -> Dict[str, Any]:
    parsed = {
        "image_path": line_result.get("image_path"),
        "store_name": extract_store_name(line_result),
        "purchase_date": extract_date(line_result),
        "items": extract_items(line_result),
        "total_amount": extract_total_amount(line_result),
        "line_result": line_result["line_result"],
    }
    print("\n===== 파싱 결과 미리보기 =====")
    print(f"상호명: {parsed['store_name']}")
    print(f"날짜: {parsed['purchase_date']}")
    print(f"합계금액: {parsed['total_amount']}")
    print("품목들:")
    for item in parsed["items"]:
        print(f"  - {item['name']} / 수량: {item['quantity']} / 단가: {item['unit_price']} / 금액: {item['line_total']}")
    print("=" * 30)
    return parsed


def load_raw_result(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parsed_result(parsed_result: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_result, f, ensure_ascii=False, indent=2)


def run_receipt_parser(ocr_raw_id: int, db_manager) -> Dict[str, Any]:
    result = db_manager.get_ocr_items(ocr_raw_id)
    items = result.data

    for item in items:
        if isinstance(item["box"], str):
            item["box"] = json.loads(item["box"])

    raw_result = {"image_path": None, "ocr_result": items}
    line_result = build_line_result(raw_result)
    return parse_receipt(line_result)
