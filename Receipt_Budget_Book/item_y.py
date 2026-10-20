# item_y.py
import json
import re

def parse_receipt_hybrid_engine(ocr_raw_id: int, db_manager):
    result = db_manager.get_ocr_items(ocr_raw_id)
    items = result.data

    for item in items:
        if isinstance(item["box"], str):
            item["box"] = json.loads(item["box"])

    ocr_result = items
    if not ocr_result:
        return []

    # 1. 영수증 내부 중심 가로 영역 탐지 (외곽 노트북 자판 노이즈 차단)
    cxs = [sum([p[0] for p in it["box"]])/4 for it in ocr_result]
    if not cxs: return []
    cxs_sorted = sorted(cxs)
    q1 = cxs_sorted[int(len(cxs_sorted) * 0.15)]
    q3 = cxs_sorted[int(len(cxs_sorted) * 0.85)]
    margin = (q3 - q1) * 0.2
    receipt_min_x = max(0, q1 - margin)
    receipt_max_x = q3 + margin

    # 2. 유효 데이터 필터링
    items = []
    store_name = ""
    
    for idx, it in enumerate(ocr_result):
        box = it["box"]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx = sum(xs) / 4
        cy = sum(ys) / 4
        if cx < receipt_min_x or cx > receipt_max_x: 
            continue
            
        val = it["text"].strip()
        
        # 상호명 추적
        if "가맹점명" in val or "상호" in val:
            store_name = val.replace("가맹점명", "").replace("가맹점", "").replace("상호명", "").replace("상호", "").replace(":", "").strip()
        elif idx == 0 or (cy < 350 and not store_name and ("식당" in val or "마트" in val or "카페" in val or "점" in val or "/" in val)):
            store_name = val.split("/")[0].strip() if "/" in val else val

        clean_name = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\(\)\-=\s]', '', val).strip()
        clean_num = "".join(filter(str.isdigit, val.replace(",", "").replace("원", "")))
        
        items.append({
            "text": val, "clean_name": clean_name, "clean_num": clean_num,
            "cx": cx, "cy": cy, "h": max(ys) - min(ys)
        })

    if not store_name: store_name = "일반 가맹점"
    store_name = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\s]', '', store_name).strip()
    
    # 오직 Y축(높이) 순서대로 텍스트 정렬
    items.sort(key=lambda x: x["cy"])

    # 💡 [핵심 수선 점] 줄바꿈 평행선 범위를 정교하게 제어하여 헤더와 1번 품목의 융합을 차단합니다.
    avg_h = sum([it["h"] for it in items]) / len(items) if items else 25
    y_threshold = avg_h * 0.38  # 줄이 아주 미세하게만 달라도 무조건 별개 행으로 분리

    lines = []
    current_line = []
    current_y = -1
    for it in items:
        if current_y == -1 or abs(it["cy"] - current_y) <= y_threshold:
            current_line.append(it)
            current_y = it["cy"] if current_y == -1 else (current_y + it["cy"]) / 2
        else:
            lines.append(current_line)
            current_line = [it]
            current_y = it["cy"]
    if current_line: lines.append(current_line)

    parsed_items = []
    start_signals = ["상품명", "메뉴명", "단가", "수량", "금액", "상 품 명"]
    absolute_end_signals = ["결제대상금액", "사용포인트", "사용_포인트", "머니사용", "카드결제", "받은금액", "바코드", "과세물품가액", "총구매액", "신용카드"]

    is_menu_zone = False
    grand_total_price = 0

    # 3. 1차 정밀 세부 품목 스캔
    for line_items in lines:
        line_items.sort(key=lambda x: x["cx"])
        joined_text = "".join([it["text"] for it in line_items]).replace(" ", "")
        
        if "합계" in joined_text or "결제금액" in joined_text or "총구매액" in joined_text:
            nums = ["".join(filter(str.isdigit, it["text"])) for it in line_items]
            nums = [int(n) for n in nums if n]
            if nums: grand_total_price = nums[-1]

        # 영수증 품목 영역 진입 판단
        has_start_piece = any(s in joined_text for s in start_signals)
        if has_start_piece and not is_menu_zone:
            is_menu_zone = True
            continue

        if is_menu_zone and any(e in joined_text for e in absolute_end_signals): 
            break
            
        if not is_menu_zone: 
            continue

        item_name_parts = []
        numbers_in_row = []

        for it in line_items:
            val = it["text"].strip()
            clean_txt = val.replace(",", "").replace("원", "").replace("（", "").replace("）", "").replace("(", "").replace(")", "").replace(".", "").strip()
            clean_num = "".join(filter(str.isdigit, clean_txt))

            if not clean_num or len(clean_num) != len(clean_txt.replace(" ", "")):
                if val.startswith("▶") or val.startswith("="): val = val[1:].strip()
                clean_name = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\(\)\-=\s]', '', val).strip()
                if clean_name and clean_name not in start_signals: 
                    item_name_parts.append(clean_name)
            else:
                num_val = int(clean_num)
                if len(clean_num) >= 7 or (len(clean_num) >= 5 and num_val < 10000) or "/" in val or ":" in val: 
                    continue
                numbers_in_row.append(num_val)

        final_name = " ".join(item_name_parts).strip()
        
        # 정산용 필터링 단어 격리
        if any(w in final_name for w in ["총품목", "면세", "과세", "부가세", "합계", "소계", "물품", "금액", "결제내역", "총구매액"]): 
            continue
        if "포장" in final_name or final_name.startswith("-") or (len(numbers_in_row) >= 1 and numbers_in_row[-1] == 0): 
            continue

        if final_name and len(numbers_in_row) >= 2:
            if len(numbers_in_row) > 3: 
                numbers_in_row = numbers_in_row[:3]

            total_price = numbers_in_row[-1]
            quantity = 1
            unit_price = total_price

            if len(numbers_in_row) == 3:
                unit_price = numbers_in_row[0]
                quantity = numbers_in_row[1]
                total_price = numbers_in_row[2]
            elif len(numbers_in_row) == 2:
                val1, val2 = numbers_in_row[0], numbers_in_row[1]
                if val1 < 10:
                    quantity = val1
                    total_price = val2
                    unit_price = total_price // quantity
                else:
                    unit_price = val1
                    total_price = val2

            if quantity >= 100 or total_price < 100 or quantity <= 0: 
                continue
                
            if 1 < len(final_name) < 30:
                parsed_items.append({
                    "item_name": final_name, "quantity": quantity, "unit_price": unit_price, "price": total_price
                })

    # 비상용 단일 카드 전표 레이어
    if not parsed_items and grand_total_price >= 100:
        parsed_items.append({
            "item_name": f"{store_name} 이용대금", "quantity": 1, "unit_price": grand_total_price, "price": grand_total_price
        })

    return parsed_items

def main():
    extracted_items = parse_receipt_hybrid_engine()
    print("\n==================================================")
    print(" 🎯 [최종 Y축 행 단위 파서 결과 리스트] ")
    print("==================================================")
    if extracted_items:
        for idx, item in enumerate(extracted_items, start=1):
            print(f" [{idx}] 품목명: {item['item_name']}\n     -> 수량: {item['quantity']}개\n     -> 단가: {item['unit_price']:,}원\n     -> 금액: {item['price']:,}원")
            print("-" * 50)
        print(f"\n[Success] 오리지널 순수 행 단위 파싱 완료! 총 {len(extracted_items)}개 품목 도출.")
    else:
        print("[Fail] 품목을 추출하는 데 실패했습니다.")

if __name__ == "__main__":
    main()