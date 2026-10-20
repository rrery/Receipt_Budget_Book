from backend.pipeline import LEGACY
from receipt_parser import _find_valid_date, DATE_PATTERNS, extract_total_amount, parse_item_line, extract_store_name


def test_dates_are_real_and_normalized():
    assert _find_valid_date('2026-02-31', DATE_PATTERNS) is None
    assert _find_valid_date('26-03-19', DATE_PATTERNS) == '2026-03-19'
    assert _find_valid_date('2026-03-1913:58', DATE_PATTERNS) == '2026-03-19'
    assert _find_valid_date('2024-02-29', DATE_PATTERNS) == '2024-02-29'


def test_total_excludes_tendered_cash():
    assert extract_total_amount({'line_result': [{'y_center': 100, 'joined_text': '합계 10,000'}, {'y_center': 120, 'joined_text': '받은금액 20,000'}]}) == 10000


def test_unit_times_quantity():
    result = parse_item_line({'texts': ['커피', '3000', '2'], 'joined_text': '커피 3000 2'}, {'has_price': True, 'has_qty': True, 'has_amount': False})
    assert result['line_total'] == 6000


def test_item_header_and_footer_are_not_store_names():
    lines = [{'joined_text': text, 'y_center': y} for text, y in [('차곡 테스트 카페', 90), ('2026-10-17', 210), ('상품명 단가 수량 금액', 300), ('구매해 주셔서 대단히 감사합니다', 1100)]]
    assert extract_store_name({'line_result': lines}) == '차곡 테스트 카페'
