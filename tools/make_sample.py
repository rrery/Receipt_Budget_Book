"""Synthetic receipt for connection testing, never used as training evidence."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, 32)
    title = ImageFont.truetype(font_path, 46)
    draw.text((80, 90), "차곡 테스트 카페", font=title, fill="black")
    for y, text in [
        (210, "2026-10-17 12:30"),
        (300, "상품명       단가      수량      금액"),
        (400, "아메리카노   4,000       2      8,000"),
        (500, "쿠키         2,000       1      2,000"),
        (680, "합계                          10,000"),
        (800, "카드 결제금액                10,000"),
        (1100, "테스트 전용 · 실제 구매 영수증 아님"),
    ]:
        draw.text((70, y), text, font=font, fill="black")
    (ROOT / "runtime").mkdir(exist_ok=True)
    image.save(ROOT / "runtime" / "test_receipt.png")


if __name__ == "__main__":
    main()
