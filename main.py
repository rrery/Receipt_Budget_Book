# main.py
import os
import requests
from database import SupabaseManager
from ysz_ocr import run_ocr

def main():
    # 1. Supabase 매니저 생성
    db_manager = SupabaseManager()
    
    print("1. Supabase Storage에서 이미지 목록을 조회하는 중...")
    images = db_manager.get_image_list(bucket_name="image")
    
    if not images:
        print("버킷에 이미지 파일이 없습니다. Supabase에 이미지를 먼저 업로드해 주세요.")
        return
        
    # 테스트를 위해 스토리지의 첫 번째 이미지 선택
    target_image = images[0]['name']
    print(f"선택된 이미지: {target_image}")
    
    # 2. 이미지의 Public URL 가져오기
    image_url = db_manager.get_image_url(target_image, bucket_name="image")
    print(f"이미지 URL: {image_url}")
    
    # 3. PaddleOCR은 인터넷 주소(URL)를 바로 읽지 못하므로, 로컬에 임시 다운로드
    local_temp_path = "temp_receipt.jpg"
    print("이미지 다운로드 중...")
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(local_temp_path, "wb") as f:
            f.write(response.content)
    else:
        print("이미지 다운로드 실패")
        return

    try:
        print("2. PaddleOCR 분석 시작...")
        # ysz_ocr.py에 있는 run_ocr 함수를 실행시켜 이미지 분석
        ocr_raw = run_ocr(local_temp_path, "receipt_ocr_raw.json")
        
        # 4. DB에 넣기 위해 텍스트 조각들을 하나로 합침
        all_text_list = [item["text"] for item in ocr_raw["ocr_result"]]
        all_text = " ".join(all_text_list)
        
        # 신뢰도 평균 계산
        scores = [item["score"] for item in ocr_raw["ocr_result"]]
        avg_confidence = sum(scores) / len(scores) if scores else 0
        
        print("3. OCR 결과를 Supabase DB에 저장 중...")
        # database.py의 insert_ocr_result 함수를 실행시켜 DB에 저장
        db_manager.insert_ocr_result(
            image_name=target_image,
            all_text=all_text,
            confidence=avg_confidence,
            total_amount=0
        )
        print("🎉 모든 프로세스가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        
    finally:
        # 작업이 끝나면 임시로 받았던 이미지 파일 삭제
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)

if __name__ == "__main__":
    main()