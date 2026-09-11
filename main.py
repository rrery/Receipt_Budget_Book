# main.py
import os
import requests
import cv2  
import numpy as np
from database import SupabaseManager
from ysz_ocr import run_ocr, PaddleOCR
from receipt_parser import run_receipt_parser

# 이미지 전처리 함수
def auto_perspective_transform(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        return False
        
    orig = img.copy()
    
    # 1. 처리 속도와 정확도를 위해 높이 800 기준으로 이미지 크기 축소
    r = 800.0 / img.shape[0]
    dim = (int(img.shape[1] * r), 800)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # 2. 에지(테두리) 검출을 위한 전처리
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    # 3. 윤곽선 찾기 및 크기 순으로 정렬
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 꼭짓점이 4개인 가장 큰 사각형 영역을 선택
        if len(approx) == 4:
            screen_cnt = approx
            break

    # 사각형 윤곽선을 찾지 못했다면 기존 이미지 유지
    if screen_cnt is None:
        print("-> [전처리] 영수증 외곽선 검출 실패 (원본 이미지로 진행)")
        return False

    # 4. 원본 이미지 크기에 맞게 좌표 복원 및 정렬 (좌상, 우상, 우하, 좌하)
    pts = screen_cnt.reshape(4, 2) / r
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    # 5. 변환 후 스캔본이 될 새 이미지의 가로/세로 최대 크기 계산
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # 6. 원근 변환 매트릭스 적용 및 이미지 잘라내기(Warping)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (max_width, max_height))

    # 가로가 더 길게 잘린 경우 세로로 자동 정렬
    if max_width > max_height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # 7. 변환된 깨끗한 이미지 덮어쓰기 저장
    cv2.imwrite(image_path, warped)
    print("-> [전처리] OpenCV 자동 원근 변환 완료 (스캔본 생성)")
    return True

def main():
    # 1. Supabase 매니저 생성
    db_manager = SupabaseManager()

    # 모델 한 번만 로드 ← 여기 추가
    print("PaddleOCR 모델 로드 중...")
    ocr_model = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1536,
        text_det_limit_type="max",
        enable_mkldnn=False,
    )

    print("1. Supabase Storage에서 이미지 목록을 조회하는 중...")
    images = db_manager.get_image_list(bucket_name="images")
    
    if not images:
        print("버킷에 이미지 파일이 없습니다. Supabase에 이미지를 먼저 업로드해 주세요.")
        return
        
    print(f"총 {len(images)}개의 이미지를 발견했습니다. 전체 분석을 시작합니다.\n" + "="*40)

    # 테스트를 위해 스토리지의  다섯 번째 이미지까지 선택
    for idx, img_info in enumerate(images, start=1):
        target_image = img_info['name']

        #if db_manager.is_image_processed(target_image):
        #    print(f"[건너뜀] {target_image} (이미 처리됨)")
        #    continue

        if idx <= 138:
            continue

        # .emptyFolderPlaceholder 같은 시스템 특수 파일은 건너넙니다.
        if target_image.startswith('.'):
            continue
            
        print(f"\n[{idx}/{len(images)}] 작업 시작 -> 파일명: {target_image}")

    
        # 2. 이미지의 Public URL 가져오기
        image_url = db_manager.get_image_url(target_image, bucket_name="images")
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
            auto_perspective_transform(local_temp_path)
        except Exception as e:
            print(f"-> [경고] 전처리 중 에러 발생 (원본 이미지로 계속 진행): {e}")
        try:
            print("2. PaddleOCR 분석 시작...")
            # ysz_ocr.py에 있는 run_ocr 함수를 실행시켜 이미지 분석
            ocr_raw = run_ocr(local_temp_path, "receipt_ocr_raw.json", ocr_model=ocr_model)
            
            # 4. DB에 넣기 위해 텍스트 조각들을 하나로 합침
            all_text_list = [item["text"] for item in ocr_raw["ocr_result"]]
            all_text = " ".join(all_text_list)
            
            # 신뢰도 평균 계산
            scores = [item["score"] for item in ocr_raw["ocr_result"]]
            avg_confidence = sum(scores) / len(scores) if scores else 0
            
            print("3. OCR 결과를 Supabase DB에 저장 중...")
            # database.py의 insert_ocr_result 함수를 실행시켜 DB에 저장
            ocr_raw_id = db_manager.insert_ocr_result(
                image_name=target_image,
                all_text=all_text,
                confidence=avg_confidence,
                ocr_items=ocr_raw["ocr_result"]
            )

            print("4. 영수증 파싱 중...")
            parsed = run_receipt_parser(ocr_raw_id, db_manager)

            print("5. 파싱 결과 DB 저장 중...")
            receipt_id = db_manager.insert_receipt(
                ocr_raw_id=ocr_raw_id,
                store_name=parsed["store_name"],
                purchased_at=parsed["purchase_date"],
                total_amount=parsed["total_amount"]
            )
            db_manager.insert_receipt_items(receipt_id, parsed["items"])

            print("🎉 모든 프로세스가 성공적으로 완료되었습니다!")
            
        except Exception as e:
            print(f"오류 발생: {e}")
            
        finally:
            # 작업이 끝나면 임시로 받았던 이미지 파일 삭제
            if os.path.exists(local_temp_path):
                os.remove(local_temp_path)
        

if __name__ == "__main__":
    main()