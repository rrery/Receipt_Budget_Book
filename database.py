from supabase import create_client

class SupabaseManager:
    def __init__(self):
        # 1. Supabase 연동 정보
        self.url = "https://djjkvygcgcmmaowbnxjq.supabase.co"
        self.key = "sb_publishable_ZX9Jjwq6LNFYMAlfu5ZGSg_cqtMwPVy"
        self.supabase = create_client(self.url, self.key)

    def get_image_list(self, bucket_name="image"):
        """Storage 버킷에 있는 파일 목록을 가져옵니다."""
        return self.supabase.storage.from_(bucket_name).list()

    def get_image_url(self, file_name, bucket_name="image"):
        """특정 이미지의 Public URL을 가져옵니다."""
        return self.supabase.storage.from_(bucket_name).get_public_url(file_name)

    def insert_ocr_result(self, image_name, all_text, confidence, total_amount=0):
        """OCR 분석 결과를 DB 테이블에 저장합니다."""
        data = {
            "image_name": image_name,
            "all_text": all_text,
            "total_amount": total_amount,
            "confidence": confidence
        }
        return self.supabase.table("receipt_ocr").insert(data).execute()