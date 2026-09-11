from supabase import create_client

class SupabaseManager:
    def __init__(self):
        # 1. Supabase 연동 정보
        self.url = "https://djjkvygcgcmmaowbnxjq.supabase.co"
        self.key = "sb_publishable_ZX9Jjwq6LNFYMAlfu5ZGSg_cqtMwPVy"
        self.supabase = create_client(self.url, self.key)

    def get_image_list(self, bucket_name="images"):
        """Storage 버킷의 전체 파일 목록을 페이지 단위로 조회합니다."""
        all_files = []
        offset = 0
        limit = 100

        while True:
            files = (
                self.supabase.storage
                .from_(bucket_name)
                .list(
                    "",
                    {
                        "limit": limit,
                        "offset": offset,
                        "sortBy": {
                            "column": "name",
                            "order": "asc"
                        }
                    }
                )
            )

            if not files:
                break

            all_files.extend(files)

            if len(files) < limit:
                break

            offset += limit

        return all_files

    def get_image_url(self, file_name, bucket_name="images"):
        """특정 이미지의 Public URL을 가져옵니다."""
        return self.supabase.storage.from_(bucket_name).get_public_url(file_name)

    def insert_ocr_result(self, image_name, all_text, confidence, ocr_items):
        # 1. ocr_raw 테이블에 영수증 저장
        raw_data = {
            "image_name": image_name,
            "all_text": all_text,
            "confidence": confidence
        }
        raw_result = self.supabase.table("ocr_raw").insert(raw_data).execute()
    
        # 2. 저장된 행의 id 가져오기
        ocr_raw_id = raw_result.data[0]["id"]
    
        # 3. ocr_raw_items에 텍스트 조각 38개 저장
        items_data = [
            {
                "ocr_raw_id": ocr_raw_id,
                "text": item["text"],
                "score": item["score"],
                "box": item["box"]
            }
            for item in ocr_items
        ]
        self.supabase.table("ocr_raw_items").insert(items_data).execute()
        return ocr_raw_id
    
    def get_ocr_items(self, ocr_raw_id: int):
        return self.supabase.table("ocr_raw_items").select("*").eq("ocr_raw_id", ocr_raw_id).execute()

    def get_ocr_raw(self, ocr_raw_id: int):
        return self.supabase.table("ocr_raw").select("*").eq("id", ocr_raw_id).single().execute()
    
    def insert_receipt(self, ocr_raw_id: int, store_name: str, purchased_at: str, total_amount: int) -> int:
        data = {
            "ocr_raw_id": ocr_raw_id,
            "store_name": store_name,
            "purchased_at": purchased_at,
            "total_amount": total_amount
        }
        result = self.supabase.table("receipts").insert(data).execute()
        return result.data[0]["id"]

    def insert_receipt_items(self, receipt_id: int, items: list) -> None:
        data = [
            {
                "receipt_id": receipt_id,
                "item_name": item["name"],
               "price": item["line_total"],
               "quantity": item["quantity"],
               "unit_price": item["unit_price"]
           }
            for item in items
        ]
        self.supabase.table("receipt_items").insert(data).execute()