import json
import logging
from pathlib import Path
import threading
from datetime import date
from backend.insights import infer_category

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, store, processor):
        self.store, self.processor = store, processor
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="receipt-worker")

    def start(self):
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET status='queued' WHERE status='processing'")
        self.thread.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=3)

    def run(self):
        while not self.stop.is_set():
            try:
                with self.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute("SELECT j.*,r.image_path,r.purchased_at FROM jobs j JOIN receipts r ON r.id=j.receipt_id WHERE j.status='queued' ORDER BY j.created_at LIMIT 1").fetchone()
                    if row:
                        db.execute("UPDATE jobs SET status='processing',error=NULL,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?", (row["id"],))
                if row is None:
                    self.wake.wait(1)
                    self.wake.clear()
                    continue
                try:
                    result = self.processor(Path(row["image_path"]))
                    parsed = result["parsed"]
                    try:
                        purchased_at = date.fromisoformat(parsed.get("purchase_date") or "").isoformat()
                        date_inferred = 0
                    except (ValueError,TypeError):
                        purchased_at = row["purchased_at"]
                        date_inferred = 1
                        parsed.setdefault("warnings",[]).append("구매 날짜를 읽지 못해 업로드 날짜에 임시로 기록했어요. 필요하면 수정해주세요.")
                    amount = parsed.get("total_amount")
                    if not isinstance(amount,int) or isinstance(amount,bool) or not 0<=amount<=10**12:
                        amount = None
                    category = infer_category(parsed.get("store_name"),parsed.get("items",[]))
                    with self.store.connect() as db:
                        db.execute("""UPDATE receipts SET store_name=?,purchased_at=?,total_amount=?,items=?,raw=?,predictions=?,engine=?,category=?,date_inferred=?,status='review',updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?""", (
                            parsed.get("store_name") or "상호명 미확인", purchased_at, amount,
                            json.dumps(parsed.get("items", []), ensure_ascii=False),
                            json.dumps({"ocr": result["raw"], "warnings": parsed.get("warnings", [])}, ensure_ascii=False),
                            json.dumps(result.get("predictions"), ensure_ascii=False), result["engine"],category,date_inferred,row["receipt_id"],
                        ))
                        db.execute("UPDATE jobs SET status='completed',updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?", (row["id"],))
                except Exception as error:
                    logger.exception("Receipt processing failed: %s", row["id"])
                    message = str(error) if isinstance(error, (ValueError, RuntimeError)) else "분석 중 오류가 발생했습니다. 서버 로그를 확인하세요."
                    with self.store.connect() as db:
                        db.execute("UPDATE jobs SET status='failed',error=?,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?", (message[:500], row["id"]))
            except Exception:
                logger.exception("Worker database error")
                self.stop.wait(1)
