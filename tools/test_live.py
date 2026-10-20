"""Real HTTP -> OCR -> trained model -> review -> calendar integration check.

Uses only the synthetic runtime/test_receipt.png, and leaves it in review state.
Run against the local preview server, never a production endpoint.
"""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import time

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Run OCR again on a new copy of the synthetic image")
    parser.add_argument("--job-id", help="Finish verification of an already uploaded synthetic test job")
    args = parser.parse_args()
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
        def call(method, path, **kwargs):
            response = client.request(method, "/api" + path, **kwargs)
            response.raise_for_status()
            return response.json()

        before = call("GET", "/calendar?month=2026-10")["total"]
        content = (ROOT / "runtime/test_receipt.png").read_bytes()
        if args.fresh:
            # Change one corner pixel to bypass deduplication, leaving the printed
            # synthetic receipt text intact. No real/training data is modified.
            with Image.open(BytesIO(content)) as image:
                image.putpixel((0, 0), tuple(os.urandom(3)))
                output = BytesIO()
                image.save(output, format="PNG")
                content = output.getvalue()
        if args.job_id:
            existing=call('GET','/jobs/'+args.job_id)
            result={'receipt_id':existing['receipt_id'],'job_id':args.job_id,'duplicate':True}
        else:
            result = call("POST", "/receipts/upload", files={"file": ("synthetic-test.png", content, "image/png")})
        print("Job", result["job_id"], flush=True)
        existing_job = call("GET", "/jobs/" + result["job_id"])
        if existing_job["status"] == "failed":
            call("POST", "/receipts/" + result["receipt_id"] + "/retry")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            job = call("GET", "/jobs/" + result["job_id"])
            if job["status"] == "failed":
                raise RuntimeError(job["error"])
            if job["status"] == "completed":
                break
            time.sleep(2)
        else:
            raise TimeoutError("Real OCR/model pipeline did not finish in 300s")
        receipt = call("GET", "/receipts/" + result["receipt_id"])
        assert "layoutlmv3" in receipt["engine"], receipt.get("raw")
        assert receipt["purchased_at"] == "2026-10-17", receipt
        assert receipt["total_amount"] == 10000, receipt
        assert len(receipt["items"]) == 2, receipt
        # Other records can be created while OCR runs. Check the current day's
        # sum instead of assuming the global ledger stayed unchanged for minutes.
        automatic=call('GET','/calendar?month=2026-10')
        day=next(day for day in automatic['days'] if day['date']==receipt['purchased_at'])
        date_records=call('GET','/receipts?purchased_at='+receipt['purchased_at']+'&limit=500')
        assert any(row['id']==receipt['id'] for row in date_records['items'])
        assert day['count']==date_records['total']
        if date_records['total']<=500:
            assert day['total']==sum(row['total_amount'] or 0 for row in date_records['items'])
        body = {key: receipt[key] for key in ("store_name", "purchased_at", "total_amount", "memo", "category", "allow_training")}
        body["items"] = [{key: item.get(key) for key in ("name", "quantity", "unit_price", "line_total")} for item in receipt["items"]]
        body["status"] = "confirmed"
        call("PATCH", "/receipts/" + receipt["id"], json=body)
        after = call("GET", "/calendar?month=2026-10")["total"]
        # Re-running this check remains valid when its old synthetic record exists.
        expected = automatic['total']
        assert after == expected, (before, after, expected)
        body["status"] = "review"
        call("PATCH", "/receipts/" + receipt["id"], json=body)
        assert call("GET", "/calendar?month=2026-10")["total"] == expected
        report = {"status": "passed", "receipt_id": receipt["id"], "engine": receipt["engine"], "store_name": receipt["store_name"], "purchased_at": receipt["purchased_at"], "total_amount": receipt["total_amount"], "item_count": len(receipt["items"]), "synthetic": True, "auto_calendar_recorded":True,"left_in_status": "review"}
        (ROOT / "runtime/live_test_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)
        call("POST", "/demo")


if __name__ == "__main__":
    main()
