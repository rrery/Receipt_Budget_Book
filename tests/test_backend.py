from io import BytesIO
import json
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.app import create_app


@pytest.fixture
def client(tmp_path):
    def processor(_):
        return {"parsed": {"store_name": "테스트 식당", "purchase_date": "2026-10-17", "total_amount": 12000, "items": []}, "raw": {"ocr_result": []}, "engine": "test-adapter"}
    with TestClient(create_app(tmp_path, processor)) as client:
        yield client


def data(**overrides):
    return {"store_name": "가게", "purchased_at": "2026-10-17", "total_amount": 5000, **overrides}


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (60, 80), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def wait_job(client, job_id):
    for _ in range(100):
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(.02)
    pytest.fail("worker timed out")


def test_calendar_edit_delete_and_month_boundary(client):
    a = client.post("/api/receipts", json=data()).json()
    client.post("/api/receipts", json=data(purchased_at="2026-11-01", total_amount=9000))
    client.post("/api/receipts", json=data(status="review", total_amount=1000))
    october = client.get("/api/calendar?month=2026-10").json()
    assert october["total"] == 6000 and october["count"] == 2
    assert client.patch("/api/receipts/" + a["id"], json=data(total_amount=7000)).status_code == 200
    assert client.get("/api/calendar?month=2026-10").json()["total"] == 8000
    assert client.delete("/api/receipts/" + a["id"]).status_code == 204
    assert client.get("/api/calendar?month=2026-10").json()["total"] == 1000
    assert client.get("/api/calendar?month=2026-11").json()["total"] == 9000


@pytest.mark.parametrize("payload", [data(purchased_at="2026-02-31"), data(total_amount=-1), data(store_name="  "), data(purchased_at=None), data(items=[{"name": "x", "quantity": 0}])])
def test_reject_bad_input(client, payload):
    assert client.post("/api/receipts", json=payload).status_code == 422


def test_upload_job_deduplication_and_confirm(client):
    content = image_bytes()
    a = client.post("/api/receipts/upload", files={"file": ("receipt.png", content, "image/png")})
    assert a.status_code == 202
    a = a.json()
    b = client.post("/api/receipts/upload", files={"file": ("other.png", content, "image/png")}).json()
    assert b["duplicate"] and b["receipt_id"] == a["receipt_id"]
    assert wait_job(client, a["job_id"])["status"] == "completed"
    receipt = client.get("/api/receipts/" + a["receipt_id"]).json()
    listed=client.get('/api/receipts').json()['items'][0]
    assert 'raw' not in listed and 'predictions' not in listed
    assert receipt["status"] == "review" and receipt["total_amount"] == 12000
    assert client.get("/api/calendar?month=2026-10").json()["total"] == 12000
    assert client.get(f'/api/receipts/{a["receipt_id"]}/image').status_code == 200
    assert client.patch('/api/receipts/'+a['receipt_id'], json=data(total_amount=12000)).status_code == 200
    assert client.get("/api/calendar?month=2026-10").json()["total"] == 12000


def test_failed_job_retry_and_recovery(tmp_path):
    attempts = []
    def processor(_):
        attempts.append(1)
        raise RuntimeError("OCR unavailable")
    with TestClient(create_app(tmp_path, processor)) as client:
        result = client.post('/api/receipts/upload', files={'file': ('x.png', image_bytes(), 'image/png')}).json()
        assert wait_job(client, result['job_id'])['status'] == 'failed'
        assert client.post('/api/receipts/'+result['receipt_id']+'/retry').status_code == 202
        assert wait_job(client, result['job_id'])['status'] == 'failed'
        assert len(attempts) == 2
        with client.app.state.store.connect() as db:
            db.execute("UPDATE jobs SET status='processing'")
    with TestClient(create_app(tmp_path, processor)) as restarted:
        assert wait_job(restarted, result['job_id'])['status'] == 'failed'
        assert len(attempts) == 3
        assert restarted.get('/api/receipts/'+result['receipt_id']).status_code == 200


def test_bad_images_and_csrf(client):
    assert client.post('/api/receipts/upload', files={'file': ('x.png', b'not an image', 'image/png')}).status_code == 415
    assert client.post('/api/demo', headers={'Origin': 'https://other.example'}).status_code == 403
    assert client.get('/api/status', headers={'Host': 'evil.example'}).status_code == 400
    assert client.get('/api/calendar?month=2026-13').status_code == 422
    assert client.get('/api/receipts/missing').status_code == 404


def test_demo_idempotent_and_persistence(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        month = client.post('/api/demo').json()['month']
        client.post('/api/demo')
        assert client.get('/api/receipts').json()['total'] == 20
        assert client.get('/api/calendar?month='+month).json()['total'] == 306000
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/receipts').json()['total'] == 20
        assert client.get('/').status_code == 200
        assert client.get('/docs').status_code == 200
