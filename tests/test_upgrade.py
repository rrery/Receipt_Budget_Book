from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
import json
import sqlite3
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.app import create_app
from backend.store import Store
from tools.export_corrections import export_candidates
from tests.test_backend import data,image_bytes,wait_job


def processor(_):
    return {'parsed':{'store_name':'카페','purchase_date':'2026-10-17','total_amount':12000,'items':[{'name':'커피','quantity':2,'unit_price':6000,'line_total':12000}]},'raw':{'ocr_result':[]},'engine':'test-adapter'}


def test_memos_stats_and_previous_month(tmp_path):
    with TestClient(create_app(tmp_path,processor)) as client:
        client.post('/api/receipts',json=data(category='식비',memo='기억할 점심',total_amount=10000))
        client.post('/api/receipts',json=data(category='카페·간식',total_amount=5000,purchased_at='2026-10-18',status='review'))
        client.post('/api/receipts',json=data(total_amount=20000,purchased_at='2026-09-30'))
        result=client.get('/api/calendar?month=2026-10').json()
        assert result['total']==15000 and result['count']==2 and result['reviewed_count']==1
        assert result['top_category']=={'category':'식비','total':10000}
        assert result['top_day']['date']=='2026-10-17'
        assert result['previous_month']['difference']==-5000 and result['previous_month']['percent']==-25
        assert client.put('/api/months/2026-10/memo',json={'memo':'가을 기록'*40}).status_code==200
        assert client.put('/api/months/2026-10/memo',json={'memo':'가'*201}).status_code==422
        assert client.post('/api/receipts',json=data(memo='가'*201)).status_code==422
        assert client.get('/api/calendar?month=2026-10').json()['memo']=='가을 기록'*40
        response=client.get('/api/months/2026-10/export.jpg')
        assert response.status_code==200 and response.headers['content-type']=='image/jpeg'
        assert Image.open(BytesIO(response.content)).size==(2160,2700)
        assert client.get('/share/2026-10').status_code==200
        assert client.get('/api/months/2026-13/export.png').status_code==422


def test_upload_visible_while_pending_and_date_amount_unknown(tmp_path):
    import threading
    release=threading.Event()
    def missing(_):
        release.wait(3)
        return {'parsed':{'purchase_date':None,'total_amount':None,'items':[]},'raw':{'ocr_result':[]},'engine':'test'}
    with TestClient(create_app(tmp_path,missing)) as client:
        value=client.post('/api/receipts/upload',files={'file':('x.png',image_bytes(),'image/png')}).json()
        month=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m')
        summary=client.get('/api/calendar?month='+month).json()
        assert summary['count']==1 and summary['pending_count']==1 and summary['unknown_amount_count']==1
        release.set();assert wait_job(client,value['job_id'])['status']=='completed'
        receipt=client.get('/api/receipts/'+value['receipt_id']).json()
        assert receipt['date_inferred']==1 and receipt['total_amount'] is None
        assert client.get('/api/calendar?month='+month).json()['count']==1


def test_corrections_preserve_provenance_consent_and_delete(tmp_path):
    with TestClient(create_app(tmp_path,processor)) as client:
        value=client.post('/api/receipts/upload',files={'file':('x.png',image_bytes(),'image/png')}).json()
        assert wait_job(client,value['job_id'])['status']=='completed'
        path='/api/receipts/'+value['receipt_id']
        body=data(store_name='수정된 카페',total_amount=13000,memo='학습에 들어가면 안 되는 비밀 메모',allow_training=True,items=[{'name':'아이스 커피','quantity':2,'unit_price':6500,'line_total':13000}])
        assert client.patch(path,json=body).status_code==200
        history=client.get(path+'/corrections').json()
        assert len(history)==1 and history[0]['before']['store_name']=='카페'
        assert history[0]['after']['store_name']=='수정된 카페' and 'items' in history[0]['changed_fields']
        assert 'memo' not in history[0]['after']
        output=tmp_path/'bundle'
        assert export_candidates(tmp_path,output)==1
        candidate=json.loads((output/'candidates.jsonl').read_text(encoding='utf-8').splitlines()[0])
        assert candidate['needs_annotation'] and not candidate['bio_labels_ready']
        assert candidate['ocr_snapshot'] and '비밀 메모' not in (output/'candidates.jsonl').read_text(encoding='utf-8')
        assert (output/candidate['image_path']).exists()
        body['memo']='메모만 바뀜';assert client.patch(path,json=body).status_code==200
        assert len(client.get(path+'/corrections').json())==1
        body['allow_training']=False;client.patch(path,json=body)
        assert export_candidates(tmp_path,tmp_path/'withdrawn')==0
        assert client.delete(path).status_code==204
        with client.app.state.store.connect() as db:assert db.execute('SELECT count(*) FROM corrections').fetchone()[0]==0


def test_public_accounts_isolate_all_record_paths(tmp_path,monkeypatch):
    monkeypatch.setenv('APP_ALLOWED_HOSTS','ledger.test')
    monkeypatch.setenv('PUBLIC_ORIGIN','https://ledger.test')
    app=create_app(tmp_path,processor,production=True)
    with TestClient(app,base_url='https://ledger.test') as a,TestClient(create_app(tmp_path,processor,production=True),base_url='https://ledger.test') as b:
        assert a.get('/api/receipts').status_code==401
        assert a.get('/api/auth/me').json()['authenticated'] is False
        ra=a.post('/api/auth/register',json={'username':'alice','password':'test-pass-12345'})
        assert ra.status_code==201 and 'HttpOnly' in ra.headers['set-cookie'] and 'Secure' in ra.headers['set-cookie']
        assert b.post('/api/auth/register',json={'username':'bob','password':'test-pass-12345'}).status_code==201
        own=a.post('/api/receipts',json=data()).json()
        assert b.get('/api/receipts').json()['total']==0
        assert b.get('/api/calendar?month=2026-10').json()['count']==0
        for suffix in ['', '/image','/corrections']:
            assert b.get('/api/receipts/'+own['id']+suffix).status_code==404
        assert b.patch('/api/receipts/'+own['id'],json=data()).status_code==404
        assert b.delete('/api/receipts/'+own['id']).status_code==404
        for client in [a,b]:
            uploaded=client.post('/api/receipts/upload',files={'file':('x.png',image_bytes(),'image/png')}).json()
            assert not uploaded['duplicate']
            assert wait_job(client,uploaded['job_id'])['status']=='completed'
            if client is a:alice_job=uploaded['job_id']
        assert b.get('/api/jobs/'+alice_job).status_code==404
        a.put('/api/months/2026-10/memo',json={'memo':'alice only'})
        assert b.get('/api/months/2026-10/memo').json()['memo']==''
        assert a.get('/api/training').status_code==404 and a.get('/docs').status_code==404
        assert a.post('/api/demo',headers={'Origin':'https://evil.example'}).status_code==403
        a.post('/api/auth/logout')
        assert a.get('/api/receipts').status_code==401
        assert a.post('/api/auth/login',json={'username':'alice','password':'wrong-pass-123'}).status_code==401
        assert a.post('/api/auth/login',json={'username':'alice','password':'test-pass-12345'}).status_code==200
        assert a.get('/api/receipts').json()['total']==2


def test_deployment_fails_closed(tmp_path,monkeypatch):
    monkeypatch.delenv('APP_ALLOWED_HOSTS',raising=False);monkeypatch.delenv('PUBLIC_ORIGIN',raising=False)
    with pytest.raises(RuntimeError):create_app(tmp_path,production=True)


def test_old_database_migration_preserves_record_and_duplicate_hash(tmp_path):
    path=tmp_path/'budget.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE receipts(id TEXT PRIMARY KEY,source TEXT,image_hash TEXT UNIQUE,image_path TEXT,store_name TEXT,purchased_at TEXT,total_amount INTEGER,items TEXT DEFAULT '[]',raw TEXT,predictions TEXT,engine TEXT DEFAULT 'manual',status TEXT DEFAULT 'review',created_at TEXT,updated_at TEXT)")
        db.execute("INSERT INTO receipts(id,source,image_hash,store_name,purchased_at,total_amount) VALUES('old','upload','original-hash','기존 가게','2026-10-01',3000)")
        db.execute("INSERT INTO receipts(id,source,created_at) VALUES('undated','upload','2026-10-01T17:00:00Z')")
    store=Store(tmp_path);store.initialize();store.initialize()
    receipt=store.get('old')
    assert receipt['store_name']=='기존 가게' and receipt['total_amount']==3000 and receipt['memo']==''
    assert receipt['content_hash']=='original-hash'
    assert store.get('old','other') is None
    assert store.get('undated')['purchased_at']=='2026-10-02' and store.get('undated')['date_inferred']==1
