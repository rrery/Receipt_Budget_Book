from contextlib import asynccontextmanager
from datetime import date, datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal
from urllib.parse import urlsplit
import uuid
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.auth import Auth
from backend.demo import SAMPLES
from backend.insights import CATEGORIES, calendar_summary, month_bounds
from backend.pipeline import ROOT, MODEL, process_receipt
from backend.sharing import month_card
from backend.store import Store
from backend.worker import Worker

MAX_UPLOAD = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


class Item(BaseModel):
    name: str = Field(min_length=1,max_length=200)
    quantity: int | None = Field(default=None,ge=1,le=100000)
    unit_price: int | None = Field(default=None,ge=0,le=10**12)
    line_total: int | None = Field(default=None,ge=0,le=10**12)


class ReceiptInput(BaseModel):
    store_name: str = Field(min_length=1,max_length=200)
    purchased_at: date | None = None
    total_amount: int | None = Field(default=None,ge=0,le=10**12)
    items: list[Item] = Field(default_factory=list,max_length=200)
    status: Literal['review','confirmed'] = 'confirmed'
    memo: str = Field(default='',max_length=200)
    category: str = '기타'
    allow_training: bool = False

    @model_validator(mode='after')
    def complete(self):
        self.store_name=self.store_name.strip()
        if not self.store_name: raise ValueError('상호명을 입력하세요.')
        if self.category not in CATEGORIES: raise ValueError('올바른 분류를 선택하세요.')
        if self.purchased_at is None: raise ValueError('달력에 기록할 날짜를 선택해주세요.')
        if self.status=='confirmed' and (self.purchased_at is None or self.total_amount is None):
            raise ValueError('검토를 마치려면 구매 날짜와 총액이 필요합니다.')
        return self


class MemoInput(BaseModel):
    memo: str = Field(default='',max_length=200)


def today():
    return datetime.now(ZoneInfo('Asia/Seoul')).date()


def create_app(directory: Path | None = None, processor=process_receipt, production: bool | None = None):
    production = production if production is not None else os.getenv('APP_ENV')=='production'
    public_origin=os.getenv('PUBLIC_ORIGIN','').rstrip('/')
    hosts=[h.strip() for h in os.getenv('APP_ALLOWED_HOSTS','').split(',') if h.strip()]
    if production and (not hosts or '*' in hosts or not public_origin.startswith('https://') or urlsplit(public_origin).hostname not in hosts):
        raise RuntimeError('공개 모드에는 HTTPS PUBLIC_ORIGIN과 정확한 APP_ALLOWED_HOSTS가 필요합니다.')
    store=Store(directory or Path(os.getenv('BUDGET_DATA_DIR',ROOT/'runtime')))
    worker=Worker(store,processor)
    auth=Auth(store,production)

    @asynccontextmanager
    async def lifespan(app):
        store.initialize();worker.start()
        yield
        worker.close()

    app=FastAPI(title='차곡 가계부',version='0.2.0',lifespan=lifespan,docs_url=None if production else '/docs',redoc_url=None,openapi_url=None if production else '/openapi.json')
    app.state.store,app.state.worker,app.state.auth=store,worker,auth
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=list(set(hosts+['127.0.0.1','localhost']+([] if production else ['testserver']))))
    app.include_router(auth.router())

    @app.middleware('http')
    async def boundaries(request: Request,call_next):
        origin=request.headers.get('origin')
        expected=public_origin if production else str(request.base_url).rstrip('/')
        if (origin and origin!=expected) or request.headers.get('sec-fetch-site')=='cross-site':
            return JSONResponse({'detail':'차곡 화면에서 다시 시도해주세요.'},status_code=403)
        try:
            if int(request.headers.get('content-length','0'))>MAX_UPLOAD+1024*1024:
                return JSONResponse({'detail':'사진은 12MB 이하로 올려주세요.'},status_code=413)
        except ValueError:
            return JSONResponse({'detail':'잘못된 요청입니다.'},status_code=400)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['X-Frame-Options']='DENY'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        if not request.url.path.startswith('/docs'):
            response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    def require(receipt_id,owner):
        value=store.get(receipt_id,owner)
        if value is None: raise HTTPException(404,'기록을 찾을 수 없습니다.')
        return value

    @app.get('/api/health')
    def health(): return {'status':'ok'}

    @app.get('/api/status')
    def status(owner=Depends(auth.owner)):
        with store.connect() as db:
            counts={row['status']:row['n'] for row in db.execute('SELECT status,count(*) n FROM receipts WHERE owner_id=? GROUP BY status',(owner,))}
            pending=db.execute("SELECT count(*) FROM jobs j JOIN receipts r ON r.id=j.receipt_id WHERE r.owner_id=? AND j.status IN ('queued','processing')",(owner,)).fetchone()[0]
        return {'receipts':counts,'pending_count':pending,'mode':'public' if production else 'local'}

    @app.get('/api/training',include_in_schema=False)
    def training():
        if production: raise HTTPException(404)
        status_path=ROOT/'runtime/training_status.json'
        result=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'status':'not_started'}
        info=MODEL/'model_info.json'
        if info.exists(): result['model']=json.loads(info.read_text(encoding='utf-8'))
        return result

    @app.get('/api/calendar')
    def calendar(month: str=Query(pattern=r'^\d{4}-\d{2}$'),owner=Depends(auth.owner)):
        return calendar_summary(store,owner,month)

    @app.get('/api/categories')
    def categories(): return list(CATEGORIES)

    @app.get('/api/receipts')
    def receipts(purchased_at: date | None=None,status: Literal['review','confirmed'] | None=None,month: str | None=None,limit: int=Query(default=200,ge=1,le=500),offset: int=Query(default=0,ge=0),owner=Depends(auth.owner)):
        clauses,args=['r.owner_id=?'],[owner]
        if purchased_at: clauses.append('r.purchased_at=?');args.append(purchased_at.isoformat())
        if status: clauses.append('r.status=?');args.append(status)
        if month:
            start,end=month_bounds(month);clauses+=['r.purchased_at>=?','r.purchased_at<?'];args += [start,end]
        where=' WHERE '+' AND '.join(clauses)
        with store.connect() as db:
            rows=db.execute('SELECT r.id,r.source,r.image_path,r.store_name,r.purchased_at,r.total_amount,r.items,NULL raw,NULL predictions,r.engine,r.status,r.created_at,r.updated_at,r.memo,r.category,r.date_inferred,r.allow_training,j.id job_id,j.status job_status,j.error job_error FROM receipts r LEFT JOIN jobs j ON j.receipt_id=r.id'+where+' ORDER BY r.purchased_at DESC,r.created_at DESC,r.id LIMIT ? OFFSET ?',(*args,limit,offset)).fetchall()
            total=db.execute('SELECT count(*) FROM receipts r'+where,args).fetchone()[0]
        values=[store.receipt(row) for row in rows]
        for value in values:
            for key in ('raw','predictions','items'):value.pop(key,None)
        return {'items':values,'total':total,'limit':limit,'offset':offset}

    @app.post('/api/receipts',status_code=201)
    def manual(body: ReceiptInput,owner=Depends(auth.owner)):
        receipt_id=uuid.uuid4().hex
        with store.connect() as db:
            db.execute('INSERT INTO receipts(id,owner_id,source,store_name,purchased_at,total_amount,items,status,memo,category,allow_training) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(receipt_id,owner,'manual',body.store_name,body.purchased_at.isoformat() if body.purchased_at else None,body.total_amount,json.dumps([i.model_dump() for i in body.items],ensure_ascii=False),body.status,body.memo,body.category,int(body.allow_training)))
        return store.get(receipt_id,owner)

    @app.post('/api/receipts/upload',status_code=202)
    async def upload(request: Request,file: UploadFile,owner=Depends(auth.owner)):
        if production:auth.limit(request,key='upload:'+owner,maximum=30)
        content=await file.read(MAX_UPLOAD+1);await file.close()
        if len(content)>MAX_UPLOAD: raise HTTPException(413,'사진은 12MB 이하로 올려주세요.')
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format not in {'JPEG','PNG','WEBP'}: raise HTTPException(415,'JPG, PNG, WebP 사진을 올려주세요.')
                if min(image.size)<32 or image.width*image.height>20_000_000: raise HTTPException(422,'사진은 최소 32×32, 최대 2천만 픽셀이어야 합니다.')
                image.load();normalized=ImageOps.exif_transpose(image).convert('RGB')
                buffer=BytesIO();normalized.save(buffer,format='PNG')
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
            raise HTTPException(415,'사진을 읽지 못했어요. 다른 사진으로 시도해주세요.')
        digest=hashlib.sha256(buffer.getvalue()).hexdigest()
        scoped=hashlib.sha256(f'{owner}:{digest}'.encode()).hexdigest()
        def existing():
            with store.connect() as db:
                return db.execute('SELECT r.id,j.id job_id FROM receipts r JOIN jobs j ON j.receipt_id=r.id WHERE r.image_hash=? AND r.owner_id=?',(scoped,owner)).fetchone()
        old=existing()
        if old: return {'receipt_id':old['id'],'job_id':old['job_id'],'duplicate':True}
        receipt_id,job_id=uuid.uuid4().hex,uuid.uuid4().hex
        path=store.directory/'uploads'/(receipt_id+'.png');path.write_bytes(buffer.getvalue())
        try:
            with store.connect() as db:
                if production:
                    count=db.execute("SELECT count(*) FROM receipts WHERE owner_id=? AND source='upload'",(owner,)).fetchone()[0]
                    if count>=int(os.getenv('MAX_RECEIPTS_PER_USER','2000')): raise HTTPException(429,'보관 한도에 도달했어요. 오래된 사진을 정리해주세요.')
                db.execute("INSERT INTO receipts(id,owner_id,source,image_hash,content_hash,image_path,engine,purchased_at,date_inferred) VALUES(?,?,'upload',?,?,?,'pending',?,1)",(receipt_id,owner,scoped,digest,str(path.resolve()),today().isoformat()))
                db.execute('INSERT INTO jobs(id,receipt_id) VALUES(?,?)',(job_id,receipt_id))
        except sqlite3.IntegrityError:
            path.unlink(missing_ok=True);old=existing()
            if old:return {'receipt_id':old['id'],'job_id':old['job_id'],'duplicate':True}
            raise
        except Exception:
            path.unlink(missing_ok=True);raise
        worker.wake.set()
        return {'receipt_id':receipt_id,'job_id':job_id,'duplicate':False,'purchased_at':today().isoformat()}

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str,owner=Depends(auth.owner)):
        result=store.job(job_id,owner)
        if result is None: raise HTTPException(404,'분석을 찾을 수 없습니다.')
        return result

    @app.post('/api/receipts/{receipt_id}/retry',status_code=202)
    def retry(receipt_id: str,owner=Depends(auth.owner)):
        require(receipt_id,owner)
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM jobs WHERE receipt_id=?',(receipt_id,)).fetchone()
            if row is None or row['status']!='failed': raise HTTPException(409,'실패한 분석만 다시 시도할 수 있어요.')
            db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=?",(row['id'],))
        worker.wake.set();return {'job_id':row['id'],'receipt_id':receipt_id}

    @app.get('/api/receipts/{receipt_id}')
    def detail(receipt_id: str,owner=Depends(auth.owner)): return require(receipt_id,owner)

    @app.patch('/api/receipts/{receipt_id}')
    def edit(receipt_id: str,body: ReceiptInput,owner=Depends(auth.owner)):
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM receipts WHERE id=? AND owner_id=?',(receipt_id,owner)).fetchone()
            if row is None: raise HTTPException(404,'기록을 찾을 수 없습니다.')
            job=db.execute('SELECT status FROM jobs WHERE receipt_id=?',(receipt_id,)).fetchone()
            if job and job['status'] in {'queued','processing'}: raise HTTPException(409,'분석이 끝나면 수정할 수 있어요.')
            before={key:row[key] for key in ('store_name','purchased_at','total_amount','category')};before['items']=json.loads(row['items'])
            after={key:getattr(body,key) for key in ('store_name','total_amount','category')}
            after['purchased_at']=body.purchased_at.isoformat() if body.purchased_at else None
            after['items']=[i.model_dump() for i in body.items]
            # raw_line is OCR provenance, not a user-editable training label.
            before['items']=[{key:item.get(key) for key in ('name','quantity','unit_price','line_total')} for item in before['items']]
            changed=[key for key in before if before[key]!=after[key]]
            if changed:
                db.execute('INSERT INTO corrections(id,receipt_id,owner_id,before_json,after_json,changed_fields,ocr_snapshot,predictions_snapshot,content_hash,engine,training_allowed) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(uuid.uuid4().hex,receipt_id,owner,json.dumps(before,ensure_ascii=False),json.dumps(after,ensure_ascii=False),json.dumps(changed),row['raw'],row['predictions'],row['content_hash'],row['engine'],int(body.allow_training)))
            db.execute("UPDATE receipts SET store_name=?,purchased_at=?,total_amount=?,items=?,status=?,memo=?,category=?,allow_training=?,date_inferred=0,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=? AND owner_id=?",(body.store_name,after['purchased_at'],body.total_amount,json.dumps(after['items'],ensure_ascii=False),body.status,body.memo,body.category,int(body.allow_training),receipt_id,owner))
            if not body.allow_training: db.execute('UPDATE corrections SET training_allowed=0 WHERE receipt_id=?',(receipt_id,))
            if job and job['status']=='failed':
                db.execute("UPDATE jobs SET status='completed',error=NULL WHERE receipt_id=?",(receipt_id,))
                db.execute("UPDATE receipts SET engine='manual-correction' WHERE id=?",(receipt_id,))
        return store.get(receipt_id,owner)

    @app.get('/api/receipts/{receipt_id}/corrections')
    def corrections(receipt_id: str,owner=Depends(auth.owner)):
        require(receipt_id,owner)
        with store.connect() as db:
            rows=db.execute('SELECT id,before_json,after_json,changed_fields,created_at,training_allowed FROM corrections WHERE receipt_id=? AND owner_id=? ORDER BY created_at DESC',(receipt_id,owner)).fetchall()
        return [{'id':r['id'],'before':json.loads(r['before_json']),'after':json.loads(r['after_json']),'changed_fields':json.loads(r['changed_fields']),'created_at':r['created_at'],'training_allowed':bool(r['training_allowed'])} for r in rows]

    @app.delete('/api/receipts/{receipt_id}',status_code=204)
    def delete(receipt_id: str,owner=Depends(auth.owner)):
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM receipts WHERE id=? AND owner_id=?',(receipt_id,owner)).fetchone()
            if row is None: raise HTTPException(404,'기록을 찾을 수 없습니다.')
            job=db.execute('SELECT status FROM jobs WHERE receipt_id=?',(receipt_id,)).fetchone()
            if job and job['status'] in {'queued','processing'}: raise HTTPException(409,'분석이 끝나면 삭제할 수 있어요.')
            db.execute('DELETE FROM receipts WHERE id=? AND owner_id=?',(receipt_id,owner))
        if row['image_path']:
            path=Path(row['image_path'])
            for candidate in (path,path.with_suffix('.ocr.json'),path.with_suffix('.error.log')):candidate.unlink(missing_ok=True)

    @app.get('/api/receipts/{receipt_id}/image')
    def receipt_image(receipt_id: str,owner=Depends(auth.owner)):
        with store.connect() as db: row=db.execute('SELECT image_path FROM receipts WHERE id=? AND owner_id=?',(receipt_id,owner)).fetchone()
        if row is None or not row[0] or not Path(row[0]).is_file():raise HTTPException(404,'사진이 없습니다.')
        return FileResponse(row[0],media_type='image/png')

    @app.get('/api/months/{month}/memo')
    def get_memo(month: str,owner=Depends(auth.owner)):
        month_bounds(month)
        with store.connect() as db:row=db.execute('SELECT memo FROM monthly_memos WHERE owner_id=? AND month=?',(owner,month)).fetchone()
        return {'month':month,'memo':row['memo'] if row else ''}

    @app.put('/api/months/{month}/memo')
    def save_memo(month: str,body: MemoInput,owner=Depends(auth.owner)):
        month_bounds(month)
        with store.connect() as db:db.execute('INSERT INTO monthly_memos VALUES(?,?,?) ON CONFLICT(owner_id,month) DO UPDATE SET memo=excluded.memo',(owner,month,body.memo))
        return {'month':month,'memo':body.memo}

    @app.get('/api/months/{month}/export.{extension}')
    def export(month: str,extension: Literal['jpg','png'],request: Request,include_memo: bool=False,owner=Depends(auth.owner)):
        if production:auth.limit(request,key='export:'+owner,maximum=12)
        summary=calendar_summary(store,owner,month)
        content=month_card(summary,'JPEG' if extension=='jpg' else 'PNG',include_memo)
        return Response(content,media_type='image/jpeg' if extension=='jpg' else 'image/png',headers={'Content-Disposition':f'attachment; filename="chagok-{month}.{extension}"'})

    @app.post('/api/demo')
    def demo(owner=Depends(auth.owner)):
        month=today().strftime('%Y-%m')
        with store.connect() as db:
            for index,(day,name,amount,category) in enumerate(SAMPLES):
                receipt_id=f'demo-{month}-{index}' if owner=='local' else f'demo-{owner}-{month}-{index}'
                db.execute("INSERT OR IGNORE INTO receipts(id,owner_id,source,store_name,purchased_at,total_amount,status,category) VALUES(?,?,'demo',?,?,?,'confirmed',?)",(receipt_id,owner,name,f'{month}-{day:02d}',amount,category))
                # Upgrade the initial five sample categories without editing user notes.
                db.execute("UPDATE receipts SET category=? WHERE id=? AND source='demo' AND category='기타'",(category,receipt_id))
        return {'month':month,'message':'샘플 영수증 20장이 준비됐어요. 실제 구매 기록은 아니에요.'}

    @app.get('/')
    def index(): return FileResponse(ROOT/'frontend/index.html',headers={'Cache-Control':'no-cache'})

    @app.get('/share/{month}',response_class=HTMLResponse)
    def share(month: str,include_memo: bool=False,owner=Depends(auth.owner)):
        month_bounds(month)
        memo='true' if include_memo else 'false'
        return HTMLResponse(f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>차곡 {month}</title><link rel="stylesheet" href="/static/print.css"><body><p>인쇄 대상에서 PDF로 저장을 선택해주세요. 다른 사람에게 공개되는 페이지는 아니에요.</p><button id="print">PDF로 저장 / 인쇄</button><img src="/api/months/{month}/export.png?include_memo={memo}" alt="{month} 월간 가계부"><script src="/static/print.js" defer></script></body></html>')

    app.mount('/static',StaticFiles(directory=ROOT/'frontend'),name='static')
    return app


app=create_app()
