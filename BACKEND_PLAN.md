# 영수증 달력 가계부 백엔드 기획서

> 최신 개선 계획과 결과: [UPGRADE_PLAN.md](UPGRADE_PLAN.md), [UPGRADE_REPORT.md](UPGRADE_REPORT.md).

작성일: 2026-10-17

구현·검증 완료 현황(2026-10-18)은 [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)를 참고한다.

## 1. 검토

기존 `Receipt_Budget_Book` 코드를 교체하지 않고 재사용한다. Supabase에서 영수증 171건, OCR 6,085행, 검수 라벨 6,007행을 읽기 전용으로 확인했다. 기존 데이터 생성·분할·학습 코드로 실제 LayoutLMv3 Linear head 가중치를 생성하고, 같은 추론 코드를 API에 연결한다. 학습 시 잘못된 좌표 등으로 제외되는 자료는 기존 검증 기준을 유지해 기록한다.

## 2. 기획

### 이번 구현 범위

이 저장소에서 실행하는 **개인용 로컬 MVP**를 만든다. FastAPI가 화면과 API를 같은 주소에서 제공하고, SQLite에 영수증·품목·처리 작업을 영속 저장한다. Supabase의 기존 연구 테이블을 수정하지 않는다. 기존 코드의 Supabase 접근은 학습 자료 생성에 사용한다. 다중 사용자 운영용 배포는 별도 단계이며 로컬 실행기는 127.0.0.1에만 바인딩한다.

| 기존 파일 | 재사용 방식 |
|---|---|
| create_dataset.py / validate_dataset.py / split_dataset.py | 검수된 실제 자료 생성·검증·분할 |
| train_layoutlmv3.py | 동결 backbone + Linear head 5-Fold 학습, holdout 평가 |
| predict_layoutlmv3.py | 모델 캐시 및 DB 어댑터를 통해 로컬 OCR 결과의 BIO 엔티티 추론 |
| ysz_ocr.py | 별도 OCR 환경에서 그대로 호출, 인식 박스 정렬만 보정 |
| receipt_parser.py / ocr_linebyline.py | 기존 규칙 기반 상호명·날짜·합계·품목 파싱 |
| main.py / database.py / item_y.py | 기존 배치/연구 도구 유지, API가 직접 실행하지 않음 |

### 처리 흐름

1. JPEG/PNG/WebP 한 장을 크기·픽셀 수·실제 형식 검사 후 고유 경로에 저장한다.
2. 이미지 해시로 중복 업로드를 방지하고 DB에 queued 작업을 기록한다.
3. 단일 작업자가 queued → processing → completed/failed 상태를 갱신한다. 서버 재시작 시 processing을 queued로 복구한다.
4. OCR 프로세스가 기존 `run_ocr`로 텍스트와 좌표를 만든다. 이번 로컬 경로는 입력 이미지를 재인코딩하여 동일 이미지에 OCR·추론을 수행한다. 원근 보정에 따른 좌표계 혼용을 피한다.
5. 규칙 파서를 기본 결과로 사용하고, 학습 모델의 상호명·날짜·총액 엔티티를 신뢰도 기준으로 결합한다. 품목은 기존 파서를 유지한다. 어떤 엔진을 사용했는지 저장한다.
6. 결과와 작업 성공을 하나의 SQLite 트랜잭션으로 저장한다. 분석 결과는 사용자가 검토·수정·확정한다. 확정된 영수증만 월별 합계에 반영한다.
7. 날짜 없는 결과는 검토 목록에서 정정한다. 수기 입력과 샘플 데이터는 OCR과 명확히 구분한다.

### 데이터/API

- receipts: id, source, image hash/path, store_name, purchased_at(date), total_amount(정수 원), items(JSON), raw OCR(JSON), model predictions(JSON), status, engine, created_at.
- jobs: id, receipt_id, status, error, timestamps. 이미지·결과는 외부 공개 디렉터리에 두지 않는다.
- `GET /api/status`: 실행 환경, OCR 준비, 가중치·학습 진행 상황.
- `POST /api/receipts/upload`: 업로드와 비동기 분석 접수.
- `GET /api/jobs/{id}`, `POST /api/receipts/{id}/retry`: 상태 확인 및 실패 재시도.
- `GET/POST /api/receipts`, `GET/PATCH/DELETE /api/receipts/{id}`: 목록·수기 입력·상세·정정·삭제.
- `GET /api/calendar?month=YYYY-MM`: 날짜별 확정 소비 합계·건수.
- `POST /api/demo`: 현재 월의 명시적 데모 자료 생성(반복 실행 시 중복 방지).
- `GET /api/receipts/{id}/image`: 해당 영수증의 이미지 조회.
- `GET /api/training`: 학습 상태와 평가 결과.
- `/docs`: 자동 API 명세 및 직접 요청 화면.

### 접근 및 운영 경계

루프백 전용·단일 사용자 모드다. Host/Origin 검사를 적용하고 CORS를 개방하지 않는다. 환경이 준비되지 않으면 OCR 실패 사유를 보여주며 성공 결과를 만들지 않는다. 재시도 버튼을 제공한다. 로그에는 인증 키를 기록하지 않는다.

운영 서비스 전환 시 Supabase Auth JWT 검증, user_id 외래키/인덱스, RLS, 비공개 Storage, 업로드 제한/요청 제한, 별도 작업 큐, 백업·모니터링을 추가한다. 이번 SQLite 구조는 실제 Supabase 연구 테이블과 섞지 않으므로 데이터 이관은 별도 마이그레이션으로 진행한다.

## 3. 작업내용

실제 변경 및 학습 산출물은 `IMPLEMENTATION_REPORT.md`, 실행법은 `README.md`에 기록한다.

## 4. 검증계획

실제 학습 데이터 검증·분할 누수 검사·5-Fold/holdout 평가를 수행한다. API는 달력 월 경계, 수정 반영, 입력 검증, 중복 업로드, 실패·재시도, DB 재시작 보존, 삭제를 검증한다. 준비된 화면과 `/docs`를 통해 사용자가 같은 흐름을 직접 재현할 수 있도록 한다.

참고: [FastAPI 업로드](https://fastapi.tiangolo.com/tutorial/request-files/), [LayoutLMv3 공식 문서](https://huggingface.co/docs/transformers/model_doc/layoutlmv3).
