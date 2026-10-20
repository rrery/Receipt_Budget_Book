# 영수증 달력 가계부 구현 및 검증 보고서

> 이 문서는 0.1 버전 기록입니다. 자동 달력 기록·새 화면·공개 배포 준비 등 최신 변경은 [UPGRADE_REPORT.md](UPGRADE_REPORT.md)를 참고하세요.

최종 정리일: 2026-10-18 (한국 시간)

## 1. 검토

요청한 방향에 따라 기존 Python 프로그램을 대체하지 않고, 기존 함수와 새 웹 API 사이에 어댑터를 추가했다. 기존 Supabase는 검수된 학습 자료를 읽는 용도로만 사용했으며 연구용 테이블을 수정하지 않았다.

Supabase 조회 당시 영수증 171건, OCR 6,085행, 검수 라벨 6,007행을 확인했다. 검수된 자료가 있는 영수증 136장 중 24장은 OCR 좌표가 이미지 범위를 벗어나 기존 검증 코드에서 제외되었다. 나머지 실제 영수증 112장을 학습 자료로 사용했다. 합성 검사용 영수증과 화면의 샘플 5건은 학습에 사용하지 않았다.

기존 코드에서 날짜·금액 파싱 오류와 CPU 최적 가중치 보관 오류를 확인해 필요한 부분만 수정했다. 운영 서비스의 사용자 인증과 Supabase RLS는 이번 로컬 구현과 구분한다.

## 2. 기획

구현 전 기획서는 [BACKEND_PLAN.md](BACKEND_PLAN.md)에 있다. 이번 결과물은 **한 컴퓨터에서 직접 테스트하는 개인용 로컬 MVP**다.

- FastAPI가 화면과 API를 `127.0.0.1:8000`에서 제공한다.
- 웹에서 추가한 기록은 SQLite와 로컬 이미지 폴더에 보관한다.
- DB에 저장된 작업을 단일 작업자가 처리하고, 서버 재시작 시 중단된 작업을 다시 대기열에 넣는다.
- 사진 분석 결과는 검토 상태로 저장하며, 확정한 기록만 날짜별·월별 소비 합계에 포함한다.
- PaddleOCR와 Torch는 환경·프로세스를 분리한다. 학습한 모델의 backbone은 캐시하고 웹 추론 시 로컬 캐시에서 읽는다.

## 3. 작업내용

### 기존 프로그램 활용 및 수정

| 파일 | 변경 또는 활용 내용 |
|---|---|
| `create_dataset.py`, `validate_dataset.py`, `split_dataset.py` | 기존 코드로 실제 검수 자료 생성·검증·분할 |
| `train_layoutlmv3.py` | 기존 동결 backbone + Linear head 학습 유지. CPU 최적 가중치 스냅샷에 `.clone()` 추가, 선택적 이미지 캐시 연결 |
| `predict_layoutlmv3.py` | 기존 BIO 엔티티 예측 유지. 로컬 OCR 어댑터 주입, 모델 한 번 로드, 웹 추론의 캐시 전용 로드 추가 |
| `ysz_ocr.py` | 기존 `run_ocr` 재사용. 인식 텍스트에 대응하는 `rec_polys` 우선 사용, 박스/텍스트/점수 길이 검사 |
| `receipt_parser.py` | 실제 달력 날짜 검사, 두 자리 연도 정규화, 받은 현금 제외, 단가×수량 계산, 품목 헤더 아래를 상호명으로 고르는 문제 보정 |
| `ocr_linebyline.py` | 기존 줄 단위 묶음 함수 재사용 |
| `main.py`, `database.py`, `item_y.py` | 기존 연구/배치 도구 유지. 새 API가 배치 `main.py`를 실행하지 않으므로 첫 138개 건너뛰기 설정은 웹 업로드에 적용되지 않음 |

### 새 백엔드

- `backend/app.py`: 업로드, 작업 조회/재시도, 영수증 CRUD, 달력 집계, 샘플 생성, 학습 현황 API.
- `backend/store.py`: SQLite 스키마·트랜잭션·인덱스와 결과 직렬화.
- `backend/worker.py`: DB 대기열 처리, 실패 기록, 서버 재시작 후 작업 복구.
- `backend/pipeline.py`: 기존 OCR/규칙 파서/모델 추론 연결. 신뢰도가 높은 상호명·날짜·총액 엔티티를 결합하고 품목 파싱은 기존 코드를 유지.
- 파일 형식·파일 크기·픽셀 수 검증, 이미지 해시 중복 방지, 고유 작업 경로, Host/Origin 검사 적용.
- 분석 중에는 수정·삭제를 차단한다. 실패 결과는 재시도 또는 수기로 정정할 수 있다.
- 인식하지 못한 날짜를 오늘 날짜로 임의로 채우지 않는다. 총액 누락·품목 합계 불일치·모델 추론 실패를 검토 화면에 표시한다.

### 직접 보는 테스트 화면

`frontend/`에 소비 달력, 월 요약, 영수증 사진 업로드, 날짜별 목록, 검토/정정 창, 수기 입력, 학습 현황을 구현했다. 샘플 자료임을 표시하고 외부 폰트 다운로드 없이 시스템 한글 글꼴을 사용한다.

실행: `start.cmd` 더블클릭 또는 아래 명령.

```powershell
.\.venv\Scripts\python.exe tools/serve.py
```

- 화면: <http://127.0.0.1:8000>
- API 테스트: <http://127.0.0.1:8000/docs>
- 자세한 순서: [README.md](README.md)

### 학습 및 실행 도구

- `tools/train.py`: 기존 코드의 학습 순서 실행, 로그·상태·5초 간격 진행 갱신. `--retrain`으로 기존 fold/최종 결과를 다시 계산.
- `tools/cache_images.py`, `image_cache.py`: 실제 이미지 바이트를 변경하지 않는 로컬 캐시와 다운로드 재시도.
- `tools/make_sample.py`: 한국어 합성 영수증 1장 생성.
- `tools/test_live.py --fresh`: 새로운 검사용 이미지 복사본으로 HTTP 업로드부터 OCR·모델·달력까지 재검증.
- `tools/check_frontend.py`: JavaScript 문법 및 HTML 요소 연결 검사.

## 4. 검증결과

### 실제 학습

자료 분할은 train 89장, validation 11장, 독립 test 12장이다. train+validation 100장으로 5-Fold 검증 후 최종 24 epoch 학습을 수행했다. 기본 LayoutLMv3는 동결하며, 학습 파라미터는 13,073개다. CPU 최적 가중치 복사 오류를 수정한 코드로 재학습해 최종 산출물과 지표를 확정했다.

| 지표 | 결과 |
|---|---:|
| 학습 전 교차 검증 평균 F1 | 2.18% |
| 학습 후 5-Fold 평균 F1 | **39.71%** |
| Fold 간 F1 표준편차 | 16.45%p |
| 독립 Test F1 | **51.98%** |
| 독립 Test Precision / Recall | 79.73% / 38.56% |
| 독립 Test 토큰 정확도 | 77.26% |
| 가중치 크기 | 52,868 bytes |

| 독립 Test 항목 | F1 |
|---|---:|
| 상호명 | 0.00% |
| 날짜 | 33.33% |
| 총액 | 40.00% |
| 품목명 | 0.00% |
| 품목 금액 | 81.69% |
| 수량 | 85.71% |
| 단가 | 41.67% |
| 주소 | 0.00% |

상호명·품목명·주소는 이번 독립 테스트에서 모델 단독으로 올바른 엔티티를 찾지 못했다. 웹서비스에서는 기존 규칙 파서의 결과를 유지하고 사용자 확인을 거친다. Fold 2 F1이 7.03%로 특히 낮아, 추가 자료 검수와 학습 개선이 필요하다.

가중치 SHA-256: `12b7f3bc97a280237925aaaa11882886ced63387f7236274fc86774f6f4eb395`.

산출물:

- 가중치: `Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/final_model/linear_head.safetensors`
- 모델 설정·최종 성능: `Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/model_info.json`
- 교차 검증: `Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/cross_validation_summary.json`
- 제외 자료·분할 근거: `Receipt_Budget_Book/data/layoutlm/dataset_build_report.json`, `split_manifest.json`
- 실행 로그: `runtime/training.log`

F1은 BIO 엔티티 단위 지표이며 영수증 전체가 정확히 맞는 비율이 아니다. 특히 일부 엔티티는 학습 성능이 낮으므로 모델 출력만으로 가계부를 자동 확정하지 않는다. 테스트 자료가 12장으로 작고 fold 사이 성능 편차도 있어, 운영 정확도를 주장할 근거로 충분하지 않다.

### API·코드 검사

- 자동 테스트 15개 통과: 월 경계, 검토 제외, 수정·삭제 반영, 날짜/금액 검증, 업로드 중복, 작업 실패·재시도, 재시작 보존, 이미지 검사, Host/Origin, CPU 가중치 스냅샷 등.
- 화면 JavaScript 구문 검사 및 참조한 HTML 요소 46개 연결 검사 통과.
- 테스트 임시 경로를 프로젝트 `runtime` 내부로 지정해 이전 샌드박스 임시 폴더 권한과 분리했다.
- Starlette의 테스트 클라이언트에 관한 deprecation 경고 1건은 테스트 실패가 아니다.

### 실제 연결 검사

합성 영수증을 HTTP로 업로드해 실제 PaddleOCR와 학습 가중치의 추론을 수행했다. 상호명 `차곡 테스트 카페`, 구매 날짜 `2026-10-17`, 총액 `10,000원`, 품목 2개를 확인했고, 확정 시 달력 합계에 반영되는 흐름을 검증했다. 검증 후 해당 기록은 사용자가 직접 검토할 수 있도록 검토 상태로 남겼다. 결과는 `runtime/live_test_result.json`에 기록한다.

이 검사는 연결 동작의 증거이며 실제 다양한 영수증에서의 인식 정확도 평가는 아니다. 실제 학습·평가는 앞의 별도 자료로 수행했다.

### 환경 및 범위

학습/백엔드: Python 3.14.7, Torch 2.10.0+xpu(CPU 실행), Transformers 5.16.1, scikit-learn 1.9.0, Supabase 2.31.0. OCR: 별도 Python 3.12.15, PaddlePaddle 3.3.1, PaddleOCR 3.7.0. 핵심 학습 버전은 `Receipt_Budget_Book/requirements-tested.txt`에 남겼다.

현재 연결된 브라우저가 없어 자동화 도구로 실제 화면 렌더링·클릭·모바일 레이아웃을 확인하지 못했다. HTTP 및 정적 코드 검사와 실제 브라우저 시각 검증을 구분한다. 사용자는 위 로컬 주소에서 직접 화면을 열어 테스트할 수 있다.

다중 사용자 로그인, Supabase 운영용 저장/RLS, 외부 배포, 대규모 작업 큐·업로드 제한 정책은 후속 범위다. 현재 금액은 원화 정수·일반 소비를 대상으로 하며 반품·복합 할인·분할 결제는 추가 설계와 실제 자료 검증이 필요하다.
