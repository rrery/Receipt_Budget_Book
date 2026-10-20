# 차곡 · 나다운 소비 기록

영수증 사진을 올리면 자동으로 소비 달력에 기록하는 가계부입니다. 기존 `Receipt_Budget_Book`의 한국어 OCR·규칙 파서·학습 모델을 그대로 재사용합니다.

## 바로 실행

**`start.cmd` 더블클릭** 또는 프로젝트 폴더에서 아래 명령을 실행하세요.

```powershell
.\.venv\Scripts\python.exe tools/serve.py
```

화면: **http://127.0.0.1:8000** · 로컬 API 테스트: **http://127.0.0.1:8000/docs**

이미 실행 중이면 주소만 열면 됩니다. 예전 화면이 보이면 브라우저에서 강력 새로고침(Ctrl+Shift+R)을 해주세요. 로컬 모드는 이 컴퓨터의 기존 기록을 보여줍니다. 공개 배포 모드는 로그인한 사람마다 기록을 분리합니다.

## 달력 사용하기

1. 맨 위 **영수증 업로드**로 사진을 올립니다. 한 번에 20장, 사진당 12MB까지 지원합니다.
2. 사진은 오늘 날짜에 분석 중으로 바로 표시됩니다. 구매 날짜와 금액을 읽으면 자동으로 해당 날짜의 합계에 반영합니다. **검토를 먼저 할 필요가 없습니다.**
3. 날짜를 읽지 못하면 업로드 날짜에 임시 표시하고, 금액은 미확인으로 남깁니다. 금액 미확인 기록은 합계에 더하지 않습니다.
4. 필요할 때 날짜를 누르고 **살펴보기**에서 상호명·날짜·분류·품목·금액을 수정하세요. 살펴본 기록으로 표시해도 합계가 중복으로 더해지지 않습니다.
5. **직접 기록**으로 수기 입력도 가능합니다. 영수증과 월간 메모는 각각 200자까지 저장합니다.
6. 달력 아래에서 지출 합계, 기록 수, 살펴본 수, 주요 분류, 가장 지출이 컸던 날과 지난달 비교를 확인합니다. 분류는 초기 키워드 규칙으로 제안하며 언제든 바꿀 수 있습니다.
7. **샘플 20장 체험**은 현재 월에 총 20건을 준비합니다. 여러 번 눌러도 중복되지 않습니다. 샘플은 목록에서 표시하며 통계와 저장 카드에 포함된 사실도 안내합니다.
8. **이번 달 저장하기**에서 JPEG로 저장하거나 **PDF / 인쇄**를 누르세요. PDF는 열린 인쇄 화면에서 브라우저의 ‘PDF로 저장’을 선택합니다. 지원하는 모바일에서는 파일 공유도 가능합니다.

저장 카드에는 상호명·영수증 사진을 넣지 않습니다. 월간 메모는 기본 제외하며 체크했을 때만 포함합니다. 저장 화면은 로그인한 본인만 열 수 있고 공개 공유 링크를 생성하지 않습니다.

## 수정 내용을 학습에 활용하기

기록의 인식 필드를 수정하면 전후 값, 원래 OCR 좌표·텍스트, 예측 결과를 별도 `corrections` 테이블에 보관합니다. **영수증·월 메모는 학습 후보에서 제외합니다.** 화면의 선택 동의가 있는 기록만 운영자가 아래 명령으로 내보낼 수 있습니다.

```powershell
.\.venv\Scripts\python.exe tools/export_corrections.py
```

출력: `runtime/training_candidates/날짜/`의 `candidates.jsonl`, 이미지, manifest. 날짜·상호명 등을 바꾼 기록은 필드 정정 자료이며 즉시 BIO 학습 정답이 되지는 않습니다. 사람이 OCR 토큰과 정답을 정렬·검수한 후 기존 학습 파이프라인에 넣어야 합니다. 사용자가 영수증을 지우면 연결된 정정 이력도 지워지고, 동의를 철회하면 새로운 내보내기에서 제외됩니다. 이미 내보낸 사본의 정리·기학습 모델에 관한 정책은 운영 단계에서 별도로 관리해야 합니다.

## 실제 모델 학습

```powershell
.\.venv\Scripts\python.exe tools/train.py
# 모든 fold와 최종 가중치를 다시 학습
.\.venv\Scripts\python.exe tools/train.py --retrain
# Supabase 자료를 다시 읽고 재학습
.\.venv\Scripts\python.exe tools/train.py --refresh-data
```

기존 데이터 생성→검증→분할→동결 LayoutLMv3 + Linear head 학습을 실행합니다. 이미지 원본은 `.cache/receipt-images`에 캐시합니다. 같은 Storage 경로의 이미지 자체를 교체한 경우 캐시도 갱신해야 합니다. 학습 상태와 로그는 `runtime/training_status.json`, `runtime/training.log`에 남습니다. 고객 메인 화면에는 학습 내용을 표시하지 않습니다.

- 가중치: `Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/final_model/linear_head.safetensors`
- 설정·성능: `Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/model_info.json`
- 모델 검증 상세: [첫 구현 보고서](IMPLEMENTATION_REPORT.md)

## 새 컴퓨터 준비

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
.\.venv\Scripts\python.exe -m pip install -r Receipt_Budget_Book/requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install uv
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.runtime'
.\.venv\Scripts\uv.exe python install 3.12
.\.venv\Scripts\uv.exe venv --python 3.12 .ocr-venv
.\.venv\Scripts\uv.exe pip install --python .ocr-venv/Scripts/python.exe -r requirements-ocr.txt
```

학습 산출물을 준비한 뒤 `tools/prepare_models.py`로 기본 모델 캐시를 받습니다. 백엔드/Torch와 PaddleOCR는 별도 환경·프로세스를 사용합니다. 다른 Supabase 학습 프로젝트를 쓰려면 환경변수 `SUPABASE_URL`, `SUPABASE_KEY`를 지정합니다. 웹 가계부는 연구용 Supabase 테이블에 쓰지 않습니다.

## 검사와 문서

GitHub에 올리거나 팀원이 코드를 받을 때는 [Git 팀 공유 안내](GIT_TEAM_GUIDE.md)를 참고하세요.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/test_live.py --fresh
```

첫 명령은 API·파서·저장·인증 회귀 검사입니다. 두 번째는 서버를 켠 상태에서 합성 영수증으로 실제 OCR→추론→자동 달력 기록을 검증합니다. 실소비 데이터와 구분된 검사용 기록을 남깁니다. 화면 코드 검사는 `requirements-dev.txt` 설치 후 `tools/check_frontend.py`로 실행합니다.

- 이번 개선: [UPGRADE_REPORT.md](UPGRADE_REPORT.md), [UPGRADE_PLAN.md](UPGRADE_PLAN.md)
- **공개 웹 배포 준비:** [DEPLOYMENT.md](DEPLOYMENT.md), `Dockerfile`, `compose.yaml`, `.env.example`
- 실제 자료: `runtime/budget.sqlite3`, `runtime/uploads/`. 업데이트 전 백업은 `runtime/backups/`에 있습니다.
- 백업은 서버를 끄고 `runtime` 전체를 복사하거나 SQLite backup API로 수행합니다. 실행 중 DB 파일만 복사하면 안 됩니다.

현재 로컬 주소는 이 컴퓨터에서만 접속됩니다. 공개 주소는 실제 서버·도메인·HTTPS 구성과 컨테이너 검증을 마친 다음 제공할 수 있습니다.
