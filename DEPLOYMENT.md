# 웹 공개 배포 준비

현재 앱은 Python OCR·Torch 모델을 실행하므로 Python 컨테이너를 실행할 수 있는 서버가 필요하다. 프론트엔드와 API는 같은 주소에서 제공한다. 이 문서는 배포 절차를 제공하며, 실제 도메인 연결이나 공개 배포를 수행한 상태는 아니다.

## 구성

`브라우저 → Caddy(HTTPS) → FastAPI(단일 프로세스) → SQLite/파일 + OCR 작업자`

- 공개 모드에서는 회원가입/로그인 후 자신의 기록만 읽고 수정한다. 비밀번호는 scrypt로 저장하며, 세션 쿠키는 HttpOnly/Secure/SameSite=Strict 속성을 사용한다.
- 사용자별 영수증, 이미지, 작업, 월 메모, 수정 이력과 통계를 분리한다. 같은 사진을 다른 계정에서 올려도 서로의 기록을 반환하지 않는다.
- `/api/training`, `/docs`는 공개 모드에서 노출하지 않는다. 모델 학습은 운영자 CLI 작업이다.
- `APP_ENV=production`에서 정확한 호스트와 HTTPS 원점이 없으면 시작을 거부한다. 인증·업로드·내보내기 요청 제한과 업로드 보관 한도를 적용한다.
- 고객 DB와 이미지, 모델 캐시는 `ledger_data` 영속 볼륨에 저장한다. 학습 가중치는 읽기 전용 `/models`로 마운트한다.

## 서버에서 실행

1. Docker Engine과 Compose를 준비한다. 초기 소규모 서비스는 CPU와 메모리 여유가 있는 서버를 사용한다. Compose는 앱 메모리를 4GB로 제한한다. 동시 처리 성능과 비용은 실제 서버에서 부하 테스트 후 정한다.
2. 저장소 코드와 `Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/` 산출물을 서버로 옮긴다. **로컬 `runtime` DB·개인 영수증·캐시·가상환경은 소스 이미지에 복사하지 않는다.** `.dockerignore`가 이를 제외한다.
3. `.env.example`을 `.env`로 복사하고 실제 `APP_DOMAIN`을 입력한다. DNS의 A/AAAA 레코드를 서버로 연결하고 외부 80/443 포트를 연다.
4. 아래 순서로 빌드, 최초 모델 준비, 실행한다.

```sh
docker compose build
docker compose run --rm app python tools/prepare_models.py
docker compose up -d
docker compose logs -f app
```

최초 준비 시 Microsoft LayoutLMv3 backbone과 한국어 OCR 모델을 내려받는다. 이미 학습한 Linear head는 다시 학습하지 않는다. 준비 후 웹 추론은 로컬 backbone 캐시를 사용한다.

Caddy는 설정한 실제 도메인의 HTTPS 인증서와 갱신을 처리한다. FastAPI 포트 8000은 외부에 공개하지 않으며, 프록시 헤더 신뢰 설정은 이 격리된 구성 안에서만 사용한다. [Caddy HTTPS 요건](https://caddyserver.com/docs/automatic-https), [FastAPI 프록시 안내](https://fastapi.tiangolo.com/advanced/behind-a-proxy/).

앱 주소: `https://설정한도메인`.

## 데이터와 계정

- 로컬 모드의 기존 자료 소유자는 `local`이다. 공개 모드의 새 계정이 이 자료를 볼 수 없게 되어 있다. 공개 서버에 기존 로컬 DB를 그대로 올리는 방식으로 이관하지 않는다.
- SQLite/WAL과 단일 작업자 구조이므로 **앱 복제본을 여러 개 띄우지 않는다.** 수평 확장이 필요하면 Postgres, 객체 스토리지, 별도 작업 큐로 바꾼다.
- 비밀번호 재설정·메일 인증·계정 탈퇴 화면은 이번 범위에 없다. 실제 고객 서비스 오픈 전 계정 복구와 개인정보 보관/삭제 절차를 마련해야 한다.
- 현재 요청 제한은 단일 앱 프로세스 기준이다. 여러 서버로 늘릴 때는 공유 저장소 기반 제한으로 바꾼다.

## 백업과 업데이트

DB 파일만 실행 중 복사하지 말고 앱을 정지한 상태에서 `ledger_data` 볼륨 전체를 백업하거나 SQLite backup API를 사용한다. 원본 영수증과 DB를 함께 복원해야 한다. 볼륨을 지우는 `docker compose down -v`는 자료 삭제 명령이므로 일반 종료에는 사용하지 않는다.

```sh
docker compose stop app
# 운영 환경의 백업 시스템으로 ledger_data 전체 백업
docker compose start app
```

코드 갱신 전 백업 후 `docker compose up -d --build`로 반영한다. 모델 산출물을 바꾼 경우 앱 재시작과 실제 사진 검증을 수행한다. 공개 모드의 세션·소유권·업로드·저장 카드 기능을 실제 HTTPS 환경에서 다시 확인한다.

## 검증 범위

HTTPS를 사용하는 테스트 클라이언트에서 계정 간 접근 차단, 로그인·로그아웃, Secure 쿠키, 원점 검사와 공개 개발 도구 차단을 검증했다. 현재 작업 환경에는 Docker와 연결된 브라우저가 없어 Linux 컨테이너 빌드·Caddy 인증서 발급·실제 모바일 브라우저 검증은 수행하지 못했다. 공개 서버에서 위 명령과 검증을 마친 뒤 운영 주소를 안내해야 한다.
