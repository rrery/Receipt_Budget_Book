# GitHub 업로드 및 팀원 실행 안내

## 검토

- 전체 프로젝트 루트에 `master` 브랜치 Git 저장소를 준비했습니다.
- 기존 `Receipt_Budget_Book/.git` 이력은 `.runtime/git-backups/legacy-20261021`에 보존했습니다. 모델 코드는 루트 저장소에서 일반 파일로 관리합니다.
- 가상환경, 모델 캐시, 학습 산출물, 실제 영수증 이미지, DB, 정정 학습 후보 및 `.env` 설정은 Git에 올리지 않습니다. `.env.example`은 공유합니다.
- 이 문서는 업로드 준비 안내입니다. 원격 연결 및 실제 push 여부는 터미널에서 확인하세요.

## 기획

소스는 GitHub로 공유하고, 학습 가중치는 팀 전용 경로로 따로 전달합니다. 팀원마다 가상환경과 모델 캐시를 새로 준비합니다. 개인 영수증이 들어 있는 `runtime` 폴더는 전달하지 않습니다.

현재 origin은 `https://github.com/minsoo9810-sys/Receipt_Budget_Book.git`이며, 기존 master 이력을 유지합니다. 새 원격 저장소를 만들 필요가 없습니다.

## 작업내용

### 1. 내가 처음 업로드하기

프로젝트 루트에서 아래 명령을 실행합니다. 커밋 대기 파일을 확인한 다음 진행하세요.

```powershell
git status
git diff --cached --stat
git commit -m "feat: receipt calendar service"
git remote -v
git push -u origin master
```

첫 push에서는 GitHub 로그인이 필요할 수 있습니다. 기존 원격 이력은 이미 연결했습니다. 강제 push는 사용하지 않습니다.

커밋 시 작성자 정보가 없다는 오류가 나면 이 저장소에 사용할 본인 정보를 지정하고 다시 커밋합니다. 이메일은 GitHub 설정의 비공개 noreply 주소도 사용할 수 있습니다.

```powershell
git config user.name "내 이름"
git config user.email "내 GitHub 이메일"
```

이후 수정분을 올릴 때는 `git add .`, `git commit -m "변경 내용"`, `git push` 순서로 진행합니다.

### 2. 팀원이 처음 받아서 실행하기

```powershell
git clone https://github.com/minsoo9810-sys/Receipt_Budget_Book.git
cd Receipt_Budget_Book
```

[README의 새 컴퓨터 준비](README.md#새-컴퓨터-준비)에 따라 의존성을 설치하세요. 기존 PC의 `.venv`와 `.ocr-venv`를 복사하지 않습니다.

아래 두 파일은 Git에 포함되지 않으므로 팀 내부에서 별도로 받아 같은 경로에 배치해야 합니다.

```text
Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/model_info.json
Receipt_Budget_Book/outputs/layoutlmv3_linear_probe/final_model/linear_head.safetensors
```

그다음 인터넷이 연결된 상태에서 모델 캐시를 준비하고 서버를 실행합니다.

```powershell
.\.venv\Scripts\python.exe tools/prepare_models.py
.\.venv\Scripts\python.exe tools/serve.py
```

브라우저에서 http://127.0.0.1:8000 을 열어 샘플 영수증으로 확인합니다. 새 PC에는 기존 개인 소비 기록이 나타나지 않습니다.

### 3. 팀원이 이후 변경분 받기

로컬 수정이 없는 상태에서 실행하세요. 수정 중이면 별도 브랜치에 커밋한 뒤 업데이트합니다. 오류가 나면 강제 초기화하지 마세요.

```powershell
git status
git switch master
git pull --ff-only origin master
```

### 4. 팀원이 수정해서 검토 요청하기

```powershell
git switch -c feature/calendar-improvement
# 코드 수정 후
git add .
git commit -m "feat: improve calendar"
git push -u origin feature/calendar-improvement
```

GitHub에서 해당 브랜치의 Pull Request를 열고 검토 후 master에 병합합니다. 팀장님 저장소로 PR을 보낼 때는 해당 저장소의 실제 병합 대상 브랜치를 선택합니다. `pull`은 변경분을 받는 명령이고, **Pull Request는 변경분을 병합해 달라는 검토 요청**입니다.

팀원에게 보낼 메시지 예시:

```text
영수증 가계부 소스 올렸어: [저장소 주소]
처음이면 clone, 이미 받았다면 로컬 수정 확인 후 master에서 git pull --ff-only origin master 해줘.
GIT_TEAM_GUIDE.md와 README대로 환경을 준비하면 돼.
학습 가중치 두 파일은 별도 팀 공유 경로로 전달할게.
```

## 검증결과

실제 준비 결과는 [GIT_PREPARATION_REPORT.md](GIT_PREPARATION_REPORT.md)를 확인하세요. Git 준비 자체는 서비스 기능을 변경하지 않습니다. 원격 주소가 없으므로 GitHub 업로드와 팀원 PC 실행은 아직 검증하지 않았습니다.
