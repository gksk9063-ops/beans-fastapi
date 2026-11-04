# 콩잎 질병 분류 · ResNet18 + FastAPI 동기/비동기

`microsoft/resnet-18`을 콩잎 3클래스(각진 잎 반점병 · 콩 녹병 · 정상)로 Fine-tuning하고 FastAPI + Celery + Redis + PostgreSQL로 서빙합니다.

| 결과 | 값 |
|---|---|
| 기준선 Head-only (3ep) | Val 0.7669 |
| Full Fine-tuning (8ep, lr 5e-5) · E4 | Val 0.9925 · Test 0.9688 |
| **최종 Full + 데이터 증강 (15ep, lr 5e-5) · E6 (서빙)** | **Val 1.0000 · Test 0.9453 · 확신도 0.9 미만 보류 시 나머지 113/113 정답** |
| 거리 변화 (Val 사진 60% 크기로 멀리) | E4 0.556 → **E6 0.992** |
| 콩잎 아님 표시 (ImageNet 특징 정규분포 거리 > 27.65) | 고양이 62.3 · 토마토 48.7 · 곤충 있는 콩잎 29.1 검출, 실제 콩잎 새 오탐 0장 (외부 사진 14장) |
| 응답 시간 (CPU, 5회 평균, E4로 측정 · E6도 같은 ResNet-18) | 동기 MISS 27.8ms · HIT 3.1ms · 비동기 접수 12.3ms |

## 실행

```bash
# 1) 학습 (WSL .venv, 제공 코드 그대로 · 옵션만 변경)
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 3 --batch-size 8 --freeze-backbone   # 기준선
python -m training.finetune --data dataset --output artifacts/beans_resnet18_full_e8 --epochs 8 --batch-size 8 --lr 5e-5   # E4 (증강 없음)
python -m training.finetune_aug --data dataset --epochs 15 --batch-size 8 --lr 5e-5 --output artifacts/beans_resnet18_aug_e15   # E6 최종 (train 증강)

# 2) .env
echo "MODEL_PATH=artifacts/beans_resnet18_aug_e15" > .env   # 이미 .env가 있으면 MODEL_PATH 줄만 수정
# (선택) 콩잎 아님 표시: ImageNet 모델 저장 + 거리 분포 만들기 (06 B-10)
python -c "from transformers import AutoImageProcessor, ResNetModel as M; M.from_pretrained('microsoft/resnet-18').save_pretrained('artifacts/resnet18_imagenet'); AutoImageProcessor.from_pretrained('microsoft/resnet-18').save_pretrained('artifacts/resnet18_imagenet')"
python -m analysis.ood_check --model artifacts/beans_resnet18_aug_e15 --feature-model microsoft/resnet-18 --tag IMGNETfeat
printf '\nLEAF_FEATURE_MODEL=artifacts/resnet18_imagenet\nLEAF_STATS_PATH=artifacts/beans_resnet18_aug_e15/ood_stats_IMGNETfeat.npz\n' >> .env

# 3) 서버 (redis · postgres · web · worker)
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

| 화면 | 주소 |
|---|---|
| 동기 | http://127.0.0.1:8000/vision/sync/ |
| 비동기 | http://127.0.0.1:8000/vision/async/ |
| 작업 목록 (추가) | http://127.0.0.1:8000/vision/jobs/ |
| Swagger | http://127.0.0.1:8000/docs |

- 테스트: `python -m pytest -q tests/test_api.py`
- 중지: `docker compose stop` (`down -v`는 DB 삭제 → 사용 금지)

## 폴더 구조

```text
fastapi_resnet18/
├── ai/
│   ├── config.py          # 환경변수·경로 (MODEL_PATH, DB, Redis)
│   ├── hf_model.py        # ResNet18 파이프라인 로드
│   ├── services.py        # 이미지 검증·추론·캐시 키
│   ├── cache_service.py   # Redis 캐시 HIT/MISS
│   ├── database.py        # PostgreSQL 연결·테이블 생성
│   ├── models.py          # image_jobs 테이블
│   ├── celery_app.py      # Celery 브로커 설정
│   ├── tasks.py           # Worker 추론 작업
│   ├── router_sync.py     # 동기 API
│   ├── router_async.py    # 비동기 접수·조회 API
│   ├── router_web.py      # HTML 화면
│   ├── router_jobs.py     # 작업 목록 API·화면 (추가)
│   └── leaf_check.py      # 콩잎 아님 거리 검사 (추가)
├── training/
│   ├── finetune.py        # Fine-tuning (제공 코드 그대로)
│   ├── finetune_aug.py    # finetune.py를 불러 train에만 증강 (추가)
│   └── augment.py         # 증강·조건 변형 함수 (추가)
├── analysis/              # 성능 분석 (추가)
│   ├── predict_split.py   #   Val/Test 예측·혼동행렬
│   ├── hsv_lesion.py      #   OpenCV 반점 면적비
│   ├── rule_review.py     #   보류 규칙 비교
│   ├── ensemble_eval.py   #   앙상블·TTA 비교
│   ├── measure_timings.py #   처리 시간 측정
│   ├── shift_eval.py      #   거리·구도·화질 바꾼 Val로 E4·E6 비교
│   ├── predict_folder.py  #   외부 사진 폴더 E4·E6 비교
│   ├── ood_check.py       #   콩잎 특징 정규분포 거리·기준값 (콩잎 아님 표시용)
│   ├── plot_predictions.py #  판정 결과 사진 격자
│   └── gradcam.py         #   Grad-CAM 판정 근거 히트맵
├── templates/             # base · sync · async · jobs(추가).html
├── static/                # style.css · result.js/css(추가)
├── tests/test_api.py      # 모의 통합 테스트
├── reports/
│   ├── 01~08_*.md         # 산출물
│   ├── data/              # timings.csv, 예측·분석 CSV
│   ├── evidence/          # 실행 캡처·응답 증빙
│   └── figures/           # 그림
├── main.py                # FastAPI 앱
├── compose.yaml           # redis · postgres · web · worker
├── Dockerfile
├── requirements.txt
├── .env.example           # .env는 업로드 안 함
├── dataset/               # train / val / test (업로드 안 함)
├── artifacts/             # 학습 모델 (업로드 안 함)
│   ├── beans_resnet18/          # 기준선 Head-only
│   ├── beans_resnet18_full_e8/  # E4 Full (이전 서빙)
│   ├── beans_resnet18_aug_e15/  # E6 Full + 증강 (서빙) + ood_stats_IMGNETfeat.npz
│   └── resnet18_imagenet/       # 콩잎 아님 검사용 ImageNet 특징 모델
└── media/jobs/            # 비동기 업로드 원본 (업로드 안 함)
```

`.env`, `dataset/`, `artifacts/`, `media/jobs/`, `external_photos/`(인터넷 사진 원본)는 GitHub에 올리지 않습니다. 자세한 설명은 [reports/08_최종기술보고서.md](reports/08_최종기술보고서.md).
