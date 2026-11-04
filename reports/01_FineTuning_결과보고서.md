# ResNet18 콩잎 3클래스 Fine-tuning 결과보고서

- 평가 대상: 콩잎 이미지 3클래스 분류 (angular_leaf_spot / bean_rust / healthy)
- 기준 모델: `microsoft/resnet-18` (ImageNet 사전학습)
- 학습 코드: 제공 `training/finetune.py` **원본 그대로** (코드 수정 없이 실행 옵션만 변경). E6만 원본을 불러 쓰는 별도 파일 `training/finetune_aug.py`로 **train 이미지 증강**을 추가함 (원본 파일은 수정 안 함)
- 학습 방식: ① 필수 기준선 Head-only (Backbone freeze) ② 비교 실험 Full Fine-tuning ③ Full + 데이터 증강 (E6)
- 입력 데이터: `dataset/train`, `dataset/val` · 최종 보관 데이터: `dataset/test` (학습·선택에 사용하지 않음)
- 모델 저장 위치: 기준선 `artifacts/beans_resnet18/` · 증강 없는 Full `artifacts/beans_resnet18_full_e8/` (E4) · **서빙 모델 `artifacts/beans_resnet18_aug_e15/` (E6)**
- 실험 상태: 실행 완료 (2026-11-04, WSL2 + CUDA GPU)

## 1. 데이터 구성

| 클래스 | Train | Validation | Test |
|---|---:|---:|---:|
| angular_leaf_spot | 345 | 44 | 43 |
| bean_rust | 348 | 45 | 43 |
| healthy | 341 | 44 | 42 |
| 합계 | **1,034** | **133** | **128** |

- 원본 폴더: `train / validation / test` → 코드 적용 폴더: `train / val / test` (`validation` 이름만 `val`로 변경, 파일 이동·재분할 없음)
- 이미지 크기: 1,295장 모두 500×500
- 데이터 누출 점검: 전체 1,295장 **md5 해시 완전 중복 0건** (split 간·split 내 모두). 유사 이미지(near-duplicate)는 점검하지 않음
- 정리한 파일: `train/healthy/healthy_train.120tore` = macOS `.DS_Store` 파일(이미지 아님) → 삭제. 코드의 확장자 필터(`.jpg .jpeg .png .webp`)로도 제외됨
- 클래스 균형: 각 split에서 클래스 비율 거의 1:1:1 → Accuracy를 주 지표로 사용

## 2. 실행 설정

| 파라미터 | E0 기준선 (필수) | E4 Full (증강 없음) | **E6 Full + 증강 (서빙)** |
|---|---|---|---|
| base_model | microsoft/resnet-18 | microsoft/resnet-18 | microsoft/resnet-18 |
| 학습 범위 | Head-only (`--freeze-backbone`, `classifier.*`만 학습) | Full (전체 가중치 학습) | Full |
| epochs | 3 | 8 | 15 |
| batch_size | 8 | 8 | 8 |
| learning_rate | 0.0002 (기본값) | 0.00005 | 0.00005 |
| optimizer | AdamW (제공 코드) | AdamW (제공 코드) | AdamW (제공 코드) |
| 데이터 증강 | 없음 (제공 코드) | 없음 (제공 코드) | **train에만**: 거리(확대 0.4~1.0 / 축소 0.5~0.9) · 뒤집기 · 회전 ±30° · 밝기·대비·채도 · 흐림·저화질·노이즈·JPEG 압축 (각 30%) |
| seed | 42 (제공 코드) | 42 (제공 코드) | 42 (증강 난수도 42) |
| 모델 저장 규칙 | Val accuracy가 이전보다 **클 때만** 저장 (best epoch) | 같음 | 같음 (Val은 증강 없이 원본) |
| output | `artifacts/beans_resnet18` | `artifacts/beans_resnet18_full_e8` | `artifacts/beans_resnet18_aug_e15` |

실행 명령 (제공 코드 실행 옵션만 사용):

```bash
# E0 기준선 (필수)
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 3 --batch-size 8 --freeze-backbone
# E4 Full (증강 없음)
python -m training.finetune --data dataset --output artifacts/beans_resnet18_full_e8 --epochs 8 --batch-size 8 --lr 5e-5
# E6 Full + 증강 (서빙)
python -m training.finetune_aug --data dataset --epochs 15 --batch-size 8 --lr 5e-5 --output artifacts/beans_resnet18_aug_e15
```

## 3. 학습 결과

### 3-1. E0 기준선 (Head-only, 3 epoch)

| Epoch | Train Loss | Validation Accuracy |
|---|---:|---:|
| 1 | 0.9924 | 0.6617 |
| 2 | 0.8108 | **0.7669** |
| 3 | 0.7078 | 0.7669 |

- Best Validation Accuracy: **0.7669 (102/133)**, epoch 2에서 저장 (epoch 3은 같은 값이라 저장 안 됨)

### 3-2. E4 Full (8 epoch, 증강 없음)

| Epoch | Train Loss | Validation Accuracy |
|---|---:|---:|
| 1 | 0.4404 | 0.9699 |
| 2 | 0.1556 | 0.9624 |
| 3 | 0.0855 | 0.9624 |
| 4 | 0.0520 | **0.9925** |
| 5 | 0.0509 | 0.9774 |
| 6 | 0.0482 | 0.9925 |
| 7 | 0.0512 | 0.9925 |
| 8 | 0.0260 | 0.9925 |

- Best Validation Accuracy: **0.9925 (132/133)**, epoch 4에서 저장
- 같은 설정 재실행 3회(rerun·r3·r4)도 모두 best 0.9925 → 4회 중 4회 같은 최고값 (GPU 연산 비결정성으로 epoch별 값은 실행마다 조금씩 다름)

### 3-2-1. E6 Full + 증강 (15 epoch, 서빙)

| Epoch | Train Loss | Validation Accuracy |
|---|---:|---:|
| 1 | 0.6107 | 0.8797 |
| 2 | 0.3853 | 0.9624 |
| 3 | 0.3350 | 0.9774 |
| 4 | 0.3068 | 0.9774 |
| 5 | 0.2635 | 0.9624 |
| 6 | 0.2365 | 0.9624 |
| 7 | 0.2600 | **1.0000** |
| 8 | 0.2042 | 0.9774 |
| 9 | 0.2432 | 0.9850 |
| 10 | 0.2152 | 0.9699 |
| 11 | 0.1813 | 0.9925 |
| 12 | 0.2019 | 0.9850 |
| 13 | 0.1975 | 1.0000 |
| 14 | 0.1800 | 0.9925 |
| 15 | 0.1545 | 0.9925 |

- Best Validation Accuracy: **1.0000 (133/133)**, epoch 7에서 저장 (epoch 13도 1.0이지만 같은 값이라 저장 안 됨)
- Train loss가 E4보다 높음 (0.15 vs 0.03): 학습 사진을 매번 일부러 어렵게 바꿨기 때문. Val은 원본 그대로
- E4와 Val 차이는 1장(0.75%p)뿐 → 원본 Val만으로는 우열을 말할 수 없음. 선택 근거는 거리·흐림 조건 Val과 외부 콩잎 사진 (06 B-9)
- 1회만 실행함 (재실행 안정성 미확인)

### 3-3. 비교 실험 전체 (같은 Val 133장, 출력 폴더 분리)

| ID | 방식 | epochs | lr | Best Val Acc | 정답 수 |
|---|---|---:|---:|---:|---:|
| E0 | Head-only (필수 기준선) | 3 | 2e-4 | 0.7669 | 102 |
| E1 | Head-only | 10 | 1e-3 | 0.9398 | 125 |
| B | Head-only | 20 | 1e-3 | 0.9624 | 128 |
| E2 | Full | 3 | 5e-5 | 0.9699 | 129 |
| E3 | Full | 5 | 1e-4 | 0.9850 | 131 |
| E4 | Full | 8 | 5e-5 | 0.9925 | 132 |
| **E6** | **Full + 증강** | **15** | **5e-5** | **1.0000** | **133** |

- 1차 선택: Val accuracy 최고 + 재실행 안정성으로 E4를 정함 → Test 1회 (0.9688)
- 2차 (E6): 외부 사진에서 거리·구도가 바뀌면 확신도가 떨어지는 문제(06 B-6)를 보고 증강 실험을 추가함. **Val 원본 + 조건 변경 Val + 외부 콩잎 사진**으로 E6를 고른 뒤 Test를 1회 측정함. Test 결과를 보고 모델을 다시 고르지 않음
- Head-only는 20 epoch까지 늘려도 0.9624 → Full과 3.0%p 차이: 고정된 ImageNet 특징만으로는 반점 모양 구분에 한계
- 상세 분석·오답 원인·규칙 기반 보류는 [06_성능비교_보고서](06_성능비교_보고서.md)의 성능 향상 절 참고

### 3-4. Test (최종 확정 후 1회만 측정)

- 제공 코드에는 Test 평가가 없어 **별도 스크립트** `analysis/predict_split.py --split test --confirm-test`로 1회 측정
- E4 Test Accuracy: **0.9688 (124/128)** · Val 0.9925 대비 2.4%p 낮음
- 오답 4장 모두 bean_rust (3장 → angular_leaf_spot, 1장 → healthy)
- **E6 Test Accuracy: 0.9453 (121/128)** · E4보다 3장 적게 맞힘. 오답 7장 (ALS→rust 3, rust→ALS 2, healthy→rust 2)의 확신도가 **모두 0.9 미만** (0.58~0.89) → R1 보류 규칙으로 모두 걸러짐 (보류 15장, 나머지 113/113 = 1.000)
- 정리: 원본 Beans Test 사진에서는 E4가 2.3%p 높고, 거리·흐림 변화와 외부 콩잎 사진에서는 E6가 높음 (06 B-9)

### 3-5. 과적합 · 과소적합 점검

best 모델로 Train을 다시 예측해 Val·Test와 비교 (`analysis/predict_split.py --split train`)

| 모델 | Train | Val | Test | 판단 |
|---|---:|---:|---:|---|
| E0 Head-only | 0.7737 (800/1034) | 0.7669 (102/133) | — | 둘 다 낮고 비슷함 → **과소적합** |
| E4 Full | 1.0000 (1034/1034) | 0.9925 (132/133) | 0.9688 (124/128) | 차이가 작음 → **과적합 징후 작음** |
| E6 Full + 증강 | 미측정 | 1.0000 (133/133) | 0.9453 (121/128) | Val–Test 차이 5.5%p. Val 1.0은 1장 차이라 낙관적인 값으로 봄 |

- E0: 고정된 backbone 특징만으로는 Train도 다 맞히지 못함. 오답 확신도 대부분 0.35~0.6
- E4: Train loss 0.44 → 0.026으로 계속 내려가는 동안 Val 정확도는 떨어지지 않음 (ep6~8 0.9925 유지)
- 한계: 제공 학습 코드가 Val loss와 epoch별 Train 정확도를 기록하지 않고 best epoch 모델만 저장 → epoch별 학습 곡선 비교는 하지 못함

## 4. 생성 파일 확인

- [x] config.json
- [x] model.safetensors
- [x] preprocessor_config.json
- [x] training_report.json

(두 폴더 `artifacts/beans_resnet18/`, `artifacts/beans_resnet18_full_e8/` 모두 확인 · 증빙 `evidence/model_folder.png`. E6 `artifacts/beans_resnet18_aug_e15/`도 같은 4개 파일 생성, `training_report.json`에 증강 설정 기록)

## 5. 서비스 연결 확인

- MODEL_PATH: `artifacts/beans_resnet18_aug_e15` (`.env`, E6로 교체 후 `docker compose up -d --force-recreate web worker`)
- 교체 확인: 동기 API 응답 `model_id = artifacts/beans_resnet18_aug_e15` (Val healthy 첫 사진 → healthy 0.674, 확인 필요 표시)
- E4 서빙 당시 동기 API 응답 (Test 각 클래스 첫 사진, 실제 실행):

| 정답 | 응답 top1 | score | 일치 |
|---|---|---:|---|
| angular_leaf_spot | angular_leaf_spot | 0.9706 | O |
| bean_rust | bean_rust | 1.0000 | O |
| healthy | healthy | 0.9991 | O |

- 증빙: `evidence/sync_angular_leaf_spot.json`, `evidence/sync_bean_rust.json`, `evidence/sync_healthy.json`

## 6. 결과 판정

- 완료: Head-only 기준선(필수) + Full Fine-tuning 비교 학습, 모델 파일 생성, 서비스 연결
- 한계
  - Val 133장에서 1장 = 0.75%p → 상위 설정끼리의 차이는 Val로 거의 구분 불가 (천장 효과)
  - 증강(E6)은 1회만 학습함. 조건 변경 Val은 학습 증강과 같은 종류의 변형이라 E6에 유리한 비교임 (그래서 외부 사진으로도 확인함)
  - E6는 원본 Beans Test에서 E4보다 3장 적게 맞힘 (0.9453 vs 0.9688)
  - 학습은 GPU, 서빙은 CPU (`device=-1`)
- 증빙: `evidence/train_log.png`, `evidence/model_folder.png`, `artifacts/*/training_report.json`
