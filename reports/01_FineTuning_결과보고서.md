# ResNet18 콩잎 3클래스 Fine-tuning 결과보고서

- 평가 대상: 콩잎 이미지 3클래스 분류 (angular_leaf_spot / bean_rust / healthy)
- 기준 모델: `microsoft/resnet-18` (ImageNet 사전학습)
- 학습 코드: 제공 `training/finetune.py` **원본 그대로** (코드 수정 없이 실행 옵션만 변경)
- 학습 방식: ① 필수 기준선 Head-only (Backbone freeze) ② 비교 실험 Full Fine-tuning
- 입력 데이터: `dataset/train`, `dataset/val` · 최종 보관 데이터: `dataset/test` (학습·선택에 사용하지 않음)
- 모델 저장 위치: 기준선 `artifacts/beans_resnet18/` · **서빙 모델 `artifacts/beans_resnet18_full_e8/`**
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

| 파라미터 | E0 기준선 (필수) | E4 최종 (서빙) |
|---|---|---|
| base_model | microsoft/resnet-18 | microsoft/resnet-18 |
| 학습 범위 | Head-only (`--freeze-backbone`, `classifier.*`만 학습) | Full (전체 가중치 학습) |
| epochs | 3 | 8 |
| batch_size | 8 | 8 |
| learning_rate | 0.0002 (기본값) | 0.00005 |
| optimizer | AdamW (제공 코드) | AdamW (제공 코드) |
| 데이터 증강 | 없음 (제공 코드) | 없음 (제공 코드) |
| seed | 42 (제공 코드) | 42 (제공 코드) |
| 모델 저장 규칙 | Val accuracy가 이전보다 **클 때만** 저장 (best epoch) | 같음 |
| output | `artifacts/beans_resnet18` | `artifacts/beans_resnet18_full_e8` |

실행 명령 (제공 코드 실행 옵션만 사용):

```bash
# E0 기준선 (필수)
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 3 --batch-size 8 --freeze-backbone
# E4 최종
python -m training.finetune --data dataset --output artifacts/beans_resnet18_full_e8 --epochs 8 --batch-size 8 --lr 5e-5
```

## 3. 학습 결과

### 3-1. E0 기준선 (Head-only, 3 epoch)

| Epoch | Train Loss | Validation Accuracy |
|---|---:|---:|
| 1 | 0.9924 | 0.6617 |
| 2 | 0.8108 | **0.7669** |
| 3 | 0.7078 | 0.7669 |

- Best Validation Accuracy: **0.7669 (102/133)**, epoch 2에서 저장 (epoch 3은 같은 값이라 저장 안 됨)

### 3-2. E4 최종 (Full, 8 epoch)

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

### 3-3. 비교 실험 전체 (같은 Val 133장, 출력 폴더 분리)

| ID | 방식 | epochs | lr | Best Val Acc | 정답 수 |
|---|---|---:|---:|---:|---:|
| E0 | Head-only (필수 기준선) | 3 | 2e-4 | 0.7669 | 102 |
| E1 | Head-only | 10 | 1e-3 | 0.9398 | 125 |
| B | Head-only | 20 | 1e-3 | 0.9624 | 128 |
| E2 | Full | 3 | 5e-5 | 0.9699 | 129 |
| E3 | Full | 5 | 1e-4 | 0.9850 | 131 |
| **E4** | **Full** | **8** | **5e-5** | **0.9925** | **132** |

- 선택 기준: Val accuracy 최고 + 재실행 안정성. Test는 선택에 사용하지 않음
- Head-only는 20 epoch까지 늘려도 0.9624 → Full과 3.0%p 차이: 고정된 ImageNet 특징만으로는 반점 모양 구분에 한계
- 상세 분석·오답 원인·규칙 기반 보류는 [06_성능비교_보고서](06_성능비교_보고서.md)의 성능 향상 절 참고

### 3-4. Test (최종 확정 후 1회만 측정)

- 제공 코드에는 Test 평가가 없어 **별도 스크립트** `analysis/predict_split.py --split test --confirm-test`로 1회 측정
- E4 Test Accuracy: **0.9688 (124/128)** · Val 0.9925 대비 2.4%p 낮음
- 오답 4장 모두 bean_rust (3장 → angular_leaf_spot, 1장 → healthy)

## 4. 생성 파일 확인

- [x] config.json
- [x] model.safetensors
- [x] preprocessor_config.json
- [x] training_report.json

(두 폴더 `artifacts/beans_resnet18/`, `artifacts/beans_resnet18_full_e8/` 모두 확인 · 증빙 `evidence/model_folder.png`)

## 5. 서비스 연결 확인

- MODEL_PATH: `artifacts/beans_resnet18_full_e8` (`.env`)
- 동기 API 응답 (Test 각 클래스 첫 사진, 실제 실행):

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
  - 데이터 증강이 없음 (제공 코드 원본 유지) → 조명·각도 변화에 대한 일반화는 미검증
  - 학습은 GPU, 서빙은 CPU (`device=-1`)
- 증빙: `evidence/train_log.png`, `evidence/model_folder.png`, `artifacts/*/training_report.json`
