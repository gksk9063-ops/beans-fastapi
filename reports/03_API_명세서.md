# API 명세서

- Base URL: `http://127.0.0.1:8000` · Swagger: `/docs`
- 서빙 모델: `artifacts/beans_resnet18_aug_e15` (E6)

> 2026-11-04 서빙 모델을 E4(`beans_resnet18_full_e8`)에서 **E6(`beans_resnet18_aug_e15`, 증강 학습)**로 교체했다 (06 B-9). 이 문서의 실제 응답·작업 이력·시간 측정은 **E4 서빙 당시 기록**이다. E4와 E6는 같은 ResNet-18 구조이고 서빙 코드는 바뀌지 않았다.

## 1. API 목록

| API ID | Method | URL | 요청 형태 | 성공 HTTP | 주요 응답 | 실패 HTTP |
|---|---|---|---|---:|---|---|
| API-01 | GET | `/health` | 없음 | 200 | `status` | 서버 미실행 시 연결 실패 |
| API-02 | POST | `/api/sync/classify/` | `multipart/form-data` / `image` | 200 | `model_id`, `top1`, `all`, `inference_ms`, `cache_hit` | 400, 413 |
| API-03 | POST | `/api/async/classify/` | `multipart/form-data` / `image` | 202 | `job_id`, `status`, `check_url` | 400, 413, 503 |
| API-04 | GET | `/api/async/jobs/{job_id}/` | 정수 `job_id` | 200 | `status`, `top1`, `all`, `inference_ms`, `cache_hit`, `error_message` | 404 |
| WEB-01 | GET | `/vision/sync/` | 없음 | 200 | HTML | — |
| WEB-02 | GET | `/vision/async/` | 없음 | 200 | HTML | — |
| API-05 (추가) | GET | `/api/async/jobs/?limit=20` | `limit` 1~100 | 200 | 최근 작업 배열 (`job_id`, `status`, `top1_label`, `top1_score`, `cache_hit`, `inference_ms`, `created_at`) | 422 (limit 범위 밖) |
| WEB-03 (추가) | GET | `/vision/jobs/` | 없음 | 200 | HTML 작업 목록 (3초 자동 갱신) | — |
| WEB-00 | GET | `/` | 없음 | 307 | `/vision/sync/`로 이동 | — |

## 2. API-02 동기 분류

| 항목 | 내용 |
|---|---|
| Content-Type | `multipart/form-data` |
| 요청 필드 | `image`: JPEG/PNG/WEBP, 5MB 이하, 1,200만 픽셀 이하 |
| 성공 | 200 OK (추론 완료 결과 포함) |
| 오류 | 400 잘못된 이미지 · 413 5MB 초과 |

실제 응답 (`dataset/test/angular_leaf_spot` 첫 사진, Cache MISS):

```json
{"model_id":"artifacts/beans_resnet18_full_e8",
 "top1":{"label":"angular_leaf_spot","score":0.9705893993377686},
 "all":[{"label":"angular_leaf_spot","score":0.9705893993377686},
        {"label":"bean_rust","score":0.026935990899801254},
        {"label":"healthy","score":0.0024746593553572893}],
 "inference_ms":2144.68,"cache_hit":false}
```
(첫 요청이라 모델 로딩 시간 포함. 이후 MISS 추론은 약 20ms)

## 3. API-03 비동기 접수

- 큐 등록 전에 이미지 유효성을 먼저 검사 → 잘못된 파일은 DB에 기록하지 않고 400
- Redis 브로커 연결 실패 시 DB에 `failed` 기록 후 503

실제 응답:
```json
{"job_id":7,"status":"queued","check_url":"/api/async/jobs/7/"}
```

## 4. API-04 상태 조회

| status | 의미 | top1 / all |
|---|---|---|
| queued | 접수됨, Worker 대기 | null |
| running | Worker 처리 중 | null |
| done | 완료 | 값 있음 |
| failed | 실패 | null, `error_message` 있음 |

실제 응답 (job 7, Cache MISS):
```json
{"job_id":7,"status":"done","model_id":"artifacts/beans_resnet18_full_e8",
 "top1":{"label":"bean_rust","score":0.9996289014816284},
 "all":[{"label":"bean_rust","score":0.9996289014816284},
        {"label":"healthy","score":0.0003314901259727776},
        {"label":"angular_leaf_spot","score":3.954617204726674e-05}],
 "inference_ms":2207.17,"cache_hit":false,"error_message":null}
```

## 5. 검증 기록 (실제 실행)

| 검증 ID | 테스트 입력 | 실제 상태코드 / 결과 | 증빙 |
|---|---|---|---|
| API-01-T01 | GET `/health` | 200 / ok | `evidence/health.png` |
| API-02-T01 | Test ALS · rust · healthy 각 1장 | 200 / 3장 모두 정답 (0.971 / 1.000 / 0.999) | `evidence/sync_*.json` |
| API-02-T02 | `.txt` 파일 | **400** `올바른 이미지 파일이 아닙니다.` | `evidence/T07_bad_input.txt` |
| API-03-T01 | Test 3장 | 202 / job 4·5·6 | `evidence/async_*.txt` |
| API-04-T01 | job 7 (새 사진) | 200 / done, MISS | `evidence/async_miss.json` |
| API-04-T02 | 없는 `job_id=999999` | **404** | `evidence/T08_not_found.txt` |
| API-03-T02 | 5MB 초과 파일 | 미실시 | — |

- `202`는 **접수**, 상태 조회 `200`의 `done`이 **완료**
- 응답 필드 `all`은 DB 컬럼 `all_scores`와 이름이 다름 (코드 그대로 유지)

## 6. 제공 코드 외 추가 기능

- `ai/router_jobs.py` (새 파일): 작업 목록 API-05 + 화면 WEB-03. 기존 `router_async.py`는 수정하지 않음
  - 추가 이유: 제공 API-04는 job **1개**만 번호로 조회 → 쌓인 작업 전체를 브라우저에서 확인할 방법이 없었음 (psql·DBeaver만 가능)
  - 실제 확인: `GET /api/async/jobs/?limit=3` → 200, job 16·15·14 반환
- `static/result.js`·`result.css`: 응답 JSON을 카드(한글 병명·확신도 막대·캐시 배지)로 표시, 확신도 0.9 미만은 "사람 확인 필요" (06 보고서 R1 규칙). **API 응답 형식은 변경 없음**
