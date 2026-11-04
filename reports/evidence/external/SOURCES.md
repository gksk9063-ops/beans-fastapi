# 외부(인터넷) 사진 확인 근거 (06 보고서 B-6, B-6-1)

- 모델: `artifacts/beans_resnet18_full_e8` (E4) · 동기 API `POST /api/sync/classify/` 응답 JSON 원문
- 사진 원본은 저작권 때문에 repo에 넣지 않음. 출처 URL만 기록
- 07~10은 화면에서 같은 사진을 한 번 넣은 뒤 다시 조회해서 `cache_hit: true`, `inference_ms: null` (예측값은 처음과 같음)

| 파일 | 사진 | 기대 | 예측 · 확신도 | 출처 |
|---|---|---|---|---|
| 01_healthy_bean | 건강한 콩잎 | healthy | healthy 0.992 | (URL 입력) |
| 02_angular_leaf_spot | 반점병 콩잎 | angular_leaf_spot | angular_leaf_spot 0.929 | (URL 입력) |
| 03_bean_rust | 녹병 콩잎 | bean_rust | bean_rust 0.821 · 확인 필요 | (URL 입력) |
| 04_bean_insect | 콩잎 + 곤충 | 3클래스 밖 | bean_rust 0.995 | (URL 입력) |
| 05_cherry_tomato | 방울토마토 | 3클래스 밖 | bean_rust 0.689 · 확인 필요 | (URL 입력) |
| 06_cat | 고양이 | 3클래스 밖 | bean_rust 0.828 · 확인 필요 | (URL 입력) |
| 07_garden_tree | 정원수 잎 | 3클래스 밖 | healthy 0.875 · 확인 필요 | (URL 입력) |
| 08_spinach | 시금치 | 3클래스 밖 | healthy 0.549 · 확인 필요 | (URL 입력) |
| 09_perilla | 깻잎 | 3클래스 밖 | angular_leaf_spot 0.815 · 확인 필요 | (URL 입력) |
| 10_bean_insect_removed | 04에서 곤충 부분만 지운 사진 | 원인 확인 | bean_rust 0.996 | 04 사진을 직접 수정 |
| 11_healthy8_dots | `dataset/test/healthy/healthy_test.8.jpg`에 둥근 갈색 점 40개 | 원인 확인 | angular_leaf_spot 0.687 · 확인 필요 | PIL, seed 0, 점 색 (160,90,30), 반지름 3~7px |
