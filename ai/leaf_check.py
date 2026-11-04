"""콩잎이 아닌 사진 표시 (06 보고서 B-10). 분류 결과(라벨·점수)는 바꾸지 않고 표시만 추가한다.

방법 (7일차 이상탐지와 같은 생각):
1) ImageNet ResNet-18 특징(512차원)을 뽑는다 (고양이·토마토 같은 물체 차이를 아는 특징)
2) Train 콩잎 특징의 정규분포(평균 μ, 공분산 Σ)까지 Mahalanobis 거리를 잰다
3) 거리가 기준값(Val 콩잎 거리의 99% 지점)보다 멀면 maybe_not_bean = True
기준값과 분포는 analysis/ood_check.py가 만든 .npz 파일에서 읽는다.
LEAF_STATS_PATH가 비어 있거나 파일이 없으면 검사를 생략하고 None을 돌려준다.
"""
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import BASE_DIR, LEAF_FEATURE_MODEL, LEAF_STATS_PATH


def load_stats(path) -> tuple[list, float]:
    """npz → [(μ, Σ⁻¹), ...], 기준값. 분포가 여러 개(클래스별)면 가장 가까운 거리를 쓴다."""
    stats = np.load(path)
    mus = [stats[k] for k in sorted(stats.files) if k.startswith("mu")]
    invs = [stats[k] for k in sorted(stats.files) if k.startswith("inv")]
    return list(zip(mus, invs)), float(stats["threshold"])


def min_distance(feature: np.ndarray, params: list) -> float:
    """특징 1개와 정규분포들 사이의 Mahalanobis 거리 중 가장 작은 값"""
    return min(float(np.sqrt((feature - mu) @ inv @ (feature - mu))) for mu, inv in params)


@lru_cache(maxsize=1)
def get_leaf_checker():
    """프로세스(web, worker)마다 처음 한 번만 특징 모델과 분포를 읽는다."""
    if not LEAF_STATS_PATH:
        return None
    stats_path = Path(LEAF_STATS_PATH)
    if not stats_path.is_absolute():
        stats_path = BASE_DIR / stats_path
    if not stats_path.is_file():
        return None
    from transformers import AutoImageProcessor, ResNetModel

    processor = AutoImageProcessor.from_pretrained(LEAF_FEATURE_MODEL)
    model = ResNetModel.from_pretrained(LEAF_FEATURE_MODEL).eval()
    params, threshold = load_stats(stats_path)
    return processor, model, params, threshold


def check_leaf(image) -> dict | None:
    checker = get_leaf_checker()
    if checker is None:
        return None
    import torch

    processor, model, params, threshold = checker
    started = time.perf_counter()
    with torch.no_grad():
        pixels = processor(images=image, return_tensors="pt")["pixel_values"]
        feature = model(pixel_values=pixels).pooler_output.flatten(1)[0].numpy()   # (512,)
    distance = min_distance(feature, params)
    return {
        "distance": round(distance, 2),
        "threshold": round(threshold, 2),
        "maybe_not_bean": distance > threshold,
        "ms": round((time.perf_counter() - started) * 1000, 2),
    }
