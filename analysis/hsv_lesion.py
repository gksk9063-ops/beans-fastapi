"""OpenCV(HSV + GrabCut + Contour)로 잎 영역 안의 반점 후보(초록이 아닌 픽셀) 면적비를 계산한다.

CNN 예측은 바꾸지 않는다. 이미지별 수치를 CSV로 남기고, 지정한 이미지는 마스크 그림을 저장한다.

예:
  python -m analysis.hsv_lesion --pred reports/data/pred_E4_val.csv --tag E4
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# 색 범위 (OpenCV HSV: H 0~179, S·V 0~255). Train 사진 3장을 보고 정한 값.
GREEN_LO, GREEN_HI = (30, 40, 40), (90, 255, 255)      # 잎 후보 (GrabCut 시작점)
# v2: 반점 = 잎 안에서 '초록이 아닌' 픽셀. v1(황색만)은 갈색·암갈색(ALS) 반점을 놓쳤음
LEAF_GREEN_LO, LEAF_GREEN_HI = (27, 30, 0), (90, 255, 255)   # 정상 잎 색
VALID_LO, VALID_HI = (0, 0, 30), (179, 255, 220)             # 너무 어두운 그림자·반사광 제외
# 반점 모양 조건: 너무 작거나 크지 않고, 길쭉하지 않은(잎맥이 아닌) 덩어리만 남김
SPOT_MIN_AREA, SPOT_MAX_AREA = 4, 1500
SPOT_MAX_ASPECT = 2.5
SPOT_MIN_EXTENT = 0.35


def leaf_mask(bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """초록 픽셀을 시작점으로 GrabCut을 돌려 가운데 잎 1장의 마스크를 만든다."""
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    h, w = green.shape
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)       # 기본: 아마 배경
    mask[green > 0] = cv2.GC_PR_FGD                        # 초록: 아마 잎
    mask[:8, :] = cv2.GC_BGD                               # 위·좌·우 테두리: 확실한 배경
    mask[:, :8] = cv2.GC_BGD
    mask[:, -8:] = cv2.GC_BGD
    center = mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center[green[h // 4: 3 * h // 4, w // 4: 3 * w // 4] > 0] = cv2.GC_FGD  # 가운데 초록: 확실한 잎
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
    leaf = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 가장 큰 덩어리 1개만 남기고, 안쪽 구멍(반점 자리)을 채운다
    n, labels, stats, _ = cv2.connectedComponentsWithStats(leaf)
    if n <= 1:
        return leaf
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = np.uint8(labels == biggest) * 255
    contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(leaf)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return cv2.erode(filled, np.ones((7, 7), np.uint8))   # 잎 가장자리(흙 경계) 제외


def spot_mask(lesion: np.ndarray) -> np.ndarray:
    """색 후보 중 반점 모양인 덩어리만 남긴다 (Contour 통계 사용)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lesion)
    keep = np.zeros_like(lesion)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not SPOT_MIN_AREA <= area <= SPOT_MAX_AREA:
            continue
        if max(w, h) / max(min(w, h), 1) > SPOT_MAX_ASPECT:
            continue
        if area / (w * h) < SPOT_MIN_EXTENT:
            continue
        keep[labels == i] = 255
    return keep


def analyze(path: str):
    bgr = cv2.imread(path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    leaf = leaf_mask(bgr, hsv)
    not_green = cv2.bitwise_not(cv2.inRange(hsv, LEAF_GREEN_LO, LEAF_GREEN_HI))
    lesion = cv2.bitwise_and(not_green, cv2.inRange(hsv, VALID_LO, VALID_HI))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 1~2픽셀 잡음 제거
    lesion = cv2.bitwise_and(lesion, leaf)
    spots = spot_mask(lesion)
    leaf_px = int((leaf > 0).sum())
    spot_px = int((spots > 0).sum())
    result = {
        "leaf_ratio": round(leaf_px / leaf.size, 4),           # 사진 중 잎 비율 (마스크 품질 확인용)
        "lesion_ratio": round(spot_px / max(leaf_px, 1), 4),   # 잎 중 반점 후보 비율
        "spot_count": cv2.connectedComponents(spots)[0] - 1,
    }
    return result, bgr, leaf, spots


def save_overlay(bgr, leaf, spots, out_path: Path, title: str):
    view = bgr.copy()
    view[leaf == 0] = (view[leaf == 0] * 0.35).astype(np.uint8)   # 잎 밖은 어둡게
    view[spots > 0] = (255, 0, 255)                                # 반점 후보는 분홍
    cv2.putText(view, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), view)


def run(args):
    rows = list(csv.DictReader(open(args.pred, encoding="utf-8")))
    fig_dir = Path(args.fig_out) / f"hsv_{args.tag}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for r in rows:
        result, bgr, leaf, spots = analyze(r["path"])
        out = {**r, **result}
        out_rows.append(out)
        # 그림 저장: 틀린 사진 + 확신도 낮은 사진
        if r["correct"] == "0" or float(r["score"]) < args.low_score:
            name = Path(r["path"]).stem
            title = f"GT={r['true']} pred={r['pred']} {float(r['score']):.2f} lesion={result['lesion_ratio']:.4f}"
            save_overlay(bgr, leaf, spots, fig_dir / f"{name}.jpg", title)

    out_csv = Path(args.data_out) / f"hsv_{args.tag}_{Path(args.pred).stem.split('_')[-1]}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # 클래스별 반점 면적비 요약: healthy와 병든 잎이 갈리는지 확인
    print("정답 클래스별 lesion_ratio (min / 중앙값 / max) · leaf_ratio 중앙값")
    for name in sorted({r["true"] for r in out_rows}):
        vals = sorted(float(r["lesion_ratio"]) for r in out_rows if r["true"] == name)
        leafs = sorted(float(r["leaf_ratio"]) for r in out_rows if r["true"] == name)
        print(f"  {name:18s} {vals[0]:.4f} / {vals[len(vals) // 2]:.4f} / {vals[-1]:.4f}   leaf {leafs[len(leafs) // 2]:.2f}")
    print(f"저장: {out_csv}, 그림 {fig_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="predict_split.py가 만든 CSV")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--low-score", type=float, default=0.9)
    parser.add_argument("--data-out", default="reports/data")
    parser.add_argument("--fig-out", default="reports/figures")
    run(parser.parse_args())
