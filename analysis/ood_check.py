"""콩잎이 아닌 사진 거르기 실험 (7일차 이상탐지 방식: 정상 특징의 정규분포 + Mahalanobis 거리).

1) Train 콩잎 사진의 ResNet-18 특징(512차원)으로 평균 μ와 공분산 Σ를 구한다 (= "콩잎 특징의 정규분포")
2) 새 사진 특징 x의 거리 d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)) 를 잰다. 멀수록 Train 콩잎과 다르다
3) 기준값 = Val 콩잎 거리의 99번째 백분위 (8일차 calibration과 같은 생각). 기준보다 멀면 "콩잎 아닐 수 있음"
Test는 쓰지 않는다. 분류 결과(softmax)는 바꾸지 않는다.

예:
  python -m analysis.ood_check --model artifacts/beans_resnet18_aug_e15 --folder external_photos
  python -m analysis.ood_check --model artifacts/beans_resnet18_aug_e15 --folder external_photos --feature-model microsoft/resnet-18 --tag IMGNETfeat
  python -m analysis.ood_check --model artifacts/beans_resnet18_aug_e15 --folder external_photos --feature-model microsoft/resnet-18 --per-class --layers 3,4 --tag IMGNET_cls_l34
결과: reports/data/ood_val.csv, reports/data/ood_external.csv, <model>/ood_stats.npz (나중에 서빙에 붙일 때 사용)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

from analysis.predict_split import list_images

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def extract(model, processor, paths, device, batch_size=32, feat_model=None, feat_processor=None, layers=(4,)):
    """사진들 → (N, 512) 특징 + (N, 3) softmax 확률. feat_model이 있으면 특징은 그 모델에서 뽑는다"""
    feat_model = feat_model or model
    feat_processor = feat_processor or processor
    feats, probs = [], []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths[start:start + batch_size]]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            feat_pixels = feat_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            hidden = feat_model.resnet(feat_pixels, output_hidden_states=True).hidden_states
            # hidden[3] = layer3 (B,256,14,14), hidden[4] = layer4 (B,512,7,7) → 위치 평균 → 이어 붙임 (7일차처럼 중간층 사용)
            pooled = torch.cat([hidden[k].mean(dim=(2, 3)) for k in layers], dim=1)
            logits = model(pixel_values=pixels).logits                     # 분류 점수 (서빙과 같은 값)
            feats.append(pooled.cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(feats), np.concatenate(probs)


def fit_gaussian(feats, shrink=0.1):
    """평균과 공분산. 512차원에 1,034장이라 공분산이 불안정 → 대각 성분 쪽으로 조금 당김(shrinkage)"""
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    d = cov.shape[0]
    cov = (1 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    return mu, np.linalg.inv(cov)


def mahalanobis(feats, mu, inv_cov):
    diff = feats - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))


class GaussianOOD:
    """per_class=False: 콩잎 전체 정규분포 1개 / True: 클래스(ALS·rust·healthy)마다 1개씩 → 가장 가까운 클래스까지 거리"""
    def __init__(self, feats, labels, per_class, shrink):
        groups = [feats[labels == c] for c in sorted(set(labels))] if per_class else [feats]
        self.params = [fit_gaussian(g, shrink) for g in groups]

    def distance(self, feats):
        return np.min([mahalanobis(feats, mu, inv) for mu, inv in self.params], axis=0)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()
    classes = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    data = Path(args.data)
    # 특징을 뽑을 모델: 기본은 분류 모델(E6). microsoft/resnet-18을 주면 ImageNet 원래 특징(고양이·토마토를 아는 특징)을 쓴다
    feat_model = AutoModelForImageClassification.from_pretrained(args.feature_model).to(device).eval() if args.feature_model else None
    feat_processor = AutoImageProcessor.from_pretrained(args.feature_model) if args.feature_model else None
    fx = dict(feat_model=feat_model, feat_processor=feat_processor, layers=tuple(int(x) for x in args.layers.split(",")))
    tag = args.tag

    train = list_images(data / "train", classes)
    val = list_images(data / "val", classes)
    train_feats, _ = extract(model, processor, [p for p, _ in train], device, **fx)
    train_labels = np.array([t for _, t in train])
    ood = GaussianOOD(train_feats, train_labels, args.per_class, args.shrink)

    val_feats, val_probs = extract(model, processor, [p for p, _ in val], device, **fx)
    val_d = ood.distance(val_feats)
    train_d = ood.distance(train_feats)
    thr = float(np.percentile(val_d, args.percentile))
    print(f"Train 거리: 중앙값 {np.median(train_d):.1f}, 최대 {train_d.max():.1f}")
    print(f"Val 거리:   중앙값 {np.median(val_d):.1f}, 최대 {val_d.max():.1f}")
    print(f"기준값 (Val {args.percentile}번째 백분위) = {thr:.1f}  → Val 콩잎 중 {int((val_d > thr).sum())}장이 기준 밖")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"ood_val_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "distance", "outside"])
        for (p, t), d in zip(val, val_d):
            w.writerow([str(p), t, round(float(d), 2), int(d > thr)])
    np.savez(Path(args.model) / f"ood_stats_{tag}.npz", threshold=thr,
             **{f"mu{i}": mu for i, (mu, _) in enumerate(ood.params)}, **{f"inv{i}": inv for i, (_, inv) in enumerate(ood.params)})

    if args.folder:
        files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
        ext_feats, ext_probs = extract(model, processor, files, device, **fx)
        ext_d = ood.distance(ext_feats)
        rows = []
        print(f"\n{'파일':<28}{'거리':>8}  {'판정':<10}{'예측':<20}{'확신도':>6}")
        for p, d, pr in zip(files, ext_d, ext_probs):
            i = int(pr.argmax())
            verdict = "콩잎 아님?" if d > thr else "콩잎 범위"
            rows.append({"file": p.name, "distance": round(float(d), 2), "ratio_to_threshold": round(float(d / thr), 2),
                         "ood": int(d > thr), "pred": classes[i], "score": round(float(pr[i]), 4)})
            print(f"{p.name:<28}{d:>8.1f}  {verdict:<10}{classes[i]:<20}{pr[i]:>6.3f}")
        with (out / f"ood_external_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n저장: {out}/ood_val_{tag}.csv, {out}/ood_external_{tag}.csv, {args.model}/ood_stats_{tag}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/beans_resnet18_aug_e15")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--folder", default="", help="외부 사진 폴더 (예: external_photos)")
    parser.add_argument("--percentile", type=float, default=99)
    parser.add_argument("--shrink", type=float, default=0.1)
    parser.add_argument("--out-dir", default="reports/data")
    parser.add_argument("--feature-model", default="", help="특징만 다른 모델에서 뽑기, 예: microsoft/resnet-18 (ImageNet)")
    parser.add_argument("--tag", default="E6feat", help="결과 파일 이름 구분용")
    parser.add_argument("--per-class", action="store_true", help="클래스별 정규분포 3개 중 가장 가까운 거리")
    parser.add_argument("--layers", default="4", help="특징 층: 4 = 마지막(512), 3,4 = layer3(256)+layer4(512)")
    run(parser.parse_args())
