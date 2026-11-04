"""폴더 안 사진(외부 사진 등)을 여러 모델로 예측해서 나란히 비교한다. 정답 라벨 없음.

예: python -m analysis.predict_folder --folder external_photos --models E4=artifacts/beans_resnet18_full_e8 E6=artifacts/beans_resnet18_aug_e15
    앙상블: ... E46=artifacts/beans_resnet18_full_e8+artifacts/beans_resnet18_aug_e15 --out reports/data/external_compare_ens.csv
서빙과 같은 pipeline("image-classification")을 사용한다.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps
from transformers import pipeline

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", default="reports/data/external_compare.csv")
    args = parser.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    # "경로1+경로2"면 두 모델 확률 평균 (앙상블)
    clfs = {tag: [pipeline("image-classification", model=p, device=-1) for p in path.split("+")] for tag, path in models.items()}
    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    rows = []
    for p in files:
        image = ImageOps.exif_transpose(Image.open(p)).convert("RGB")   # services.load_rgb_image와 같은 처리
        row = {"file": p.name}
        for tag, clf_list in clfs.items():
            avg = {}
            for clf in clf_list:
                for item in clf(image, top_k=3):
                    avg[item["label"]] = avg.get(item["label"], 0) + item["score"] / len(clf_list)
            label = max(avg, key=avg.get)
            row[f"{tag}_label"] = label
            row[f"{tag}_score"] = round(avg[label], 4)
        rows.append(row)
        print(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {args.out}")
