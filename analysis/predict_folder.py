"""폴더 안 사진(외부 사진 등)을 여러 모델로 예측해서 나란히 비교한다. 정답 라벨 없음.

예: python -m analysis.predict_folder --folder external_photos --models E4=artifacts/beans_resnet18_full_e8 E6=artifacts/beans_resnet18_aug_e15
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
    clfs = {tag: pipeline("image-classification", model=path, device=-1) for tag, path in models.items()}
    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    rows = []
    for p in files:
        image = ImageOps.exif_transpose(Image.open(p)).convert("RGB")   # services.load_rgb_image와 같은 처리
        row = {"file": p.name}
        for tag, clf in clfs.items():
            top = clf(image, top_k=1)[0]
            row[f"{tag}_label"] = top["label"]
            row[f"{tag}_score"] = round(top["score"], 4)
        rows.append(row)
        print(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {args.out}")
