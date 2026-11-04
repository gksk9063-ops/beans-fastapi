"""여러 모델의 확률 평균(앙상블)과 좌우반전 TTA를 평가한다. 학습 코드·서빙 코드는 바꾸지 않는다.

예 (Val):
  python -m analysis.ensemble_eval --split val --tag ENS4 \
      --models artifacts/beans_resnet18_full_e8 artifacts/beans_resnet18_full_e8_rerun \
               artifacts/beans_resnet18_full_e8_r3 artifacts/beans_resnet18_full_e8_r4
  python -m analysis.ensemble_eval --split val --tag ENS4_TTA --tta --models (위와 같음)
  python -m analysis.ensemble_eval --split val --tag E4_TTA --tta --models artifacts/beans_resnet18_full_e8
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .predict_split import list_images


def model_probs(model_dir, records, tta, device, batch_size):
    """모델 1개로 모든 이미지의 클래스 확률을 구한다. tta면 원본+좌우반전 평균."""
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()
    classes = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    out = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            images = [Image.open(p).convert("RGB") for p, _ in records[start:start + batch_size]]
            views = [images, [ImageOps.mirror(im) for im in images]] if tta else [images]
            probs = 0
            for view in views:
                pixels = processor(images=view, return_tensors="pt")["pixel_values"].to(device)
                probs = probs + torch.softmax(model(pixel_values=pixels).logits, dim=-1).cpu()
            out.append(probs / len(views))
    return classes, torch.cat(out)


def accuracy(probs, records, classes):
    pred = probs.argmax(dim=-1).tolist()
    return sum(classes[i] == true for i, (_, true) in zip(pred, records)) / len(records)


def run(args):
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Test는 최종 방법 확정 후 1번만 확인합니다. 맞다면 --confirm-test를 붙이세요.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes, records, total = None, None, 0
    for model_dir in args.models:
        if records is None:
            # 첫 모델의 클래스 순서로 이미지 목록을 만든다
            first = AutoModelForImageClassification.from_pretrained(model_dir)
            classes = [first.config.id2label[i] for i in range(len(first.config.id2label))]
            records = list_images(Path(args.data) / args.split, classes)
        model_classes, probs = model_probs(model_dir, records, args.tta, device, args.batch_size)
        if model_classes != classes:
            raise SystemExit(f"클래스 순서가 다릅니다: {model_dir}")
        print(f"  단일 {model_dir}{' +TTA' if args.tta else ''}: {accuracy(probs, records, classes):.4f}")
        total = total + probs

    ens = total / len(args.models)
    correct = 0
    rows = []
    for (path, true), prob in zip(records, ens):
        idx = int(prob.argmax())
        ok = int(classes[idx] == true)
        correct += ok
        row = {"path": str(path), "true": true, "pred": classes[idx],
               "score": round(float(prob[idx]), 4), "correct": ok}
        for i, name in enumerate(classes):
            row[f"p_{name}"] = round(float(prob[i]), 4)
        rows.append(row)

    out_csv = Path(args.out) / f"pred_{args.tag}_{args.split}.csv"   # predict_split.py와 같은 형식
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{args.tag} / {args.split}] 모델 {len(args.models)}개{' +TTA' if args.tta else ''} "
          f"accuracy = {correct}/{len(rows)} = {correct / len(rows):.4f}")
    print("틀린 이미지:")
    for r in rows:
        if not r["correct"]:
            print(f"  {r['path']}  정답={r['true']}  예측={r['pred']}  score={r['score']}")
    low = sum(1 for r in rows if r["score"] < 0.9)
    print(f"score < 0.9 이미지 수: {low}  (R1 보류 대상)")
    print(f"저장: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tta", action="store_true", help="좌우반전 평균")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="reports/data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confirm-test", action="store_true")
    run(parser.parse_args())
