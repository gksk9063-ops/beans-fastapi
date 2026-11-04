"""Val 사진을 거리·구도·빛 조건별로 바꿔서 모델들의 정확도와 확신도를 비교한다. (Test는 안 씀)

예:
  python -m analysis.shift_eval --models E4=artifacts/beans_resnet18_full_e8 E6=artifacts/beans_resnet18_aug_e15 --preview
  # 앙상블 (E4+E6 확률 평균):
  python -m analysis.shift_eval --models E4=artifacts/beans_resnet18_full_e8 E6=artifacts/beans_resnet18_aug_e15 E46=artifacts/beans_resnet18_full_e8+artifacts/beans_resnet18_aug_e15 --out reports/data/shift_eval_ens.csv
결과: reports/data/shift_eval.csv (+ --preview면 reports/figures/shift_preview/ 에 조건별 예시 사진)
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from analysis.predict_split import list_images
from training.augment import SHIFT_CONDITIONS


def evaluate(model_path, records, condition, device, batch_size, threshold):
    # "경로1+경로2"면 두 모델의 softmax 확률을 평균 (앙상블). 클래스 순서는 학습 때 같은 sorted 순서
    paths = model_path.split("+")
    processor = AutoImageProcessor.from_pretrained(paths[0])
    models = [AutoModelForImageClassification.from_pretrained(p).to(device).eval() for p in paths]
    classes = [models[0].config.id2label[i] for i in range(len(models[0].config.id2label))]
    correct, scores, held = 0, [], 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = [condition(Image.open(p).convert("RGB")) for p, _ in batch]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            probs = torch.stack([torch.softmax(m(pixel_values=pixels).logits, dim=-1) for m in models]).mean(0).cpu()
            for (_, true), prob in zip(batch, probs):
                idx = int(prob.argmax())
                correct += int(classes[idx] == true)
                scores.append(float(prob[idx]))
                held += int(prob[idx] < threshold)
    n = len(records)
    return {"accuracy": round(correct / n, 4), "correct": f"{correct}/{n}",
            "mean_score": round(sum(scores) / n, 4), f"held_lt_{threshold}": held}


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = dict(item.split("=", 1) for item in args.models)
    first = next(iter(models.values()))
    cfg = AutoModelForImageClassification.from_pretrained(first).config
    classes = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    records = list_images(Path(args.data) / "val", classes)

    if args.preview:   # 조건별로 사진이 어떻게 바뀌는지 1장씩 저장
        prev_dir = Path("reports/figures/shift_preview")
        prev_dir.mkdir(parents=True, exist_ok=True)
        sample = Image.open(records[0][0]).convert("RGB")
        for name, cond in SHIFT_CONDITIONS.items():
            cond(sample).save(prev_dir / f"{name}.jpg")
        print(f"예시 사진 저장: {prev_dir}")

    rows = []
    for cond_name, cond in SHIFT_CONDITIONS.items():
        for tag, path in models.items():
            row = {"condition": cond_name, "model": tag,
                   **evaluate(path, records, cond, device, args.batch_size, args.threshold)}
            rows.append(row)
            print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="태그=모델경로, 예: E4=artifacts/beans_resnet18_full_e8")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="reports/data/shift_eval.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--preview", action="store_true")
    run(parser.parse_args())
