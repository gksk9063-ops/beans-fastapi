"""학습된 모델로 한 split의 모든 이미지를 예측해 CSV와 혼동행렬을 저장한다.

예:
  python -m analysis.predict_split --model artifacts/beans_resnet18 --split val --tag E0
  python -m analysis.predict_split --model artifacts/beans_resnet18_full_e8 --split val --tag E4
  # Test는 최종 모델 확정 후 1번만:
  python -m analysis.predict_split --model artifacts/beans_resnet18_full_e8 --split test --tag E4 --confirm-test
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # finetune.py와 같은 필터


def list_images(split_dir: Path, classes: list[str]):
    """폴더 이름 = 정답 라벨. finetune.py와 같은 방식으로 이미지를 모은다."""
    records = []
    for name in classes:
        for path in sorted((split_dir / name).iterdir()):
            if path.is_file() and path.suffix.lower() in EXTS:
                records.append((path, name))
    return records


def run(args):
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Test는 최종 모델 확정 후 1번만 확인합니다. 맞다면 --confirm-test를 붙이세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()

    # 모델 config에 저장된 클래스 순서를 그대로 사용 (학습 때 순서와 동일)
    classes = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    records = list_images(Path(args.data) / args.split, classes)

    rows = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            images = [Image.open(p).convert("RGB") for p, _ in batch]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            probs = torch.softmax(model(pixel_values=pixels).logits, dim=-1).cpu()
            for (path, true), prob in zip(batch, probs):
                idx = int(prob.argmax())
                row = {
                    "path": str(path),
                    "true": true,
                    "pred": classes[idx],
                    "score": round(float(prob[idx]), 4),
                    "correct": int(classes[idx] == true),
                }
                for i, name in enumerate(classes):
                    row[f"p_{name}"] = round(float(prob[i]), 4)
                rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"pred_{args.tag}_{args.split}.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 혼동행렬: 행 = 정답, 열 = 예측
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for r in rows:
        matrix[r["true"]][r["pred"]] += 1
    cm_csv = out_dir / f"cm_{args.tag}_{args.split}.csv"
    with cm_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + classes)
        for t in classes:
            writer.writerow([t] + [matrix[t][p] for p in classes])

    correct = sum(r["correct"] for r in rows)
    print(f"[{args.tag} / {args.split}] accuracy = {correct}/{len(rows)} = {correct / len(rows):.4f}")
    print("혼동행렬 (행=정답, 열=예측)")
    print("".ljust(20) + "".join(c[:12].rjust(14) for c in classes))
    for t in classes:
        print(t.ljust(20) + "".join(str(matrix[t][p]).rjust(14) for p in classes))
    print("틀린 이미지:")
    for r in rows:
        if not r["correct"]:
            print(f"  {r['path']}  정답={r['true']}  예측={r['pred']}  score={r['score']}")
    print(f"저장: {pred_csv}, {cm_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tag", required=True, help="실험 ID, 예: E0, E4")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="reports/data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confirm-test", action="store_true")
    run(parser.parse_args())
