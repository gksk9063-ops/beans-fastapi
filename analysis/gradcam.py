"""Grad-CAM: 모델이 사진의 어느 부분을 보고 판정했는지 히트맵으로 그린다.

방법
  1) 마지막 합성곱 단계(resnet.encoder.stages[-1], 7x7x512)의 출력 A를 저장한다.
  2) 판정한 클래스 점수(logit)를 A로 미분해 기울기 G를 구한다.
  3) 채널별 가중치 w = G의 7x7 평균, CAM = ReLU(sum(w * A)) -> 0~1로 정규화.
  4) 7x7 CAM을 224x224로 늘려 모델 입력 사진 위에 겹친다. 빨갈수록 판정에 크게 기여한 곳.

사진 한 장당 3칸: 모델 입력 | 예측 클래스 근거 | 정답 클래스 근거(틀린 사진만)
테두리: 빨강 = 틀림, 주황 = 맞췄지만 보류(score<0.9), 초록 = 맞음

python -m analysis.gradcam --model artifacts/beans_resnet18_aug_e15 --csv reports/data/pred_E6_test.csv --out reports/figures/gradcam_E6_test.png
python -m analysis.gradcam --model artifacts/beans_resnet18_aug_e15 --folder external_photos --out external_photos/gradcam_E6_external.png
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

SHORT = {"angular_leaf_spot": "ALS", "bean_rust": "rust", "healthy": "healthy"}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
RED, ORANGE, GREEN = "#d62728", "#ff7f0e", "#2ca02c"


def short(name):
    return SHORT.get(name, name)


class GradCAM:
    def __init__(self, model_dir):
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir).eval()
        self.labels = self.model.config.id2label
        self.name2id = {v: int(k) for k, v in self.labels.items()}
        layer = self.model.resnet.encoder.stages[-1]
        layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        output.retain_grad()
        self.act = output

    def input_image(self, pixels):
        """정규화된 입력을 다시 0~1 사진으로 되돌린다 (모델이 실제로 본 224x224)."""
        mean = np.array(self.processor.image_mean).reshape(3, 1, 1)
        std = np.array(self.processor.image_std).reshape(3, 1, 1)
        img = pixels[0].detach().numpy() * std + mean
        return np.clip(img.transpose(1, 2, 0), 0, 1)

    def run(self, path, extra_class=None):
        image = Image.open(path).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"]
        logits = self.model(pixel_values=pixels).logits
        probs = logits.softmax(-1)[0]
        pred = int(probs.argmax())
        targets = [pred] + ([self.name2id[extra_class]] if extra_class else [])
        cams = []
        for t in targets:
            self.model.zero_grad()
            if self.act.grad is not None:
                self.act.grad = None
            logits[0, t].backward(retain_graph=True)
            weights = self.act.grad[0].mean(dim=(1, 2))                 # (512,)
            cam = torch.relu((weights[:, None, None] * self.act[0]).sum(0))  # (7,7)
            cam = cam / (cam.max() + 1e-8)
            cam = torch.nn.functional.interpolate(
                cam[None, None], size=pixels.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0].detach().numpy()
            cams.append((self.labels[t], float(probs[t]), cam))
        return self.input_image(pixels), cams


def pick_from_csv(csv_path, hold):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    picked = [r for r in rows if r["correct"] != "1" or float(r["score"]) < hold]
    picked.sort(key=lambda r: (r["correct"] == "1", float(r["score"])))
    return [(Path(r["path"]), r["true"]) for r in picked]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--csv", help="pred_*.csv: 틀린 사진 + 보류 사진만")
    p.add_argument("--folder", help="폴더 안 사진 전부 (정답 없음)")
    p.add_argument("--hold", type=float, default=0.9)
    p.add_argument("--per-row", type=int, default=2, help="한 줄에 사진 몇 장")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.csv:
        items = pick_from_csv(a.csv, a.hold)
    elif a.folder:
        items = [(f, None) for f in sorted(Path(a.folder).iterdir())
                 if f.suffix.lower() in EXTS and not f.stem.startswith(("external_grid", "gradcam"))]
    else:
        p.error("--csv 또는 --folder 중 하나가 필요해요")

    gc = GradCAM(a.model)
    rows = math.ceil(len(items) / a.per_row)
    fig, axes = plt.subplots(rows, 3 * a.per_row, figsize=(3 * a.per_row * 2.4, rows * 2.7),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for i, (path, true) in enumerate(items, 1):
        img, cams = gc.run(path)
        pred_name, score, _ = cams[0]
        wrong = true is not None and pred_name != true
        if wrong:
            img, cams = gc.run(path, extra_class=true)
        color = RED if wrong else ORANGE if score < a.hold else GREEN
        tag = f"#{i} {Path(path).stem.split('_')[-1] if true else ''}".strip()
        if true:
            tag += f"  true {short(true)}"
        print(f"#{i:<3} {path}  pred {pred_name} {score:.3f}" + (f"  true {true}" if true else ""))

        r, c0 = divmod(i - 1, a.per_row)
        c0 *= 3
        panels = [(img, None, tag)] + [
            (img, cam, f"why {short(n)} {s:.2f}") for n, s, cam in cams
        ]
        for k, (base, cam, title) in enumerate(panels):
            ax = axes[r, c0 + k]
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(base)
            if cam is not None:
                ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(3)
            ax.set_title(title, fontsize=8, color=color)

    fig.suptitle("Grad-CAM (last conv stage): red = region that pushed the decision  |  "
                 "border red=wrong, orange=held, green=ok  |  all dark = no positive evidence", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / (rows * 2.7)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"저장: {a.out} ({len(items)}장)")


if __name__ == "__main__":
    main()
