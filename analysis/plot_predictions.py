"""판정 결과를 사진 격자로 그린다.

모드 1 (--csv): pred_*.csv 에서 틀린 사진 + 보류(score < --hold) 사진을 그린다.
    빨간 테두리 = 틀림, 주황 테두리 = 맞췄지만 보류.
모드 2 (--ood-csv): 외부 사진 판정 + 콩잎 거리 검사 결과를 그린다.
    빨간 테두리 = 콩잎 아님? (거리 > 기준), 초록 = 통과.

python -m analysis.plot_predictions --csv reports/data/pred_E6_test.csv --out reports/figures/test_review_E6.png
python -m analysis.plot_predictions --ood-csv reports/data/ood_external_IMGNETfeat.csv --folder external_photos --out external_photos/external_grid_E6.png
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

SHORT = {"angular_leaf_spot": "ALS", "bean_rust": "rust", "healthy": "healthy"}
RED, ORANGE, GREEN = "#d62728", "#ff7f0e", "#2ca02c"


def short(name):
    return SHORT.get(name, name)


def draw_grid(items, out, title, cols):
    """items: [(image_path, caption, color)]"""
    rows = max(1, math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    axes = [axes] if rows * cols == 1 else list(getattr(axes, "flat", axes))
    for ax in axes:
        ax.axis("off")
    for ax, (path, caption, color) in zip(axes, items):
        img = Image.open(path).convert("RGB")
        img.thumbnail((256, 256))
        ax.imshow(img)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(4)
        ax.set_title(caption, fontsize=8, color=color)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"저장: {out} ({len(items)}장)")


def test_review(csv_path, out, hold, cols):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    picked = [r for r in rows if r["correct"] != "1" or float(r["score"]) < hold]
    # 틀린 것 먼저, 그다음 점수 낮은 순
    picked.sort(key=lambda r: (r["correct"] == "1", float(r["score"])))
    items = []
    for r in picked:
        wrong = r["correct"] != "1"
        name = Path(r["path"]).stem.split("_")[-1]  # 예: test.14
        mark = "WRONG" if wrong else "HOLD"
        cap = f"{mark} {short(r['true'])}/{name}\ntrue {short(r['true'])} -> pred {short(r['pred'])} {float(r['score']):.2f}"
        items.append((r["path"], cap, RED if wrong else ORANGE))
    n_wrong = sum(r["correct"] != "1" for r in rows)
    n_hold = sum(float(r["score"]) < hold for r in rows)
    title = (f"{Path(csv_path).stem}: {len(rows)} images, wrong {n_wrong}, "
             f"held (score<{hold}) {n_hold}  |  red=wrong, orange=held but correct")
    draw_grid(items, out, title, cols)


def external_review(ood_csv, folder, out, cols):
    rows = list(csv.DictReader(open(ood_csv, encoding="utf-8")))
    items = []
    print("번호  파일")
    for i, r in enumerate(rows, 1):
        path = Path(folder) / r["file"]
        if not path.exists():
            print(f"  건너뜀(없음): {path}")
            continue
        flag = r["ood"] == "1"
        dist = float(r["distance"])
        cap = (f"#{i} {short(r['pred'])} {float(r['score']):.2f}\n"
               f"dist {dist:.1f}" + (" NOT BEAN?" if flag else " ok"))
        items.append((path, cap, RED if flag else GREEN))
        print(f"#{i:<3} {r['file']}")
    title = "External photos: E6 label/score + leaf distance  |  red=maybe not bean, green=passed"
    draw_grid(items, out, title, cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="pred_*.csv (Test 오답·보류 격자)")
    p.add_argument("--ood-csv", help="ood_external_*.csv (외부 사진 격자)")
    p.add_argument("--folder", default="external_photos")
    p.add_argument("--hold", type=float, default=0.9)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.csv:
        test_review(a.csv, a.out, a.hold, a.cols)
    elif a.ood_csv:
        external_review(a.ood_csv, a.folder, a.out, a.cols)
    else:
        p.error("--csv 또는 --ood-csv 중 하나가 필요해요")


if __name__ == "__main__":
    main()
