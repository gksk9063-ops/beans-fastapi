"""CNN 예측은 그대로 두고 '확인 필요(보류)' 규칙만 붙였을 때 효과를 비교한다.

규칙 기준값(lesion 임계값)은 Train으로만 정하고, Val에서 비교한다. Test는 쓰지 않는다.

예:
  python -m analysis.rule_review --train-hsv reports/data/hsv_E4_train.csv --eval-hsv reports/data/hsv_E4_val.csv --tag E4
"""
import argparse
import csv
from pathlib import Path


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["score"] = float(r["score"])
        r["lesion_ratio"] = float(r["lesion_ratio"])
        r["correct"] = int(r["correct"])
    return rows


def percentile(values, pct):
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))
    return values[k]


def evaluate(rows, hold_fn):
    held = [r for r in rows if hold_fn(r)]
    kept = [r for r in rows if not hold_fn(r)]
    errors = [r for r in rows if not r["correct"]]
    return {
        "보류 수": len(held),
        "보류율": round(len(held) / len(rows), 4),
        "보류된 오답": sum(1 for r in held if not r["correct"]),
        "전체 오답": len(errors),
        "보류 제외 정확도": round(sum(r["correct"] for r in kept) / max(len(kept), 1), 4),
        "보류 제외 오답": sum(1 for r in kept if not r["correct"]),
    }


def run(args):
    train = load(args.train_hsv)
    rows = load(args.eval_hsv)

    # 기준값: Train의 정답 healthy 잎 중 lesion_ratio 상위 (100 - pct)%에 해당하는 값
    healthy_train = [r["lesion_ratio"] for r in train if r["true"] == "healthy"]
    t_lesion = percentile(healthy_train, args.pct)
    print(f"Train healthy {len(healthy_train)}장의 lesion_ratio {args.pct}번째 백분위 = {t_lesion:.4f} → HSV 기준값")

    rules = {
        "R0 규칙 없음 (CNN 단독)": lambda r: False,
        f"R1 점수 < {args.score}": lambda r: r["score"] < args.score,
        f"R2 healthy 예측 & lesion ≥ {t_lesion:.4f}": lambda r: r["pred"] == "healthy" and r["lesion_ratio"] >= t_lesion,
        "R3 = R1 또는 R2": lambda r: r["score"] < args.score or (r["pred"] == "healthy" and r["lesion_ratio"] >= t_lesion),
    }

    out_rows = []
    for name, fn in rules.items():
        result = {"규칙": name, **evaluate(rows, fn)}
        out_rows.append(result)

    print(f"\n[{args.tag}] 평가 데이터: {args.eval_hsv} ({len(rows)}장)")
    header = list(out_rows[0].keys())
    print(" | ".join(header))
    for r in out_rows:
        print(" | ".join(str(r[h]) for h in header))

    print("\n오답 이미지의 규칙 판정:")
    for r in rows:
        if not r["correct"]:
            print(f"  {Path(r['path']).name}  정답={r['true']} 예측={r['pred']} score={r['score']} "
                  f"lesion={r['lesion_ratio']:.4f} → R1 {'보류' if r['score'] < args.score else '통과'}, "
                  f"R2 {'보류' if r['pred'] == 'healthy' and r['lesion_ratio'] >= t_lesion else '통과'}")

    out_csv = Path(args.out) / f"rule_review_{args.tag}_{Path(args.eval_hsv).stem.split('_')[-1]}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hsv", required=True)
    parser.add_argument("--eval-hsv", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--score", type=float, default=0.9, help="CNN 확신도 보류 기준")
    parser.add_argument("--pct", type=float, default=95, help="Train healthy lesion_ratio 백분위")
    parser.add_argument("--out", default="reports/data")
    run(parser.parse_args())
