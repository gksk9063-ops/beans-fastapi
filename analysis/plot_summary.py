"""06 보고서 그래프: 처리 시간(5회) + 방식별 Val 정확도 -> reports/figures/performance_summary.png"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("reports/data/timings.csv", encoding="utf-8")))
cases = ["sync_miss", "sync_hit", "async_accept", "async_done"]
labels = ["Sync MISS", "Sync HIT", "Async accept", "Async done"]
vals = [[float(r["total_ms"]) for r in rows if r["case"] == c] for c in cases]
avg = [sum(v) / len(v) for v in vals]
err = [[a - min(v) for a, v in zip(avg, vals)], [max(v) - a for a, v in zip(avg, vals)]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
bars = ax1.bar(labels, avg, yerr=err, capsize=5, color=["#4f6d94", "#2f855a", "#78aafa", "#c05621"])
for b, a, v in zip(bars, avg, vals):
    ax1.text(b.get_x() + b.get_width() / 2, max(v) + 2, f"{a:.1f} ms", ha="center", fontweight="bold")
ax1.set_title("Response time (E4, CPU, 5 runs, warm-up excluded)")
ax1.set_ylabel("ms (bar = mean, line = min~max)")

names = ["E0\nHead 3ep", "E1\nHead 10ep", "B\nHead 20ep", "E2\nFull 3ep", "E3\nFull 5ep", "E4\nFull 8ep"]
acc = [0.7669, 0.9398, 0.9624, 0.9699, 0.9850, 0.9925]
bars = ax2.bar(names, acc, color=["#9aa5b1"] * 3 + ["#78aafa"] * 2 + ["#2b6cb0"])
for b, a in zip(bars, acc):
    ax2.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.4f}", ha="center", fontweight="bold")
ax2.set_ylim(0.7, 1.02)
ax2.set_title("Validation accuracy (133 images): gray Head-only / blue Full")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/figures/performance_summary.png")
print("saved reports/figures/performance_summary.png")
