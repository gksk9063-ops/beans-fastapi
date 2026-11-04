"""동기·비동기·캐시 처리 시간을 측정해 reports/data/timings.csv로 저장한다. (서버·worker 실행 중이어야 함)

측정 항목 (각 --runs 회, 기본 5):
  sync_miss     : 처음 보내는 사진 → 모델 추론 (Redis MISS)
  sync_hit      : 방금 보낸 같은 사진 → Redis HIT
  async_accept  : 비동기 접수 응답(202)까지 시간
  async_done    : 접수부터 status=done 조회까지 시간 (worker 추론 + 폴링 포함)
첫 요청은 모델 로딩(cold start)이 섞이므로 warm-up 1회는 기록하지 않는다.

예:
  python -m analysis.measure_timings --runs 5
"""
import argparse
import csv
import time
from pathlib import Path

import httpx

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def post_image(client, url, path):
    with open(path, "rb") as f:
        return client.post(url, files={"image": (path.name, f, "image/jpeg")})


def run(args):
    base = args.base.rstrip("/")
    images = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    # 같은 사진이 이전 실행에서 캐시됐을 수 있으므로 --offset으로 매번 다른 사진을 고른다
    need = 2 * args.runs + 2
    picked = images[args.offset: args.offset + need]
    if len(picked) < need:
        raise SystemExit(f"사진이 부족합니다: {len(picked)} < {need}")

    rows = []
    with httpx.Client(timeout=60) as client:
        # warm-up: web, worker 모두 모델을 메모리에 올린다 (기록 안 함)
        post_image(client, f"{base}/api/sync/classify/", picked[0])
        job = post_image(client, f"{base}/api/async/classify/", picked[1]).json()
        while client.get(f"{base}/api/async/jobs/{job['job_id']}/").json()["status"] not in ("done", "failed"):
            time.sleep(0.05)

        for i in range(args.runs):
            # 동기 MISS → 같은 사진으로 HIT
            img = picked[2 + i]
            t0 = time.perf_counter()
            r = post_image(client, f"{base}/api/sync/classify/", img).json()
            rows.append({"case": "sync_miss", "run": i + 1, "image": img.name,
                         "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                         "inference_ms": r.get("inference_ms"), "cache_hit": r.get("cache_hit")})
            t0 = time.perf_counter()
            r = post_image(client, f"{base}/api/sync/classify/", img).json()
            rows.append({"case": "sync_hit", "run": i + 1, "image": img.name,
                         "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                         "inference_ms": r.get("inference_ms"), "cache_hit": r.get("cache_hit")})

            # 비동기: 캐시에 없는 새 사진
            img = picked[2 + args.runs + i]
            t0 = time.perf_counter()
            job = post_image(client, f"{base}/api/async/classify/", img).json()
            accept_ms = (time.perf_counter() - t0) * 1000
            while True:
                state = client.get(f"{base}/api/async/jobs/{job['job_id']}/").json()
                if state["status"] in ("done", "failed"):
                    break
                time.sleep(args.poll)
            done_ms = (time.perf_counter() - t0) * 1000
            rows.append({"case": "async_accept", "run": i + 1, "image": img.name,
                         "total_ms": round(accept_ms, 1), "inference_ms": None, "cache_hit": None})
            rows.append({"case": "async_done", "run": i + 1, "image": img.name,
                         "total_ms": round(done_ms, 1), "inference_ms": state.get("inference_ms"),
                         "cache_hit": state.get("cache_hit")})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'case':14s} {'평균 total_ms':>14s} {'min':>8s} {'max':>8s}  cache_hit")
    for case in ("sync_miss", "sync_hit", "async_accept", "async_done"):
        vals = [r["total_ms"] for r in rows if r["case"] == case]
        hits = {str(r["cache_hit"]) for r in rows if r["case"] == case}
        print(f"{case:14s} {sum(vals) / len(vals):14.1f} {min(vals):8.1f} {max(vals):8.1f}  {','.join(sorted(hits))}")
    print(f"저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--images", default="dataset/test")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--offset", type=int, default=10, help="앞쪽(이미 보낸) 사진을 건너뛸 개수")
    parser.add_argument("--poll", type=float, default=0.05, help="상태 조회 간격(초)")
    parser.add_argument("--out", default="reports/data/timings.csv")
    run(parser.parse_args())
