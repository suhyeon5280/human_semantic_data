#!/usr/bin/env python3
"""1단계: 에셋 준비 — 사람 PNG마다 실제 키 H_real을 한 번만 부여한다.

결과는 data/human_assets.json에 고정 저장된다. 합성을 여러 번 돌려도 같은 에셋은
항상 같은 키를 갖도록 하기 위함 (에셋 ID -> 3D 크기가 흔들리면 GT가 오염된다).

키 범위는 요청받은 1.60~1.85m를 기본으로 하되, 캡션으로 알 수 있는 경우만 조정한다.
어린이 에셋에 1.85m를 주면 depth 대비 픽셀 크기가 어긋나 눈에 띄게 이상해지기 때문.
"""

import argparse
import json
import random
from pathlib import Path

from human_annotations import load as load_annotations

# (판별 키워드, (최소, 최대)) — 위에서부터 먼저 걸리는 것을 적용
HEIGHT_RULES = [
    (("girl", "boy", "child"),          (1.45, 1.70)),   # 아동/청소년
    (("elderly", "an old woman", "an old man"), (1.55, 1.75)),
    (("woman",),                        (1.55, 1.75)),
    (("man",),                          (1.65, 1.88)),
]
DEFAULT_RANGE = (1.60, 1.80)   # "a person ..." 등 성별 불명


def pick_range(captions: list[str]) -> tuple[str, tuple[float, float]]:
    blob = " | ".join(captions).lower()
    for keys, rng in HEIGHT_RULES:
        if any(k in blob for k in keys):
            return keys[0], rng
    return "person", DEFAULT_RANGE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/human_assets.json")
    ap.add_argument("--images", default="data/human_cropped")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--force", action="store_true", help="이미 있어도 새로 뽑는다 (GT 재생성 필요)")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"이미 존재합니다: {out_path}  (다시 뽑으려면 --force)")
        return

    ann = load_annotations()
    rng = random.Random(args.seed)
    assets, bucket_count = {}, {}

    for i in sorted(ann):
        caps = ann[i]["captions"]
        bucket, (lo, hi) = pick_range(caps)
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
        assets[str(i)] = {
            "id": i,
            "file": ann[i]["file"],
            "image": str(Path(args.images) / ann[i]["file"]),
            "H_real": round(rng.uniform(lo, hi), 3),
            "height_bucket": bucket,
            "captions": caps,
        }

    out_path.write_text(json.dumps({"seed": args.seed, "assets": assets},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    heights = [a["H_real"] for a in assets.values()]
    print(f"에셋 {len(assets)}개 -> {out_path}")
    print(f"키 범위 {min(heights):.2f}~{max(heights):.2f}m (평균 {sum(heights)/len(heights):.2f}m)")
    print("분류:", ", ".join(f"{k} {v}개" for k, v in sorted(bucket_count.items())))


if __name__ == "__main__":
    main()
