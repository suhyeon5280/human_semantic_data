#!/usr/bin/env python3
"""합성 결과 폴더를 훑어 두 개의 색인을 만든다.

  manifest.jsonl   — 샘플 1개당 한 줄. "이 이미지에 누가 들어있나"
  assets_used.json — 에셋 1개당 한 항목. "이 사람이 어느 샘플에 쓰였나" (역색인)

역색인이 핵심이다. 나중에 어떤 에셋의 캡션을 고치거나 잘못 잘린 PNG를 발견했을 때,
그 에셋이 들어간 샘플만 골라서 다시 만들 수 있어야 한다. 샘플별 gt.json만 있으면
매번 전체를 다시 뒤져야 한다.

여러 번 나눠 돌려도 폴더를 통째로 다시 스캔하므로 항상 최신 상태가 된다.

  python3 synth_manifest.py --dir data/synth_preview
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ASSETS_PATH = Path("data/human_assets.json")


def build(out_dir: Path, assets: dict) -> tuple[int, dict]:
    samples, usage = [], defaultdict(lambda: {"as_target": 0, "as_distractor": 0, "samples": []})

    for gt_path in sorted(out_dir.glob("*_gt.json")):
        g = json.loads(gt_path.read_text(encoding="utf-8"))
        people = []
        for p in g["people"]:
            aid = p["asset_id"]
            people.append({
                "asset_id": aid,
                "file": p["asset_file"],
                "is_target": p["is_target"],
                "depth_m": p["depth_m"],
                "bbox_xyxy": p["bbox_xyxy"],
                "visible_ratio": p["visible_ratio"],
            })
            u = usage[aid]
            u["as_target" if p["is_target"] else "as_distractor"] += 1
            u["samples"].append(g["name"])
        samples.append({
            "name": g["name"],
            "image": f"{g['name']}.jpg",
            "episode": g["episode"],
            "frame": g["frame"],
            "prompt": g["prompt"],
            "target_asset_id": g["target_asset_id"],
            "num_people": g["num_people"],
            "asset_ids": [p["asset_id"] for p in g["people"]],
            "people": people,
        })

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    used = {}
    for aid, u in sorted(usage.items()):
        meta = assets.get(str(aid), {})
        used[str(aid)] = {
            "id": aid,
            "file": meta.get("file"),
            "H_real": meta.get("H_real"),
            "captions": meta.get("captions", []),
            "times_used": u["as_target"] + u["as_distractor"],
            "as_target": u["as_target"],
            "as_distractor": u["as_distractor"],
            "samples": u["samples"],
        }
    unused = sorted(int(k) for k in assets if int(k) not in usage)

    index = {
        "num_samples": len(samples),
        "num_people_placed": sum(s["num_people"] for s in samples),
        "num_assets_used": len(used),
        "num_assets_total": len(assets),
        "unused_asset_ids": unused,
        "assets": used,
    }
    (out_dir / "assets_used.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(samples), index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/synth_preview")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    assets = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))["assets"]
    n, index = build(out_dir, assets)

    print(f"샘플 {n}개 -> {out_dir/'manifest.jsonl'}")
    print(f"에셋 {index['num_assets_used']}/{index['num_assets_total']}종 사용, "
          f"인물 {index['num_people_placed']}명 배치 -> {out_dir/'assets_used.json'}")
    if index["assets"]:
        top = sorted(index["assets"].values(), key=lambda a: -a["times_used"])[:3]
        print("최다 사용: " + ", ".join(
            f"#{a['id']}({a['times_used']}회)" for a in top))
    print(f"미사용 에셋 {len(index['unused_asset_ids'])}종")


if __name__ == "__main__":
    main()
