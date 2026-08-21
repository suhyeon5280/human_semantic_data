#!/usr/bin/env python3
"""human 이미지 캡션 주석 관리 유틸.

저장 포맷은 JSONL (data/human_annotations.jsonl), 한 줄에 이미지 하나:
    {"id": 1, "file": "human_1.png", "image": "data/human_cropped/human_1.png",
     "width": 306, "height": 943, "captions": ["a man in a blue kurta", ...]}

파이썬에서는 dict로 다루고, 디스크에는 JSONL로 저장한다.
    from human_annotations import load, save, add_caption
    ann = load()                      # {1: {...}, 2: {...}}
    ann[3]["captions"].append("a man wearing headphones")
    save(ann)

CLI:
    python human_annotations.py stats            # 통계
    python human_annotations.py show 3           # 한 장 보기
    python human_annotations.py grep "red"       # 캡션 검색
    python human_annotations.py flatten out.jsonl  # (이미지, 캡션1개) 단위로 펼치기
    python human_annotations.py csv out.csv      # 스프레드시트로 검수용 내보내기
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "data" / "human_annotations.jsonl"


def load(path: Path = DEFAULT_PATH) -> dict[int, dict]:
    """JSONL을 {id: record} 딕셔너리로 읽는다."""
    ann = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                ann[rec["id"]] = rec
    return ann


def save(ann: dict[int, dict], path: Path = DEFAULT_PATH) -> None:
    """{id: record} 딕셔너리를 id 순으로 JSONL에 쓴다."""
    with Path(path).open("w", encoding="utf-8") as f:
        for i in sorted(ann):
            f.write(json.dumps(ann[i], ensure_ascii=False) + "\n")


def add_caption(ann: dict[int, dict], image_id: int, caption: str) -> None:
    """중복이 아니면 캡션을 추가한다."""
    caps = ann[image_id]["captions"]
    if caption not in caps:
        caps.append(caption)


def flatten(ann: dict[int, dict]) -> list[dict]:
    """(이미지, 캡션 1개) 단위로 펼친다 — 학습 샘플 형태."""
    return [
        {"image": rec["image"], "caption": c, "id": rec["id"], "caption_idx": k}
        for rec in ann.values()
        for k, c in enumerate(rec["captions"])
    ]


def _stats(ann):
    counts = [len(r["captions"]) for r in ann.values()]
    words = {w for r in ann.values() for c in r["captions"] for w in c.lower().split()}
    print(f"이미지 {len(ann)}개, 캡션 {sum(counts)}개 "
          f"(이미지당 {min(counts)}~{max(counts)}개, 평균 {sum(counts)/len(counts):.1f})")
    print(f"고유 캡션 {len({c for r in ann.values() for c in r['captions']})}개, 어휘 {len(words)}개")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stats", "show", "grep", "flatten", "csv"])
    ap.add_argument("arg", nargs="?", help="show=id, grep=검색어, flatten/csv=출력 경로")
    ap.add_argument("--path", default=DEFAULT_PATH, help="주석 JSONL 경로")
    args = ap.parse_args()

    ann = load(Path(args.path))

    if args.cmd == "stats":
        _stats(ann)
    elif args.cmd == "show":
        rec = ann[int(args.arg)]
        print(f"{rec['file']}  ({rec['width']}x{rec['height']})")
        for c in rec["captions"]:
            print("  -", c)
    elif args.cmd == "grep":
        q = args.arg.lower()
        for i in sorted(ann):
            hits = [c for c in ann[i]["captions"] if q in c.lower()]
            if hits:
                print(f"#{i}: " + " | ".join(hits))
    elif args.cmd == "flatten":
        rows = flatten(ann)
        with Path(args.arg).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(rows)}개 샘플 -> {args.arg}")
    elif args.cmd == "csv":
        with Path(args.arg).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "file", "caption_1", "caption_2", "caption_3", "caption_4"])
            for i in sorted(ann):
                caps = ann[i]["captions"]
                w.writerow([i, ann[i]["file"]] + caps + [""] * (4 - len(caps)))
        print(f"{len(ann)}행 -> {args.arg}")


if __name__ == "__main__":
    main()
