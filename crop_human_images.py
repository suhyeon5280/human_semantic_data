#!/usr/bin/env python3
"""누끼딴 PNG의 투명 여백을 잘라내 사람 크기에 딱 맞게 crop.

알파 채널의 bounding box를 구해 사람 영역만 남기며, 투명 배경(RGBA)은 그대로 유지합니다.
몸이 잘리지 않도록 알파가 있는 픽셀 전체를 감싸는 최소 사각형만 사용합니다.

사용 예)
    python crop_human_images.py --dry-run                 # 미리보기
    python crop_human_images.py                           # data/human -> data/human_cropped
    python crop_human_images.py --margin 8                # 사방 8px 여백 남기고 crop
    python crop_human_images.py --margin-pct 2            # 사람 크기의 2% 여백
    python crop_human_images.py --inplace                 # 원본 폴더에 덮어쓰기
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".webp", ".tif", ".tiff"}  # 알파를 지원하는 포맷만


def alpha_bbox(alpha: np.ndarray, threshold: int) -> tuple[int, int, int, int] | None:
    """알파 >= threshold 인 픽셀을 모두 포함하는 (left, top, right, bottom). 없으면 None."""
    mask = alpha >= threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def expand(box, margin_px: int, margin_pct: float, w: int, h: int):
    """여백을 더하되 이미지 경계를 넘지 않게 clamp."""
    left, top, right, bottom = box
    m = margin_px
    if margin_pct:
        m += int(round(max(right - left, bottom - top) * margin_pct / 100.0))
    return (max(0, left - m), max(0, top - m), min(w, right + m), min(h, bottom + m))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/human", help="원본 폴더 (기본: data/human)")
    ap.add_argument("--dst", default="data/human_cropped", help="저장 폴더 (기본: data/human_cropped)")
    ap.add_argument("--inplace", action="store_true", help="원본 폴더에 덮어쓰기 (--dst 무시)")
    ap.add_argument("--threshold", type=int, default=10,
                    help="사람으로 볼 최소 알파값 0~255. 낮을수록 반투명 가장자리까지 포함 (기본: 10)")
    ap.add_argument("--margin", type=int, default=0, help="사방에 남길 여백 픽셀 (기본: 0)")
    ap.add_argument("--margin-pct", type=float, default=0.0, help="사람 크기 대비 여백 %% (기본: 0)")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"폴더를 찾을 수 없습니다: {src}")
    dst = src if args.inplace else Path(args.dst).expanduser().resolve()

    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"이미지 파일이 없습니다: {src}")

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for p in files:
        im = Image.open(p).convert("RGBA")  # 팔레트(P) 이미지도 알파 보존하며 변환
        w, h = im.size
        box = alpha_bbox(np.array(im)[..., 3], args.threshold)

        if box is None:
            print(f"  [skip] {p.name}: 불투명 픽셀이 없습니다")
            skipped += 1
            continue

        box = expand(box, args.margin, args.margin_pct, w, h)
        cw, ch = box[2] - box[0], box[3] - box[1]

        if args.dry_run:
            print(f"  {p.name}: {w}x{h} -> {cw}x{ch}  (bbox={box})")
        else:
            im.crop(box).save(dst / f"{p.stem}.png")  # RGBA PNG = 투명 배경 유지
        done += 1

    where = "(dry-run, 저장 안 함)" if args.dry_run else f"-> {dst}"
    print(f"\n{done}개 처리, {skipped}개 건너뜀  {where}")


if __name__ == "__main__":
    main()
