#!/usr/bin/env python3
"""data/human 폴더의 이미지 파일들을 human_1, human_2, ... 로 순차 rename.

사용 예)
    python rename_human_images.py --dry-run          # 미리보기 (실제 변경 없음)
    python rename_human_images.py                    # 실행
    python rename_human_images.py --pad 3            # human_001.png 형태
    python rename_human_images.py --sort name        # 파일명 순 정렬
"""

import argparse
import csv
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def collect(folder: Path, sort_key: str) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if sort_key == "mtime":
        files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    else:
        files.sort(key=lambda p: p.name)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="data/human", help="대상 폴더 (기본: data/human)")
    ap.add_argument("--prefix", default="human_", help="새 파일명 접두사 (기본: human_)")
    ap.add_argument("--start", type=int, default=1, help="시작 번호 (기본: 1)")
    ap.add_argument("--pad", type=int, default=0, help="번호 자리수 zero-padding (기본: 0 = 패딩 없음)")
    ap.add_argument("--sort", choices=["mtime", "name"], default="mtime",
                    help="정렬 기준: mtime=다운로드(수정) 시각순, name=파일명순 (기본: mtime)")
    ap.add_argument("--dry-run", action="store_true", help="실제로 바꾸지 않고 결과만 출력")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"폴더를 찾을 수 없습니다: {folder}")

    files = collect(folder, args.sort)
    if not files:
        raise SystemExit(f"이미지 파일이 없습니다: {folder}")

    plan = []
    for i, src in enumerate(files, start=args.start):
        num = str(i).zfill(args.pad) if args.pad else str(i)
        plan.append((src, folder / f"{args.prefix}{num}{src.suffix.lower()}"))

    print(f"대상: {folder}  ({len(plan)}개, 정렬={args.sort})")
    for src, dst in plan[:5]:
        print(f"  {src.name} -> {dst.name}")
    if len(plan) > 5:
        print(f"  ... ({len(plan) - 5}개 더)  마지막: {plan[-1][0].name} -> {plan[-1][1].name}")

    if args.dry_run:
        print("\n[dry-run] 실제 변경 없음.")
        return

    # 1단계: 임시 이름으로 옮겨 기존 파일명과의 충돌 방지
    tmp_pairs = []
    for idx, (src, dst) in enumerate(plan):
        tmp = folder / f".__rename_tmp_{idx}{src.suffix.lower()}"
        src.rename(tmp)
        tmp_pairs.append((tmp, dst, src.name))

    # 2단계: 최종 이름으로 변경
    mapping = []
    for tmp, dst, orig_name in tmp_pairs:
        tmp.rename(dst)
        mapping.append((orig_name, dst.name))

    map_path = folder / "rename_mapping.csv"
    with map_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original", "renamed"])
        w.writerows(mapping)

    print(f"\n완료: {len(mapping)}개 rename. 매핑 저장 -> {map_path}")


if __name__ == "__main__":
    main()
