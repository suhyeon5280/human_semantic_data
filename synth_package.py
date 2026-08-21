#!/usr/bin/env python3
"""합성 데이터셋 -> OmniVLA LeLaN 학습 폴더 구조로 패키징.

main.py의 package_for_omnivla()와 같은 출력 형태를 만든다.

    out_root/<source_episode>/image/00000000.jpg          (224x224)
    out_root/<source_episode>/pickle_nomad/00000000.pkl   (객체 리스트)

로더(vint_train/data/lelan_dataset.py)는 폴더 안 파일을 0부터 연속 인덱스로 읽고,
현재 인덱스 iv에 대해 image_path[iv-1], [iv-2] 를 시간적 컨텍스트로 사용한다.
그래서 프레임은 원본 촬영 순서대로 정렬해 재번호해야 한다.

알아둘 제약 두 가지
-------------------
1) 합성 샘플은 프레임 간격 3 이상으로 "무작위" 선택했기 때문에 간격이 균일하지 않다
   (3, 6, 12 ...). 원본 파이프라인은 정확히 stride=3 균일이다. 컨텍스트 프레임이
   때때로 실제보다 먼 과거가 된다.
2) 더 근본적으로, 컨텍스트 프레임(i-1, i-2)은 **다른 합성 샘플**이라 거기 서 있는
   사람이 현재 프레임의 사람과 다르다. 프레임마다 독립적으로 사람을 심는 방식에서는
   피할 수 없다 (같은 사람을 여러 프레임에 일관되게 두려면 카메라 odometry가 필요한데
   데이터셋에 없다). 컨텍스트는 주로 자아운동 정보로 쓰이므로 치명적이진 않지만,
   "컨텍스트에 보이던 사람이 사라진다"는 점은 알고 있어야 한다.

--layout minisequence 를 쓰면 샘플마다 [원본 직전 프레임 N장 + 합성 프레임 1장]으로
된 독립 폴더를 만든다. 위 2번이 해소되는 대신(컨텍스트에 사람이 아예 없음) 시퀀스가
매우 짧아져서, 로더의 인덱스 계산 방식에 따라 안 맞을 수 있다. 어느 쪽이 맞는지는
실제 OmniVLA 학습 설정에서 확인해야 한다.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

OMNIVLA_IMG_SIZE = 224
IMAGE_MODE = "stretch"          # main.py 기본값과 동일 (전체 FOV 유지)
OMNIVLA_KEYS = ("pose_median", "pose_median_norm", "nomad_traj_norm", "prompt", "bbox")


def transform_image(src: Path, dst: Path, size: int, mode: str) -> None:
    im = Image.open(src).convert("RGB")
    if mode == "center_square":
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im.resize((size, size)).save(dst, quality=95)


def slim(objs: list, keep_debug: bool) -> list:
    if keep_debug:
        return objs
    return [{k: o[k] for k in OMNIVLA_KEYS if k in o} for o in objs]


def pack_episode_layout(samples, synth_dir, out_root, args) -> tuple[int, int]:
    """main.py와 동일한 구조: 원본 에피소드별로 묶어 촬영 순서대로 재번호."""
    by_ep = defaultdict(list)
    for s in samples:
        by_ep[s["episode"]].append(s)

    frames = objs_total = 0
    for ep, items in sorted(by_ep.items()):
        items.sort(key=lambda s: int(s["frame"]))
        img_dir = out_root / ep / "image"
        pkl_dir = out_root / ep / "pickle_nomad"
        img_dir.mkdir(parents=True, exist_ok=True)
        pkl_dir.mkdir(parents=True, exist_ok=True)

        n_obj = 0
        for i, s in enumerate(tqdm(items, desc=f"패키징 {ep}", leave=False)):
            transform_image(synth_dir / f"{s['name']}.jpg", img_dir / f"{i:08d}.jpg",
                            args.img_size, args.image_mode)
            pkl_src = synth_dir / f"{s['name']}.pkl"
            objs = pickle.loads(pkl_src.read_bytes()) if pkl_src.exists() else []
            if args.require_traj:
                objs = [o for o in objs if o.get("nomad_traj_norm") is not None]
            (pkl_dir / f"{i:08d}.pkl").write_bytes(pickle.dumps(slim(objs, args.keep_debug)))
            n_obj += len(objs)

        print(f"  [완료] {ep}: 프레임 {len(items)}개, 객체 {n_obj}개")
        frames += len(items)
        objs_total += n_obj
    return frames, objs_total


def pack_minisequence_layout(samples, synth_dir, episodes_dir, out_root, args) -> tuple[int, int]:
    """샘플마다 독립 폴더: [원본 직전 프레임 N장] + [합성 프레임 1장].

    컨텍스트에 합성 사람이 섞이지 않는다. 대신 시퀀스 길이가 N+1로 매우 짧다.
    """
    frames = objs_total = 0
    for s in tqdm(samples, desc="패키징 (minisequence)"):
        img_dir = out_root / s["name"] / "image"
        pkl_dir = out_root / s["name"] / "pickle_nomad"
        img_dir.mkdir(parents=True, exist_ok=True)
        pkl_dir.mkdir(parents=True, exist_ok=True)

        idx = int(s["frame"])
        seq = []
        for back in range(args.context, 0, -1):
            p = episodes_dir / s["episode"] / f"{idx - back:06d}.jpg"
            if p.exists():
                seq.append(p)
        seq.append(synth_dir / f"{s['name']}.jpg")          # 마지막이 합성 프레임

        pkl_src = synth_dir / f"{s['name']}.pkl"
        objs = pickle.loads(pkl_src.read_bytes()) if pkl_src.exists() else []
        if args.require_traj:
            objs = [o for o in objs if o.get("nomad_traj_norm") is not None]

        for i, p in enumerate(seq):
            transform_image(p, img_dir / f"{i:08d}.jpg", args.img_size, args.image_mode)
            # 컨텍스트 프레임의 pkl은 빈 리스트 (로더가 정합만 맞추고 건너뛴다)
            payload = slim(objs, args.keep_debug) if i == len(seq) - 1 else []
            (pkl_dir / f"{i:08d}.pkl").write_bytes(pickle.dumps(payload))

        frames += len(seq)
        objs_total += len(objs)
    return frames, objs_total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", default="data/synth_dataset")
    ap.add_argument("--episodes-dir", default="data/episodes")
    ap.add_argument("--out", default="data/omnivla_dataset")
    ap.add_argument("--layout", choices=["episode", "minisequence"], default="episode")
    ap.add_argument("--context", type=int, default=3, help="minisequence 레이아웃의 컨텍스트 길이")
    ap.add_argument("--img-size", type=int, default=OMNIVLA_IMG_SIZE)
    ap.add_argument("--image-mode", choices=["stretch", "center_square"], default=IMAGE_MODE)
    ap.add_argument("--keep-debug", action="store_true", help="디버그 필드까지 유지 (기본은 slim)")
    ap.add_argument("--require-traj", action="store_true",
                    help="nomad_traj_norm이 채워진 객체만 포함 (step2b 실행 후 권장)")
    args = ap.parse_args()

    synth_dir, out_root = Path(args.synth_dir), Path(args.out)
    manifest = synth_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"[CRITICAL] {manifest} 없음. synth_manifest.py를 먼저 실행하세요.")

    samples = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    # main.py와 동일: 객체가 없는 프레임(0명 negative)도 image/pkl을 만들고 pkl은 빈
    # 리스트로 둔다. 로더가 연속 인덱스를 요구하므로 빼면 시간축 정합이 깨진다.
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"[PACKAGING] 합성 데이터 -> OmniVLA 구조  (layout={args.layout}, "
          f"{args.img_size}x{args.img_size} {args.image_mode}, slim={not args.keep_debug})")
    print("=" * 66)

    if args.layout == "episode":
        frames, objs = pack_episode_layout(samples, synth_dir, out_root, args)
    else:
        frames, objs = pack_minisequence_layout(samples, synth_dir, Path(args.episodes_dir),
                                                out_root, args)

    print("-" * 66)
    print(f"[완료] 샘플 {len(samples)}개 -> 프레임 {frames}개 / 객체 {objs}개")
    print(f"  출력: {out_root.resolve()}")
    if objs == 0:
        print("  [WARN] 객체가 0개입니다. --require-traj를 켰다면 step2b(NoMaD)를 먼저 돌리세요.")
    print("  ── OmniVLA data_config.yaml 힌트 ──")
    print("     metric_waypoint_spacing: 0.12")
    print(f"     data_image_folder = data_pickle_folder = {out_root.resolve()}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
