#!/usr/bin/env python3
"""합성 보행자 데이터셋 전체 파이프라인 — 한 번에 실행.

    python3 synth_pipeline.py --all --num 600                # 처음부터 전부
    python3 synth_pipeline.py --all --skip-gpu                # GPU 없는 머신 (합성까지만)

    # GPU 머신: 옮겨온 합성 데이터로 궤적 채우고 학습 폴더까지 한 번에
    python3 synth_pipeline.py --stage nomad,package \
        --synth-dir synth_for_gpu/synth_dataset \
        --episodes-dir synth_for_gpu/episodes \
        --out data/omnivla_dataset

단계
----
  depth      step1_metric3d_engine.py   [GPU] [조건부]  원본 프레임의 metric depth
  calibrate  synth_calibrate.py                        에피소드별 지면 평면 피팅
  assets     synth_assets.py                           에셋별 실제 키 H_real (1회)
  compose    synth_compose.py                          합성 + GT + pkl 뼈대
  nomad      step2b_nomad_synth_engine.py [GPU]        nomad_traj_norm 채우기
  package    synth_package.py                          OmniVLA 학습 폴더 구조


Metric3D는 다시 돌려야 하나? — 아니오 (조건부)
----------------------------------------------
depth를 쓰는 곳은 **합성 시 가림(occlusion) 판정 한 군데뿐**이고, 거기 필요한 것은
"원본 프레임"의 depth다. 그건 data/episodes/*/metric_depth/에 이미 있다.

  - 사람 배치: depth를 안 쓴다. 거리 d를 먼저 뽑고 물리 카메라 높이(0.561m)로
    발 위치를 역산한다 (에피소드별 depth 스케일 오차에 면역).
  - 3D 위치 GT: 우리가 심었으므로 정확히 안다. depth 추정이 필요 없다.
  - step2b: GT pose를 쓰므로 depth를 안 쓴다.
  - 패키징: depth를 안 쓴다.

**합성된 이미지에 Metric3D를 다시 돌리는 것은 오히려 해롭다.** 붙여넣은 사람의
거리를 추정하려 들 텐데, 우리는 그 값을 이미 정확히 알고 있다. 추정치로 덮어쓰면
GT가 오염된다.

따라서 depth 단계는 **metric_depth가 없는 에피소드가 있을 때만** 실행된다.
새 에피소드를 추가하면 그때 자동으로 걸린다.


Qwen(step3)은? — 필요 없다
--------------------------
step3는 SAM이 찾은 정체불명의 객체에 자연어 라벨을 붙이는 단계다. 합성 사람은
어떤 에셋인지 알고 있고, 캡션도 미리 달아뒀다(data/human_annotations.jsonl).
게다가 distractor와 헷갈리지 않는 캡션을 골라 쓰므로 Qwen보다 정확하다.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STAGES = ["depth", "calibrate", "assets", "compose", "nomad", "package"]
GPU_STAGES = {"depth", "nomad"}

EPISODES_DIR = Path("data/episodes")
SYNTH_DIR = Path("data/synth_dataset")
GROUND_FIT = Path("data/episode_ground_fit.json")
ASSETS_JSON = Path("data/human_assets.json")


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'=' * 66}\n[{desc}] {' '.join(cmd)}\n{'=' * 66}")
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd)
    ok = r.returncode == 0
    print(f"---> {'완료' if ok else '실패(코드 %d)' % r.returncode} ({time.time() - t0:.1f}초)")
    return ok


def episodes_missing_depth(eps_dir: Path = EPISODES_DIR) -> list[str]:
    if not eps_dir.exists():
        return []
    missing = []
    for ep in sorted(d for d in eps_dir.iterdir() if d.is_dir()):
        n_img = len(list(ep.glob("*.jpg")))
        n_depth = len(list((ep / "metric_depth").glob("*_depth.npy")))
        if n_img and n_depth < n_img:
            missing.append(f"{ep.name} ({n_depth}/{n_img})")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="전체 단계 실행")
    ap.add_argument("--stage", default="", help=f"쉼표 구분. 가능: {','.join(STAGES)}")
    ap.add_argument("--skip-gpu", action="store_true", help="GPU 필요 단계(depth, nomad) 건너뜀")
    ap.add_argument("--num", type=int, default=600, help="합성할 샘플 수 상한")
    ap.add_argument("--frame-gap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/omnivla_dataset", help="패키징 출력 루트")
    ap.add_argument("--synth-dir", default=str(SYNTH_DIR),
                    help="합성 결과 위치. 다른 머신에서 압축 풀었으면 그 경로를 준다")
    ap.add_argument("--episodes-dir", default=str(EPISODES_DIR),
                    help="원본 주행 프레임 위치 (nomad --context real 에 필요)")
    ap.add_argument("--layout", choices=["episode", "minisequence"], default="episode")
    ap.add_argument("--context", choices=["real", "repeat"], default="real",
                    help="step2b의 obs 컨텍스트 구성 방식")
    ap.add_argument("--goal-mode", choices=["square", "pad", "bbox"], default="square",
                    help="step2b의 goal 이미지 구성 (square 권장)")
    ap.add_argument("--force-assets", action="store_true",
                    help="H_real을 다시 뽑는다 (기존 GT와 어긋나므로 재합성 필요)")
    ap.add_argument("--dry-run", action="store_true", help="실행할 단계만 출력")
    args = ap.parse_args()

    stages = STAGES if args.all else [s.strip() for s in args.stage.split(",") if s.strip()]
    if not stages:
        ap.error("--all 또는 --stage 중 하나가 필요합니다.")
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"알 수 없는 단계: {bad}. 가능: {STAGES}")

    if args.skip_gpu:
        skipped = [s for s in stages if s in GPU_STAGES]
        stages = [s for s in stages if s not in GPU_STAGES]
        if skipped:
            print(f"[INFO] --skip-gpu: {skipped} 건너뜀")

    # depth 단계는 정말 필요할 때만
    if "depth" in stages:
        missing = episodes_missing_depth(Path(args.episodes_dir))
        if missing:
            print(f"[INFO] metric_depth가 없는 에피소드 {len(missing)}개: {missing}")
        else:
            print("[INFO] 모든 에피소드에 metric_depth가 있습니다 -> depth 단계 건너뜀.")
            print("       (합성 이미지에 Metric3D를 다시 돌리면 안 됩니다. 사람의 거리는 GT입니다.)")
            stages.remove("depth")

    if "assets" in stages and ASSETS_JSON.exists() and not args.force_assets:
        print(f"[INFO] {ASSETS_JSON} 이미 존재 -> assets 단계 건너뜀 (--force-assets로 재생성)")
        stages.remove("assets")

    print(f"\n실행할 단계: {stages}")
    if args.dry_run:
        return

    synth_dir, eps_dir = args.synth_dir, args.episodes_dir
    cmds = {
        "depth": (["step1_metric3d_engine.py"], "1/6 Metric3D depth [GPU]"),
        "calibrate": (["synth_calibrate.py"], "2/6 지면 평면 피팅"),
        "assets": (["synth_assets.py"] + (["--force"] if args.force_assets else []),
                   "3/6 에셋 H_real 부여"),
        "compose": (["synth_compose.py", "--num", str(args.num),
                     "--frame-gap", str(args.frame_gap), "--seed", str(args.seed),
                     "--quiet", "--out", synth_dir], "4/6 합성 + GT"),
        "nomad": (["step2b_nomad_synth_engine.py", "--synth-dir", synth_dir,
                   "--episodes-dir", eps_dir, "--context", args.context,
                   "--goal-mode", args.goal_mode],
                  "5/6 NoMaD 궤적 [GPU]"),
        "package": (["synth_package.py", "--synth-dir", synth_dir,
                     "--episodes-dir", eps_dir, "--out", args.out,
                     "--layout", args.layout, "--require-traj"],
                    "6/6 OmniVLA 패키징"),
    }

    t0 = time.time()
    for s in stages:
        cmd, desc = cmds[s]
        if not run(cmd, desc):
            print(f"\n[중단] '{s}' 단계 실패. 고친 뒤 --stage {','.join(stages[stages.index(s):])} 로 이어가세요.")
            sys.exit(1)

    print(f"\n{'=' * 66}\n[전체 완료] {time.time() - t0:.1f}초")
    sd = Path(args.synth_dir)
    if sd.exists():
        n = len(list(sd.glob("*_gt.json")))
        print(f"  합성 샘플     : {n}개  ({sd})")
    if Path(args.out).exists():
        print(f"  학습용 패키지 : {Path(args.out).resolve()}")

    # nomad를 건너뛰었다면 반드시 알려준다 — 이 상태로는 학습이 안 된다
    pkls = sorted(sd.glob("*.pkl")) if sd.exists() else []
    if pkls and "nomad" not in stages:
        import pickle
        pending = sum(1 for p in pkls[:50]
                      for o in pickle.loads(p.read_bytes())
                      if o.get("nomad_traj_norm") is None)
        if pending:
            print("\n  [주의] nomad_traj_norm이 비어 있습니다. GPU 머신에서 아래를 실행하세요:")
            print("     python3 step2b_nomad_synth_engine.py --synth-dir data/synth_dataset")
            print("     python3 synth_package.py --require-traj   # 그 다음 다시 패키징")
    print("=" * 66)


if __name__ == "__main__":
    main()
