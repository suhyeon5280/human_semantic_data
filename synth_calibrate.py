#!/usr/bin/env python3
"""에피소드별 지면 평면 피팅 -> 합성에 필요한 기하 파라미터 산출.

step1을 다시 돌리려면 Metric3D 모델 + GPU가 필요하고 프레임 1700장을 재생성해야 한다.
대신 이 스크립트는 **이미 만들어진 depth npy**에서 지면 평면을 역산해서,
합성기가 쓸 두 값을 에피소드별로 뽑는다.

  1) horizon_v : 지평선 행 (카메라 피치를 흡수)
  2) depth_k   : 기존 depth를 참 스케일로 되돌리는 배율
                 true_depth ~= stored_depth * depth_k

원리
----
평지 핀홀 모델에서 지면 픽셀의 깊이는   Z(v) = f_y * h / (v - v0)
따라서  1/Z 는 v 에 대해 선형이다.  1/Z = (v - v0)/a,  a = f_y * h

측정 depth로 (a, v0)를 피팅하면 a로부터 "그 depth가 암묵적으로 주장하는 카메라 높이"
h_implied = a / f_y 가 나온다. 카메라 높이는 공식 스펙으로 0.561m 고정이므로,
h_implied가 0.561에서 벗어난 만큼이 곧 depth 스케일 오차다.

  depth_k = 0.561 / h_implied

핵심: 합성기는 사람 배치에 depth를 쓰지 않는다. 물리 카메라 높이(0.561m)와
horizon_v만으로 발 위치를 역산하므로, depth 스케일 오차가 사람 크기를 오염시키지
않는다. depth_k는 오직 occlusion 판정에서 장면 깊이와 사람 깊이를 같은 자로
재기 위해 쓴다.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# --- 540x360 FrodoBots 카메라 (step1_metric3d_engine.py와 동일한 값) ---
FX, FY = 214.97, 254.92
CX, CY = 281.15, 174.19
CAMERA_HEIGHT_M = 0.561          # EarthRover Zero 공식 스펙: 지면에서 561mm

FIT_ROWS = (250, 356)            # 지면으로 신뢰할 수 있는 행 범위 (하단부)
FIT_COLS = (200, 340)            # 중앙부만 — 렌즈 왜곡과 측면 구조물 회피
FIT_Z_RANGE = (0.3, 6.0)         # depth가 6.582에서 포화되므로 그 아래만 사용
MAX_FIT_FRAMES = 40


def fit_frame(depth: np.ndarray):
    """한 프레임에서 (a, v0)를 피팅. 실패하면 None."""
    vs = np.arange(*FIT_ROWS)
    z = np.array([np.median(depth[v, FIT_COLS[0]:FIT_COLS[1]]) for v in vs])
    ok = (z > FIT_Z_RANGE[0]) & (z < FIT_Z_RANGE[1])
    if ok.sum() < 40:
        return None
    # 1/Z = v/a - v0/a  -> 기울기 s=1/a, 절편 i=-v0/a
    s, i = np.polyfit(vs[ok], 1.0 / z[ok], 1)
    if s <= 0:
        return None
    return 1.0 / s, -i / s


def calibrate_episode(ep_dir: Path, verbose: bool = True) -> dict | None:
    depth_files = sorted((ep_dir / "metric_depth").glob("*_depth.npy"))
    if not depth_files:
        return None
    step = max(1, len(depth_files) // MAX_FIT_FRAMES)
    picked = depth_files[::step][:MAX_FIT_FRAMES]

    a_list, v0_list = [], []
    for f in picked:
        r = fit_frame(np.load(f).astype(np.float32))
        if r is not None:
            a_list.append(r[0])
            v0_list.append(r[1])
    if len(a_list) < 5:
        return None

    a = float(np.median(a_list))
    v0 = float(np.median(v0_list))
    h_implied = a / FY
    depth_k = CAMERA_HEIGHT_M / h_implied

    # 산포 — 좁아야 "진짜 스케일 편향", 넓으면 지면 가정 자체가 흔들린 것
    h_all = np.array(a_list) / FY
    spread = float(np.percentile(h_all, 90) / np.percentile(h_all, 10))

    info = {
        "horizon_v": round(v0, 2),
        "depth_k": round(depth_k, 4),
        "h_implied_m": round(h_implied, 4),
        "pitch_deg": round(float(np.degrees(np.arctan((CY - v0) / FY))), 2),
        "spread_p90_p10": round(spread, 2),
        "num_frames_fitted": len(a_list),
        "num_frames_total": len(depth_files),
    }
    if verbose:
        flag = "  [WARN] 산포 큼 - 지면 가정 재확인" if spread > 1.8 else ""
        print(f"  {ep_dir.name}: horizon_v={info['horizon_v']:.1f}  "
              f"pitch={info['pitch_deg']:+.1f}도  "
              f"h_implied={h_implied:.3f}m  depth_k={depth_k:.3f}  "
              f"(spread {spread:.2f}x, {len(a_list)}/{len(depth_files)} 프레임){flag}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="data/episodes")
    ap.add_argument("--out", default="data/episode_ground_fit.json")
    args = ap.parse_args()

    root = Path(args.episodes)
    eps = sorted(d for d in root.iterdir() if d.is_dir())
    print(f"지면 평면 피팅 (카메라 높이 {CAMERA_HEIGHT_M}m 고정, fy={FY})\n")

    out = {"camera_height_m": CAMERA_HEIGHT_M, "fx": FX, "fy": FY, "cx": CX, "cy": CY,
           "episodes": {}}
    for ep in eps:
        info = calibrate_episode(ep)
        if info:
            out["episodes"][ep.name] = info
        else:
            print(f"  {ep.name}: 피팅 실패 (지면 샘플 부족)")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{len(out['episodes'])}개 에피소드 -> {args.out}")


if __name__ == "__main__":
    main()
