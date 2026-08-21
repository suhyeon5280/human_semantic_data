#!/usr/bin/env python3
"""주행 프레임에 사람 PNG를 기하학적으로 올바르게 합성한다.

  python3 synth_compose.py --num 5 --out data/synth_preview --seed 7

핵심 설계 두 가지
-----------------
1) **배치에 depth를 쓰지 않는다.**
   depth 맵에서 픽셀을 골라 그 값을 읽는 대신, 거리 d를 먼저 뽑고 물리 카메라
   높이(0.561m)와 지평선으로 발 위치를 역산한다.

       v_foot = horizon_v + fy * 0.561 / d

   이유 A: 2~6m 구간이 세로로 48픽셀뿐이라 픽셀 하나의 depth 노이즈가 거리 오차로
           크게 증폭된다.
   이유 B: 에피소드마다 depth 스케일이 1.4배까지 어긋나 있는데(synth_calibrate.py),
           이 방식은 물리 높이만 쓰므로 그 오차에 면역이다.
   depth 맵은 occlusion 판정에만 쓰고, 그때만 depth_k로 단위를 맞춘다.

2) **비등방 리사이즈.**
   FrodoBots-2K가 1024x576을 540x360으로 비균등 리사이즈해서 fx != fy 다
   (214.97 vs 254.92). 사람 PNG는 정상 비율 사진이라 등방 리사이즈하면 전원이
   가로로 16% 뚱뚱해진다.

       h_px = fy * H_real / d
       w_px = h_px * (orig_w / orig_h) * (fx / fy)
"""

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

from synth_manifest import build as build_manifest

# ---------------------------------------------------------------- 설정
GROUND_FIT_PATH = Path("data/episode_ground_fit.json")
ASSETS_PATH = Path("data/human_assets.json")
EPISODES_DIR = Path("data/episodes")

D_RANGE = (1.6, 4.5)      # 근거리 위주. 상한을 6.0->4.5로 당겨 인물을 크게 잡는다.
MAX_TOP_CROP = 0.15       # 머리가 상단에서 잘려도 되는 최대 비율.
                          # 카메라가 0.561m로 낮아서 1.89m보다 가까우면 머리가 프레임을
                          # 벗어난다. 그 아래까지 쓰려면 약간의 크롭을 허용해야 한다.
                          # (1.6m에서 8.6% 잘림 — 실제 보행자 데이터에도 흔한 구도)
U_HALF_WIDTH = 105        # 중심(cx)에서 좌우 105px — 이 안에서 렌즈 왜곡 < 4px
MAX_OVERLAP_FRAC = 0.45   # 사람끼리 좌우로 겹쳐도 되는 최대 비율.
                          # 근거리 인물은 폭이 80px을 넘어 중앙 210px 대역에 3명이 안 들어간다.
                          # 겹침 자체는 depth 버퍼가 이미 올바르게 처리하므로(먼 사람부터
                          # 그리며 버퍼 갱신), 완전 분리를 강제할 이유가 없다. 실제 보행자
                          # 무리도 서로 겹쳐 보인다.
MIN_FRAME_GAP = 5         # 같은 에피소드 안에서 고를 프레임 사이의 최소 간격.
                          # 주행 영상이라 인접 프레임은 사실상 같은 장면이다.
PLACE_ATTEMPTS = 80       # 사람 1명당 배치 재시도 횟수. 탈락 사유는 사실상 전부 겹침이라,
                          # 시도를 늘리면 3인 장면 성공률이 올라간다.
OCC_MARGIN_M = 0.30       # depth 노이즈 허용치. 장면이 이만큼 더 앞이어야 가림으로 인정
GROUND_TOL = (0.55, 1.9)  # 발 지점 장면깊이/d 가 이 범위를 벗어나면 지면이 아님
# step2_nomad_sam_engine.py와 반드시 동일해야 하는 값 (pkl 스키마 호환)
METRIC_WAYPOINT_SPACING = 0.12
OMNIVLA_IMG_SIZE = 224

BRIGHTNESS_ALPHA = 0.5    # 밝기 매칭 강도 (1.0=완전일치, 0=미적용)
BRIGHTNESS_CLIP = (0.70, 1.30)
FEATHER_SIGMA = 0.8       # 경계 페더링

# 장면당 인원 분포 — distractor가 있어야 언어 조건이 실제로 학습된다
N_PEOPLE_CHOICES = [0, 1, 2, 3]
N_PEOPLE_WEIGHTS = [0.10, 0.40, 0.35, 0.15]

STOPWORDS = {"a", "an", "the", "in", "and", "with", "of", "on", "while", "his", "her",
             "their", "is", "to", "at", "over", "under", "from", "up", "seen", "wearing"}


# ---------------------------------------------------------------- 캡션 충돌
def content_words(caption: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", caption.lower()) if w not in STOPWORDS}


def collide(cap_a: str, cap_b: str, thresh: float = 0.5) -> bool:
    """두 캡션이 같은 사람을 가리킬 만큼 비슷한가 (Jaccard)."""
    a, b = content_words(cap_a), content_words(cap_b)
    if not a or not b:
        return True
    return len(a & b) / len(a | b) >= thresh


def choose_prompt(target_caps: list[str], distractor_caps: list[str]) -> str | None:
    """distractor와 헷갈리지 않는 것 중 '가장 짧은' 캡션을 고른다.

    혼자면 짧은 캡션("a man in a blue shirt"), 비슷한 사람이 있으면 자세한 캡션이
    자동으로 선택되어 난이도 커리큘럼이 공짜로 생긴다.
    """
    for cap in sorted(target_caps, key=len):
        if not any(collide(cap, d) for d in distractor_caps):
            return cap
    return None


def sample_cast(rng, assets: dict, n: int, max_try: int = 60):
    """캡션이 서로 충돌하지 않는 에셋 n명 + 타겟 프롬프트를 뽑는다."""
    ids = list(assets)
    for _ in range(max_try):
        picked = rng.sample(ids, n)
        target = picked[0]
        distractor_caps = [c for p in picked[1:] for c in assets[p]["captions"]]
        prompt = choose_prompt(assets[target]["captions"], distractor_caps)
        if prompt:
            return picked, prompt
    return None, None


# ---------------------------------------------------------------- 기하
def foot_row(d: float, horizon_v: float, fy: float, cam_h: float) -> float:
    return horizon_v + fy * cam_h / d


def sample_depth(rng) -> float:
    """역깊이(1/d) 균등 샘플링.

    d에 균등하게 뽑으면 화면상 크기가 1/d로 찌그러져 먼(작은) 인물이 과다해진다.
    픽셀 크기가 1/d에 비례하므로, 1/d에 균등하게 뽑아야 '화면에서 차지하는 크기'가
    고르게 퍼지고 자연스럽게 근거리 쪽에 무게가 실린다.
    """
    lo, hi = 1.0 / D_RANGE[1], 1.0 / D_RANGE[0]
    return 1.0 / rng.uniform(lo, hi)


def person_size(H_real: float, d: float, orig_w: int, orig_h: int, fx: float, fy: float):
    h_px = fy * H_real / d
    w_px = h_px * (orig_w / orig_h) * (fx / fy)     # 비등방 보정
    return max(2, int(round(w_px))), max(2, int(round(h_px)))


def sample_placements(rng, n, cam, depth_true, asset_list):
    """겹치지 않고 지면 위에 놓이는 배치 n개. 실패하면 가능한 만큼만."""
    H, W = depth_true.shape
    placed = []
    for asset in asset_list:
        for _ in range(PLACE_ATTEMPTS):
            d = sample_depth(rng)
            vf = foot_row(d, cam["horizon_v"], cam["fy"], cam["camera_height_m"])
            aw, ah = asset["_orig_size"]
            w_px, h_px = person_size(asset["H_real"], d, aw, ah, cam["fx"], cam["fy"])

            lo = int(cam["cx"] - U_HALF_WIDTH + w_px / 2)
            hi = int(cam["cx"] + U_HALF_WIDTH - w_px / 2)
            if hi <= lo:
                continue
            uf = rng.uniform(lo, hi)

            x0, y0 = int(round(uf - w_px / 2)), int(round(vf - h_px))
            # y0 < 0 이면 머리가 상단에서 잘린다. 허용치 안이면 크롭해서 쓴다.
            if -y0 > MAX_TOP_CROP * h_px or int(round(vf)) >= H or x0 < 0 or x0 + w_px > W:
                continue
            # 이미 놓인 사람과의 겹침 한도 검사 (완전 분리가 아니라 부분 겹침까지 허용)
            if any(abs(uf - p["u_foot"]) < (w_px + p["w_px"]) / 2 * (1 - MAX_OVERLAP_FRAC)
                   for p in placed):
                continue
            # 발 지점이 정말 그 거리의 지면인가 (앞에 벽/차가 있거나, 지면이 아닌 배경이면 탈락)
            vi, ui = int(round(vf)), int(round(uf))
            patch = depth_true[max(0, vi - 2):vi + 3, max(0, ui - 3):ui + 4]
            patch = patch[patch > 0.05]
            if patch.size == 0:
                continue
            ratio = float(np.median(patch)) / d
            if not (GROUND_TOL[0] <= ratio <= GROUND_TOL[1]):
                continue

            placed.append({"asset": asset, "d": d, "u_foot": uf, "v_foot": vf,
                           "x0": x0, "y0": y0, "w_px": w_px, "h_px": h_px})
            break
    return placed


# ---------------------------------------------------------------- 렌더링
def match_brightness(person_bgr, alpha, scene_patch):
    """사람 밝기를 배치 주변 장면에 부분적으로 맞춘다 (완전 일치는 부자연스러움)."""
    m = alpha > 0.3
    if m.sum() < 10 or scene_patch.size == 0:
        return person_bgr
    p_lum = float(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)[m].mean())
    s_lum = float(cv2.cvtColor(scene_patch, cv2.COLOR_BGR2GRAY).mean())
    if p_lum < 1:
        return person_bgr
    gain = float(np.clip((s_lum / p_lum) ** BRIGHTNESS_ALPHA, *BRIGHTNESS_CLIP))
    return np.clip(person_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def scene_blur_sigma(frame_bgr) -> float:
    """원본이 흐리면 사람도 살짝 흐리게. Laplacian 분산으로 대충 판단."""
    lap = cv2.Laplacian(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if lap > 300:
        return 0.0
    return float(np.clip((300 - lap) / 300 * 1.2, 0.0, 1.2))


def compose(frame_bgr, depth_true, placements, cam):
    """먼 사람부터 그리며 depth 버퍼를 갱신 -> 사람끼리의 가림도 같은 코드로 처리."""
    H, W = depth_true.shape
    canvas = frame_bgr.astype(np.float32)
    buffer = depth_true.copy()
    inst = np.zeros((H, W), np.uint8)
    blur_sigma = scene_blur_sigma(frame_bgr)
    records = []

    for idx, p in enumerate(sorted(placements, key=lambda q: -q["d"]), start=1):
        rgba = cv2.imread(p["asset"]["image"], cv2.IMREAD_UNCHANGED)
        rgba = cv2.resize(rgba, (p["w_px"], p["h_px"]), interpolation=cv2.INTER_AREA)
        full_silhouette = int((rgba[..., 3] > 128).sum())   # 크롭 전 실루엣 (가시율 분모)

        x0, y0 = p["x0"], p["y0"]
        if y0 < 0:                      # 머리가 프레임 위로 나간 만큼 잘라낸다
            rgba, y0 = rgba[-y0:], 0
        person, a = rgba[..., :3], rgba[..., 3].astype(np.float32) / 255.0
        x1, y1 = x0 + rgba.shape[1], y0 + rgba.shape[0]

        person = match_brightness(person, a, frame_bgr[y0:y1, x0:x1])
        if blur_sigma > 0.05:
            person = cv2.GaussianBlur(person, (0, 0), blur_sigma)

        # occlusion: 장면(또는 앞서 놓인 사람)이 더 앞이면 그 픽셀은 그리지 않는다
        a = a * (buffer[y0:y1, x0:x1] > p["d"] - OCC_MARGIN_M)
        visible_ratio = float((a > 0.5).sum() / max(1, full_silhouette))
        if visible_ratio < 0.15:      # 거의 다 가려지면 배치 취소
            continue
        a = cv2.GaussianBlur(a, (0, 0), FEATHER_SIGMA)      # 경계 페더링

        a3 = a[..., None]
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a3) + person.astype(np.float32) * a3

        solid = a > 0.5
        buffer[y0:y1, x0:x1][solid] = p["d"]
        inst[y0:y1, x0:x1][solid] = idx

        ys, xs = np.where(solid)
        records.append({**p, "inst_id": idx, "visible_ratio": round(visible_ratio, 3),
                        "bbox": [int(x0 + xs.min()), int(y0 + ys.min()),
                                 int(x0 + xs.max()), int(y0 + ys.max())]})

    return np.clip(canvas, 0, 255).astype(np.uint8), buffer, inst, records


# ---------------------------------------------------------------- 출력
def write_omnivla_pkl(path: Path, gt: dict, img_hw: tuple[int, int]) -> None:
    """step2가 내보내는 것과 같은 스키마의 pkl (객체 dict들의 리스트).

    주의: `nomad_traj_norm`은 채울 수 없어 None으로 둔다. 이건 GPU가 없어서가 아니라
    개념적인 문제다 — LeLaN의 행동 라벨은 로봇이 실제로 그 객체를 향해 간 미래 궤적인데,
    합성으로 끼워넣은 사람에게는 로봇이 간 적이 없다. 셋 중 하나를 정해야 한다.
      (a) 로봇의 실제 진행 경로 위에만 사람을 배치한다 (궤적을 그대로 라벨로 씀)
      (b) 사람을 향하는 직선 궤적을 합성해서 넣는다
      (c) 합성 이미지에 NoMaD를 다시 돌린다
    지금은 (c)를 전제로 자리만 비워두고 needs_nomad_traj 플래그를 세운다.
    """
    import pickle

    H, W = img_hw
    sx, sy = OMNIVLA_IMG_SIZE / W, OMNIVLA_IMG_SIZE / H
    objs = []
    for p in gt["people"]:
        if p["prompt"] is None:          # 구별 불가한 사람은 학습 목표로 쓰지 않는다
            continue
        fwd, left = p["pose_robot_fwd_left"]
        pose = np.array([[fwd, left]], dtype=np.float64)
        x1, y1, x2, y2 = p["bbox_xyxy"]
        objs.append({
            "pose_median": pose,
            "pose_median_norm": pose / METRIC_WAYPOINT_SPACING,
            "nomad_traj_norm": None,                       # ← 위 주석 참조
            "prompt": [(p["prompt"],)],                    # step3와 동일한 [(label,), ...] 형식
            "bbox": np.array([[int(y1 * sy), int(y2 * sy),
                               int(x1 * sx), int(x2 * sx)]], dtype=np.int64),
            "obj_detect": True,
            "bbox_orig_540x360": [x1, y1, x2, y2],
            "pose_mean": np.array(p["pose_camera_xyz"], dtype=np.float64),
            # ── 합성 데이터 표식 ──
            "synthetic": True,
            "needs_nomad_traj": True,
            "asset_id": p["asset_id"],
            "all_captions": p["captions"],
        })
    with path.open("wb") as f:
        pickle.dump(objs, f)


def save_sample(out_dir: Path, gt: dict, canvas, depth_stored, inst) -> None:
    name = gt["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(out_dir / f"{name}_depth_synth.npy", depth_stored.astype(np.float16))
    cv2.imwrite(str(out_dir / f"{name}_mask.png"),
                inst * (255 // max(1, gt["num_people"])))
    (out_dir / f"{name}_gt.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    write_omnivla_pkl(out_dir / f"{name}.pkl", gt, canvas.shape[:2])


def empty_sample(rng, ep_name, frame_stem, frame_path, frame, depth_stored, cam,
                 assets, out_dir):
    """사람 0명 장면 (negative sample).

    프롬프트는 장면에 **없는** 사람을 가리키도록 무작위 에셋에서 뽑는다. 그래야
    "지시한 대상이 화면에 없다"를 판단하는 hard negative가 된다.

    주의: OmniVLA pkl 스키마에는 '대상 없음' 개념이 없어서 pkl은 빈 리스트가 된다.
    이 샘플들을 실제로 쓰려면 학습 쪽에 negative 처리를 따로 넣어야 한다.
    """
    absent = assets[rng.choice(list(assets))]
    name = f"{ep_name.replace('episode_', 'ep')}_{frame_stem}_t000"
    gt = {
        "name": name, "episode": ep_name, "frame": frame_stem,
        "source_image": str(frame_path),
        "prompt": absent["captions"][0],
        "target_present": False,
        "absent_asset_id": absent["id"],
        "target_asset_id": None,
        "num_people": 0,
        "camera": cam,
        "people": [],
        "notes": "negative sample — 지시 대상이 화면에 없음. pkl은 빈 리스트.",
    }
    save_sample(out_dir, gt, frame, depth_stored, np.zeros(frame.shape[:2], np.uint8))
    gap = np.full((frame.shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(str(out_dir / f"{name}_compare.jpg"), np.hstack([frame, gap, frame]))
    return gt


# ---------------------------------------------------------------- 메인
def build_sample(rng, ep_name, frame_stem, cam, assets, out_dir):
    ep_dir = EPISODES_DIR / ep_name
    frame_path = ep_dir / f"{frame_stem}.jpg"
    depth_path = ep_dir / "metric_depth" / f"{frame_stem}_depth.npy"
    frame = cv2.imread(str(frame_path))
    if frame is None or not depth_path.exists():
        return None

    depth_stored = np.load(depth_path).astype(np.float32)
    depth_true = depth_stored * cam["depth_k"]          # occlusion 비교용 단위 통일

    n = rng.choices(N_PEOPLE_CHOICES, weights=N_PEOPLE_WEIGHTS)[0]
    if n == 0:
        return empty_sample(rng, ep_name, frame_stem, frame_path, frame,
                            depth_stored, cam, assets, out_dir)
    ids, prompt = sample_cast(rng, assets, n)
    if ids is None:
        return None

    asset_list = []
    for k, aid in enumerate(ids):
        a = dict(assets[aid])
        rgba = cv2.imread(a["image"], cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            return None
        a["_orig_size"] = (rgba.shape[1], rgba.shape[0])
        a["_is_target"] = (k == 0)
        asset_list.append(a)

    placements = sample_placements(rng, n, cam, depth_true, asset_list)
    if not placements or not any(p["asset"]["_is_target"] for p in placements):
        return None

    canvas, buffer_true, inst, records = compose(frame, depth_true, placements, cam)
    if not any(r["asset"]["_is_target"] for r in records):
        return None

    target = next(r for r in records if r["asset"]["_is_target"])
    others = [r for r in records if not r["asset"]["_is_target"]]
    name = (f"{ep_name.replace('episode_', 'ep')}_{frame_stem}"
            f"_t{target['asset']['id']:03d}"
            + ("_d" + "-".join(f"{r['asset']['id']:03d}" for r in others) if others else ""))

    people = []
    for r in records:
        d, uf, vf = r["d"], r["u_foot"], r["v_foot"]
        X = (uf - cam["cx"]) * d / cam["fx"]
        Y = (vf - cam["cy"]) * d / cam["fy"]
        # 배치된 사람 각각에 대해, 나머지와 헷갈리지 않는 자기 프롬프트를 따로 구한다.
        # 그러면 distractor도 그 자체로 유효한 학습 샘플이 된다 (같은 이미지, 다른 목표).
        own = choose_prompt(r["asset"]["captions"],
                            [c for o in records if o is not r for c in o["asset"]["captions"]])
        people.append({
            "inst_id": r["inst_id"],
            "asset_id": r["asset"]["id"],
            "asset_file": r["asset"]["file"],
            "is_target": r["asset"]["_is_target"],
            "prompt": own,                    # 구별 불가하면 None (그 사람은 목표로 못 씀)
            "H_real_m": r["asset"]["H_real"],
            "depth_m": round(d, 3),
            "foot_uv": [round(uf, 1), round(vf, 1)],
            "bbox_xyxy": r["bbox"],
            "pixel_size_wh": [r["w_px"], r["h_px"]],
            "visible_ratio": r["visible_ratio"],
            "pose_camera_xyz": [round(X, 3), round(Y, 3), round(d, 3)],
            "pose_robot_fwd_left": [round(d, 3), round(-X, 3)],   # step2 규약과 동일
            "captions": r["asset"]["captions"],
        })

    gt = {
        "name": name, "episode": ep_name, "frame": frame_stem,
        "source_image": str(frame_path), "prompt": prompt,
        "target_present": True,
        "target_asset_id": target["asset"]["id"],
        "num_people": len(records),
        "camera": cam,
        "people": people,
        "notes": "depth_synth.npy는 원본과 같은 stored 단위(true = stored * depth_k).",
    }

    save_sample(out_dir, gt, canvas, buffer_true / cam["depth_k"], inst)

    # 검수용 비교 이미지 (원본 | 합성)
    gap = np.full((frame.shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(str(out_dir / f"{name}_compare.jpg"), np.hstack([frame, gap, canvas]))
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--out", default="data/synth_preview")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frame-gap", type=int, default=MIN_FRAME_GAP,
                    help="같은 에피소드 안에서 고를 프레임 사이의 최소 간격")
    ap.add_argument("--quiet", action="store_true", help="샘플별 출력 생략")
    args = ap.parse_args()

    fit = json.loads(GROUND_FIT_PATH.read_text())
    assets = json.loads(ASSETS_PATH.read_text())["assets"]
    rng = random.Random(args.seed)
    out_dir = Path(args.out)

    frames, capacity = [], 0
    for ep_name in fit["episodes"]:
        eps_frames = sorted((EPISODES_DIR / ep_name).glob("*.jpg"))
        frames.extend((ep_name, p.stem) for p in eps_frames)
        capacity += len(eps_frames) // max(1, args.frame_gap)
    rng.shuffle(frames)

    if args.num > capacity:
        print(f"[WARN] 요청 {args.num}개 > 프레임 간격 {args.frame_gap} 기준 최대 {capacity}개. "
              f"최대치까지만 생성됩니다. 더 필요하면 --frame-gap을 줄이세요.")

    used_frames = {}          # 에피소드별로 이미 고른 프레임 번호
    made, tried, rejected = [], 0, {"frame_gap": 0, "no_placement": 0}
    for ep_name, stem in frames:
        if len(made) >= args.num:
            break
        # 인접 프레임은 사실상 같은 장면이므로 건너뛴다
        idx = int(stem)
        chosen = used_frames.setdefault(ep_name, [])
        if any(abs(idx - f) < args.frame_gap for f in chosen):
            rejected["frame_gap"] += 1
            continue

        tried += 1
        cam = {"fx": fit["fx"], "fy": fit["fy"], "cx": fit["cx"], "cy": fit["cy"],
               "camera_height_m": fit["camera_height_m"],
               "horizon_v": fit["episodes"][ep_name]["horizon_v"],
               "depth_k": fit["episodes"][ep_name]["depth_k"]}
        gt = build_sample(rng, ep_name, stem, cam, assets, out_dir)
        if not gt:
            rejected["no_placement"] += 1
            continue

        chosen.append(idx)
        made.append(gt)
        if not args.quiet:
            ppl = ", ".join(f"#{p['asset_id']}@{p['depth_m']:.1f}m"
                            + ("(타겟)" if p["is_target"] else "") for p in gt["people"]) or "없음"
            print(f"[{len(made)}] {gt['name']}  \"{gt['prompt']}\"  | {ppl}")

    print(f"\n{len(made)}개 생성 ({tried}개 프레임 시도, "
          f"프레임간격 탈락 {rejected['frame_gap']}, 배치실패 {rejected['no_placement']}) -> {out_dir}")

    dist = {}
    for g in made:
        dist[g["num_people"]] = dist.get(g["num_people"], 0) + 1
    print("인원 분포: " + "  ".join(
        f"{k}명 {v}개({v/len(made)*100:.0f}%)" for k, v in sorted(dist.items())) if made else "")

    # 색인 갱신. 폴더를 통째로 다시 스캔하므로 여러 번 나눠 돌려도 항상 최신 상태가 된다.
    if made:
        n, index = build_manifest(out_dir, assets)
        print(f"색인: 샘플 {n}개, 에셋 {index['num_assets_used']}/{index['num_assets_total']}종 사용 "
              f"-> manifest.jsonl / assets_used.json")


if __name__ == "__main__":
    main()
