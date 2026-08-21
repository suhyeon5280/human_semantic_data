#!/usr/bin/env python3
"""합성 데이터셋 전용 step2 — NoMaD만 돌려서 `nomad_traj_norm`을 채운다.

원본 step2와 무엇이 다른가
--------------------------
원본 step2는 (1) SAM으로 객체 후보를 찾고 (2) depth로 3D 위치를 추정한 뒤
(3) NoMaD 궤적과 hindsight 매칭해서 "로봇이 실제로 다가간 객체"를 골라낸다.

합성 데이터에서는 (1)(2)가 **이미 정답으로 주어져 있다.** 우리가 사람을 직접
심었으므로 마스크·bbox·3D 위치가 GT다. 그래서 SAM을 아예 로드하지 않고
(체크포인트 2.4GB, VRAM, 추론시간 전부 절약), depth 기반 pose 추정도 하지 않는다.
남는 것은 (3) NoMaD 궤적 생성뿐이다.

  원본: SAM 마스크 -> depth로 pose 추정 -> NoMaD 매칭 -> 통과한 것만 저장
  이것: GT pose -> NoMaD 매칭 -> nomad_traj_norm 채워서 pkl 갱신

또 하나 중요한 차이는 **obs 컨텍스트**다. 원본은 에피소드를 순서대로 훑으므로
obs_queue에 직전 프레임들이 자연히 쌓인다. 합성 샘플은 서로 무관한 프레임에서
뽑혔기 때문에 큐를 그대로 두면 엉뚱한 장면이 컨텍스트로 새어든다.
반드시 샘플마다 큐를 비우고 다시 채운다 (--context 옵션 참조).

사용법 (GPU 있는 머신에서)
--------------------------
  python3 step2b_nomad_synth_engine.py \
      --synth-dir data/synth_dataset \
      --episodes-dir data/episodes \
      --nomad-yaml models/nomad.yaml \
      --nomad-ckpt models/nomad_vla_checkpoint.pth

  # 이어서 돌리기 (이미 채워진 pkl은 건너뜀)
  python3 step2b_nomad_synth_engine.py --resume

step3(Qwen)는 합성 사람에 대해서는 돌릴 필요가 없다. prompt가 이미 GT로 들어있다.
"""

import argparse
import json
import pickle
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from diffusers import DDPMScheduler
from tqdm import tqdm

base_path = Path(__file__).resolve().parent
sys.path.append(str(base_path))
sys.path.append(str(base_path / "diffusion_policy"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork                    # noqa: E402
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn    # noqa: E402
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402

# ── step2_nomad_sam_engine.py와 반드시 동일해야 하는 값들 ──
IMAGE_ASPECT_RATIO = 4 / 3
METRIC_WAYPOINT_SPACING = 0.12
DIST_THRESHOLD = 1.2
DIST_THRESHOLD_NORM = DIST_THRESHOLD / METRIC_WAYPOINT_SPACING   # = 10.0
NUM_TRAJ_SAMPLES = 30
OMNIVLA_IMG_SIZE = 224


def _center_crop_and_resize(img_rgb: np.ndarray, size: int = 96) -> np.ndarray:
    """NoMaD 학습 분포와 동일한 전처리: center_crop(4:3) -> resize(size, size).

    step2_nomad_sam_engine.py의 동명 함수와 같은 동작이어야 한다.
    """
    h, w = img_rgb.shape[:2]
    target_ratio = IMAGE_ASPECT_RATIO
    if w / h > target_ratio:                      # 너무 넓다 -> 좌우를 자른다
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        cropped = img_rgb[:, x0:x0 + new_w]
    else:                                         # 너무 높다 -> 위아래를 자른다
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        cropped = img_rgb[y0:y0 + new_h, :]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def _ego_action_to_nomad_traj_norm(best_traj: np.ndarray) -> np.ndarray:
    """(8,2) 궤적 -> OmniVLA `nomad_traj_norm` (8,4) = (x, y, cos, sin).

    step2와 동일한 변환. 진행 방향은 연속한 waypoint의 차분으로 구하고,
    마지막 스텝은 직전 방향을 그대로 쓴다.
    """
    traj = np.asarray(best_traj, dtype=np.float64).reshape(8, 2)
    deltas = np.diff(traj, axis=0, prepend=np.zeros((1, 2)))
    yaw = np.arctan2(deltas[:, 1], deltas[:, 0])
    for i in range(1, 8):                          # 정지 구간은 직전 방향 유지
        if np.hypot(*deltas[i]) < 1e-8:
            yaw[i] = yaw[i - 1]
    return np.stack([traj[:, 0], traj[:, 1], np.cos(yaw), np.sin(yaw)], axis=-1)


class NomadSynthEngine:
    """NoMaD만 담은 경량 엔진 (SAM 없음)."""

    def __init__(self, nomad_yaml, nomad_ckpt, device="cuda"):
        print("=" * 66)
        print("[STEP 2b] NoMaD only (SAM 미로드) — 합성 데이터 궤적 생성")
        print("=" * 66)
        self.device = device

        with open(nomad_yaml, "r") as f:
            self.config = yaml.safe_load(f)

        enc_size = self.config.get("encoding_size", 256)
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=enc_size,
            context_size=self.config.get("context_size", 3),
            mha_num_attention_heads=self.config.get("mha_num_attention_heads", 4),
            mha_num_attention_layers=self.config.get("mha_num_attention_layers", 4),
            mha_ff_dim_factor=self.config.get("mha_ff_dim_factor", 4),
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=enc_size,
            down_dims=self.config.get("down_dims", [64, 128, 256]),
            cond_predict_scale=self.config.get("cond_predict_scale", False),
        )
        self.nomad = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=DenseNetwork(embedding_dim=enc_size),
        ).to(device)

        state_dict = torch.load(nomad_ckpt, map_location=device)
        if not any(k.startswith(("vision_encoder.", "noise_pred_net.", "dist_pred_net."))
                   for k in state_dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        r = self.nomad.load_state_dict(state_dict, strict=False)
        print(f"---> load_state_dict missing={len(r.missing_keys)}, unexpected={len(r.unexpected_keys)}")
        if r.missing_keys or r.unexpected_keys:
            print("[WARN] missing/unexpected key가 0이 아닙니다. 구조-체크포인트 불일치 점검 필요.")
        self.nomad.eval()

        self.num_diffusion_iters = self.config.get("num_diffusion_iters", 10)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.context_size = self.nomad.vision_encoder.context_size
        self.obs_queue = deque(maxlen=self.context_size + 1)

        cfg_paths = [base_path / "data" / "data_config.yaml",
                     base_path / "vint_train" / "data" / "data_config.yaml"]
        cfg = next((p for p in cfg_paths if p.exists()), None)
        if cfg is not None:
            stats = yaml.safe_load(cfg.read_text())["action_stats"]
            self.action_min = torch.tensor(stats["min"], device=device, dtype=torch.float32)
            self.action_max = torch.tensor(stats["max"], device=device, dtype=torch.float32)
            print(f"---> action_min={self.action_min.cpu().numpy()}, action_max={self.action_max.cpu().numpy()}")
        else:
            print("[WARN] data_config.yaml 없음 -> 기본값 사용 (원본 step2와 동일)")
            self.action_min = torch.tensor([-2.5, -4.0], device=device)
            self.action_max = torch.tensor([5.0, 4.0], device=device)

        self.metric_waypoint_spacing = METRIC_WAYPOINT_SPACING
        print(f"---> context_size={self.context_size}, diffusion_iters={self.num_diffusion_iters}, "
              f"best-of-N={NUM_TRAJ_SAMPLES}")

    # ------------------------------------------------------------------ 전처리
    def _to_tensor_normalized(self, img_96: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_96.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        return (t - mean) / std

    def build_obs(self, context_frames: list[np.ndarray], current_rgb: np.ndarray) -> torch.Tensor:
        """샘플 하나의 obs 스택을 처음부터 새로 만든다.

        합성 샘플끼리는 서로 무관하므로 큐를 반드시 비우고 시작한다.
        컨텍스트가 모자라면 가장 오래된 프레임으로 앞을 채운다 (원본 step2가
        에피소드 첫 프레임에서 하는 것과 같은 처리).
        """
        self.obs_queue.clear()
        for f in context_frames:
            self.obs_queue.append(self._to_tensor_normalized(_center_crop_and_resize(f, 96)))
        cur_96 = _center_crop_and_resize(current_rgb, 96)
        self._last_obs_96 = cur_96
        self.obs_queue.append(self._to_tensor_normalized(cur_96))
        while len(self.obs_queue) < self.obs_queue.maxlen:
            self.obs_queue.appendleft(self.obs_queue[0])
        return torch.cat(list(self.obs_queue), dim=0).unsqueeze(0)

    # ------------------------------------------------------------------ 추론
    def sample_trajectory(self, obs_stacked, goal_rgb, target_xy):
        """step2._sample_trajectory와 동일. N개 시드 중 target에 가장 가까운 궤적 선택."""
        norm_goal = self._to_tensor_normalized(_center_crop_and_resize(goal_rgb, 96)).unsqueeze(0)
        mask = torch.zeros(1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            cond = self.nomad("vision_encoder", obs_img=obs_stacked,
                              goal_img=norm_goal, input_goal_mask=mask)
            cond_rep = cond.repeat_interleave(NUM_TRAJ_SAMPLES, dim=0)
            naction = torch.randn(NUM_TRAJ_SAMPLES, 8, 2, device=self.device)
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters, device=self.device)
            for t in self.noise_scheduler.timesteps:
                t_b = t.reshape(1).repeat(NUM_TRAJ_SAMPLES).to(self.device)
                pred = self.nomad("noise_pred_net", sample=naction, timestep=t_b,
                                  global_cond=cond_rep)
                naction = self.noise_scheduler.step(pred, t, naction).prev_sample

            deltas = ((naction + 1.0) / 2.0) * (self.action_max - self.action_min) + self.action_min
            waypoints = torch.cumsum(deltas, dim=1)                     # (N, 8, 2)

            target = torch.tensor([target_xy[0] / self.metric_waypoint_spacing,
                                   target_xy[1] / self.metric_waypoint_spacing],
                                  device=self.device, dtype=torch.float32)
            dists = torch.norm(waypoints.reshape(-1, 2) - target, dim=1)
            k = int(torch.argmin(dists).item())
            return waypoints[k // 8].cpu().numpy(), float(dists[k].item()), k % 8


# ---------------------------------------------------------------- 컨텍스트
def load_context(episodes_dir: Path, episode: str, frame: str, n: int,
                 mode: str, current_bgr: np.ndarray) -> list[np.ndarray]:
    """obs 컨텍스트로 쓸 직전 프레임 n장을 RGB로 반환.

    mode="real"  : 원본 에피소드의 직전 프레임들. 자아운동(ego-motion) 정보가 살아 있어
                   NoMaD가 학습 때 본 분포에 가깝다. 단 그 프레임들에는 합성 사람이
                   없으므로, 사람이 마지막 프레임에서 갑자기 나타난 것처럼 보인다.
    mode="repeat": 합성된 현재 프레임만 반복. 사람은 일관되게 보이지만 로봇이
                   정지해 있는 것처럼 보여 예측 궤적이 짧아질 수 있다.

    기본값은 real이다. NoMaD의 궤적은 자아운동에 크게 의존하는데, repeat은 그
    정보를 통째로 없애기 때문이다.
    """
    if mode == "repeat":
        return [cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)] * n

    idx = int(frame)
    frames = []
    for back in range(n, 0, -1):
        p = episodes_dir / episode / f"{idx - back:06d}.jpg"
        img = cv2.imread(str(p)) if p.exists() else None
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


# ---------------------------------------------------------------- 메인
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", default="data/synth_dataset")
    ap.add_argument("--episodes-dir", default="data/episodes")
    ap.add_argument("--nomad-yaml", default="models/nomad.yaml")
    ap.add_argument("--nomad-ckpt", default="models/nomad_vla_checkpoint.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--context", choices=["real", "repeat"], default="real")
    ap.add_argument("--dist-threshold", type=float, default=DIST_THRESHOLD_NORM,
                    help="NoMaD 궤적이 이만큼(정규화 단위) 안으로 못 들어오면 그 사람은 "
                         "obj_detect=False로 표시. 0을 주면 필터를 끄고 전부 채운다.")
    ap.add_argument("--resume", action="store_true", help="이미 채워진 pkl은 건너뛴다")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 처리 (동작 확인용)")
    args = ap.parse_args()

    synth_dir, eps_dir = Path(args.synth_dir), Path(args.episodes_dir)
    gt_files = sorted(synth_dir.glob("*_gt.json"))
    if args.limit:
        gt_files = gt_files[:args.limit]
    if not gt_files:
        sys.exit(f"[CRITICAL] {synth_dir}에 *_gt.json이 없습니다.")

    engine = NomadSynthEngine(args.nomad_yaml, args.nomad_ckpt, args.device)
    print(f"---> 컨텍스트 모드: {args.context} "
          f"({'원본 에피소드 직전 프레임' if args.context == 'real' else '현재 프레임 반복'})")

    stat = {"samples": 0, "skipped": 0, "negative": 0,
            "people": 0, "matched": 0, "too_far": 0}
    min_ds = []

    for gt_path in tqdm(gt_files, desc="Step 2b: NoMaD -> 합성 데이터"):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        pkl_path = synth_dir / f"{gt['name']}.pkl"
        if not pkl_path.exists():
            continue

        if not gt["people"]:                       # negative sample — 채울 객체가 없다
            stat["negative"] += 1
            continue

        objs = pickle.loads(pkl_path.read_bytes())
        if args.resume and objs and all(o.get("nomad_traj_norm") is not None for o in objs):
            stat["skipped"] += 1
            continue

        bgr = cv2.imread(str(synth_dir / f"{gt['name']}.jpg"))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        ctx = load_context(eps_dir, gt["episode"], gt["frame"],
                           engine.context_size, args.context, bgr)
        obs = engine.build_obs(ctx, rgb)

        by_asset = {p["asset_id"]: p for p in gt["people"]}
        for o in objs:
            person = by_asset.get(o["asset_id"])
            if person is None:
                continue
            stat["people"] += 1

            x1, y1, x2, y2 = person["bbox_xyxy"]
            goal = rgb[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
            if goal.size == 0:
                continue

            traj, min_d, matched_idx = engine.sample_trajectory(
                obs, goal, tuple(person["pose_robot_fwd_left"]))
            min_ds.append(min_d)

            reachable = args.dist_threshold <= 0 or min_d < args.dist_threshold
            o["nomad_traj_norm"] = _ego_action_to_nomad_traj_norm(traj)
            o["ego_action"] = traj
            o["matched_step_idx"] = matched_idx
            o["nomad_match_dist"] = min_d
            o["nomad_obs_img"] = engine._last_obs_96
            o["nomad_goal_img"] = _center_crop_and_resize(goal, 96)
            o["obj_detect"] = bool(reachable)
            o["needs_nomad_traj"] = False
            o["nomad_context_mode"] = args.context

            stat["matched" if reachable else "too_far"] += 1

        pkl_path.write_bytes(pickle.dumps(objs))
        stat["samples"] += 1

    print("\n" + "=" * 66)
    print(f"처리한 샘플      : {stat['samples']}개 (건너뜀 {stat['skipped']}, "
          f"negative {stat['negative']})")
    print(f"궤적 생성한 인물 : {stat['people']}명")
    print(f"  임계 통과      : {stat['matched']}명  (obj_detect=True)")
    print(f"  너무 멀어 탈락 : {stat['too_far']}명  (obj_detect=False, 데이터는 남김)")
    if min_ds:
        a = np.array(min_ds)
        print(f"  min_dist 분포  : median {np.median(a):.2f}, "
              f"p90 {np.percentile(a, 90):.2f}, max {a.max():.2f} "
              f"(임계 {args.dist_threshold:.1f})")
        if np.median(a) > args.dist_threshold:
            print("  [WARN] 중앙값이 임계를 넘었습니다. 사람 배치 위치가 로봇 진행 경로와 "
                  "많이 어긋난다는 뜻이므로, synth_compose의 U_HALF_WIDTH/D_RANGE를 "
                  "재검토하거나 --dist-threshold를 조정하세요.")
    print("=" * 66)


if __name__ == "__main__":
    main()
