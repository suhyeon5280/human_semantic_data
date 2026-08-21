import os
import pickle
import sys
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from diffusers import DDPMScheduler
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

# 1. 프로젝트 최상단 거실 경로 강제 등록
base_path = Path(__file__).resolve().parent
sys.path.append(str(base_path))

# 2. 겉껍데기 폴더를 뚫고 진짜 소스코드가 있는 곳을 다이렉트 주입
diffusion_path = base_path / "diffusion_policy"
sys.path.append(str(diffusion_path))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# ====================== [이미지 전처리 옵션] ======================
# NoMaD 학습 시 transform_images() 함수(vint_train/data/data_utils.py)에서
# IMAGE_ASPECT_RATIO = 4/3 기준으로 center crop 후 96x96 resize를 수행함.
# obs와 goal 모두 동일한 전처리를 적용해야 학습-추론 분포가 일치함.
IMAGE_ASPECT_RATIO = 4 / 3  # 가로:세로 = 4:3
# =================================================================

# ====================== [필터링 옵션] ======================
DIST_THRESHOLD = 1.2
METRIC_WAYPOINT_SPACING = 0.12
DIST_THRESHOLD_NORM = DIST_THRESHOLD / METRIC_WAYPOINT_SPACING  # = 10.0

BOTTOM_DEADZONE_RATIO = 0.80
# bbox 넓이 중 하단 deadzone(BOTTOM_DEADZONE_RATIO 아래)과 겹치는 넓이가
# 이 비율 이상이면 "주로 하단에 걸친 객체"로 보고 제외한다.
DEADZONE_OVERLAP_RATIO = 0.50
# crop 사각형(bbox) 넓이가 전체 프레임의 이 비율 이상일 때만 후보로 인정.
# (세그멘테이션 마스크 픽셀 수가 아니라, 실제로 잘리는 crop 사각형 넓이 기준)
MIN_BBOX_AREA_RATIO = 0.15
CAMERA_HEIGHT_M = 0.561
MIN_OBJECT_HEIGHT_M = 0.08
MIN_CROP_STD = 8.0
# =============================================================

# ====================== [처리 구간 옵션] ======================
# 이름순 정렬에서 이 에피소드부터(포함) 처리한다. 앞 구간은 파일 검사 없이 통째로
# 건너뛴다. 마지막으로 중단된 지점이 episode_0469라 여기서 재개한다.
# (앞 에피소드를 다시 훑으면 "객체 미채택 프레임"마다 pkl이 없어서 out_pkl.exists()
#  스킵이 안 먹히고 SAM을 다시 돌리므로 몇 주가 걸린다.)
# 전체를 처음부터 돌리려면 "" 로 비워둘 것.
EPISODE_FROM = "episode_0469"
# ==============================================================

# ====================== [프레임레이트 보정 옵션] ======================
SOURCE_FPS = 10
TARGET_FPS = 3
FRAME_STRIDE = max(1, round(SOURCE_FPS / TARGET_FPS))  # = 3
# =====================================================================

# ====================== [LeLaN best-of-N 샘플링 옵션] ======================
NUM_TRAJ_SAMPLES = 30
# =============================================================================

# ====================== [OmniVLA 출력 형식 옵션] ======================
# OmniVLA LeLaN 로더는 관측 이미지를 224 높이로 읽는다. bbox를 224 공간으로
# 환산할 때 쓰는 목표 크기(디버그/원본LeLaN 호환용, 로더는 bbox 미사용).
OMNIVLA_IMG_SIZE = 224
# =====================================================================

# ====================== [DEBUG] 점검 옵션 ======================
DEBUG_VERBOSE_FIRST_N = 5

# NoMaD에 들어간 obs/goal 이미지(96x96)를 nomad_inputs/에 PNG로도 저장할지 여부.
# 이 이미지는 pkl 안 nomad_obs_img/nomad_goal_img 필드에 이미 들어있어 완전 중복이며,
# 검수 도구 test.py의 fallback 용도일 뿐 파이프라인(step3/패키징/학습)은 안 읽는다.
# 기본 False(저장 안 함). 검수용 PNG가 필요하면 True로.
SAVE_NOMAD_DEBUG = False
# ================================================================


def _center_crop_and_resize(img_rgb: np.ndarray, size: int = 96) -> np.ndarray:
    """
    NoMaD 학습 시 transform_images()와 동일한 전처리:
      1. 원본 이미지를 4:3 비율로 center crop
      2. (size x size)로 resize
    obs/goal 모두 이 함수를 거쳐야 학습-추론 분포가 일치함.
    """
    pil_img = PILImage.fromarray(img_rgb)
    w, h = pil_img.size

    # [중요] 'w > h'가 아니라 실제 종횡비(w/h)를 4:3과 비교해야 한다.
    # NoMaD는 카메라 비율이 4:3이라 가정하고 학습/배포되므로 입력은 무조건 4:3여야 하고,
    # 절대로 zero 패딩이 끼어선 안 된다(비전 모델이 실재하지 않는 검은 경계를 특징으로 오해함).
    # 종횡비로 분기하면 항상 원본 이하 크기를 잘라내
    # 패딩이 전혀 생기지 않는다. (전체 프레임 obs는 w/h=1.5>4/3라 결과가 기존과 동일)
    if w / h > IMAGE_ASPECT_RATIO:
        # 4:3보다 가로가 넓으면: 세로는 그대로 두고 가로만 h*(4/3)으로 잘라냄
        pil_img = TF.center_crop(pil_img, (h, int(round(h * IMAGE_ASPECT_RATIO))))
    else:
        # 4:3보다 세로가 높으면: 가로는 그대로 두고 세로만 w/(4/3)으로 잘라냄
        pil_img = TF.center_crop(pil_img, (int(round(w / IMAGE_ASPECT_RATIO)), w))

    pil_img = pil_img.resize((size, size))
    return np.array(pil_img)


def _ego_action_to_nomad_traj_norm(best_traj: np.ndarray) -> np.ndarray:
    """
    step2의 궤적(best_traj)을 OmniVLA `nomad_traj_norm` (8,4) 형식으로 변환한다.

    best_traj: (8, 2) — 정규화된 누적 로컬 위치. col0=전방(x), col1=좌(+)(y).
        단위가 meters / METRIC_WAYPOINT_SPACING 라, OmniVLA nomad_traj_norm의
        col0,1과 스케일이 그대로 일치한다(실측 데이터로 검증됨).

    반환: (8, 4) float32 = (x, y, cos(yaw), sin(yaw)).
        yaw_i = atan2(Δy_i, Δx_i), 첫 스텝은 원점(0,0) 기준. OmniVLA 원본과
        소수점까지 일치하는 것을 실측으로 확인함.
    """
    xy = np.asarray(best_traj, dtype=np.float32)                    # (8, 2)
    prev = np.vstack([np.zeros((1, 2), dtype=np.float32), xy[:-1]])  # 직전 위치(첫 스텝은 원점)
    d = xy - prev                                                    # 각 구간 변위
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return np.stack([xy[:, 0], xy[:, 1], np.cos(yaw), np.sin(yaw)], axis=1).astype(np.float32)


class NomadSamModularEngine:

    def __init__(self, sam_ckpt, nomad_yaml, nomad_ckpt, device="cuda"):
        print("==================================================================")
        print("[STEP 2 ENGINE] Assembling LeLaN Official NoMaD & SAM into VRAM...")
        print("==================================================================")
        self.device = device

        sam_path = Path(sam_ckpt)
        if not sam_path.exists():
            alt_path = sam_path.parent / "sam_vit_h_4b8939.pth"
            if alt_path.exists():
                sam_path = alt_path

        print(f"---> Target SAM Checkpoint: {sam_path.resolve()}")
        self.sam = sam_model_registry["vit_h"](checkpoint=str(sam_path)).to(device).eval()

        self.mask_gen = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )

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

        dist_pred_network = DenseNetwork(embedding_dim=enc_size)

        self.nomad = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        ).to(device)

        state_dict = torch.load(nomad_ckpt, map_location=device)
        has_model_params = any(
            k.startswith(("vision_encoder.", "noise_pred_net.", "dist_pred_net."))
            for k in state_dict.keys()
        )
        if not has_model_params and "model" in state_dict:
            print("[DEBUG][WARN] 체크포인트가 {'model': ...} 형태로 감싸져 있는 것 같습니다. 'model' 키로 다시 시도합니다.")
            state_dict = state_dict["model"]

        load_result = self.nomad.load_state_dict(state_dict, strict=False)
        print(f"[DEBUG] NoMaD load_state_dict missing_keys={len(load_result.missing_keys)}, "
              f"unexpected_keys={len(load_result.unexpected_keys)}")
        if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
            print("[DEBUG][WARN] missing/unexpected key가 0이 아닙니다. 모델 구조-체크포인트 불일치 가능성 점검 필요!")

        self.nomad.eval()

        self.num_diffusion_iters = self.config.get("num_diffusion_iters", 10)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.obs_queue = deque(maxlen=self.nomad.vision_encoder.context_size + 1)
        # NoMaD에 들어가는 현재 obs(정규화 전 96x96 RGB)를 저장용으로 캐싱
        self._last_obs_96 = None

        config_paths = [
            base_path / "data" / "data_config.yaml",
            base_path / "vint_train" / "data" / "data_config.yaml"
        ]
        data_config_path = next((p for p in config_paths if p.exists()), None)

        if data_config_path is not None:
            with open(data_config_path, "r") as f:
                data_conf = yaml.safe_load(f)
            self.action_min = torch.tensor(data_conf['action_stats']['min'], device=device, dtype=torch.float32)
            self.action_max = torch.tensor(data_conf['action_stats']['max'], device=device, dtype=torch.float32)
            print(f"---> [스케일 로드 완료] 진짜 action_min: {self.action_min.cpu().numpy()}, action_max: {self.action_max.cpu().numpy()}")
        else:
            print("[경고] data_config.yaml을 찾을 수 없어 기본값으로 초기화하옵니다.")
            self.action_min = torch.tensor([-2.5, -4.0], device=device)
            self.action_max = torch.tensor([5.0, 4.0], device=device)

        self.metric_waypoint_spacing = METRIC_WAYPOINT_SPACING

        # [수정] SlamMate/vSLAM-on-FrodoBots-2K의 Robot_Zero.yaml 실제 원문
        # (Camera.type: "PinHole", 원본 1024x576, fx=202.907, fy=202.601,
        #  cx=240.540, cy=159.135)을 540x360으로 non-uniform 환산한 값.
        # 기존 206.492/263.424/270.0/180.0은 전부 검증되지 않은 잘못된 값이었고,
        # 실제로는 약 2배씩 과대평가되어 있었음 (compute_pose의 X/Y, 즉 좌우
        # 위치와 물체 높이 계산에 직접 영향을 준 값이라 중요함).
        # [최종 수정] IRL-UVA-24-01 공식 캘리브레이션(정제 후, 오차 0.0773px, 원본 1024x576)
        # + castacks/earthrovers_ros front_camera.yaml 교차검증. 540x360 non-uniform
        # 리사이즈(scale_x=0.5273, scale_y=0.6250) 반영. step1과 반드시 동일해야 함.
        self.fx = 214.97
        self.fy = 254.92
        self.cx = 281.15
        self.cy = 174.19
        print(f"---> [렌즈 체결 완료] fx:{self.fx:.3f}, fy:{self.fy:.3f}, cx:{self.cx}, cy:{self.cy}")
        # CAMERA_HEIGHT_M=0.35는 FrodoBots팀 답변(전체 높이 380mm, 카메라 상단부 장착) 기반
        # 추정치. step1의 ground-scale 캘리브레이션과 반드시 같은 값을 써야 함 (현재 둘 다 0.35).
        print(f"---> [프레임레이트 보정] 원본 {SOURCE_FPS}Hz -> 목표 {TARGET_FPS}Hz, FRAME_STRIDE={FRAME_STRIDE} "
              f"(실제 적용 Hz ≈ {SOURCE_FPS / FRAME_STRIDE:.2f}Hz)")
        print(f"---> [Diffusion 스텝] num_diffusion_iters={self.num_diffusion_iters} (학습/추론 동일하게 사용)")
        print(f"---> [LeLaN best-of-N] NUM_TRAJ_SAMPLES={NUM_TRAJ_SAMPLES} (N개 노이즈 시드 중 target에 가장 가까운 1개 선택)")
        print(f"---> [이미지 전처리] center_crop(4:3) -> resize(96x96) [NoMaD 학습 분포 일치]")
        print("---> [안착 성공] NoMaD 백본 결합 완료!")

    def _to_tensor_normalized(self, img_96: np.ndarray) -> torch.Tensor:
        """96x96 numpy array -> 정규화된 torch tensor (3, 96, 96)"""
        tensor = torch.from_numpy(img_96.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        return (tensor - mean) / std

    def _push_obs_to_queue(self, img_rgb: np.ndarray) -> torch.Tensor:
        """
        obs 전처리: NoMaD 학습과 동일하게 center_crop(4:3) -> resize(96x96) -> 정규화
        기존: cv2.resize(img_rgb, (96, 96))로 단순 축소 → 학습 분포와 불일치
        변경: _center_crop_and_resize() 적용 → 학습 분포 일치
        """
        img_96 = _center_crop_and_resize(img_rgb, size=96)
        # NoMaD에 실제로 들어가는 현재 obs(정규화 전 96x96 RGB)를 저장용으로 보관
        self._last_obs_96 = img_96
        norm_obs = self._to_tensor_normalized(img_96)

        self.obs_queue.append(norm_obs)
        while len(self.obs_queue) < self.obs_queue.maxlen:
            self.obs_queue.append(norm_obs)

        return torch.cat(list(self.obs_queue), dim=0).unsqueeze(0)  # (1, context+1, 3, 96, 96)

    def _is_low_information_crop(self, crop_img: np.ndarray) -> bool:
        if crop_img is None or crop_img.size == 0:
            return True
        std_val = float(np.std(crop_img.astype(np.float32)))
        return std_val < MIN_CROP_STD

    def _sample_trajectory(self, obs_stacked: torch.Tensor, goal_rgb: np.ndarray, target_xy: tuple):
        """
        [LeLaN 공식 방식 반영] 같은 goal 조건에 대해 노이즈 시드를 N개(NUM_TRAJ_SAMPLES) 다르게 뽑아
        N개의 서로 다른 8-step trajectory를 배치로 병렬 생성하고, N×8개의 모든 waypoint 중
        target_xy에 가장 가까운 waypoint를 포함하는 trajectory 1개를 선택해서 반환한다.

        goal 전처리: NoMaD 학습과 동일하게 center_crop(4:3) -> resize(96x96) -> 정규화
        변경: _center_crop_and_resize() 적용 → 학습 분포 일치

        Returns:
            best_traj: (8, 2) np.ndarray
            min_dist: float, 정규화 단위
            matched_step_idx: int, 0~7
        """
        # goal 전처리: center_crop(4:3) -> 96x96 resize -> 정규화
        goal_96 = _center_crop_and_resize(goal_rgb, size=96)
        norm_goal = self._to_tensor_normalized(goal_96).unsqueeze(0)  # (1, 3, 96, 96)

        mask = torch.zeros(1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            cond = self.nomad(
                "vision_encoder",
                obs_img=obs_stacked,
                goal_img=norm_goal,
                input_goal_mask=mask,
            )  # (1, enc_size)

            cond_rep = cond.repeat_interleave(NUM_TRAJ_SAMPLES, dim=0)  # (N, enc_size)
            naction = torch.randn(NUM_TRAJ_SAMPLES, 8, 2, device=self.device)
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters, device=self.device)
            for t in self.noise_scheduler.timesteps:
                t_batched = t.reshape(1).repeat(NUM_TRAJ_SAMPLES).to(self.device)
                pred = self.nomad(
                    "noise_pred_net",
                    sample=naction,
                    timestep=t_batched,
                    global_cond=cond_rep,
                )
                naction = self.noise_scheduler.step(pred, t, naction).prev_sample

            deltas_norm = naction  # (N, 8, 2)
            deltas_real = ((deltas_norm + 1.0) / 2.0) * (self.action_max - self.action_min) + self.action_min
            waypoints = torch.cumsum(deltas_real, dim=1)  # (N, 8, 2)

            target_norm = torch.tensor([
                target_xy[0] / self.metric_waypoint_spacing,
                target_xy[1] / self.metric_waypoint_spacing,
            ], device=self.device, dtype=torch.float32)
            flat_wp = waypoints.reshape(-1, 2)
            dists = torch.norm(flat_wp - target_norm, dim=1)
            min_idx = torch.argmin(dists)
            min_dist = float(dists[min_idx].item())

            flat_idx = int(min_idx.item())
            traj_id = flat_idx // 8
            matched_step_idx = flat_idx % 8

            best_traj = waypoints[traj_id].cpu().numpy()  # (8, 2)

            return best_traj, min_dist, matched_step_idx

    def compute_pose(self, seg_mask: np.ndarray, depth_map: np.ndarray) -> tuple:
        v, u = np.where(seg_mask)
        if len(v) < 20:
            return None, None
        Z = depth_map[v, u]
        valid = Z > 0.1
        if not np.any(valid):
            return None, None
        u_v, v_v, Z_v = u[valid], v[valid], Z[valid]
        X = (u_v - self.cx) * Z_v / self.fx
        Y = (v_v - self.cy) * Z_v / self.fy
        pts = np.stack([X, Y, Z_v], axis=-1)
        return np.mean(pts, axis=0), np.median(pts, axis=0)

    def _passes_candidate_filters(self, bbox, bbox_area, img_h, img_w, p_mean):
        x1, y1, x2, y2 = bbox

        # deadzone: 화면 하단(BOTTOM_DEADZONE_RATIO 아래) 영역과 bbox가 겹치는 넓이가
        # bbox 넓이의 DEADZONE_OVERLAP_RATIO 이상이면(=주로 하단에 걸쳐 있으면) 제외.
        # 기존엔 중심(center_y)만 봐서, 키 큰 객체가 바닥에 붙어 있어도 중심이 경계
        # 위라 안 걸리는 문제가 있었음(높이>=144px면 위치 무관하게 통과).
        deadzone_top = BOTTOM_DEADZONE_RATIO * img_h
        overlap_h = max(0.0, y2 - max(y1, deadzone_top))
        overlap_area = overlap_h * (x2 - x1)
        if bbox_area > 0 and overlap_area > DEADZONE_OVERLAP_RATIO * bbox_area:
            return False, "deadzone"

        # crop되는 사각형(bbox) 넓이가 프레임의 MIN_BBOX_AREA_RATIO 미만이면 탈락
        frame_area = img_h * img_w
        if bbox_area < MIN_BBOX_AREA_RATIO * frame_area:
            return False, "too_small"

        if p_mean is not None:
            object_height_above_floor = CAMERA_HEIGHT_M - p_mean[1]
            if object_height_above_floor < MIN_OBJECT_HEIGHT_M:
                return False, "floor_level"

        return True, None

    def process_episode(self, ep_dir: Path):
        images_dir = ep_dir / "images"
        if not images_dir.exists():
            images_dir = ep_dir

        depths_dir = ep_dir / "metric_depth"
        output_dir = ep_dir / "stage2_singular_dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not images_dir.exists() or not depths_dir.exists():
            return

        self.obs_queue.clear()
        all_img_files = sorted(images_dir.glob("*.jpg"))
        img_files = all_img_files[::FRAME_STRIDE]

        valid_cnt = 0

        debug_count = 0
        skipped_no_depth = 0
        skipped_existing_pkl = 0
        min_d_list = []
        zero_mask_count = 0
        filtered_deadzone = 0
        filtered_too_small = 0
        filtered_floor_level = 0
        filtered_low_info = 0
        no_candidate_survived = 0

        for img_path in tqdm(
            img_files,
            desc=f"Step 2: NoMaD+SAM -> {ep_dir.name} (stride={FRAME_STRIDE}, ~{SOURCE_FPS/FRAME_STRIDE:.1f}Hz)",
        ):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = bgr.shape

            if w != 540 or h != 360:
                continue

            obs_stacked = self._push_obs_to_queue(rgb)

            out_pkl = output_dir / f"{img_path.stem}.pkl"
            depth_path = depths_dir / f"{img_path.stem}_depth.npy"

            if out_pkl.exists():
                skipped_existing_pkl += 1
                continue
            if not depth_path.exists():
                skipped_no_depth += 1
                continue

            # step1이 float16으로 저장한 depth를 명시적으로 float32로 올려서 사용
            # (연산 정밀도/범위 안정성 확보; np.load 자체는 저장 dtype으로 읽힘)
            depth_map = np.load(str(depth_path)).astype(np.float32)
            masks = self.mask_gen.generate(rgb)

            if len(masks) == 0:
                zero_mask_count += 1
                continue

            # 이 프레임에서 채택된 OmniVLA 객체 dict들의 리스트 (pkl은 이 리스트를 저장)
            frame_objects = []
            min_d = float("inf")           # 디버그 출력용 best 추적
            best_mean, best_traj = None, None
            survived_any = False

            for m in masks:
                p_mean, p_med = self.compute_pose(m["segmentation"], depth_map)
                if p_mean is None:
                    continue

                x, y, wb, hb = m["bbox"]
                bbox = [int(x), int(y), int(x + wb), int(y + hb)]
                # 실제로 잘리는 crop 사각형(bbox) 넓이 기준으로 크기 필터링
                bbox_area = float(wb * hb)

                ok, reason = self._passes_candidate_filters(bbox, bbox_area, h, w, p_mean)
                if not ok:
                    if reason == "deadzone":
                        filtered_deadzone += 1
                    elif reason == "too_small":
                        filtered_too_small += 1
                    elif reason == "floor_level":
                        filtered_floor_level += 1
                    continue

                crop_candidate = rgb[max(0, int(y)) : min(h, int(y + hb)), max(0, int(x)) : min(w, int(x + wb))]
                if crop_candidate.size == 0:
                    continue

                if self._is_low_information_crop(crop_candidate):
                    filtered_low_info += 1
                    continue

                survived_any = True

                obj_forward = p_mean[2]
                obj_lateral_ros = -p_mean[0]

                traj_candidate, dist, matched_step_idx = self._sample_trajectory(
                    obs_stacked, crop_candidate, (obj_forward, obj_lateral_ros)
                )

                # 디버그용 best 추적
                if dist < min_d:
                    min_d = dist
                    best_mean = p_mean
                    best_traj = traj_candidate

                # NoMaD 궤적이 이 객체에 충분히 닿는 경우만 OmniVLA 객체로 채택
                if dist >= DIST_THRESHOLD_NORM:
                    continue

                # ── OmniVLA 스키마로 객체 dict 구성 ──
                nomad_traj_norm = _ego_action_to_nomad_traj_norm(traj_candidate)      # (8,4) (x,y,cos,sin)
                # pose_median: [[전방, 좌(+)]] 로봇좌표(m). compute_pose는 카메라좌표(X우,Y하,Z전)라 변환.
                pose_med_robot = np.array([[p_med[2], -p_med[0]]], dtype=np.float64)  # (1,2)
                # bbox → 224 공간 [top,bottom,left,right] (540x360 -> 224x224 리사이즈 가정, 디버그/원본LeLaN 호환용)
                sx, sy = OMNIVLA_IMG_SIZE / w, OMNIVLA_IMG_SIZE / h
                x1, y1, x2, y2 = bbox
                bbox_224 = np.array([[int(y1 * sy), int(y2 * sy), int(x1 * sx), int(x2 * sx)]], dtype=np.int64)
                nomad_goal_96 = _center_crop_and_resize(crop_candidate, size=96)

                frame_objects.append({
                    # ── OmniVLA 로더가 실제로 읽는 필드 ──
                    "pose_median": pose_med_robot,                                      # (1,2) m [[전방,좌]]
                    "pose_median_norm": pose_med_robot / self.metric_waypoint_spacing,  # (1,2)
                    "nomad_traj_norm": nomad_traj_norm,                                 # (8,4) (x,y,cos,sin)
                    "prompt": None,                                                     # step3에서 [(label,),...]로 채움
                    "bbox": bbox_224,                                                   # (1,4) [top,bottom,left,right] @224
                    # ── 디버그/데이터 검토용 (OmniVLA 로더는 무시) ──
                    "obj_detect": True,
                    "obj_inst": crop_candidate,
                    "nomad_obs_img": self._last_obs_96,     # 정규화 전 96x96 RGB
                    "nomad_goal_img": nomad_goal_96,        # 정규화 전 96x96 RGB
                    "pose_mean": p_mean,                    # 카메라좌표 원본 (X,Y,Z)
                    "bbox_orig_540x360": [x1, y1, x2, y2],  # 원본 해상도 bbox [x1,y1,x2,y2]
                    "ego_action": traj_candidate,           # (8,2) 원본 궤적
                    "matched_step_idx": matched_step_idx,
                })

            if not survived_any:
                no_candidate_survived += 1
            if min_d != float("inf"):
                min_d_list.append(min_d)

            if debug_count < DEBUG_VERBOSE_FIRST_N:
                print(f"\n[DEBUG][{img_path.name}]")
                print(f"  SAM 마스크 개수: {len(masks)}, 필터 통과: {survived_any}, 채택 객체 수: {len(frame_objects)}")
                if best_mean is not None:
                    obj_height = CAMERA_HEIGHT_M - best_mean[1]
                    mws = self.metric_waypoint_spacing
                    print(f"  최적 사물 ROS 좌표(전진, 좌우) = {best_mean[2]:.2f}m, {-best_mean[0]:.2f}m (높이 {obj_height:.2f}m)")
                    print(f"  trajectory 끝점(8번째 스텝) = {best_traj[-1][0]*mws:.2f}m, {best_traj[-1][1]*mws:.2f}m")
                    print(f"  최종 정합 오차 min_d = {min_d:.4f} (정규화) = {min_d*mws:.3f}m "
                          f"(임계값 {DIST_THRESHOLD}m = {DIST_THRESHOLD_NORM:.1f})")
                else:
                    print(f"  best_mean=None (모든 후보 탈락 혹은 기하 연산 실패)")
            debug_count += 1

            # 채택된 객체가 하나라도 있으면 "객체 리스트"로 저장 (OmniVLA 형식)
            if frame_objects:
                # 검수용 obs/goal PNG는 SAVE_NOMAD_DEBUG일 때만 (pkl 필드와 중복)
                if SAVE_NOMAD_DEBUG:
                    nomad_input_dir = output_dir / "nomad_inputs"
                    nomad_input_dir.mkdir(parents=True, exist_ok=True)
                    if self._last_obs_96 is not None:
                        PILImage.fromarray(self._last_obs_96).save(
                            nomad_input_dir / f"{img_path.stem}_obs.png"
                        )
                    for k, obj in enumerate(frame_objects):
                        PILImage.fromarray(obj["nomad_goal_img"]).save(
                            nomad_input_dir / f"{img_path.stem}_obj{k}_goal.png"
                        )

                with open(out_pkl, "wb") as f:
                    pickle.dump(frame_objects, f)   # ← 단일 dict가 아니라 "객체 dict들의 리스트"

                valid_cnt += 1

        print(f"\n[DEBUG][{ep_dir.name}] ===== 에피소드 통계 =====")
        print(f"  원본 프레임 수: {len(all_img_files)} -> stride={FRAME_STRIDE} 적용 후 처리 대상: {len(img_files)}")
        print(f"  스킵(이미 pkl 존재, obs_queue는 그래도 갱신됨): {skipped_existing_pkl}")
        print(f"  필터 탈락 누적 - 데드존: {filtered_deadzone}, 너무작음: {filtered_too_small}, "
              f"바닥높이: {filtered_floor_level}, 저정보crop: {filtered_low_info}")
        print(f"  모든 후보 탈락 프레임: {no_candidate_survived}")
        if len(min_d_list) > 0:
            arr = np.array(min_d_list)
            print(f"  min_d 통계: min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")
        print(f"   └── 정산 완료: {valid_cnt}/{len(img_files)}개 피클 매칭 성공")
        print("=" * 60)


if __name__ == "__main__":
    SAM_CKPT = "models/sam_vit_h.pth"
    NOMAD_YAML = "models/nomad.yaml"
    NOMAD_CKPT = "models/nomad_vla_checkpoint.pth"

    engine = NomadSamModularEngine(
        sam_ckpt=SAM_CKPT, nomad_yaml=NOMAD_YAML, nomad_ckpt=NOMAD_CKPT
    )
    DATASET_ROOT = Path("data/processed_dataset/ready_for_labeling")
    episode_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    _names = [d.name for d in episode_dirs]

    # 이름순 정렬에서 [FROM, UNTIL] 구간(양끝 포함)만 처리. 각각 비어있으면 처음/끝까지.
    #   FROM  : 파일 상단 EPISODE_FROM 상수 (재개 지점)
    #   UNTIL : main.py가 넘기는 LELAN_EPISODE_UNTIL 환경변수
    _ep_from = EPISODE_FROM.strip()
    _ep_until = os.environ.get("LELAN_EPISODE_UNTIL", "").strip()
    _start, _end = 0, len(episode_dirs)

    if _ep_from:
        if _ep_from in _names:
            _start = _names.index(_ep_from)
            print(f"[INFO] 에피소드 시작점: {_ep_from} (앞의 {_start}개 건너뜀)")
        else:
            print(f"[WARN] EPISODE_FROM='{_ep_from}' 못 찾음 -> 처음부터 처리")
    else:
        print("[INFO] EPISODE_FROM 비어있음 -> episode 처음부터 전체 처리")

    if _ep_until:
        if _ep_until in _names:
            _end = _names.index(_ep_until) + 1
            print(f"[INFO] 에피소드 끝점: {_ep_until}까지")
        else:
            print(f"[WARN] EPISODE_UNTIL='{_ep_until}' 못 찾음 -> 끝까지 처리")

    if _start >= _end:
        print(f"[WARN] 시작점이 끝점보다 뒤입니다(start={_start}, end={_end}) -> 처리할 에피소드 없음")
        episode_dirs = []
    else:
        episode_dirs = episode_dirs[_start:_end]
        print(f"[INFO] 처리 대상 에피소드 {len(episode_dirs)}개 "
              f"({episode_dirs[0].name} ~ {episode_dirs[-1].name}) / 전체 {len(_names)}개")

    for ep in episode_dirs:
        engine.process_episode(ep)
