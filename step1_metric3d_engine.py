import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from mmengine.config import Config, DictAction
except ModuleNotFoundError:
    print(
        "[CRITICAL] mmengine 패키지가 누락되었습니다. 'pip install mmengine'을 타건하십시오."
    )
    sys.exit(1)

fake_mmcv = MagicMock()
fake_mmcv_utils = MagicMock()
fake_mmcv_utils.Config = Config
fake_mmcv_utils.DictAction = DictAction

sys.modules["mmcv"] = fake_mmcv
sys.modules["mmcv.utils"] = fake_mmcv_utils

# ====================== [FrodoBots 540x360 카메라 스펙 - 최종 검증본] ======================
# IRL-UVA-24-01 공식 캘리브레이션 리포트(정제 후, 재투영 오차 0.0773px, 원본 1024x576)
# + castacks/earthrovers_ros의 front_camera.yaml로 교차검증된 값.
# (이전에 썼던 202.9/407.86 계열 수치는 해상도가 잘못 라벨링된 예비 캘리브레이션이었음 -
#  cx=241이 1024 폭의 중심(512)과 너무 멀어 실제로는 ~480x360에서 찍힌 것으로 판명됨.)
#
# 원본(1024x576) 정제 캘리브레이션:
#   fx=407.860, fy=407.866, cx=533.301, cy=278.699
#   D=[k1=-0.2172, k2=0.0537, p1=0.001853, p2=-0.002105, k3=-0.006000]  (원본 해상도 기준, 참고용)
#
# FrodoBots-2K는 이 1024x576 원본을 ffmpeg으로 540x360에 비균등(non-uniform) 리사이즈해서
# 배포함 (16:9 -> 3:2, 가로/세로 스케일이 다름):
#   scale_x = 540/1024 = 0.5273,  scale_y = 360/576 = 0.6250
#
# 주의: 왜곡계수 D는 원본(등방 픽셀) 기준이라, 이미 비균등 리사이즈된 540x360 이미지에는
# 수학적으로 올바르게 적용할 방법이 없음 (원본 1024x576 raw 프레임이 있어야 cv2.undistort
# 가능). 현재는 fx/fy/cx/cy 선형 스케일링까지만 반영, D는 참고용으로만 남겨둠.
REAL_CAM_FX = 214.97   # 407.860 * 0.5273
REAL_CAM_FY = 254.92   # 407.866 * 0.6250
REAL_CAM_CX = 281.15   # 533.301 * 0.5273
REAL_CAM_CY = 174.19   # 278.699 * 0.6250
REAL_CAM_DIST_COEFFS_ORIGINAL_RES = (-0.2172, 0.0537, 0.001853, -0.002105, -0.006000)  # 참고용, 미적용

METRIC3D_CANONICAL_FX = 1000.0
MODEL_INPUT_H, MODEL_INPUT_W = 616, 1064

# ====================== [지면 기준 스케일 보정] ======================
# 기존의 EMPIRICAL_SCALE_CORRECTION=0.1은 근거 없는 상수였음.
# do_test.py 공식 스케일링 수식을 검증한 결과 canonical_to_real_scale 계산
# 자체는 정확했고, 남은 오차는 "Metric3D가 자동차 높이(~1.6m) 기준으로
# 지면까지의 시선각을 해석하는데, 실제 로봇 카메라 높이(~0.2m)와 달라서
# 생기는 도메인 갭"으로 확인됨.
# -> 이미지 하단(지면 영역)의 실제 기하학적 거리를 핀홀 공식으로 계산해
#    raw depth와 비교, median(이론값/raw값)을 로봇/카메라별로 한 번만
#    추정해서 고정 상수로 사용한다.
CAMERA_HEIGHT_M = 0.561   # [확정값] EarthRover Zero 공식 스펙 문서의 전면 카메라 위치
                          # XYZ = (0, 184, 561) mm (원점=지면 기준) -> 지면에서 561mm.
                          # (이전 주석의 "전체 높이 380mm 기반 추정" 은 근거가 틀렸으므로 폐기.
                          #  380mm는 카메라 높이와 무관한 수치였음.)
                          # 실측 검증: episode_0763의 지면 평면을 역산하면 0.557m로 스펙과 0.7% 일치.
                          # 이 값이 확정 상수가 되면서, 지면 역산 결과가 561mm에서 벗어나는 만큼은
                          # 전부 "depth 스케일 오차"로 귀속시켜 진단할 수 있게 됨.

# 지평선(소실선) 행. 이론상으로는 cy와 같아야 하지만, 실제로는 카메라가 약 3도 아래를
# 향하고 있어서 cy(174.19)보다 위쪽에 맺힌다. 3개 에피소드에서 지면 평면을 피팅한 결과
# v0 = 159.4 / 160.6 / 165.3 로 일관되게 cy보다 작았다.
#   pitch = atan((174.19 - 160.0) / 254.92) = 3.2도 (하향)
# cy를 그대로 쓰면 z_theory가 캘리브레이션 ROI에서 10~18% 부풀려지고, 그 편향이
# ground_scale에 실려 전체 depth를 같은 비율로 왜곡시킨다.
HORIZON_V = 160.0

GROUND_ROI_V_RANGE = (0.70, 0.90)   # 이미지 세로 70~90% 구간을 "지면"으로 간주
GROUND_ROI_U_RANGE = (0.35, 0.65)   # 좌우 가장자리 왜곡을 피해 중앙부만 사용
NUM_CALIBRATION_FRAMES = 50

# 에피소드별 스케일 캐시. 전역 스칼라 하나로는 안 되는 이유:
# Metric3D는 학습 모델이라 장면 종류/조명/하늘 비율에 따라 metric 편향이 달라진다.
# 실제로 전역 스칼라를 적용한 결과물에서 지면을 역산하면 카메라 높이가
# 0.557 / 0.441 / 0.378 m 로 1.5배나 벌어졌다 (물리적으로는 전부 0.561m여야 함).
SCALE_CACHE_PATH = Path("data/processed_dataset/ground_scale_correction.json")
# ==============================================================================


class Metric3DModularEngine:

    def __init__(self, device="cuda"):
        print("==================================================================")
        print("👁️ [STEP 1 ENGINE] Sourcing Metric3D (ViT-Large) into Jetson Thor...")
        print("==================================================================")
        self.device = device
        self.model = torch.hub.load(
            "yvanyin/metric3d", "metric3d_vit_large", pretrain=True
        ).to(device).eval()
        print("➔ ✨ [안착 성공] 정석 파서가 완벽 복원되었습니다!")

    def _resize_with_padding(self, img_rgb: np.ndarray):
        h, w = img_rgb.shape[:2]
        scale = min(MODEL_INPUT_H / h, MODEL_INPUT_W / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = MODEL_INPUT_H - new_h
        pad_w = MODEL_INPUT_W - new_w
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(123, 116, 103),
        )
        return img_padded, scale, new_h, new_w

    def _infer_raw_metric_depth(self, img_rgb: np.ndarray):
        """
        Metric3D 추론 + 공식 canonical_to_real_scale까지만 적용한 depth (미터 단위이긴 하나
        아직 로봇-카메라 도메인 갭 보정 전). do_test.py와 대수적으로 동일한 공식.
        반환: (H, W) numpy 배열, 미터 단위, 지면 보정 전.
        """
        h, w, _ = img_rgb.shape
        img_padded, resize_scale, new_h, new_w = self._resize_with_padding(img_rgb)

        full_img_tensor = (
            torch.from_numpy(img_padded).permute(2, 0, 1).float().to(self.device)
        )
        img_input = full_img_tensor.unsqueeze(0)

        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(self.device)
        img_input = (img_input - mean) / std

        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({"input": img_input})
            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[:new_h, :new_w]
            depth_restored = torch.nn.functional.interpolate(
                pred_depth[None, None, :, :], size=(h, w), mode="bilinear", align_corners=True
            )

            # do_test.py 공식: ori_focal = (fx+fy)/2, canonical_to_real_scale = ori_focal/1000
            real_camera_fx_in_model = REAL_CAM_FX * resize_scale
            real_camera_fy_in_model = REAL_CAM_FY * resize_scale
            ori_focal_in_model = (real_camera_fx_in_model + real_camera_fy_in_model) / 2.0
            canonical_to_real_scale = ori_focal_in_model / METRIC3D_CANONICAL_FX

            depth_restored = depth_restored * canonical_to_real_scale
            depth_map_np = depth_restored.squeeze().cpu().numpy()

        torch.cuda.synchronize()
        return depth_map_np

    def calibrate_ground_scale(self, image_paths, camera_height_m=CAMERA_HEIGHT_M,
                                num_frames=NUM_CALIBRATION_FRAMES, verbose=True):
        """
        화면 하단 '지면'으로 추정되는 영역에서, 핀홀 카메라 기하학으로 계산한
        이론적 지면 거리 Z_theory(v) = fy * h / (v - HORIZON_V) 와 raw depth의 비율
        median(Z_theory / Z_raw)을 구해 GROUND_SCALE로 반환한다.

        반환값은 스칼라가 아니라 dict다. percentile 폭을 함께 남겨야 나중에
        "스케일이 진짜 어긋난 것"과 "ROI에 지면 아닌 물체가 섞인 것"을 구분할 수 있다.
        폭이 좁은데 median이 1에서 멀면 -> 진짜 스케일 편향.
        폭이 넓으면 -> ROI 오염이므로 스케일을 신뢰하면 안 됨.
        """
        if len(image_paths) > num_frames:
            idx = np.linspace(0, len(image_paths) - 1, num_frames).astype(int)
            sample_paths = [image_paths[i] for i in idx]
        else:
            sample_paths = image_paths

        ratios = []
        for img_path in tqdm(sample_paths, desc="Calibrating ground scale"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            if (w, h) != (540, 360):
                continue

            raw_depth = self._infer_raw_metric_depth(img_rgb)

            v0, v1 = int(h * GROUND_ROI_V_RANGE[0]), int(h * GROUND_ROI_V_RANGE[1])
            u0, u1 = int(w * GROUND_ROI_U_RANGE[0]), int(w * GROUND_ROI_U_RANGE[1])

            for v in range(v0, v1):
                if v <= HORIZON_V:
                    continue  # 원근 공식이 성립하지 않는 영역(지평선 위)은 skip
                z_theory = (REAL_CAM_FY * camera_height_m) / (v - HORIZON_V)
                if not (0.05 < z_theory < 20.0):
                    continue
                z_raw_row = raw_depth[v, u0:u1]
                valid = (z_raw_row > 0.05) & (z_raw_row < 50.0)
                if valid.sum() == 0:
                    continue
                ratios.extend((z_theory / z_raw_row[valid]).tolist())

        if len(ratios) == 0:
            raise RuntimeError(
                "지면 영역에서 유효한 raw depth 샘플을 하나도 얻지 못했습니다. "
                "GROUND_ROI_V_RANGE/U_RANGE, CAMERA_HEIGHT_M 값을 확인하세요."
            )

        ratios = np.array(ratios)
        ground_scale = float(np.median(ratios))
        p5, p95 = float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))
        spread = p95 / p5 if p5 > 0 else float("inf")

        result = {
            "ground_scale": ground_scale,
            "p5": p5,
            "p95": p95,
            "spread": spread,          # p95/p5. 2.0을 넘으면 ROI 오염 의심
            "num_samples": int(len(ratios)),
            "num_frames": len(sample_paths),
            "camera_height_m": camera_height_m,
            "horizon_v": HORIZON_V,
        }

        if verbose:
            print("\n==================== [지면 기준 스케일 캘리브레이션] ====================")
            print(f"  샘플 프레임 수        : {len(sample_paths)}")
            print(f"  유효 픽셀 샘플 수     : {len(ratios)}")
            print(f"  비율(이론/raw) median : {ground_scale:.4f}")
            print(f"  비율 5~95 percentile  : {p5:.4f} ~ {p95:.4f}  (spread {spread:.2f}x)")
            if spread > 2.0:
                print("  [WARN] spread > 2.0 -> ROI에 지면 아닌 물체가 섞였을 가능성. 스케일 신뢰도 낮음.")
            print("==========================================================================\n")

        return result

    def process_episode(self, images_dir: Path, output_depth_dir: Path, ground_scale: float):
        output_depth_dir.mkdir(parents=True, exist_ok=True)
        img_files = sorted(images_dir.glob("*.jpg"))

        if not img_files:
            return

        debug_count = 0
        for img_path in tqdm(
            img_files, desc=f"Step 1: Metric3D -> {images_dir.parent.name}"
        ):
            out_path = output_depth_dir / f"{img_path.stem}_depth.npy"
            if out_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            raw_depth_map = self._infer_raw_metric_depth(img_rgb)
            depth_map_np = np.clip(raw_depth_map * ground_scale, 0.0, 100.0).astype(np.float16)

            np.save(str(out_path), depth_map_np)

            # ====================== [DEBUG] 통계 로그 ======================
            if debug_count < 3:
                print(f"\n[DEBUG][{img_path.name}] ground_scale={ground_scale:.4f}")
                print(f"  └─ raw(보정 전) 통계   : min={raw_depth_map.min():.2f}m, mean={raw_depth_map.mean():.2f}m, max={raw_depth_map.max():.2f}m")
                print(f"  └─ 최종(보정 후) 통계  : min={depth_map_np.min():.2f}m, mean={depth_map_np.mean():.2f}m, max={depth_map_np.max():.2f}m")
            debug_count += 1
            # ================================================================

    def save_debug_visualization(self, img_path: Path, depth_map_np: np.ndarray, out_path: Path):
        """
        RGB와 컬러맵 depth를 나란히 붙여서 저장 -> 눈으로 검증할 때 사용.
        """
        img_bgr = cv2.imread(str(img_path))
        depth_clipped = np.clip(depth_map_np, 0, 10.0)  # 근거리 대비 위주로 보기 좋게 10m로 클립
        depth_norm = (depth_clipped / 10.0 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        combined = np.concatenate([img_bgr, depth_color], axis=1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), combined)


if __name__ == "__main__":
    engine = Metric3DModularEngine()
    DATASET_ROOT = Path("data/processed_dataset/ready_for_labeling")

    if not DATASET_ROOT.exists():
        print(f"[CRITICAL] 경로를 찾을 수 없습니다: {DATASET_ROOT.resolve()}")
        sys.exit(1)

    episode_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    # main.py가 넘긴 컷오프: 이름순 정렬에서 이 이름까지(포함)만 처리(비어있으면 전체)
    _ep_until = os.environ.get("LELAN_EPISODE_UNTIL", "").strip()
    if _ep_until:
        _names = [d.name for d in episode_dirs]
        if _ep_until in _names:
            episode_dirs = episode_dirs[:_names.index(_ep_until) + 1]
            print(f"[INFO] 에피소드 컷오프: {_ep_until}까지 {len(episode_dirs)}개만 처리")
        else:
            print(f"[WARN] EPISODE_UNTIL='{_ep_until}' 못 찾음 -> 전체 처리")
    all_images_dirs = []
    for ep_dir in episode_dirs:
        images_dir = ep_dir / "images"
        if not images_dir.exists():
            images_dir = ep_dir
        all_images_dirs.append(images_dir)

    # ------------------------------------------------------------------
    # 1) 지면 기준 스케일 캘리브레이션 - 에피소드별로 따로 구한다.
    #
    #    예전에는 모든 에피소드 프레임을 한 통에 섞어 스칼라 하나를 뽑았는데,
    #    Metric3D의 metric 편향이 장면마다 다르기 때문에 그 하나로는 못 맞춘다.
    #    (전역 스칼라 적용 후 지면 역산 결과: 0.557 / 0.441 / 0.378 m — 전부
    #     0.561m여야 하는데 1.5배가 벌어졌음.)
    # ------------------------------------------------------------------
    cache = {}
    if SCALE_CACHE_PATH.exists():
        cache = json.loads(SCALE_CACHE_PATH.read_text())
        # 구버전(전역 스칼라 1개) 캐시는 버리고 다시 잡는다.
        if "episodes" not in cache:
            print(f"[WARN] 구버전 전역 캐시 발견 -> 폐기하고 에피소드별로 다시 캘리브레이션합니다.")
            cache = {}
    cache.setdefault("episodes", {})
    cache["camera_height_m"] = CAMERA_HEIGHT_M
    cache["horizon_v"] = HORIZON_V

    for ep_dir, images_dir in zip(episode_dirs, all_images_dirs):
        ep_name = ep_dir.name
        if ep_name in cache["episodes"]:
            gs = cache["episodes"][ep_name]["ground_scale"]
            print(f"[INFO] [{ep_name}] 캐시된 ground_scale 사용: {gs:.4f}")
            continue

        calib_pool = sorted(images_dir.glob("*.jpg"))[::5]   # 5프레임마다 하나씩 후보로
        if not calib_pool:
            print(f"[WARN] [{ep_name}] 캘리브레이션에 쓸 이미지가 없습니다. 건너뜁니다.")
            continue

        print(f"\n[INFO] [{ep_name}] 캘리브레이션 시작 (후보 {len(calib_pool)}장)")
        cache["episodes"][ep_name] = engine.calibrate_ground_scale(calib_pool)

    if not cache["episodes"]:
        print("[CRITICAL] 캘리브레이션된 에피소드가 하나도 없습니다.")
        sys.exit(1)

    SCALE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALE_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"[INFO] 에피소드별 ground_scale {len(cache['episodes'])}개를 {SCALE_CACHE_PATH}에 저장했습니다.")
    for name, info in sorted(cache["episodes"].items()):
        print(f"    {name}: scale={info['ground_scale']:.4f}  spread={info['spread']:.2f}x")

    # ------------------------------------------------------------------
    # 2) 검증용 시각화: 캘리브레이션에 쓰인 첫 3장을 RGB+depth 컬러맵으로 저장
    #    -> 눈으로 직접 "바닥/장애물 거리가 그럴듯한지" 확인할 것.
    # ------------------------------------------------------------------
    debug_out_dir = Path("data/processed_dataset/_depth_debug_preview")
    preview_paths = sorted(all_images_dirs[0].glob("*.jpg"))[:3] if all_images_dirs else []
    preview_scale = cache["episodes"][episode_dirs[0].name]["ground_scale"] if preview_paths else 1.0
    for p in preview_paths:
        img_rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        raw_depth = engine._infer_raw_metric_depth(img_rgb)
        final_depth = np.clip(raw_depth * preview_scale, 0.0, 100.0)
        engine.save_debug_visualization(p, final_depth, debug_out_dir / f"{p.stem}_check.jpg")
    if preview_paths:
        print(f"[INFO] 검증용 미리보기 이미지 {len(preview_paths)}장을 {debug_out_dir}에 저장했습니다. "
              f"직접 열어서 depth가 그럴듯한지 확인하세요.")

    # ------------------------------------------------------------------
    # 3) 본 처리
    # ------------------------------------------------------------------
    for ep_dir, images_dir in zip(episode_dirs, all_images_dirs):
        info = cache["episodes"].get(ep_dir.name)
        if info is None:
            print(f"[WARN] [{ep_dir.name}] ground_scale이 없어 건너뜁니다.")
            continue
        engine.process_episode(images_dir, ep_dir / "metric_depth",
                               ground_scale=info["ground_scale"])
        torch.cuda.empty_cache()

    print("\n🎉 [STEP 1 종료] 밀집 깊이 행렬 연성 완료!")
