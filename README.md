# LeLaN_Data_plus — 합성 보행자 주행 데이터셋 생성

주행 영상 프레임에 누끼 딴 사람 PNG를 **기하학적으로 올바르게** 합성해서,
언어 조건부 내비게이션(LeLaN / OmniVLA) 학습용 데이터셋을 만든다.

핵심은 "사람을 붙여넣는 것"이 아니라 **GT를 정확히 아는 상태로 붙여넣는 것**이다.
우리가 심었으므로 마스크·bbox·3D 위치·언어 프롬프트가 모두 정답이다.
그래서 SAM(객체 탐지)과 Qwen(라벨링)을 돌릴 필요가 없다.

---

## 1. 빠른 시작

> **다른 머신에서 학습시킬 계획이면 2절부터 읽으면 된다.** 무엇을 옮기고
> 어디서 무엇을 돌리는지 순서대로 정리해 뒀다.

```bash
# 전체 파이프라인 (GPU 없으면 --skip-gpu)
python3 synth_pipeline.py --all --num 600

# 어떤 단계가 돌지 먼저 확인
python3 synth_pipeline.py --all --dry-run

# 일부 단계만
python3 synth_pipeline.py --stage compose,package
```

최종 산출물:

| 경로 | 내용 |
|---|---|
| `data/synth_dataset/` | 합성 이미지 + GT + 색인 (사람이 검수하는 형태) |
| `data/omnivla_dataset/` | OmniVLA 학습 폴더 구조 (`main.py` 출력과 동일) |

---

## 2. 다른 머신에서 학습하기 — 전체 순서

노트북에서 합성까지 하고, GPU 머신에서 NoMaD와 학습을 돌리는 것을 전제로 한다.

| 머신 | 하는 일 | 필요한 것 |
|---|---|---|
| 노트북 (CPU) | 에셋 준비 → 캘리브레이션 → **합성** | 주행 프레임 + depth + 사람 PNG |
| GPU 머신 | **NoMaD 궤적** → 패키징 → 학습 | 합성 결과 + 주행 프레임(jpg만) |

핵심: **depth는 GPU 머신에 옮길 필요가 없다.** 합성이 끝나면 depth를 쓰는 곳이
없다(5-1 참조). 그래서 635MB를 통째로 뺄 수 있다.

### 2-1. 노트북: 합성까지

```bash
python3 synth_pipeline.py --all --skip-gpu --num 600
```

끝나면 이렇게 나온다:

```
data/synth_dataset/     466개 샘플, 샘플당 파일 6개
  ├── manifest.jsonl        샘플 -> 인물 색인
  └── assets_used.json      인물 -> 샘플 역색인
```

확인:

```bash
python3 -c "import json;print(sum(1 for _ in open('data/synth_dataset/manifest.jsonl')))"
# 466
```

### 2-2. GPU 머신으로 옮길 것

**약 107MB.** 아래 4개만 옮기면 된다.

| 옮길 것 | 용량 | 왜 필요한가 |
|---|---|---|
| 코드 저장소 | 176 KB | `git clone` |
| `data/synth_dataset/*.jpg` (`_compare.jpg` 제외) | 28 MB | NoMaD의 obs·goal 입력 |
| `data/synth_dataset/*_gt.json`, `*.pkl`, `manifest.jsonl` | 4 MB | GT pose·bbox, 갱신 대상 pkl |
| `data/episodes/*/*.jpg` | 75 MB | `--context real`의 직전 프레임 |
| `data/human_assets.json` | 84 KB | 에셋 ID ↔ H_real 대조 |

**옮기지 않아도 되는 것:**

| 제외 | 용량 | 이유 |
|---|---|---|
| `data/episodes/*/metric_depth/` | 635 MB | 합성이 끝나면 depth를 쓰지 않는다 |
| `data/synth_dataset/*_depth_synth.npy` | 173 MB | 학습에 쓰이지 않는 부산물 |
| `data/synth_dataset/*_compare.jpg` | 57 MB | 사람 눈 검수용 |
| `data/synth_dataset/*_mask.png` | 1.9 MB | GT bbox로 충분 |
| `data/human_cropped/` | 21 MB | 합성이 이미 끝났다 |

rsync 예시 (`--dry-run`으로 규칙을 검증했다):

```bash
# 코드
git clone https://github.com/suhyeon5280/human_semantic_data.git
cd human_semantic_data

# 합성 결과 (검수용·부산물 제외)
# 주의: rsync는 규칙을 "먼저 맞는 것"으로 적용한다. --exclude='*_compare.jpg'가
# --include='*.jpg'보다 앞에 와야 한다. 순서를 바꾸면 검수용 57MB가 그대로 따라온다.
rsync -av --progress \
  --exclude='*_compare.jpg' \
  --include='*.jpg' --include='*_gt.json' --include='*.pkl' \
  --include='manifest.jsonl' --include='assets_used.json' \
  --exclude='*' \
  <노트북>:~/LeLaN_Data_plus/data/synth_dataset/ data/synth_dataset/

# 주행 프레임 jpg만 (metric_depth 635MB 제외)
rsync -av --progress \
  --exclude='metric_depth/' --include='*/' --include='*.jpg' --exclude='*' \
  <노트북>:~/LeLaN_Data_plus/data/episodes/ data/episodes/

# 에셋 메타
rsync -av <노트북>:~/LeLaN_Data_plus/data/human_assets.json data/
```

전송 후 확인 — 이 숫자가 나와야 한다:

```bash
ls data/synth_dataset/*.jpg | grep -vc _compare   # 466
ls data/synth_dataset/*.pkl | wc -l               # 466
ls data/episodes/*/*.jpg | wc -l                  # 1709
find data/episodes -name '*.npy' | wc -l          # 0   (안 옮겨도 된다)
```

> `--context repeat`으로 돌릴 거면 `data/episodes/`도 필요 없다. 다만 권장값은
> `real`이다 (8절 참조 — 정지로 오인되면 궤적이 전부 탈락한다).

### 2-3. GPU 머신: NoMaD 궤적 채우기

> **이 저장소만으로는 실행되지 않는다.** `step2b`는 NoMaD 본체를 import하는데,
> 그 코드와 체크포인트는 원본 `LeLaN_Data` 저장소에 있다. 필요한 것:
>
> | 필요 | 위치 |
> |---|---|
> | `vint_train/` 패키지 | `LeLaN_Data/vint_train/` |
> | `models/nomad.yaml` | `LeLaN_Data/models/` |
> | `models/nomad_vla_checkpoint.pth` (73MB) | `LeLaN_Data/models/` |
> | `vint_train/data/data_config.yaml` (action_stats) | `LeLaN_Data/vint_train/data/` |
> | `diffusers`, `diffusion_policy` | pip 설치 |
>
> SAM 체크포인트(273MB)는 **필요 없다** — `step2b`는 SAM을 로드하지 않는다.

`step2b`는 자기 파일이 있는 디렉토리를 `sys.path`에 넣는다
(`base_path = Path(__file__).resolve().parent`). 그래서 **스크립트를 `LeLaN_Data`
안에 두고 거기서 실행하면** `vint_train`, `models/`, `data_config.yaml`을 알아서 찾는다.

아래 명령은 **전부 GPU 머신에서** 실행한다 (노트북에서는 돌지 않는다).

```bash
cd ~/LeLaN_Data
cp ~/human_semantic_data/step2b_nomad_synth_engine.py .     # 또는 심볼릭 링크

# 의존성 확인 (GPU 머신)
python3 -c "import torch, diffusers; from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D; print('OK', torch.cuda.is_available())"
```

`OK True`가 나와야 한다. `False`면 그 머신이 CUDA를 못 잡은 것이다.

```bash
# 1) 스모크 테스트 — 모델이 뜨고 궤적이 나오는지만 본다
python3 step2b_nomad_synth_engine.py \
  --synth-dir ~/LeLaN_Data_plus/data/synth_dataset \
  --episodes-dir ~/LeLaN_Data_plus/data/episodes \
  --limit 5
```

`load_state_dict missing=0, unexpected=0`이 아니면 모델 구조와 체크포인트가
어긋난 것이니 여기서 멈춰야 한다.

```bash
# 2) 컨텍스트 방식 비교 — min_dist 중앙값이 낮은 쪽을 택한다
python3 step2b_nomad_synth_engine.py --synth-dir ... --limit 50 --context real
python3 step2b_nomad_synth_engine.py --synth-dir ... --limit 50 --context repeat
```

마지막 요약에서 이걸 본다:

```
min_dist 분포 : median 6.2, p90 11.4, max 18.0  (임계 10.0)
```

- 중앙값이 10보다 **한참 아래** → 그대로 전체 진행
- 중앙값이 10 **근처거나 위** → 다른 `--context`로 재시도.
  양쪽 다 높으면 배치 문제다 (`U_HALF_WIDTH`를 좁혀 노트북에서 재합성)

```bash
# 3) 전체 (466샘플 / 객체 732개)
python3 step2b_nomad_synth_engine.py \
  --synth-dir ~/LeLaN_Data_plus/data/synth_dataset \
  --episodes-dir ~/LeLaN_Data_plus/data/episodes \
  --context real
```

`--synth-dir`의 `*.pkl` 466개가 제자리에서 갱신되고 `needs_nomad_traj`가 `False`가 된다.
중단됐으면 `--resume`으로 이어서 돌린다.

### 2-4. 패키징 — GPU 불필요, 노트북에서 해도 된다

`synth_package.py`는 PIL과 tqdm만 쓴다. torch도, GPU도 필요 없다.
그래서 **GPU 머신에서 갱신된 `*.pkl` 466개(1.9MB)만 회수해서 노트북에서 돌리는 것이
가장 가볍다.**

```bash
# GPU 머신 -> 노트북: pkl만 (1.9MB)
rsync -av <GPU머신>:~/LeLaN_Data_plus/data/synth_dataset/*.pkl data/synth_dataset/

# 노트북에서
python3 synth_package.py --require-traj
```

`--require-traj`는 궤적이 채워진 객체만 남긴다. 결과:

```
data/omnivla_dataset/<episode>/image/00000000.jpg        224x224
data/omnivla_dataset/<episode>/pickle_nomad/00000000.pkl
```

**여기부터 학습에 필요한 건 이 폴더 13MB뿐이다.**

확인:

```bash
python3 -c "
import pickle, glob
fs = glob.glob('data/omnivla_dataset/*/pickle_nomad/*.pkl')
objs = [o for f in fs for o in pickle.load(open(f,'rb'))]
print('프레임', len(fs), '객체', len(objs))
print('궤적 채워진 객체', sum(1 for o in objs if o['nomad_traj_norm'] is not None))
print('키', sorted(objs[0]))
"
```

`궤적 채워진 객체`가 0이면 2-3을 건너뛴 것이다.

### 2-5. 학습 등록

OmniVLA / LeLaN 학습 설정의 `data_config.yaml`에 시퀀스 폴더로 등록한다.

```yaml
metric_waypoint_spacing: 0.12
data_image_folder:  /절대경로/data/omnivla_dataset/
data_pickle_folder: /절대경로/data/omnivla_dataset/
```

각 `<episode>` 폴더가 `go_stanford2` 스타일 시퀀스 역할을 한다.
로더는 폴더 안 파일을 0부터 연속 인덱스로 읽고, 현재 인덱스 `iv`에 대해
`image_path[iv-1]`, `[iv-2]`를 시간적 컨텍스트로 쓴다 — **9절 1번 한계를 반드시
확인하고 시작할 것.**

### 2-6. 전부 GPU 머신에서 하고 싶다면

노트북 결과를 옮기지 않고 처음부터 다시 돌릴 수도 있다. 이때는 원본 데이터가 필요하다.

```
data/episodes/…                  주행 프레임 + metric_depth (없으면 step1이 만든다)
data/human_cropped/…             사람 PNG 194개
data/human_annotations.jsonl     캡션
data/human_assets.json           있으면 그대로, 없으면 자동 생성
```

```bash
python3 synth_pipeline.py --all --num 600 --seed 2026
python3 synth_package.py --require-traj
```

**`--seed`를 노트북과 같게 주면 동일한 데이터셋이 나온다.**
단 `data/human_assets.json`이 있어야 한다 — 없으면 `H_real`이 새로 뽑혀서
사람 크기가 달라진다 (11절 참조).

---

## 3. 입력 준비

```
data/
├── episodes/
│   ├── episode_0761/
│   │   ├── 000000.jpg  ...          # 540x360 주행 프레임
│   │   └── metric_depth/
│   │       └── 000000_depth.npy     # step1(Metric3D) 산출물, float16
│   └── ...
└── human_cropped/
    └── human_1.png  ...             # 알파 채널 있는 사람 PNG (누끼 + 타이트 크롭)
```

> **저장소에는 코드만 있다.** `data/` 아래는 전부 별도 관리 대상이다 —
> 주행 프레임(`episodes/`), 사람 PNG 에셋(`human_cropped/`), 에셋 캡션
> (`human_annotations.jsonl`)을 먼저 배치해야 파이프라인이 돌아간다.
> `human_assets.json`과 `episode_ground_fit.json`은 파이프라인이 직접 만든다.

사람 PNG를 처음부터 만드는 경우:

```bash
python3 rename_human_images.py            # 다운로드한 파일명 -> human_1, human_2, ...
python3 crop_human_images.py              # 투명 여백 제거, 알파 유지
```

캡션은 `data/human_annotations.jsonl`에 에셋당 4개씩 들어 있다
(짧은 참조 표현 → 색+의복 → 소지품/행동 → 전체 묘사). 관리 도구:

```bash
python3 human_annotations.py stats
python3 human_annotations.py show 3
python3 human_annotations.py grep "leather"
python3 human_annotations.py csv review.csv     # 엑셀로 검수
```

---

## 4. 파이프라인 단계

| # | 단계 | 스크립트 | GPU | 비고 |
|---|---|---|---|---|
| 1 | depth | `step1_metric3d_engine.py` | ● | **조건부** — metric_depth 없는 에피소드만 |
| 2 | calibrate | `synth_calibrate.py` | | 에피소드별 지면 평면 피팅 |
| 3 | assets | `synth_assets.py` | | 에셋별 실제 키 `H_real` (1회만) |
| 4 | compose | `synth_compose.py` | | 합성 + GT + pkl 뼈대 |
| 5 | nomad | `step2b_nomad_synth_engine.py` | ● | `nomad_traj_norm` 채우기 |
| 6 | package | `synth_package.py` | | OmniVLA 폴더 구조 |

`synth_manifest.py`는 4단계 끝에 자동 실행되어 색인 2종을 갱신한다.

### Metric3D를 다시 돌려야 하나? — 아니오

depth를 쓰는 곳은 **합성 시 가림(occlusion) 판정 한 군데뿐**이고,
거기 필요한 건 *원본 프레임*의 depth다. 이미 있다.

- 사람 배치: depth를 안 쓴다 (아래 5-1 참조)
- 3D 위치 GT: 우리가 심었으므로 정확히 안다
- step2b, 패키징: depth를 안 쓴다

**합성된 이미지에 Metric3D를 다시 돌리면 해롭다.** 붙여넣은 사람의 거리를
추정하려 들 텐데 우리는 그 값을 이미 정확히 알고 있다. 추정치로 덮어쓰면 GT가 오염된다.

새 에피소드를 추가하면 `synth_pipeline.py`가 자동으로 감지해서 그때만 실행한다.

### Qwen(step3)은? — 필요 없다

step3는 SAM이 찾은 정체불명 객체에 자연어 라벨을 붙이는 단계다.
합성 사람은 어떤 에셋인지 알고 캡션도 미리 달아뒀다. 게다가 같은 장면의
다른 사람과 헷갈리지 않는 캡션을 골라 쓰므로 Qwen보다 정확하다.

---

## 5. 기하 — 왜 이렇게 계산하는가

### 5-1. 배치에 depth를 쓰지 않는다

depth 맵에서 픽셀을 골라 그 값을 읽는 대신, **거리 `d`를 먼저 뽑고**
물리 카메라 높이로 발 위치를 역산한다.

```
v_foot = horizon_v + fy × 0.561 / d
```

이유 두 가지:

1. 2~6m 구간이 세로로 **48픽셀뿐**이다. 픽셀 하나의 depth 노이즈가 거리 오차로 크게 증폭된다.
2. 에피소드마다 depth 스케일이 **1.4배까지 어긋나 있다**. 이 방식은 물리 높이만
   쓰므로 그 오차에 면역이다.

depth는 가림 판정에만 쓰고, 그때만 `depth_k`로 단위를 맞춘다.

### 5-2. 비등방 리사이즈 (놓치기 쉬움)

FrodoBots-2K는 1024×576을 540×360으로 **비균등** 리사이즈해서 배포한다
(`scale_x=0.5273`, `scale_y=0.6250`). 그래서 `fx ≠ fy`다.

```
fx = 214.97,  fy = 254.92,  cx = 281.15,  cy = 174.19
```

사람 PNG는 정상 비율 사진이므로 **등방 리사이즈하면 전원이 가로로 16% 뚱뚱해진다.**

```python
h_px = fy * H_real / d
w_px = h_px * (orig_w / orig_h) * (fx / fy)     # fx/fy = 0.8433
```

### 5-3. 카메라 파라미터

| 값 | 출처 |
|---|---|
| 카메라 높이 **0.561 m** | EarthRover Zero 공식 스펙 `XYZ = (0, 184, 561) mm`. episode_0763의 지면 역산 결과 0.557m로 0.7% 일치 |
| 지평선 `horizon_v ≈ 160` | 실측 피팅. `cy=174.19`보다 위 → 카메라 약 3° 하향 |
| 배치 범위 1.6~4.5 m | 하한은 머리 크롭 허용치, 상한은 depth 포화(6.582m) 여유 |
| 좌우 대역 `cx ± 105 px` | 이 안에서 렌즈 왜곡 < 4px |

전신이 프레임에 완전히 들어오는 최소 거리는 **1.89m**다. 그보다 가까우면 머리가
잘리므로 `MAX_TOP_CROP = 0.15`로 15%까지 크롭을 허용한다 (1.6m에서 8.6% 잘림).

### 5-4. 거리 샘플링은 역깊이 균등

`d`에 균등하게 뽑으면 화면상 크기가 `1/d`로 찌그러져 작은(먼) 인물이 과다해진다.
픽셀 크기가 `1/d`에 비례하므로 **`1/d`에 균등하게** 뽑는다.

| | 균등 | 역깊이 균등 |
|---|---|---|
| 거리 중앙값 | 3.95 m | **2.36 m** |
| 화면상 키 중앙값 | 113 px (31%) | **189 px (53%)** |

---

## 6. 장면 구성

### 인원 분포

| 인원 | 비율 | 이유 |
|---|---|---|
| 0명 | 10% | negative — 지시 대상이 화면에 없는 hard negative |
| 1명 | 40% | 깔끔한 단일 목표 |
| 2명 | 35% | distractor 있음 |
| 3명 | 15% | distractor 2명 |

**distractor가 있어야 언어 조건이 실제로 학습된다.** 사람이 하나뿐이면
"파란 셔츠 입은 사람에게 가"와 "사람에게 가"가 같은 뜻이 되어 모델이 속성을
읽을 이유가 없어진다.

### 캡션 충돌 회피 = 난이도 커리큘럼

같은 장면에 "a man in a navy blazer"가 두 명 있으면 그건 라벨 노이즈다.
그래서 **나머지와 헷갈리지 않는 것 중 가장 짧은 캡션**을 고른다 (Jaccard ≥ 0.5면 충돌).

- 혼자면 짧은 캡션 → `"a man in a blue shirt"`
- 비슷한 사람이 있으면 자세한 캡션 → `"a man in a light blue shirt and khaki shorts"`

배치된 사람 **전원**이 각자의 프롬프트를 갖는다. distractor도 그 자체로
유효한 학습 목표가 된다 (같은 이미지, 다른 목표).

### 겹침과 가림

먼 사람부터 그리며 **depth 버퍼를 갱신**하므로 사람-장면 가림과 사람-사람 가림이
같은 코드로 처리된다. 좌우 겹침은 45%까지 허용한다(`MAX_OVERLAP_FRAC`) —
근거리 인물은 폭이 80px을 넘어 완전 분리를 강제하면 3인 장면이 거의 안 나온다.

---

## 7. 출력 형식

### `data/synth_dataset/` — 검수·재현용

샘플당 파일 6개:

| 파일 | 내용 |
|---|---|
| `{name}.jpg` | 합성 이미지 540×360 |
| `{name}_depth_synth.npy` | 사람 영역을 `d`로 덮은 depth (원본과 같은 stored 단위) |
| `{name}_mask.png` | 인스턴스 마스크 (0=배경, 1..N=사람) |
| `{name}_gt.json` | bbox, 3D 위치, 에셋 ID, 프롬프트, 캡션, 카메라 파라미터 |
| `{name}.pkl` | step2 스키마 (OmniVLA 로더용) |
| `{name}_compare.jpg` | 검수용 원본\|합성 (학습에 불필요) |

**파일명이 곧 색인이다**: `ep0761_000017_t074_d095-075`
= 에피소드 0761, 프레임 17, 타겟 에셋 #74, distractor #95·#75

전체 색인 2종:

- `manifest.jsonl` — 샘플 1개당 한 줄. "이 이미지에 누가 들어있나"
- `assets_used.json` — 에셋 1개당 한 항목. **"이 사람이 어느 샘플에 쓰였나" (역색인)**

역색인이 있어야 나중에 어떤 에셋의 캡션을 고쳤을 때 **영향받는 샘플만 골라
다시 만들 수 있다.** `unused_asset_ids`도 함께 저장한다 — 대량 생성 후에도 이 목록이
크게 남아 있으면 그 에셋들이 배치 제약에 계속 걸리고 있다는 신호다.

### `data/omnivla_dataset/` — 학습용

`main.py`의 `package_for_omnivla()`와 동일한 구조:

```
episode_0761/image/00000000.jpg          # 224x224, stretch(crop 없음)
episode_0761/pickle_nomad/00000000.pkl   # 객체 리스트
```

- 프레임은 촬영 순서대로 **0부터 연속 재번호** (로더가 `image_path[iv-1]`을 컨텍스트로 읽음)
- 객체 없는 프레임(0명 negative)도 image/pkl을 만들고 pkl은 **빈 리스트** — 시간축 정합 유지
- slim: `OMNIVLA_KEYS` 5개만 (`pose_median`, `pose_median_norm`, `nomad_traj_norm`, `prompt`, `bbox`)

`data_config.yaml` 등록:

```yaml
metric_waypoint_spacing: 0.12
data_image_folder:  <절대경로>/data/omnivla_dataset/
data_pickle_folder: <절대경로>/data/omnivla_dataset/
```

---

## 8. GPU 머신에서 할 일

> `step2b`는 `LeLaN_Data` 안에서 실행해야 한다 (NoMaD 코드·체크포인트가 거기 있다).
> 자세한 절차는 2-3절.

```bash
# 1) 컨텍스트 방식 비교 (선택) — min_dist 중앙값이 낮은 쪽을 택한다
python3 step2b_nomad_synth_engine.py --limit 50 --context real
python3 step2b_nomad_synth_engine.py --limit 50 --context repeat

# 2) 전체
python3 step2b_nomad_synth_engine.py --synth-dir data/synth_dataset

# 3) 궤적 있는 객체만 재패키징
python3 synth_package.py --require-traj
```

### `--context`가 왜 중요한가

NoMaD의 obs 스택은 `[직전 3프레임 + 현재]`이고, 여기서 **자아운동(ego-motion)**을 읽는다.

| | 내용 | 위험 |
|---|---|---|
| `real` (기본) | 원본 에피소드의 직전 프레임 | 그 프레임엔 합성 사람이 없어 마지막에 갑자기 나타남 |
| `repeat` | 합성 프레임만 반복 | 로봇이 **정지**한 것으로 보여 궤적이 원점 근처에 머묾 |

`_sample_trajectory`는 궤적 30개 중 사람 위치에 가장 가까운 것을 고르고,
그 거리 `min_dist`가 임계 10.0(=1.2m)을 넘으면 `obj_detect=False`가 된다.
사람은 전방 1.6~4.5m = **13~37 유닛**에 있고, 8스텝 최대 도달거리가 40유닛이라
**여유가 크지 않다.** 정지로 오인되면 곧바로 전멸한다.

원본 step2는 임계를 넘은 객체를 **버리지만**, 여기서는 버리지 않고
`obj_detect=False`로 표시만 하고 데이터는 남긴다. 학습 로더에서 거르면 된다.

---

## 9. 알려진 한계

1. **컨텍스트 프레임에 다른 사람이 서 있다.** `omnivla_dataset`의 인덱스 `i-1`은
   다른 합성 샘플이라 거기 있는 사람이 현재 프레임의 사람과 다르다.
   프레임마다 독립적으로 사람을 심는 방식에서는 피할 수 없다 — 같은 사람을 여러
   프레임에 일관되게 두려면 카메라 odometry가 필요한데 데이터셋에 없다.
   `synth_package.py --layout minisequence`가 대안이지만 시퀀스가 4프레임으로
   짧아져 로더 인덱스 계산과 안 맞을 수 있다. **실제 학습 설정에서 확인 필요.**

2. **프레임 간격이 불균일하다.** 원본은 정확히 stride 3인데, 우리는 "간격 3 이상
   무작위"라 3·6·12로 들쭉날쭉하다.

3. **0명 샘플은 pkl이 빈 리스트다.** OmniVLA 스키마에 "대상 없음" 개념이 없다.
   이미지와 프롬프트는 `gt.json`에 온전히 있으니 학습 쪽에 negative 처리를 넣으면 살릴 수 있다.

4. **그림자가 없다.** 인물이 지면에서 살짝 떠 보이는 주된 원인. 발밑 타원 알파
   하나만 깔아도 크게 개선된다.

5. **"지면"과 "보행 가능면"을 구분하지 못한다.** 발 지점 검증은 장면 깊이가 `d`와
   모순되지 않는지만 본다. 풀숲이나 연석 위처럼 깊이는 맞지만 사람이 서 있을 곳이
   아닌 자리에도 배치될 수 있다.

6. **depth가 6.582m에서 포화된다.** 픽셀의 7~10%가 그 값에 붙어 있다(주로 하늘).
   그 너머 기하는 복원 불가. 배치 상한 4.5m는 이 한계에서 온 값이다.

7. **배경 다양성이 에피소드 수에 갇힌다.** 466개 샘플의 배경은 293+27+146개
   장면이 전부다. 늘리려면 프레임을 더 쪼개기보다 **에피소드를 추가**하는 편이 낫다.

---

## 10. 조정 손잡이

`synth_compose.py` 상단 상수:

| 상수 | 기본값 | 효과 |
|---|---|---|
| `D_RANGE` | `(1.6, 4.5)` | 좁히면 인물 크기 편차 감소 |
| `U_HALF_WIDTH` | `105` | 좁히면 사람이 정면에 모임 → NoMaD `min_dist` 개선 |
| `MAX_OVERLAP_FRAC` | `0.45` | 높이면 3인 장면 성공률 ↑ |
| `MAX_TOP_CROP` | `0.15` | 높이면 더 가까이 배치 가능 |
| `N_PEOPLE_WEIGHTS` | `[.10,.40,.35,.15]` | 인원 분포 |
| `BRIGHTNESS_ALPHA` | `0.5` | 1.0=장면 밝기에 완전 일치 (부자연스러움) |
| `OCC_MARGIN_M` | `0.30` | depth 노이즈 허용치 |
| `PLACE_ATTEMPTS` | `80` | 배치 재시도 |

`step2b`에서 `min_dist` 중앙값이 임계를 넘으면 `U_HALF_WIDTH`를 좁히거나
`D_RANGE` 상한을 낮추는 것이 정석이다.

---

## 11. 재현성

- `data/human_assets.json`의 `H_real`은 시드 고정으로 **한 번만** 생성된다.
  `--force-assets`로 다시 뽑으면 기존 GT와 어긋나므로 **재합성이 필요하다.**
- `synth_compose.py --seed`가 배치·캐스팅 전부를 결정한다. 같은 시드 + 같은
  에셋 파일이면 동일한 데이터셋이 나온다.
- `synth_manifest.py`는 폴더를 통째로 다시 스캔하므로 여러 번 나눠 돌려도 색인이 항상 최신이다.

---

## 12. 파일 목록

| 파일 | 역할 |
|---|---|
| `synth_pipeline.py` | **전체 오케스트레이터** — 여기서 시작 |
| `synth_calibrate.py` | 에피소드별 지면 평면 피팅 → `data/episode_ground_fit.json` |
| `synth_assets.py` | 에셋별 `H_real` → `data/human_assets.json` |
| `synth_compose.py` | 합성 본체 |
| `synth_manifest.py` | 색인 2종 생성 |
| `synth_package.py` | OmniVLA 폴더 구조로 패키징 |
| `step2b_nomad_synth_engine.py` | NoMaD 궤적 생성 (SAM 미로드) |
| `human_annotations.py` | 캡션 관리 유틸 |
| `rename_human_images.py` | 에셋 파일명 정규화 |
| `crop_human_images.py` | 에셋 투명 여백 제거 |
| `step1_metric3d_engine.py` | (원본) Metric3D depth — 새 에피소드에만 |
| `step2_nomad_sam_engine.py` | (원본) SAM+NoMaD — 합성 데이터엔 미사용 |
