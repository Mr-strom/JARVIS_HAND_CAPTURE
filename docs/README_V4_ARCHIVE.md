# JARVIS-Hand 4.0 — personal learning and fixed-base humanoid experiments

The verified personalized LEAP system is preserved, with a tested full-body **G1 + two Shadow Hands** extension. The pelvis is fixed; this is **not dynamic balance or locomotion**. Both arms and both hands move through actual MuJoCo actuators under Earth gravity. An individual sweep measured responses from 50/50 arm/hand channels, and a 60-second run completed without numerical warnings.

Personal right-hand predictions contribute through an explicit **LEAP FK geometry → normalized Shadow ROM adapter**. The adapter is analytical and experimental; it is not a newly trained Shadow policy. Left-hand and arm behaviors are scripted/feedback-driven. The standalone LEAP comparison remains the strongest validated learning result.

**Manipulation status:** the original LEAP recovery remains 0/5 valid holds. The humanoid tabletop attempts also scored 0/5 single and 0/5 assisted valid lift-and-holds. Deep-contact attempts are aborted and rejected. No reliable bimanual grasp advantage is claimed.

[Full v4 evidence, failures, model selection and limitations](HUMANOID_V4_REPORT.md)

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' scripts\fetch_g1_shadow.py
& '.\.venv\Scripts\python.exe' humanoid_control.py --viewer --seconds 30
& '.\.venv\Scripts\python.exe' humanoid_personal_replay.py
```

The first command fetches a pinned, checksum-verified model and preserves its composite Unitree/Shadow licenses. Existing Phase 2/learning requirements apply; no new framework is needed. The extended regression suite has **21 passing tests**. Actual G1 palm tracking error is on the order of 3 cm, with about 0.0053 rad soft-limit excursions, so precision tracking and zero raw joint-limit violations are not claimed.

The private captioned technical demo is `private/JARVIS_Hand_v4_technical_demo.mp4`. The v4 public code/results archive and a **separate private-artifact approval manifest** are prepared under `submission/v4/`. Do not publish the whole submission folder: personal recordings, weights, raw traces and private videos require explicit approval. Nothing has been pushed or uploaded.

## Preserved personalized-learning submission

Train a small CPU neural network on the operator's own hand recordings, then reproduce those motions with a 16-actuator LEAP Hand in MuJoCo. The verified Phase 2 live mirror remains intact alongside the trained model.

**Measured:** 34 personal recordings; 3,672 usable hand detections; five held-out test clips. The personal MLP reduced held-out fingertip geometry-target error from **61.32 mm to 32.96 mm**, and pinch-distance error from **41.12 mm to 16.68 mm**, relative to ROM mapping. Optimization was more accurate than the neural model. These are kinematic proxy metrics, not external pose ground truth.

The trained model is a **79→128→128→16 CPU PyTorch MLP**, distilled from bounded optimization through FK validated against our exact LEAP model. It performs neural motion retargeting. It is not an autonomous VLA or manipulation policy.

Read [learning, dataset, evaluation and limitations](LEARNING_AND_MANIPULATION.md) and [pinned reuse research](RESEARCH_REUSE.md). Quantitative results are in `reports/personal_model_results.json`, `reports/method_comparison.csv`, and `reports/personal_learning_results.png`.

### Run the trained model

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' neural_live.py
```

The checkpoint is `private/learning/personal_mlp.pt`. The additive live mode passed a camera/runtime smoke test; **neural pose-following was visually verified on personal-video replay, not established by that sparse live smoke test**. Use `live_mirror.py` for the original visually verified ROM live control.

### Reproduce personal training

First create the Phase 2 environment using the preserved instructions below. Then:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes -r requirements-learning.lock
& '.\.venv\Scripts\python.exe' personal_data.py --source data\personal\raw
& '.\.venv\Scripts\python.exe' train_personal.py
& '.\.venv\Scripts\python.exe' replay_personal.py
& '.\.venv\Scripts\python.exe' scripts\summarize_learning.py
& '.\.venv\Scripts\python.exe' -m pytest -q tests
```

Training uses actual recordings, whole-recording splits, training-only input normalization, early stopping, and cached optimization labels. Videos, caches, calibration, weights and comparison imagery remain private and are excluded from the code bundle. The locally inspected comparison is `private/learning/personal_comparison.mp4`: original held-out footage/landmarks, baseline physics, and trained-neural physics. It samples valid observations at a nominal 12 FPS for presentation and compresses missing-observation periods; physics evaluation uses original timestamps.

For a fresh personal-learning reproduction, first run `live_mirror.py`, press **C** to collect a valid personal calibration, and quit. This creates the `private/` workspace and `private/calibration.json` required by training. The supplied public source archive intentionally contains neither this calibration nor the private checkpoint; use your own recordings/calibration or obtain approval to distribute the proposed sample/checkpoint in the separate manifest.

### Object interaction: experimental, unsuccessful grasp benchmark

A separate scene implements a free 35 g cylinder, a physically controlled XYZ wrist, simulator-state observations, normal-contact force feedback, and a bounded state controller. **Both baseline and contact correction scored 0/5 valid two-second holds.** Brief physical lifts are not counted as successful grasps. This starts from a seeded palm pregrasp and does not demonstrate tabletop pick-and-place. A simulator RGB color-segmentation option is evaluated separately; the controller uses ground truth. No full arm, autonomous behavior cloning or language model is integrated.

```powershell
& '.\.venv\Scripts\python.exe' manipulation.py --render
& '.\.venv\Scripts\python.exe' scripts\benchmark_perception.py
```

**17 automated tests pass.** The Phase 2 core code still matches its checkpoint hashes. CPU camera benchmarks found about 7 distinct frames/s in current lighting; MSMF's higher return rate consisted largely of repeated images, so no camera FPS improvement is claimed. Total physical end-to-end latency remains unmeasured.

## Preserved Phase 2 live mirror

The working Windows application uses an OpenCV webcam, CPU MediaPipe Hand Landmarker (21 landmarks), and the right LEAP Hand in MuJoCo. Your actual detected finger geometry drives the robot's 16 actuators. The wrist stays fixed; the little finger has no independent actuator.

## Run the live mirror

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' live_mirror.py
```

Show your **right hand**, fully visible, with the palm generally toward the camera. The left pane shows the selfie-mirrored camera and 21 landmarks; the right pane shows the simulated hand. Only the selected right hand controls the robot. `--camera 1` selects another device; `--hand left` or `--hand any` changes the selection if needed. The detected label is reported in the metrics. No camera recording is substituted for live input.

- **C:** collect eight seconds of calibration. Repeatedly open, curl, spread your fingers and oppose the thumb across your palm. At least 30 valid detections are required; channels with too little observed range retain their prior/default calibration.
- **Q / Esc:** exit. Close the window to exit as well.
- The last calibration loads automatically from `private/calibration.json`.
- Tracking loss holds the last request for 0.3 seconds, then smoothly returns to neutral. Reacquisition is filtered and rate-limited. Frames older than 250 ms at inference completion are rejected.

Optional bounded verification run:

```powershell
& '.\.venv\Scripts\python.exe' live_mirror.py --seconds 60 --output live_check
```

Metrics go to `reports/live_check_metrics.json`. Derived joint telemetry and calibration stay in ignored `private/`. No video is saved by default. Add `--evidence` to save two local diagnostic screenshots containing camera imagery; these are also ignored and are not published.

## Phase 2 setup and validation

The existing Phase 1 environment needs no new Python packages. Fetch the official versioned model once:

```powershell
& '.\.venv\Scripts\python.exe' scripts\fetch_hand_landmarker.py
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pytest -q tests
```

For a fresh environment, first follow the reproduction commands below, then fetch this model. Its SHA-256 is checked by the download script. MediaPipe Tasks HandLandmarker is the supported 21-point hand API used by installed MediaPipe 0.10.32; this application does not depend on the removed legacy `mp.solutions.hands` interface.

The first actual human test lasted 117 seconds: 715 tracked frames, 10.31 webcam FPS, 27.63 robot/display updates per second, 21.10 ms average MediaPipe processing, and 51.90 ms average read-completion-to-render delay. There were zero control/joint-limit violations and zero MuJoCo warnings. Calibration collected 60 samples and personalized all 16 channels. These measurements include periods with the hand absent; the detection ratio is not a detector-accuracy score.

**Latency limitation:** the 51.90 ms value is a measured software-pipeline proxy, not exposure-to-display latency. Camera buffering, display scanout, filtering and actuator response add variable delay. The original zero correlation-lag bin was unresolved at the native camera sampling interval, not evidence of zero latency. See `reports/phase2_live1_analysis.json` for the corrected interpretation, and `PHASE2_REPORT.md` for full results and limitations.

The supplied mapping is geometric and approximate: calibrated flexion angles, signed palm-relative splay, and thumb opposition proxies. Occlusion and monocular depth errors can produce imperfect thumb/knuckle poses. No pose-accuracy benchmark or object-grasp success is claimed.

## Reset investigation

The old synthetic-viewer loop assumed simulation time could never restart. An intentional reset at step 13534 (27.068 simulated seconds) reproduced the reported exception with no numerical warnings. Both the old and corrected LEAP loops passed separate 180-second tests without intervention. The new guard recognizes resets only at the passive-viewer synchronization boundary; unexpected time changes, nonfinite states, changed timestep, and numerical warnings still fail. Faults now save model/run context in a `*_failure.json` report. The live app has no MuJoCo reset UI.

## Phase 1 actuator-only baseline

A tested, standalone Windows MuJoCo simulation of the right LEAP Hand. All 16 actuators follow a bounded synthetic curl/spread cycle. The simulation uses actual actuator controls and physics stepping; it does not animate joint positions directly.

The commands below are the retained synthetic Phase 1 baseline, separate from `live_mirror.py`. No grasping, training, or publication is implemented in either application.

## Run the installed simulation

In PowerShell:

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' simulate.py --viewer --seconds 30
```

The window displays opening/curling fingers and closes after 30 seconds. Mouse controls are provided by MuJoCo. Closing early ends the run and marks the requested-duration check incomplete. No environment activation or execution-policy change is needed.

Save a six-second video, GIF, two snapshots, actual control/joint trajectories, and metrics:

```powershell
& '.\.venv\Scripts\python.exe' simulate.py --seconds 6 --render --output leap_demo
```

Outputs are written to `reports/`. The supplied `reports/leap_demo.gif` and `reports/leap_demo.mp4` were generated and inspected during Phase 1. They are synthetic actuator demonstrations, not user recordings.

## Reproduce the environment

Requires existing CPython **3.11.9 x64**, internet access for setup, and a working Windows OpenGL driver. The following commands install into `.venv` only. Do not replace an unrelated existing environment.

```powershell
# Run from repository root
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -c "import sys; assert sys.version_info[:3] == (3,11,9), sys.version"
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes --only-binary=:all: -r requirements.lock
& '.\.venv\Scripts\python.exe' scripts\fetch_leap.py
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' simulate.py --seconds 6 --render --output leap_demo
& '.\.venv\Scripts\python.exe' -m pytest -q tests
```

The lockfile covers 32 resolved PyPI packages for Windows x64/Python 3.11. It includes approved later-phase candidates (MediaPipe and SciPy); they are not used by the standalone LEAP controller. Only OpenCV Contrib is installed, avoiding conflicting `cv2` distributions. No CUDA, uv, WSL, or global package changes are required.

The installed environment also contains somehand 0.3.0 for the retained Dex5 trial. To reproduce that optional trial separately:

```powershell
& '.\.venv\Scripts\python.exe' scripts\fetch_assets.py
& '.\.venv\Scripts\python.exe' -m pip install --no-deps .\third_party\somehand
& '.\.venv\Scripts\python.exe' scripts\probe_environment.py
& '.\.venv\Scripts\python.exe' experiments\dex5_attempt.py --seconds 6 --output dex5_tuned
```

The probe is diagnostic: its raw Dex5 dynamics produce instability warnings. The tuned trial is retained as a failed acceptance case, not the working demo. Its local source comes from a verified archive at a pinned revision; the optional source-wheel build is separate from the hashed PyPI wheel lock.

## Validation and results

```powershell
& '.\.venv\Scripts\python.exe' simulate.py --seconds 60 --output stability_60s
& '.\.venv\Scripts\python.exe' -m pytest -q tests
```

- 60 simulated seconds / 30,000 physics steps, all 16 actuators moving.
- Zero MuJoCo warnings, nonfinite states, resets, control-bound violations, and observed joint-limit excursions.
- Zero initial self-penetration. This gentle motion had no contacts, so contact/grasp behavior remains untested.
- Two six-second runs produced exactly equal sampled controls, joint positions, and final states on this machine/environment. This is not a cross-platform determinism guarantee.
- A six-second 960×720 render produced 181 frames and completed its simulation/render loop in 3.27 s. This excludes model loading and media encoding and is not webcam FPS or tracking latency.
- An eight-second interactive viewer run completed in 8.00 s.
- Three validation tests passed. See `PHASE1_REPORT.md` and raw `reports/*_metrics.json` for evidence and limitations.

## Why LEAP replaced Dex5

Dex5 loaded through somehand's core API, but its raw asset model became unstable at 0.010 s and began with 24.75 mm of self-penetration. A conservative integration/control trial stabilized stepping but retained that penetration. We switched to the preapproved Menagerie fallback approximately eight minutes after starting Phase 1, inside the 45-minute limit.

LEAP has three fingers plus a thumb; there is no independent little finger. Its model revision is pinned to `c1a4eeb85694ae1dffe33ff1797d4e528928a133`. XML, meshes, and physics parameters remain unchanged.

## Attribution and data

See `THIRD_PARTY_NOTICES.md` and `licenses/`. Downloaded assets, third-party source, the virtual environment, and private imagery/calibration/telemetry are excluded by `.gitignore`. Nothing has been pushed or published. The preserved Phase 2 milestone is described here; current personalized-learning results are at the top of this README.
