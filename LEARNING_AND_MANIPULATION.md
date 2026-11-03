# Personalized learning and manipulation status

The working Phase 2 system has been preserved. `checkpoints/phase2_verified.zip` and its SHA-256 inventory capture the original code and reports; a separate ignored calibration backup captures private settings. All original core files and tests still match that inventory. No Git repository was present, so the checkpoint is a backup rather than a Git commit.

## What was implemented and measured

- Personal video discovery, exact hashing, timestamped cached MediaPipe extraction, geometry categories, handedness labels/reflection, and recording provenance. Original files are untouched.
- Differentiable LEAP FK adapted from MIT-licensed handrobot; 20 random-pose checks against this exact MuJoCo model gave maximum absolute point difference **1.45e-8 m**.
- A **28,816-parameter 79→128→128→16 MLP**, trained on actual personal recordings using bounded optimization-derived joint targets. Output sigmoid enforces conservative ranges intersected with actual actuator limits. AdamW, seed 42, early stopping on whole held-out validation recordings.
- Actual MuJoCo replay of all five test clips for both baseline and neural control, with common rate limiting and tracking-gap fallback. Zero physics warnings and zero observed joint-limit excursions in those replays.
- A private three-panel video showing original held-out human footage with newly detected landmarks, Phase 2 baseline physics, and trained-neural physics: `private/learning/personal_comparison.mp4`. The generated video has **473 frames at 1920×520**. It was visually inspected.
- An additive `neural_live.py` entry point retaining the original filtering/loss fallback. Its 15-second webcam smoke test completed; only one hand frame was detected in that test, so **live neural pose-following quality is not claimed as visually validated**. Recorded-neural replay is validated.
- Camera read/backend benchmarks counting both delivered images and distinct frame hashes.
- A separate physical cylinder scene, force-limited XYZ wrist, contact/normal-force observations, feedback-driven phases, lift/hold/recovery evaluation, and RGB orange-object segmentation benchmark.

## Dataset

**34 recordings**, all unique by SHA-256; **3,905 sampled frames**, **3,672 valid detections**, **94.03% coverage**. The originals are under `data/personal/raw/`; their arbitrary filenames/folders are not used as gesture labels. Simple geometry assigns open, curled, pinch, mixed-flexion or unknown categories; these labels are descriptive and are not verified manipulation outcomes.

The pipeline uses a nominal 12 Hz integer source-frame stride (effective rates can vary, for example 10–15 Hz), preserving source decoded-frame indexes and timestamps. Original metadata, orientation flags, detected-handedness counts, quality counts and output hashes are retained locally. Invalid frames remain explicitly marked rather than being filled with invented motion. Features include normalized landmarks, joint bends, thumb proximity, palm basis, image-wrist displacement proxies, velocity and acceleration. Distances/orientation can be derived directly from the cached landmarks.

The split is **24 train / 5 validation / 5 test recordings**, corresponding to **2,280 / 471 / 921 valid frames**. Exact duplicate hashes cannot cross partitions. A conservative pose-sequence/duration near-duplicate screen found no candidate pairs; it is a heuristic, not proof of no semantic similarity. Whole recordings, never random frames, are partitioned. Capture sessions/backgrounds may be shared across partitions; this tests held-out clips from one person, not generalization to a new session or person.

Monocular world landmarks are estimates. The true mirror state cannot be inferred reliably from arbitrary video alone; the pipeline records that ambiguity, reads video orientation metadata, and canonicalizes the tracker-labeled left hand through reflection. It does not claim verified global 3D coordinates or perfect handedness. Multiple visible people/hands and handedness flicker remain quality risks.

## Training method and labels

Inputs are 63 normalized landmark coordinates plus 16 geometric joint features. Input standardization is fit on training recordings only. No public hand dataset or synthetic hand examples were used for training.

The teacher minimizes fingertip-position MSE plus 0.25 times pinch-distance MSE through verified differentiable FK, constrained to the same conservative operating ranges. Each human digit's shape is proportion-adjusted to the robot chain length and anchored at that robot digit's base. Targets therefore express a chosen kinematic correspondence; they are **not measured robot joint labels or guaranteed reachable physical poses**. The teacher runs 100 Adam steps per batch, retaining the best geometry objective for each frame including its ROM initialization.

The MLP is trained to regress those normalized teacher joint commands with MSE and AdamW weight decay. It stopped after **113 epochs**; best validation loss was **0.0398649** normalized joint MSE. Full train/validation histories are in JSON/CSV and plotted in `reports/personal_learning_results.png`. There is no autonomous imitation policy or robot-success labeling of video footage.

An initial target-frame sign error put human curl opposite LEAP curl. That run was rejected and retained as `reports/initial_coordinate_error_results.json`; the corrected transform is human `(forward, thumb-side, curl)` to robot `(x, y, -z)`. A regression test now verifies the curl direction. All reported final results below use corrected targets and retraining.

## Held-out results

| Method | Fingertip target error | Pinch-distance error | Raw command acceleration norm |
|---|---:|---:|---:|
| Phase 2 ROM | 61.32 mm | 41.12 mm | 148.23 rad/s² |
| Bounded optimization teacher | 27.83 mm | 8.93 mm | 182.11 rad/s² |
| Personal MLP | 32.96 mm | 16.68 mm | 141.25 rad/s² |

All methods had zero command-limit violations. The neural model improves these geometry proxies over ROM, while optimization is more accurate than the MLP. The acceleration statistic is computed on raw commands over short contiguous intervals, before shared execution filtering; it is not a perceptual jitter score. The geometry error uses the target correspondence above, not external motion-capture ground truth or settled simulator fingertip accuracy. It is not evidence of improved grasp success.

Neural forward-pass latency was **0.0847 ms** on CPU. `reports/inference_benchmark.json` additionally measures the complete warmed inference adapter and baseline, and separately times a single-frame 100-step optimizer. Rendering, tracking, camera delay and physics settling are excluded from those inference numbers.

## Camera performance

At current lighting, DirectShow delivered about **7.27 distinct frames/s** at both 640×480 and 320×240. Reducing resolution did not improve that measurement. Mean blocking camera read was approximately **137 ms**, substantially longer than the previously measured roughly 20 ms landmark inference.

MSMF returned about **26.60 images/s**, but only **58 distinct image hashes over 8.08 s** (roughly 7.18 distinct images/s). Calling that a 27 FPS camera improvement would count repeated images. No such improvement is claimed. The original latest-frame capture thread remains; it does not queue stale frames. Exposure settings were not changed, and physical sensor/display latency remains unmeasured. These before/after comparisons were taken in different lighting sessions, not a controlled illumination experiment.

## Object and manipulation benchmark

The separate scene uses a **35 g cylinder, radius 22 mm, height 50 mm**, a table/floor support and a three-axis physically actuated wrist. Wrist actuators are limited to 30 N per axis; measured peak was about 10.51 N. The original hand scene is untouched. The benchmark seeds the cylinder above the palm and lets gravity settle it: **this is a seeded pregrasp experiment, not tabletop reach-and-pick**.

The controller uses object ground-truth position/velocity, relative alignment, distinct digit contacts and normal forces. It executes SETTLE→ALIGN→CLOSE→CONTACT_CHECK→LIFT→HOLD→RELEASE or RECOVER. Thresholds use feedback; timeouts bound failures. The corrected variant reduces closure above 8 N contact force and adds bounded closure when contact is missing. No object attachment, equality constraint or direct repositioning after initialization is used.

Success requires at least 2 cm lift above the settled reference, no table contact, at least two distinct contacting digits, and object speed under 0.15 m/s continuously for two seconds. Identical offsets `[-8,-4,0,4,8] mm` were used for both controllers.

**Final outcome: baseline 0/5 successes; contact-aware correction 0/5 successes.** Maximum lifts were about 7.62 cm and 7.70 cm, but maximum valid holds were only 0.108 s and 0.100 s. Both controllers lost sustained multi-digit contact. All ten final trials had zero MuJoCo warnings and zero observed joint-limit excursions. An earlier correction attempt also failed and is retained separately. Raw contact-loss edges are logged as unfiltered contact changes; they are **not reported as verified slip events**, because solver contact flicker needs further interpretation. Release logic exists but was not reached through a successful hold.

This demonstrates movable-base physics and contact-feedback instrumentation, but **does not establish reliable grasping or a beneficial grasp correction**. No successful robot demonstrations exist to train autonomous behavior cloning. Full-arm assembly and optional language planning were deferred to protect the working personal-learning submission.

## Perception scope

Orange-color RGB segmentation was evaluated separately against simulator geometry-ID masks on three offset views. Mask IoU was **0.808–0.898**, with centroid error **0.19–0.43 pixels**. This is a controlled simulator-color benchmark, not semantic recognition, real-camera depth estimation, or the observation source used by the grasp controller. The task controller uses explicitly labeled simulator ground truth.

## Reproduction and private artifacts

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes -r requirements-learning.lock
& '.\.venv\Scripts\python.exe' personal_data.py --source data\personal\raw
& '.\.venv\Scripts\python.exe' train_personal.py
& '.\.venv\Scripts\python.exe' replay_personal.py
& '.\.venv\Scripts\python.exe' scripts\summarize_learning.py
& '.\.venv\Scripts\python.exe' neural_live.py
```

The preserved baseline remains `live_mirror.py`. Personal raw videos, processed caches, calibration, weights, derived motion and comparison videos stay under ignored `data/` or `private/`. Do not publish them without explicit approval. Public-ready code and aggregate results can be packaged separately; nothing has been pushed or uploaded.

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests
& '.\.venv\Scripts\python.exe' manipulation.py --render
& '.\.venv\Scripts\python.exe' scripts\benchmark_perception.py
```

The test suite has **17 passing tests**, including original Phase 2 regressions, FK equivalence, split isolation, normalization, finite bounded learned outputs, curl convention and physical free-object/force-limited-wrist checks.
