# Final integration and release decision

Status: **local candidate, validated; publication approval required**. v5 remains the default; v6 is optional. No retraining, new robot, grasp controller, license, push or upload.

## Verified behavior and evidence

The selector `python -m v6.launch` delegates neural directly to unchanged `neural_live.py`; optimizer/hybrid use the isolated v6 application. Existing source, weights, calibration and 314 protected artifacts are preserved; frozen v5 RC1 is not overwritten. No dependency installation was needed.

The personal MLP has 28,816 parameters, trained from 34 recordings. Five held-out recordings provide 921 valid observations; the v6 quality gate accepts a common 866. Accepted raw fingertip/pinch errors: neural 31.11/15.62 mm; optimizer 22.06/7.48 mm; hybrid 27.28/11.74 mm. These are proportion-adjusted geometry proxies, not physical ground truth. Original all-921 results are separate: ROM 61.32/41.12, neural 32.96/16.68, teacher optimizer 27.83/8.93 mm.

Hybrid is not the accuracy leader. Physical replay against the v5 neural-filter comparison worsened 49.93→50.10 mm; commanded acceleration is bounded at 30 rad/s². Prediction was rejected. Shadow neural transfer did not improve alignment; G1 pelvis remains fixed and left-hand coordination scripted. LEAP, single-hand G1 and assisted G1 each remain at 0/5 valid grasp holds. No autonomy or locomotion claim.

## Attended live verification, November 3, 2026

Both ran 30 seconds on actual webcam input. Strict right-only v5 selection initially accepted zero detections; its existing `--hand any` option then tracked the sole visible hand. This does not justify rewriting default behavior. The operator confirmed: **“Both followed; no obvious improvement.”**

| Measurement | Neural (existing any-hand option) | Hybrid |
|---|---:|---:|
| Captured FPS | 8.62 | 7.49 |
| Tracked observations/s | 6.20 | 7.10 |
| Robot/display FPS | 27.50 | 29.72 |
| Read-complete to first render, mean | 54.24 ms | 59.26 ms |
| Same delay, p95 | 71.50 ms | 75.03 ms |
| Displayed frame age, mean / p95 | Not instrumented in successful run | 109.64 / 170.47 ms |
| MuJoCo warnings / joint-limit violations | 0 / 0 | 0 / 0 |

Hybrid accepted 213 frames, with 94 neural and 119 precision observations and 10 missing observations; neural accepted 186 and logged three tracking-loss transitions, with hold/return behavior. All 16 neural joints moved. Camera imagery was not saved. Hybrid counts content-hashed unique frames; successful neural run counts fresh capture sequences without duplicate-image analysis. Runs used different gestures/exposure conditions, not a controlled paired accuracy benchmark. Physical motion latency and objective visible jitter/pinch accuracy remain unmeasured. MediaPipe emitted its existing feedback-tensor/projection advisory messages, not MuJoCo numerical warnings.

## Final validation and integration decision

**Recommend including v6 only as an optional experiment; retain v5 as default.** All 35 tests passed in the working project and clean extraction (25 existing + 6 v6 + 4 selection checks). Before restoring private inputs, 30 passed and five skipped: the original two personal-artifact skips plus three v6 checkpoint/calibration skips. The old test message "Training not yet complete" means the checkpoint is absent in that extraction; training was already completed.

The sample replay decoded 80 frames, detected 49, and moved all 16 actuators over 6.668 simulated seconds: zero warnings and limit violations. G1 viewer launch/stepping passed for three seconds with zero numerical warnings; its unchanged soft joint constraints allowed a maximum 0.005294 rad excursion, disclosed rather than hidden. The original 921-frame LEAP metrics reproduced within the existing 0.001 mm assertion. All 42 non-timing Shadow summary values matched exactly.

Portable v6 reproduction used all five held-out clips. Neural/hybrid metrics matched exactly. One optimizer clip changed by -0.008384 mm raw tip / -0.141914 mm physical error because its previous timeout did not recur; other records matched. Runtime timing is not deterministic. All replay checks passed without new numerical instability. No default simulation code was changed. Source-equivalence and final archive checks accompany FINAL_VALIDATION.json and FINAL_ARCHIVE_CHECK.json.

All 314 protected hashes and frozen RC1 remain unchanged. The final demo fully decodes to 1,620 frames at 30 FPS / 54 seconds, with no audio. Selected output frames and the results card were visually inspected; this is not exhaustive privacy clearance. New-code inclusion is local only; no Git merge, remote push or publication was performed.

## Reproduction

Use Python 3.11.9 x64, Windows/OpenGL, and the existing hashed requirements. From a fresh extraction:

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -c "import sys; assert sys.version_info[:3] == (3,11,9)"
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes --only-binary=:all: -r requirements.lock
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes -r requirements-learning.lock
& '.\.venv\Scripts\python.exe' scripts\fetch_leap.py
& '.\.venv\Scripts\python.exe' scripts\fetch_g1_shadow.py
& '.\.venv\Scripts\python.exe' scripts\fetch_hand_landmarker.py
& '.\.venv\Scripts\python.exe' scripts\restore_release_artifacts.py
& '.\.venv\Scripts\python.exe' -m pytest -q -rs tests v6/test_adaptive.py v6/test_selection.py
& '.\.venv\Scripts\python.exe' -m v6.launch --controller neural --hand any --seconds 30
& '.\.venv\Scripts\python.exe' -m v6.launch --controller optimizer --seconds 30
& '.\.venv\Scripts\python.exe' -m v6.launch --controller hybrid --seconds 30
& '.\.venv\Scripts\python.exe' scripts\release_sample_smoke.py
& '.\.venv\Scripts\python.exe' scripts\release_leap_metrics_smoke.py
& '.\.venv\Scripts\python.exe' -m v6.reproduce_release
& '.\.venv\Scripts\python.exe' cross_embodiment.py --dataset private\final\heldout_landmarks
& '.\.venv\Scripts\python.exe' humanoid_control.py --viewer --seconds 10
```

This run reuses the pinned local environment and asset cache; it is not a fresh wheel-download test. Sample replay needs consented sample/model/calibration. Full held-out comparisons need optional held-out inputs; training cannot be reproduced from the sample alone. Wall-clock solver timeout fallbacks make small numerical/timing variation possible.

## Demonstration and approvals

New local archive: `JARVIS-Hand-v5-default-v6-optional-REQUIRES-APPROVAL.zip`, SHA in `CANDIDATE_MANIFEST.json`. It contains 13 pending-approval personal artifacts, including a replacement 54-second video at `release_artifacts/JARVIS_Hand_final_demo.mp4`. Reused real footage retains playback speed. Optimizer comparison is a measured table, not newly rendered optimizer footage. Original G1 full-body launch remains separately runnable.

See `EXACT_RELEASE_PROPOSAL.md` for every path/hash/risk. Minimum sample/demo set is five artifacts; eight additional held-out artifacts support metric reproduction. Raw recordings stay private. The earlier review of v5 sample frames does not clear every new v6 demo pane. No license is applied; MIT and copyright identity remain awaiting approval.

## GitHub description and submission summary

Suggested description: **CPU personalized hand retargeting: learned LEAP control, optional geometric precision, and honestly evaluated transfer to fixed-base G1/Shadow Hands.**

Summary: We collected 34 personal recordings and trained a small neural LEAP retargeter. It improves over ROM geometry mapping, while optimization remains more accurate. An optional hybrid trades CPU cost and smoother bounded commands against precision and response delay. Real webcam control and deterministic-input physical replay are demonstrated. Transfer to Shadow did not improve alignment and physical grasp holds remain unsuccessful. Learned right-hand motion, analytical retargeting, scripted left-hand coordination and simulator-state feedback are explicitly distinguished.

## Publishing steps — owner action only after approval

1. Approve the exact archive/hash, personal files and their reuse terms, original-code license/copyright holder, and destination.
2. Finish frame-by-frame privacy clearance of the new demo; replace or withhold it if needed, then regenerate its hash and candidate.
3. Extract the approved candidate into a separate public staging folder. Apply only the approved original-code license; preserve all upstream notices.
4. Exclude `.venv`, `assets`, `private`, `data`, runtime logs and raw recordings. Stage explicit reviewed source/docs/results and approved `release_artifacts`; inspect staged filenames and diff. Do not use an unreviewed blanket `git add .`.
5. Inspect the existing remote before creating a publication commit. Configure `https://github.com/Mr-strom/JARVIS_HAND_CAPTURE.git` only in approved staging; do not overwrite remote history or force-push. Push only after explicit publication approval.
6. Link the approved repository/demo in the competition form, state the limitations above, and submit before November 4, 2026, 23:59 IST. No upload or submission has been performed here.

Submission checklist: validation receipt; candidate hashes; license and attribution; personal-artifact consent; new-demo privacy clearance; runnable commands; honest quantitative claims; owner-approved repository push and portal submission.
