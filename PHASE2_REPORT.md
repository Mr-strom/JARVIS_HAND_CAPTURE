# JARVIS-Hand — Phase 2 results

Live testing: November 2–3, 2026, IST. Final verification: November 3.

**Outcome: functional live mirroring demonstrated with real webcam hand detections.** The application estimates finger geometry, drives actual MuJoCo actuator controls, shows webcam landmarks beside the robot, calibrates personal ranges, filters motion, and returns toward neutral on tracking loss. This is approximate geometric retargeting, not a precision pose-reconstruction or grasp benchmark.

No object grasping, model training, GitHub publishing, or Phase 3 demonstration collection was started. The LEAP model XML, meshes, timestep, masses, damping, collision settings, and gains remain upstream originals. Only display-buffer size and camera view are configured by our code.

## Task A — reported reset

### Provenance and diagnosis

The exact original `RuntimeError: Unexpected simulation reset at step 13534` was not present in the saved project logs or an attached terminal. The old message existed in both `simulate.py` (LEAP) and the retained `experiments/dex5_attempt.py`; the text alone cannot prove the original command/model. The user answered “yes” when asked whether Reset/Backspace was used. Given the working LEAP viewer command from Phase 1, a manual LEAP viewer reset is the supported explanation, but the original unarchived command is not independently recovered.

The archived Dex5 numerical failure is a different event: BADQACC at 0.010 s in `reports/environment_probe.json`, on November 2 at about 18:49 IST. It must not be conflated with the reported step-13534 exception.

Executed reproductions:

1. **Unmodified old LEAP loop:** 180 simulated seconds / 90,000 steps passed without intervention, warnings, or resets. Evidence: `reports/phase2_reset_reproduction_metrics.json`.
2. **Exact old error reproduced deliberately:** reset LEAP data immediately before zero-based step 13534. Prior time: 27.068 s; after the next step: 0.002 s; old expected time: 27.070 s. The old guard raised the exact reported message with all numerical warning counters zero. Evidence: `reports/reset_injection_original_guard.json`. This is a controlled reproduction, not a claim that we recovered the user's historical process.
3. **Corrected loop:** another 180 simulated seconds / 90,000 steps passed. Evidence: `reports/phase2_reset_fixed_metrics.json`.
4. **Actual viewer Backspace test:** Backspace reset time from 18.134 s to zero at cumulative step 9067. The new code recorded that event and completed all 15,000 steps / 30 seconds without warnings, joint-limit excursions, or crashes. Evidence: `reports/phase2_reset_ui_verified_metrics.json`.
5. An earlier 45-second viewer smoke test passed but received no reset: its attempted key input was interrupted by user window activity. It is retained as `reports/phase2_viewer_reset_metrics.json` and is not counted as the actual reset regression.

### Fix

`physics_guard.py` checks time continuity, finite position/velocity/acceleration, the fixed timestep, and MuJoCo warning counters around every physics step. A warning, numerical reset during `mj_step`, or unexplained external time jump still raises `PhysicsFault`.

Only a backward-to-zero time change observed directly around passive `viewer.sync()` is acknowledged as a viewer reset. There is no physics stepping inside that boundary. Acknowledgment also requires finite state and zero warnings. Cumulative run duration and frame pacing now use completed step count rather than assuming that local `data.time` never restarts. Failure reports include the model, requested duration, viewer mode and traceback in `reports/<output>_failure.json`.

The live app uses an offscreen MuJoCo renderer and OpenCV window, so it has no MuJoCo Reset control. Numerical guards remain enabled there. Automated tests deliberately inject an enormous velocity and verify that the resulting numerical failure still stops execution; the corresponding QVEL warning in `MUJOCO_LOG.TXT` is test-generated, not a live-run failure.

## Task B — implementation

Entry point: `live_mirror.py`. No synthetic actuator trajectory is called by this application. The synthetic Phase 1 `simulate.py` command remains a separate baseline.

- A capture thread opens camera 0 using DirectShow (MSMF fallback), requests 640×480 at 30 FPS, and retains only the latest frame. Delivered camera rate is measured rather than assumed.
- MediaPipe **0.10.32 Tasks HandLandmarker** runs with an explicit CPU delegate, VIDEO mode and monotonic timestamps. It provides 21 image landmarks, 21 world landmarks, and handedness. This is the supported hand-detection interface in our installed package; no GPU-specific inference dependency was added.
- Input and preview are consistently selfie-mirrored. Default selection is the detected right hand; other sides are ignored. `--hand left`/`any` are available explicitly. Partial or unstable handedness can therefore cause temporary loss rather than switching hands silently.
- A palm-relative basis removes rigid wrist orientation. Index/middle/ring MCP flexion and splay plus PIP/DIP bending map by **joint name**, not an assumed actuator order. The little-finger MCP anchors palm orientation only; its PIP/DIP/tip have no independent actuator mapping.
- LEAP positive splay direction was checked by forward kinematics. Human motion toward the thumb uses the opposite sign for LEAP's rotation actuator.
- Thumb MCP/IP use bending angles. Thumb CMC and axial opposition use planar orientation and palm-width-normalized thumb-tip proximity to the ring MCP. Those two thumb channels are heuristics, not anatomical one-to-one joint recovery.
- Eight-second calibration collects valid feature samples and uses the 5th/95th percentiles. It requires at least 30 valid frames and a minimum observed span per channel. Unexcited channels keep their previous/default range. The first live calibration collected **60 samples and personalized 16/16 channels**; the final session loaded this saved calibration.
- Normalized features map to conservative operating ranges intersected with model actuator limits: MCP up to 1.25 rad, PIP 1.5, DIP 1.15, splay ±0.35; thumb caps 1.0/1.05/1.1/1.2 rad. These are controller target restrictions, not modifications to the model's physical limits.
- Exponential smoothing uses a 60 ms time constant, with a 2.5 rad/s command slew limit. Actual joint-limit excursions are measured separately from command bounds.
- Tracking loss holds the last requested pose for 0.3 s, then returns toward zero/neutral with a 200 ms smoothing constant and the same slew limit. Reacquisition does not jump directly to the new pose. The final code rejects frames older than 250 ms at inference completion.
- Physics runs at the model's 2 ms fixed timestep with bounded catch-up work. Dropped catch-up time is reported; it is not hidden as real-time simulation. Any MuJoCo warning or joint-limit excursion greater than 0.02 rad stops the application and saves an error report.
- One simple split-screen window shows landmarks, robot, tracking state, detected-frame count, update rate and inference timing. `C` calibrates; `Q`/Esc exits. Wrist pose stays fixed.

## Task C — actual measurements

These are measurements from this laptop, not estimates copied from another repository.

| Metric | Initial human test, Nov 2 | Final regression, Nov 3 |
|---|---:|---:|
| Wall duration | 116.99 s | 45.02 s |
| Camera frames delivered | 1,204 | 335 |
| Webcam FPS | **10.31** | **7.49** |
| Frames processed | 1,200 | 335 |
| Valid selected-hand detections | **715** | **106** |
| MediaPipe time, mean | **21.10 ms** | **19.69 ms** |
| MediaPipe time, p95 | 26.06 ms | 25.28 ms |
| Robot/display updates | **27.63 FPS** | **28.32 FPS** |
| Physics steps | 58,256 | 22,352 |
| Simulated duration | 116.512 s | 44.704 s |
| Camera read complete → rendered frame, mean | **51.90 ms** | **50.60 ms** |
| Same software-pipeline delay, p95 | 70.94 ms | 69.51 ms |
| Command-limit violations | **0** | **0** |
| Joint-limit violation steps (>0.0001 rad) | **0** | **0** |
| Maximum joint-limit excursion | **0 rad** | **0 rad** |
| MuJoCo warnings | **0** | **0** |
| Tracking-loss return transitions | 4 | 3 |
| Personalized calibration channels | 16/16 | 16/16 loaded |
| Runtime error | None | None |

Raw results: `reports/phase2_live1_metrics.json`, `reports/phase2_live_final_metrics.json`. The initial session ended normally shortly before its two-minute cap. The final session completed its 45-second cap.

### Latency interpretation

The approximately **51–52 ms** figure is a measured software-pipeline proxy: timestamp immediately after `VideoCapture.read()` returns, through inference, physics update and rendering. It does not include camera exposure/buffering or actual screen presentation. Nor does it mean that a large finger movement has finished settling: filtering, the slew limiter and robot dynamics add movement-dependent delay. For example, the 2.5 rad/s limiter alone requires at least 0.5 s to traverse a 1.25 rad command change.

The first diagnostic cross-correlation returned a zero-lag bin. That result was too coarse: tracked samples were roughly 110 ms apart, and interpolation across occlusions cannot improve actual timing resolution. The raw report is preserved, but **its `median_lag_ms: 0` must not be treated as measured zero latency**. Corrected interpretation is in `reports/phase2_live1_analysis.json`; the final app labels sub-resolution lag estimates unavailable. A regression test protects this distinction.

Therefore: software frame-to-render timing is measured; full physical human-motion-to-screen latency is **not reliably established** by this test. An external high-frame-rate reference or suitable synchronized timing target would be needed for that measurement.

### Visual evidence

I inspected the actual live window and local diagnostic snapshots. The webcam showed the user's hand with detected landmarks; different open/curled finger configurations produced visibly different LEAP configurations. The controller path consumes only those camera-derived features. During absence/loss the robot returned toward neutral, rather than continuing an animated sequence.

Evidence stays local in ignored `private/phase2_live1_open.png`, `private/phase2_live1_curled.png`, and the corresponding final-run screenshots. The first pair spans different points in the session, including calibration; it is evidence of live behavior, not a controlled pose-accuracy comparison. Derived joint telemetry is also kept in `private/` and is not public-repository material without approval.

The final scene was visibly much darker. Tracking was intermittent and sometimes labeled a presented hand as the nonselected side. No accuracy claim is derived from detected/processed ratios: the sessions also intentionally included no-hand periods, hand removal and calibration. Good lighting and a fully visible palm are needed for a convincing demo.

### Automated validation

**11 tests passed** (`reports/phase2_tests_final.log`): the three Phase 1 checks plus Phase 2 geometry/wrist invariance, no independent little-finger actuator, named-joint isolation and splay direction, tracking-loss/reacquisition behavior, limits/rate filtering/calibration persistence, intentional-versus-unexplained resets, numerical-fault rejection, and unresolved latency-bin handling.

The exact installed package set is unchanged from Phase 1. No global Python modifications were made. The official Hand Landmarker asset is versioned and SHA-256 checked; its manifest is `reports/hand_landmarker_manifest.json`.

## Run commands

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' live_mirror.py
```

If the hand-landmark model is absent:

```powershell
& '.\.venv\Scripts\python.exe' scripts\fetch_hand_landmarker.py
```

Validation:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests
& '.\.venv\Scripts\python.exe' simulate.py --seconds 180 --output long_check
```

## Known limitations and stop point

1. Camera throughput was 7.49–10.31 FPS in the measured sessions despite the 30 FPS request; robot/display updates were about 28 FPS. This is responsive coarse mirroring, not verified 30 FPS visual tracking.
2. Low light, occlusion, camera buffering, depth ambiguity and handedness changes degrade tracking. A closed fist can hide finger segments from the single camera.
3. Thumb CMC/axial mapping and MCP geometry are approximations. ROM calibration is user/session dependent and does not establish anatomical accuracy. No quantitative pose-error study was performed.
4. Small self-contacts occurred during live retargeting: maximum penetration was about **1.38 mm** initially and **3.50 mm** in the final low-light run. No warning or joint-limit failure occurred, but contact quality is not validated for grasping. No contacts were disabled or upstream physics altered.
5. Full physical end-to-end latency remains unmeasured; only the stated software-pipeline timing and exploratory response analysis are available.
6. Calibration, screenshots and derived telemetry are private local files. Nothing was committed, pushed or published.

**Stop after Phase 2.** No grasping, learning/training or publishing has been attempted.
