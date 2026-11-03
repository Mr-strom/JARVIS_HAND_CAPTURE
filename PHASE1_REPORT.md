# Phase 1 results — November 2, 2026 IST

**Outcome: PASS using MuJoCo Menagerie right LEAP Hand.** The robot visibly opens and curls in the inspected rendered frames and exported animation. The interactive viewer also ran successfully. Work stops at Phase 1; no camera was opened in this phase.

Project: `C:\path\jarvis-hand`.

## Environment

Created `.venv` using existing Python 3.11.9 x64. `sys.prefix` points to this project environment; `sys.base_prefix` remains the existing Python311 installation. No global package or system settings were changed.

Installed direct pins: MuJoCo 3.11.0, NumPy 2.2.6, MediaPipe 0.10.32, OpenCV Contrib 4.12.0.88, SciPy 1.15.3, PyYAML 6.0.3, pytest 8.4.2, and somehand 0.3.0 from source revision `c4525805a82d1d6991bbed3ef74923f5d86d23b6`. No dependency conflicts occurred; `pip check` passed. Imports of MediaPipe Tasks, OpenCV, MuJoCo, and somehand core succeeded. MediaPipe inference has not been tested or integrated.

`requirements.lock` pins the 32 resolved PyPI wheels with hashes. `reports/installed_versions.txt` records the actual installed environment; `reports/pip_install.json` and `reports/somehand_install.json` preserve installation metadata. The optional somehand source build has separate pinned-source provenance, rather than a reproducible-wheel claim.

## Dex5 investigation and errors

Phase 1 began at approximately 18:42:47 IST. Dex5 was rejected approximately eight minutes later, comfortably within the maximum 45-minute integration window.

1. Initial asset layout mismatch caused a `FileNotFoundError`: somehand validates its configured `assets/mjcf/...` path before a caller can override it. Corrected extraction to the expected project-local layout. Original error retained in `reports/initial_path_error.json` and `reports/first_step_test.log`.
2. Core API construction then succeeded: Dex5 right, 20 DoF, 20 actuators, 15 vector constraints. No webcam/CLI or `termios` path was used.
3. Raw asset physics produced a BADQACC warning at 0.010 s. After 1,000 requested steps, simulation time had reset to 0.010 s and maximum absolute velocity was approximately 6.69 million rad/s. Initial contacts: 13; maximum initial penetration: 0.0247459 m. A finite final array did not establish stability. See `reports/environment_probe.json` and `reports/probe.log`.
4. Explicit implicitfast integration, 1 ms steps, damping 0.05, armature 0.0001 and position gain 3 stabilized a six-second trial. No warnings or joint-limit excursions occurred, but ~24.75 mm penetration persisted, and only 19/20 joints exceeded the 0.05 rad motion threshold. This trial failed acceptance and is preserved in `experiments/dex5_attempt.py` and `reports/dex5_tuned_metrics.json`.
5. Switched to LEAP instead of disabling collisions or treating the Dex5 viewer as a validated physical model. somehand source and downloaded XML were not modified.

The external asset archive checksum matches the publisher's LFS SHA-256. Its model card declares Apache-2.0, but no per-model notices were found in the archive. Dex5 assets remain local and ignored by version control.

## Working model and controller

Menagerie right LEAP at revision `c1a4eeb85694ae1dffe33ff1797d4e528928a133`, scene `leap_hand/scene_right.xml`. MIT model license is preserved. All 22 files in the downloaded LEAP folder have recorded hashes and pinned URLs.

The standalone script uses the upstream 2 ms timestep and implicitfast integrator. It preserves model masses, collisions, damping, gains and limits. It varies only bounded actuator targets through `data.ctrl`, then calls `mj_step`. No per-frame writes to `qpos`, no object, no attachment, and no prerecorded third-party trajectory are involved. Motion is explicitly synthetic.

## Executed results

| Check | Result |
|---|---|
| Initial load and actuator cycle | PASS, 16/16 joints moved; measured ranges ~0.441–0.702 rad |
| Stability | PASS, 60 simulated seconds, 30,000 steps |
| State/time checks | Finite at every checked step; no unexpected simulation resets |
| MuJoCo warnings | 0 for all warning categories |
| Control-limit violations | 0 |
| Joint-limit excursions | 0 rad maximum; 0 steps beyond 0.0001 rad tolerance |
| Initial and trajectory penetration | 0 m; no contacts during this gentle cycle |
| Maximum absolute joint velocity | 3.3505 rad/s |
| Headless execution | 60 simulated seconds in 4.50 wall seconds; includes Python validation overhead |
| Render | 181 frames, 960×720, nominal 30 FPS output; render/simulation loop 3.27 s for six simulated seconds |
| Visual inspection | Open and curled PNGs show distinct finger poses; video decoded successfully |
| Interactive viewer | PASS, 8 simulated seconds in 8.0004 wall seconds |
| Repeatability | Two independent six-second runs: maximum difference 0 for saved samples, final qpos, and final qvel |
| Automated checks | 3 passed in 2.30 s |

Render loop time excludes model loading and encoding. Nominal video FPS is an export setting, not measured live tracking throughput. The viewer result is a short local smoke test. Repeatability applies only to this fixed model, control sequence, and installed environment; it is not yet personal-data replay.

## Deliverables

- `simulate.py`: working standalone simulator and metric/video export.
- `requirements.in`, `requirements.lock`, install reports and version inventory.
- `scripts/fetch_leap.py`: pinned, integrity-checked model download preserving notices.
- `scripts/fetch_assets.py`, `scripts/probe_environment.py`, `experiments/dex5_attempt.py`: retained Dex5 investigation.
- `tests/test_phase1.py`: mapping/bounds, repeatability/motion, and decoded-video tests.
- `reports/leap_demo.mp4`, `reports/leap_demo.gif`, open/curled snapshots and actual control/state samples.
- `README.md`: exact setup, viewer, render and validation commands.
- `THIRD_PARTY_NOTICES.md` and `licenses/`: attribution and license copies.

## Limits and next gate

No webcam integration, human landmarks, calibration, personal recordings, retargeting quality, tracking latency, object dynamics, or grasp success were tested. LEAP lacks an independent little finger. Contact behavior still needs a separate physical test before any grasp claim. No GitHub push or publication occurred.

Phase 1 is complete. Stop here until the user authorizes the next phase.
