# Adaptive Precision Retargeting Engine — initial experiment

Branch: `experiment/adaptive-precision-v6`; source checkpoint `verified/v5-local`. No local repository existed, so it was initialized with a source-only checkpoint using an explicit local checkpoint author, not the owner's identity. No remote push/fetch or license grant. Existing GitHub URL could not be read through the browser; it is not a blocker for local experiments.

Frozen v5 RC1 remains at its original path, SHA256 `87d38204b73cfc46d2f011f568de412bc0d21f148ef2d71100d251b5fa42b2e8`. Baseline: **25 tests pass in11.76s**. New code/reports stay under `v6/`; private trajectories/videos under ignored `private/v6/`. Existing recordings, checkpoint, calibration, physics and v5 packages are read-only inputs.

Reuse: `learning/runtime.py`, `train_personal.py` target/objective conventions and Network; `retargeting.py`; `learning/leap_fk.py`; `simulate.py`/`physics_guard.py`; cached `reports/dataset_audit.json` and personal NPZs. No retraining or dependency installation planned.

Findings: capture already uses a bounded latest-frame thread; detection/rendering share the main loop. Archived camera measurements found repeated MSMF frames, not real throughput improvement. The old100-step Torch optimizer has measured warmed single-frame runtimes1507–1529ms (`reports/inference_benchmark.json`), not a viable synchronous control budget. Forward-pass timing excludes geometry/adapter overhead.

Experiment: analytically differentiated MuJoCo FK with the existing fingertip/pinch objective, bounded small solve initialized by the personal MLP; confidence/continuity checks; hysteretic pinch precision mode; bounded hold/return on unreliable input; velocity/acceleration-limited execution. Live optimization and vision workers use separate scratch state and latest-only queues; only the main thread mutates actual simulation data.

Selection uses original validation recordings; the five final test clips have already informed earlier work and are disclosed as a reused test set. Compare ROM, MLP, bounded optimizer and adaptive control under identical timestamps/safety filtering, plus deterministic injected dropouts. Measure raw geometry and physical tracking separately, warmed p50/p95 computation, timeout rate and mode switching. A simple prediction ablation is optional; do not train another network without evidence.

Merge only if regression/safety checks pass and a meaningful measured tradeoff survives. No new humanoid or grasp tuning is planned; v5 remains the fallback.
