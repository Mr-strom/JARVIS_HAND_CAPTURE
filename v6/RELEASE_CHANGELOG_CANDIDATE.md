# Candidate changelog — local experiment, not approved release

- Add isolated Adaptive Precision Retargeting Engine with preserved personal MLP, analytic bounded geometry refinement, hysteresis/dwell and timeout fallback.
- Add geometry/continuity quality proxies and command position/velocity/acceleration safety with hold/neutral tracking-loss recovery.
- Add optional latest-only vision worker; actual MuJoCo stepping remains on the main thread. Unattended display throughput improves, software frame age worsens; attended live verification is pending.
- Add validation/test comparisons, explicit coverage, timing scopes, geometric categories, deterministic dropout trials and prediction/filter/warm-start ablations.
- Add six tests and a29.03s private before/after video using genuine personal source timestamps and fresh physics.
- Preserve all old source/runtime entry points, weights, calibration, cached data/assets, failed grasp experiments and v5 RC1. No retraining, dependency installation, new robot, grasp controller, LLM/VLA or locomotion.
- Recommend optional precision functionality only; v5 stays default and ready-to-submit. No license applied; no artifact publication, GitHub push or submission upload authorized/performed.
