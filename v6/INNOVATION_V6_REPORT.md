# INNOVATION V6 — measured optional precision engine

## Decision

**Keep frozen v5 RC1 as the submission/default. Recommend merging v6 only as an opt-in recorded-data precision experiment, not a replacement default or a claim of improved live latency.** No actual branch merge/publication/license was performed. The always-on bounded optimizer remains more accurate than the hybrid.

## Preservation and reuse

Local branch `experiment/adaptive-precision-v6`, baseline commit `a3b93eda28c7cc6563ba915d948a4cc6cefda467`. No repository existed locally; a source-only Git checkpoint was initialized with an explicitly local checkpoint author. Private files were not staged. User-supplied GitHub URL was not pushed or fetched. All implementation additions live under `v6/`; frozen simulator, retargeting, model weights, calibration, data, original tests and v5 package are unchanged.

Initial baseline25 tests passed. Final suite31 tests passed (25 original +6 new). No package installation, neural retraining, new robot or grasp-controller work occurred. The314 protected-file/RC1 hashes are checked at delivery. The completed release audit was used, not repeated.

## Implementation

- Existing personal79-128-128-16 MLP supplies ordinary commands, solver initialization and every timeout fallback. No external trajectory replaces personal recordings.
- `geometry.py` evaluates the original fingertip MSE +.25 pinch MSE objective with analytic MuJoCo FK/Jacobians. Jacobian finite-difference maximum error in the initial probe was3.4e-11; tests also compare FK and targets to the exact preserved implementation. Each solver owns scratch MjData; no actual joint positions are assigned by retargeting.
- Four residual evaluations, eight-millisecond cooperative deadline, same conservative LEAP bounds. Late results are discarded for neural output. Solver runs in the vision worker online, never in the physics/render loop. A single native numerical call cannot be preempted; this is a cooperative deadline, not hard real-time scheduling.
- Modes A NEURAL, B PRECISION and C RECOVERY are explicit. Optional numerical blending Mode D was not added without evidence; transitions use shared motion safety instead of assuming angle averages preserve anatomy.
- Precision enters on pinch<.55 palm widths, neural tip proxy error>.040m or pinch proxy error>.020m, with speed<4 palm widths/s. Exit thresholds differ (.75/.030m/.014m or speed>6), with.25s minimum dwell. Bad/stale observations override normal tracking immediately.
- Quality uses finite/nondegenerate coordinates, plausible bone lengths and bone-length continuity; these are proxies, not MediaPipe per-landmark confidence. Handedness scores are not mislabeled as tracking confidence. Live unmirrored observations follow the dataset left-label reflection convention; the original app remains untouched.
- Common v6 safety: position bounds,2.5rad/s command velocity,30rad/s² command acceleration, discrete braking margins,0.3s hold then smooth neutral return. These are command bounds, not a claim of hard physical joint-speed/acceleration constraints in the unchanged position-servo model.
- Latest-only camera and vision handoffs use locks and immutable published ownership. The main thread alone mutates real MuJoCo state. Solver/vision use independent data. Old camera capture was already threaded; the new split also removes detection from the render loop.

## Protocol and coverage

Parameter and solver iteration selection used the five original validation recordings. The original five test clips have already informed previous project design: this is a reused held-out recording comparison, not a new prospective test or independent-person/session generalization. No final-test threshold tuning followed inspection of the reported results.

Each method received the same original timestamps and invalid observations. The test has921 original valid observations; the common quality gate accepted866, or94.03%. The55 additional rejected detections are not silently dropped: all921 contribute to physical metrics; raw geometry comparisons use the identical866 accepted-observation subset. These raw numbers therefore must not be substituted for the frozen all921-frame61.32/32.96/27.83mm table without the coverage caveat.

Four primary methods share the same safety filter. A fifth neural-v5-filter ablation isolates the old filter while retaining the same input quality gate. Thus that ablation is the old filter in this matched protocol, not a rerun of the complete unmodified historical webcam experiment. Actual physics steps500Hz, control roughly29.4Hz, with causal previous commands advanced to each observation. No object, wrist motion or modified finger physics.

Eight-millisecond budget and four evaluations were selected before test evaluation. Initial validation with eight evaluations had excessive deadline fallback; retained initial reports document this. Physical filter trials.045/.025/.015s were evaluated on validation only; faster settings gave little physical benefit and more acceleration norm, so.045s remained. Warm-up calls precede timing.

## Held-out results

| Method | Raw tip mm | Raw pinch mm | Physical tip mm, all921 | Mapping mean ms | Pooled mapping p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| rom | 60.065 | 40.829 | 70.976 | 0.263 | 0.258 / 0.365 |
| neural | 31.111 | 15.621 | 52.516 | 0.460 | 0.453 / 0.625 |
| optimizer | 22.064 | 7.477 | 47.610 | 3.474 | 3.476 / 4.888 |
| hybrid | 27.278 | 11.742 | 50.096 | 1.806 | 0.739 / 3.941 |

Hybrid raw tip improvement versus matched neural: 12.32%; pinch improvement: 24.84%. Hybrid mean mapping computation is48.02% lower than always optimizing, while accuracy is worse. The hybrid used neural output498 times and precision368 times; RECOVERY count135 includes original invalid plus quality-rejected observations.

Against the old-filter ablation, physical fingertip error is **49.932 -> 50.096mm**, a small worsening, not an accuracy win. Mean command acceleration vector norm is78.809 -> 63.190rad/s². Maximum per-joint discrete command acceleration is147.059 -> 30.000rad/s². Tracking RMSE against applied commands is0.03872 -> 0.03437rad. Lower actuator tracking error does not prove better human geometry.

Hybrid clean solver timeouts0/368; always-optimizer1/866. Hybrid switch rate1.075/s. Zero MuJoCo warnings, command-limit violations and observed joint-limit excursions across the evaluated methods/scenarios. All retained failure/timeout paths remain visible in logs.

## Full-controller timing

An additional paired wall timer measured already-extracted world21x3 -> canonical/joint features -> decision -> safe command, warmed10 calls per recording/method. It excludes camera, MediaPipe, physics and rendering. Safety was called once per observation for this CPU benchmark, rather than the physical replay cadence. Forward-pass, mapping and full-controller costs are not conflated.

| Method | Full world-to-command mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| neural_v5 | 1.614 | 1.863 | 2.059 |
| neural | 2.731 | 2.815 | 3.197 |
| optimizer | 5.614 | 5.661 | 6.435 |
| hybrid | 4.120 | 3.395 | 6.263 |

The old100-step Torch/Adam solver had prior measured warmed times1507–1529ms, first call4168ms. That file existed despite the prompt describing its timing as unestablished. The new solver uses the same objective/bounds but analytic Jacobians, neural initialization and a different bounded iteration policy; do not advertise an equal-accuracy solver speedup without that distinction.

## Categories and tracking loss

Fast threshold3.288 palm widths/s is the validation75th percentile. Categories are geometry heuristics and can overlap, not manually verified action labels. Open/close198 frames across5 test clips; pinch24 across3; fast315 across5. Individual-finger proxy has only12 frames from1 clip, insufficient for a general claim.

Injected failures withhold input450ms every4s (2..2.45s modulo4), identical for every method. This is simulated tracking loss, not physical camera failure. Geometry during missing input is still scored against withheld source targets where available.

- neural_v5_filter: 22/22 gaps recovered within0.1rad maximum of its clean command for2 observed samples; median delay0.578s. This is a sample-resolution command-convergence proxy, not true motion latency.
- neural: 22/22 gaps recovered within0.1rad maximum of its clean command for2 observed samples; median delay0.681s. This is a sample-resolution command-convergence proxy, not true motion latency.
- hybrid: 22/22 gaps recovered within0.1rad maximum of its clean command for2 observed samples; median delay0.680s. This is a sample-resolution command-convergence proxy, not true motion latency.

Hybrid dropout physical tip error52.521mm versus old-filter neural52.199mm. Recovery is smoother but slower; do not claim universally better robustness or latency.

## Camera and online check

No persistent exposure change was made. DirectShow640x480 delivered237 exact-hash-unique frames (~29.92fps) in the capture-only test. The320x240 request still returned640x480 and about14.96fps; MJPG also about14.96fps. Media Foundation timed out and was abandoned. These are separate same-session observations, not a controlled historical comparison.

Unattended paired online runs measured about29.71 vs29.85 unique camera fps and25.16 vs29.59 display fps. Both detected zero hands; the user deferred attended testing. Unique successfully tracked fps was therefore0. No live neural/hybrid following or hand-motion latency improvement is claimed. Original software read-completion-to-render mean51.07ms, p9566.40ms; worker mean60.12ms, p9576.88ms. Async display cadence improved but software delay worsened. The historical7–10fps camera figure cannot be attributed to a code improvement because conditions/driver/exposure were not controlled across sessions.

True exposure-to-display/physical hand-to-robot latency remains unverified. `LATENCY_PROFILE.json` includes every capture, MediaPipe, frame age, retargeter, filter, step and render measurement, including limitations and failed backend runs. The restricted camera attempt failed; the authorized local desktop camera runs completed.

## Optional prediction and ablations

A validation-only461-transition, next-observed-frame predictor screen at mean96.4ms horizon gave mean landmark error0.5475 palm widths for hold-last,0.6193 for bounded constant velocity and0.5198 for a.06s landmark lowpass. Constant-velocity overshoot proxy0.01180 palm widths. Prediction was rejected; no temporal network or runtime prediction mode was added. That landmark lowpass is a comparison using the same time constant, not the literal robot filter.

An80-validation-pose ablation at identical4-evaluation budget gave ROM-initialized optimizer29.98/13.55mm tip/pinch versus personal-neural initialization23.36/10.46mm. This supports a useful role for the learned warm start, not superiority over all optimizers or new-user generalization. Full details are in `HYBRID_ABLATION.md`.

## Demonstration, limitations and release boundary

Private video `private/v6/adaptive_precision_before_after.mp4` is29.03s at30fps with source-time-preserving pinch, faster finger movement and labeled injected dropout comparisons. Both models are freshly stepped, not posed by assigning joint states. Its privacy display uses a separate VIDEO tracker with preserved aspect ratio only to mask/draw the displayed hand; algorithm inputs remain the cached source observations. Missing display-mask detections remain black. Plots and per-frame error overlays expose subtle differences instead of hiding them. No playback speed gain or grasp success is implied.

G1/Shadow, scripted left hand and failed0/5 grasp experiments were not modified; no new cross-embodiment/grasp improvement is claimed. Existing25 tests include the preserved humanoid checks. New video/derived data are private and need separate release review; the old13-artifact list does not authorize them. MIT remains proposed only; no license/holder was guessed.

Reproduction commands and bundle smoke results are in `README.md` / `delivery_verification.json`. Results are hardware/session observations, not real-time guarantees or externally calibrated pose truth.