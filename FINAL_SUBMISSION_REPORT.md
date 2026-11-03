# Final submission freeze — JARVIS-Hand

Prepared November 3, 2026, IST. Deadline November 4, 2026, 23:59 IST.

## Submission claim

Personalized CPU neural retargeting for LEAP is the strongest verified contribution. Fixed-base G1/Shadow provides an executed cross-embodiment experiment, not a general-purpose learned humanoid policy. This final experiment finds that the trained LEAP pathway **does not improve Shadow fingertip alignment** through the existing adapter. Physical grasp attempts remain unsuccessful.

No retraining, new package installation, robot replacement, GitHub push or upload occurred in this final stage. The optional temporal predictor was omitted to secure the tested submission. No LLM, VLA or locomotion was added.

## Preservation and execution

Actual v4 source, reports, model assets, predictions, weights, raw-video proposal and demonstrations were inspected. The initial regression passed **21/21** tests. Four existing videos decoded completely: personal comparison 473 frames, Shadow transfer 180, assisted task 265 and v4 technical demo 1,111.

`checkpoints/pre_final_verified.zip` is a recoverable full project-artifact checkpoint: 482 files, about 699 MB of source assets before compression, including raw recordings, model assets, weights, calibration, cached features, reports and previous submission files. It excludes the virtual environment, redundant downloads, third-party source checkout and earlier backups. Restore into a separate directory and use the pinned environment instructions; do not overwrite the working project indiscriminately. Its SHA inventory is `checkpoints/pre_final_hashes.json`. Original Python modules, assets, data, calibration, weights and videos are checked again at delivery. The original README is retained at `docs/README_V4_ARCHIVE.md`.

The new extended suite passes **25 tests**. New checks cover normalized-coordinate invariance, Shadow curl sign and coupled tendon reference, three bounded side-specific pathways on the same input, invalid input rejection, and unchanged stable-hold thresholds. The initial new test incorrectly assumed MCP separation was purely lateral; inspection showed the forward offset, and the expected 3D span was corrected to 46.581568 mm. No benchmark metric or model was changed to satisfy that test. `pip check` reports no broken requirements.

An extracted public source archive separately passed **23 tests / 2 expected skips** without private weights or calibration. It reused this installed venv and copied the already verified model assets into an isolated smoke directory; it was not a clean dependency installation or a new network-download test. The private restoration was then checked separately before the final package freeze.

## Cross-embodiment protocol

Script: `cross_embodiment.py`. Original v4 `embodiment_adapter.py` and `humanoid_control.py` are reused unchanged. All three pathways receive the same cached, valid observations from recording IDs 5, 10, 24, 29 and 33, the original five held-out clips. There are **921 observations per pathway** and **97.014 simulated seconds per pathway**. No parameter was selected against these test results.

1. **Direct ROM:** actual human joint features and existing personal calibration directly populate named Shadow proximal, tendon-sum, splay and thumb commands. This is a simple analytical baseline, not an optimization method.
2. **Geometry from ROM LEAP:** original calibrated ROM LEAP predictions feed LEAP FK and the unchanged v4 Shadow geometry-to-ROM adapter.
3. **Geometry from neural LEAP:** the actual preserved 28,816-parameter checkpoint consumes 79 features, predicts LEAP commands, and feeds the same FK/Shadow adapter.

The little finger and unused thumb channels stay neutral. Arm targets stay at the initial configuration under bounded physical control; gravity can still cause dynamic tracking error. Left-hand learning is absent. The model includes the whole anchored humanoid, but the comparison isolates its right hand with no object.

At each source timestamp, physical state is evaluated after simulating the interval under the **previous** observed command. This is causal, not an anticipatory current-frame command applied into the past. Simulation timestep is 2 ms; source timestamps are rounded up to the next integration step. Inputs are held across invalid-observation gaps. The original live application's tracking-loss fallback remains separate and unchanged; this offline protocol is not a replacement live controller.

### Frames and metrics

Human canonical x points from little MCP toward index/thumb, y points along the projected wrist-to-middle MCP direction, and z is their cross product. Human points have wrist origin and index/little MCP span one before evaluation. Shadow world positions transform into the moving right-palm frame, then `[x,z,-y]` aligns its axes to human thumb-side/forward/curl. Both point sets are recentered at their mean index/middle/ring MCP and divided by their own index/little MCP span; Shadow's 3D span is 0.046581568 m. No scale, rotation or offset is fit to test poses.

- Fingertip error: mean Euclidean distance between corresponding normalized index/middle/ring/thumb tip endpoints.
- Direction agreement: mean cosine between normalized base-to-tip vectors for those four digits.
- Pinch error: mean absolute difference between normalized thumb-to-index/middle/ring tip distances.
- Shadow tips are distal local +z mesh-axis endpoint proxies, not calibrated tactile-pad contact centers. Human tips are monocular MediaPipe estimates. These metrics include embodiment proportions and chosen correspondence; they are not external ground truth.
- Dynamic tracking: RMSE of actual actuator length minus the rate-limited command, divided by each actuator's control span. This makes joint angles and tendon angle sums dimensionless without confusing the two transmissions.
- Joint limits: count each right-hand joint/physics-step excursion above 1e-6 rad and report maximum raw excursion. Such events are correlated soft-constraint samples, not independent failures. Commands are bounded separately.
- Stability: inspect warnings, finite state, reset conditions and every-step maximum contact penetration. Numerical stability does not itself establish physical task success.

Kinematic reference metrics are reported separately. Scratch data set joint-transmitted commands and equally divide each tendon J0 target into J1/J2. This explicit approximation is not imposed on real dynamic physics, which determines the coupled-joint distribution. The original v4 demo's mixed radian tracking RMSE is not reused as a cross-hand geometry metric.

### Results: equal-weight recording means

| Pathway | Physical tip error (palm widths) | Direction cosine | Pinch error (palm widths) | Normalized actuator tracking RMSE |
|---|---:|---:|---:|---:|
| Direct ROM | 0.917776 | 0.763420 | 0.495508 | 0.065539 |
| ROM → LEAP FK → Shadow | 0.895194 | 0.764395 | 0.550318 | 0.030715 |
| Personal MLP → LEAP FK → Shadow | 0.944445 | 0.737731 | 0.543908 | 0.029172 |

The neural pathway is worse than both alternatives for physical fingertip alignment and direction. Its pinch error is slightly lower than geometry-from-ROM but higher than direct ROM. It tracks its smaller mapped actuator commands more closely; that does not establish better human geometry. Differences are descriptive five-recording means, not statistical significance or generalization to unseen people/hands.

| Pathway | Kinematic tip error | Kinematic direction | Kinematic pinch error |
|---|---:|---:|---:|
| Direct ROM | 0.741406 | 0.865943 | 0.425574 |
| ROM → LEAP FK → Shadow | 0.754412 | 0.845327 | 0.453273 |
| Personal MLP → LEAP FK → Shadow | 0.809506 | 0.824984 | 0.452993 |

Both kinematic and physical evidence limit the adapter's effectiveness. Likely causes include training solely for LEAP's geometry, the analytical adapter's loss of fingertip/pinch information, heuristic thumb mapping and Shadow's coupled anatomy. These are diagnoses to investigate, not proven causal ablations.

| Pathway | Raw joint-step excursions | Maximum excursion (rad) | Maximum penetration (m) | Numerical warnings | Command violations |
|---|---:|---:|---:|---:|---:|
| Direct ROM | 85,163 | 0.0094776 | 0.000256971 | 0 | 0 |
| ROM via LEAP | 60,226 | 0.0006648 | 0.000233068 | 0 | 0 |
| Neural via LEAP | 61,291 | 0.0006171 | 0.000217428 | 0 | 0 |

Landmark-to-command mean CPU time in the portable reproduction run was **0.642 / 7.654 / 7.820 ms**, respectively. This includes canonical/joint-feature extraction and whichever ROM/MLP/FK operations the pathway executes. It excludes MediaPipe, camera buffering, simulation, control smoothing, rendering and display. It is not exposure-to-robot-motion latency. Per-recording p95 values and timing scope are recorded in JSON; aggregate timing is a mean of recording means. Hardware load and execution order can affect it.

The final portable pack contains only cached arrays and anonymous clip indices, not original RGB or absolute source paths. An actual second evaluation from that pack reproduced **all 42 non-timing summary values exactly on this machine**. Timing varied as expected. This validates the derived-data replay path, not cross-platform bitwise determinism. See `reports/cross_portable_validation.json`.

Evidence: `reports/cross_embodiment_results.json`, `reports/cross_embodiment.csv`, `reports/final_evaluation.png`; private per-frame geometry records under `private/final/cross_clip_*.json`.

## One bounded grasp recovery

The source of truth was v4: original diagnosis showed poor opposing contacts, tracking error and friction saturation; the previous surface-fit controller also failed, and deep humanoid contacts were rejected. The earlier recovery work was read from actual files, not assumed to have run separately.

`final_grasp_recovery.py` adds **one controller configuration**, with a 20-minute execution cap. It was evaluated on the same five original 35 g cylinder offsets `[-8,-4,0,4,8] mm`. One repeated execution added telemetry only, without controller retuning. This is a seeded palm-pregrasp benchmark, not a tabletop pickup or direct comparison to the different 80 g humanoid object scene.

A subsequent center-trial render enlarged the camera view only; all six checked physical outcome/tracking values matched its recorded trial exactly (`reports/final_grasp_render_validation.json`). It is not an additional tuned controller or success trial.

Changes are physically explicit: thumb axial preposition; actuator-driven wrist X alignment; independent digit closure based on measured contact forces; finger motor torque bounds ±0.4 Nm; original position gain 3 with increased velocity damping 0.08; bounded free-motion tracking correction; opposed contact normal cosine below -0.5 and thumb contact for 0.2 s before lift; lift speed 15 mm/s; recovery on contact loss. Wrist motors retain ±30 N. Geometry, mass, object friction, collision masks and object free joint are unchanged. No object state is modified after initialization. The original scripts and benchmark results are untouched.

Success remains ≥2 cm lift above settled reference, ≥2 distinct contacting digits, no table contact and object speed <0.15 m/s **continuously for 2 s**. Safety gates reject any trial with penetration >5 mm or joint excursion >0.02 rad. Passing the safety gate is not counted as grasp success.

Actual outcome: **0/5 successes; 0 s maximum valid hold**. Maximum transient lift was **20.380 mm**, maximum penetration **1.800 mm**, maximum joint excursion **0**, and there were **zero numerical warnings / integrity aborts**. Mean joint-command tracking RMSE was **0.012106 rad**, but opposing-contact fraction was only **0.006714** averaged over the full trajectories. All five trials recovered without establishing sustained opposing contacts. The gradual lift phase was therefore implemented but not reached in these trials; no claim is made that its effect was validated.

Peak summed digit contact force reached **234.19 N including seeded-object impact** despite motor limits. Torque ceilings do not cap collision impulse forces. The old 0.211-rad median maximum target error and the new mean RMSE differ in scope/statistic; they are not a matched precision-improvement measurement. The archived v4 0/5 recovery and the unchanged original baseline remain available. No grasp-success improvement or strong contact-force safety claim is supported. Manipulation work is frozen after this attempt.

Evidence: `reports/final_grasp_recovery.json` (including phase/contact traces), `reports/final_grasp_summary.json`, `reports/final_grasp.csv`, `private/final/grasp_recovery.mp4`. Original humanoid results remain 0/5 single and 0/5 assisted with seven penetration rejections.

## Demonstrations and privacy

The final **48.6 s, 1280×720, 15 FPS, silent** video is `private/final/JARVIS_Hand_final_demo.mp4`. Its manifest names exact source clips and intervals. Segment A combines actual personal frames with LEAP ROM/neural physical replay. Segment B shows the actual full fixed-base G1 personal-pose montage and the synchronized three-pathway comparison. Segment C shows actual unsuccessful free-cylinder contact closure and recovery, with force/opposition/hold overlays.

The v4 montage samples different held-out poses rather than reproducing original timing, and its arms/left hand are scripted. The new comparison uses matching source frame indices; presentation frames are exported nominally at 12 FPS before composition. Gaps can be compressed. Actual evaluation preserves source timestamps. None of these export rates is a measured webcam frame rate.

Both original videos proposed by v4 visibly contain a face, body and private room. Their original audio was not transcribed or cleared. They are **not recommended for release**. A local derivative keeps only hand-hull RGB, replaces everything outside it with black, enlarges the hand and omits audio. The minimum proposed sample covers source timestamps 5–13 s, with eighty cached samples exported at nominal 12 FPS (6.67 s presentation duration). Its matching JSON preserves original timestamps and canonical features. All originals remain intact.

The demonstration also masks/zooms its human panes. Redaction detections succeeded on 336/473 LEAP presentation frames and 109/144 transfer frames; missing detections deliberately show black. Those counts describe a new privacy pass, not the original dataset's tracking coverage. Landmark annotations in presentation may come from that privacy pass; model control uses the original cached valid observations. Contact sheets and selected frames were inspected. Cropping/masking reduces disclosure but does not guarantee anonymity; artifacts still require explicit owner review and approval.

All exported videos are written by a video-only encoder. Final technical validation checks audio-stream absence and full video decoding. No original conversations are included in these derivatives. No raw-video transcription or complete forensic privacy audit is claimed. Personal landmark geometry, calibration and weights are still personal derived artifacts.

## Submission packaging and release boundary

`submission/final/JARVIS-Hand-final-public-code-results.zip` contains runnable original/new source, tests, locked requirements, model download scripts, documentation, third-party notices, synthetic LEAP test video and aggregate results. It excludes raw data, private media, checkpoints, calibration, personal weights and landmark arrays. Public copies of dataset split metadata remove source/cache paths. A path/secret-pattern scan is a heuristic, not a secret-absence guarantee. Owner original-code licensing remains a decision before public release; third-party licenses are preserved unchanged.

`submission/final/JARVIS-Hand-private-review-bundle.zip` is **private and awaiting approval**: one silent hand-masked sample, its landmarks, the unchanged trained checkpoint, calibration, all five held-out landmark caches for evaluation, and the final masked demonstration. The companion private artifact manifest records content hashes and approval=false. It does not include the 34 raw recordings. It is enough to run the trained network and held-out cross-embodiment evaluation, not enough to retrain the full reported personal model. Dataset publication permission and checkpoint permission remain separate from code publication/push/upload permission.

Exact setup and run commands are in README. Models are fetched at existing pinned revisions; the current venv requires no changes. The final delivery audit records preservation, archive validation, extracted-archive smoke results, video checks and unresolved release decisions. Nothing has been published or uploaded.
