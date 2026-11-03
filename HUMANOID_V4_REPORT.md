# JARVIS-Hand 4.0 — verified extension and submission status

Date: November 3, 2026, IST. Deadline: November 4, 2026, 23:59 IST.

**Selected deliverable: fixed-base G1 arm/hand control, an experimental geometry adapter for personal right-hand motion, and the preserved validated standalone learned LEAP system. Reliable humanoid grasping and learned bimanual coordination were not achieved.**

The research goal remains “Personalized Human Demonstration Learning for Contact-Adaptive Bimanual Humanoid Manipulation.” That title describes the goal, not a completed capability claim.

## Capability ledger

| Capability | Implementation/evidence | Status |
|---|---|---|
| Personal LEAP MLP | Existing 34-video training, grouped test results and five physics replays | Preserved and previously visually verified |
| Full-body G1 + two Shadow Hands | Pinned assembled model, native MuJoCo on Windows | Loaded, stepped, rendered |
| Left/right arm control | Named five-joint G1 groups, Cartesian IK, bounded PD and gravity feedforward | Physically exercised and visually verified |
| Both hands | Twenty position actuators per side, including two Shadow wrist actuators | Independent groups; opening/curling demonstrated |
| Personal motion into right Shadow Hand | Personal MLP → LEAP FK geometry → normalized Shadow ROM | Experimental transfer demonstrated, not precision-validated |
| Left-hand behavior | Scripted support/closure and feedback state controller | Explicitly programmed, not learned |
| Balance/locomotion | Pelvis fixed in source model | Not implemented or validated |
| Single-hand grasp correction | Diagnosed contact geometry and tested bounded correction | 0/5 valid holds |
| Humanoid object task | Free cylinder, table, feedback phases and real contacts | Attempted; 0/5 valid lift/holds for each configuration |
| Bimanual assistance benefit | Matched five-offset trials, with integrity gates | No grasp improvement; preliminary stabilization counts only |
| Autonomous imitation/LLM | None | Not implemented |

## Preservation and regression

The existing actual source, trained checkpoint, calibration, dataset manifest, model assets, results and submission archive were inspected. `PHASE0_REPORT.md` is in the parent workspace; the other reports are in this project. No retraining was performed. The preserved metrics remain ROM 61.32/41.12 mm, optimizer 27.83/8.93 mm, neural 32.96/16.68 mm for estimated fingertip/pinch geometry targets.

Before modification, **17 tests passed**. A source/calibration/weight backup was created at `checkpoints/pre_v4_verified.zip`, with source hashes in `checkpoints/pre_v4_hashes.json`. No Git repository existed, so this is a safe archive checkpoint rather than a commit. New modules were added alongside the original application. The original simulator, live mirror, retargeting, learning code, manipulation benchmark and weights were retained. Documentation and submission builders were extended.

The extended suite passes **21 tests**: the original regressions plus fixed-base/Earth-gravity/group separation, Cartesian IK transform checks, named side-specific adapter constraints, and a free task object without attachment constraints. `reports/v4_tests.log` records execution.

## Model selection and licenses

### A — selected: darth-alexus/humanoid_shadow_g1

Revision `249acdb565b4040fbfff7d5265629d61ae82b5d0`. Source: https://github.com/darth-alexus/humanoid_shadow_g1/tree/249acdb565b4040fbfff7d5265629d61ae82b5d0 . Archive SHA-256: `671300ff916197a4942ab010047198f47cd139ffae43b93502a3a44c1bd078ca`.

Downloaded 87 files, approximately 44.9 MB uncompressed. The composite license preserves Unitree's BSD-style conditions and Shadow's Apache-2.0 terms; integration modifications are Apache-2.0. Full notice is copied to `licenses/g1-shadow-COMPOSITE.txt`. A pinned download/integrity script and per-file hash manifest are supplied. No new framework dependencies were installed.

The compiled model has **73 scalar joints and 65 actuators**, rather than relying on the filename's “29dof” label. Each G1 arm has five actuators; each hand has twenty actuators including its two wrist joints and tendon-coupled fingers. Four tendons per hand couple middle/distal angle sums. Those transmissions are not interchangeable with LEAP joints.

Important source facts: the floating-base joint is commented out, and the example `scene.xml` uses gravity `0 0 -0.3`. Both example and robot XML were loaded and stepped. Our runtime uses **Earth gravity `0 0 -9.81` with the already-fixed pelvis**, so no locomotion/balance claim is made. Unclassified G1 position actuators defaulted to gain 1. Our separate controller explicitly sets bounded PD gain 100, velocity gain 6, arm/wrist force limits 25 Nm and other G1 actuator bounds 80 Nm; feedforward gravity compensation goes through those bounded actuators. Shadow gains and geometry remain source values, and joint control ranges are enforced by name.

The model loaded within the first few minutes and reached a concrete two-arm/two-hand rendering/control milestone well inside the 45-minute initial integration limit.

### B — reference only: luckyrobots/g1-manipulation-challenge

Revision `0e9d1f9b61772905c9f7997b527f221dc7800fc9`. Source/README and `run.py` were inspected. It supplies G1/Dex3 assets and ONNX walker/reacher policies, with MuJoCo/ONNX Runtime/OpenCV dependencies. No root LICENSE was found and GitHub license metadata is null. No code, policies or assets were reused. Repository performance and “grab” descriptions were not treated as our validated results.

### C — MuJoCo Menagerie

Inspected revision `98a70c569c4c1b6da467a20012f766ee1f8dc521`, including G1 and arm/hand families. Existing LEAP remains pinned to its prior verified revision. MuJoCo's documented MJCF `<attach>` mechanism can assemble submodels with prefixes, but was unnecessary because candidate A already assembled both hands. No untested hand attachment was substituted for the working model. Documentation: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-attach . Licenses are per model.

### D — reference only: HumanoidBench

Inspected revision `cb1189039151c8aadaaa987b442da54383c87fab`, saved in `reports/research/carlosferrazza_humanoid-bench.json`. Its MIT code carries separate upstream model attributions. The setup pins MuJoCo/MJX 3.1.6, dm_control 1.0.20, torch 2.3.1 and other training dependencies, unlike this working environment. It includes H1 with Shadow Hands and many manipulation tasks, but the whole benchmark/training stack was not installed. No policy performance was borrowed as a submission claim.

## G1 control measurements

`humanoid_control.py` provides a common left/right arm and hand interface. `scipy.optimize.least_squares` uses MuJoCo Cartesian Jacobians on scratch data to solve arm targets; only actuator controls are written during physical execution. Pixel coordinates are not used. Scratch FK/IK data is separate from actual physics state.

- **60 simulated seconds / 30,000 steps**, zero numerical warnings.
- Headless wall time **26.87 s**, real-time factor **2.23×**.
- Six-second rendered run: **177 frames**, approximately **1.14×** real-time including rendering. Export FPS is nominal 30; it is not camera FPS.
- Process memory about **475 MiB**, CPU about **99% of one core** during the headless measurement. This excludes the separate camera/training process.
- Median actual-versus-kinematic palm-command errors were approximately **31.6 mm left and 27.9 mm right**, p95 roughly 41–42 mm. Tiny numerical IK residuals do not imply equally small dynamic tracking error.
- Maximum raw joint-limit excursion approximately **0.00529 rad**; therefore zero joint-limit violations are **not** claimed for G1. Commands stay bounded. The source model uses soft constraints.
- Maximum penetration in the motion/stability demonstration approximately **1.35 mm**. Task contacts are evaluated separately.
- An actual eight-second interactive viewer run also completed.

The individually commanded actuator sweep measured a response exceeding 5 mrad from **50/50 tested channels**: ten arm and forty hand/wrist actuators, with zero numerical warnings. It is recorded in `reports/g1_actuator_sweep.json`. This validates physical response, not precision tracking or isolated single-joint motion in tendon-coupled fingers.

## Cross-embodiment contribution and limits

`ShadowGeometryAdapter` computes a representation from **learned LEAP FK keypoint positions**: segment bends, direction and splay. It then maps normalized geometry into named Shadow proximal/tendon/thumb commands, intersected with the new model's limits. LEAP's sixteen numerical joint values are never copied into the sixty-five-actuator array. Little-finger control is not invented from missing supervision. Shadow commands use actuator IDs resolved from side-specific names.

The right-hand transfer demo uses 180 sampled held-out personal predictions. It is a pose montage, not original-time replay. Left hand and arm motions are scripted and clearly captioned. Mean right Shadow joint/tendon-command tracking RMSE was approximately **0.209 rad or tendon angle-sum radians**. This mixed command-space statistic is not human fingertip accuracy. No Shadow-specific neural training or geometric precision improvement was established.

Two independent executions produced exactly equal saved joint states, desired controls and actuator lengths on this machine (`reports/g1_transfer_repeatability.json`). The rendered model movement and changing finger poses were visually inspected. Video: `private/g1_personal_transfer.mp4`. The original learned LEAP comparison remains the stronger validated learning result.

## Original grasp failure diagnosis and bounded correction

The original benchmark was instrumented without changing its control or physics. Six hundred samples record contact positions, signed normals on the object, normal/tangential forces, friction utilization proxies, joint-command error and wrist/object motion in `private/grasp_contact_trace.json`.

During the selected lift window:

- Multi-digit contact fraction: **57.9%**.
- Opposing-contact fraction, normal cosine below -0.5: **15.7%**.
- Median maximum joint-target error: **0.211 rad**.
- Friction-utilization proxy reached **1.0** on loaded contacts.

The evidence supports **poor contact opposition, controller/geometry tracking error and premature readiness based on contact count alone**, with friction saturation consistent with slip risk. Contact-count changes are not treated as verified physical slipping. The original wrist remained numerically stable; its previous peak actuator force was below the 30 N limit, so wrist force saturation was not the primary diagnosis.

`grasp_recovery.py` tries the smallest targeted alternative: kinematic fingertip targets on opposing cylinder surfaces, opposing-contact dwell before lift, and a cosine-profile lift. Object mass, friction and geometry and the original two-second success criterion remain unchanged. **0/5 valid holds** resulted. The correction usually could not establish sufficiently persistent opposing contacts, so it declined to lift. The original 0/5 baseline and prior correction results remain available; this is not reported as a successful grasp fix or learned grasp policy.

## Humanoid tabletop/assistance experiment

Scene: a free cylinder of **80 g, radius 25 mm, height 100 mm** on a table at top height 0.81 m. This is a new embodiment experiment, not a direct replication of the 35 g LEAP pregrasp benchmark. Five object X offsets `[-8,-4,0,4,8] mm` are used in both single and assisted modes. Object mass/friction/geometry are identical between compared modes. There is no weld, object child attachment, teleporting after initialization, or disabled collision shortcut.

The left controller approaches a support pose; the right controller approaches and closes to a posture derived from a personal-MLP prediction through the adapter. The state machine checks actual contacts, object motion and right-hand contact dwell, reduces left support when right contact is ready, and attempts lift/hold or recovery. The supporting left behavior is programmed, not learned. Scene observations include actual pose/velocity/contact locations/normals/forces, distinctly labeled simulator ground truth.

Initial failures included table/forearm clearance and waiting for an obstructed end-effector position despite actual object contact. A documented table-front clearance correction and slower approaches were tested, then actual contact was allowed to trigger closure. All prior configurations and failed traces were retained.

Some runs produced brief stable contact, but deeper penetrations reached 28–32 mm in preliminary attempts. Those are **not accepted as physical task successes**. Final execution aborts any trial whose contact penetration exceeds 5 mm, or joint excursion exceeds 0.02 rad. It preserves the error and counts the attempt as a failure. The final files include seven penetration aborts across ten attempts.

Final primary task outcome: **0/5 single-hand and 0/5 assisted valid lift-and-holds**. No bimanual grasp advantage was established.

The final integrity-gated suite recorded four aborts in single-hand mode and three in assisted mode. Average combined observation/control/step times were approximately 0.83 ms and 0.87 ms; mean real-time factors were about 2.26× and 1.99×. These task runs are distinct from the clean 60-second motion benchmark. Aggregate CSV/JSON and the failure-inclusive plot are `reports/v4_experiments.csv`, `reports/v4_experiment_summary.json` and `reports/v4_experiments.png`.

A separately defined preliminary stabilization criterion—upright object within 15 mm of its initial planar location, speed below 0.05 m/s, actual right-hand contact for 0.5 s, table support allowed—was met in **1/5 single and 2/5 assisted** attempts after integrity rejection. Left contact is not required by this shared criterion. These are small deterministic offset trials, not evidence of robust assistance or a statistically established improvement. The lift/hold criterion remains unchanged and unmet.

Deep-contact diagnostic traces and the final integrity-aborted results are kept separately. No contact-based failure predictor or false-alarm rate was validated. No LLM/VLM was added.

## Artifacts and exact commands

```powershell
# Run from repository root
& '.\.venv\Scripts\python.exe' scripts\fetch_g1_shadow.py
& '.\.venv\Scripts\python.exe' humanoid_control.py --viewer --seconds 30
& '.\.venv\Scripts\python.exe' humanoid_personal_replay.py
& '.\.venv\Scripts\python.exe' bimanual_task.py --trials 5 --render
& '.\.venv\Scripts\python.exe' -m pytest -q tests
```

The original fallback commands remain `neural_live.py`, `live_mirror.py`, and `replay_personal.py`. Existing weights and cached landmarks are reused. No package migration is needed; the existing requirements and CPU-learning lockfile apply.

Files are organized as:

```text
live_mirror.py / neural_live.py / simulate.py   preserved standalone entry points
learning/ / train_personal.py                  preserved learning and FK
humanoid_control.py                           fixed-base G1/Shadow model and named controls
embodiment_adapter.py                          explicit geometry-to-Shadow ROM adapter
humanoid_personal_replay.py                    recorded personal transfer experiment
diagnose_grasp.py / grasp_recovery.py           original-grasp instrumentation and bounded alternative
bimanual_task.py                               feedback-driven object/task attempts
scripts/                                      downloads, sweeps, demo and submission builders
tests/ / licenses/ / reports/                  validation, attribution, quantitative records
data/ / private/ / checkpoints/                private, ignored recordings/weights/footage/backups
submission/                                   review archives and private-artifact approval manifest
```

Demonstrations are local/private: `private/learning/personal_comparison.mp4`, `private/g1_both_arms_hands.mp4`, `private/g1_personal_transfer.mp4`, `private/g1_assisted_task.mp4`, and the captioned `private/JARVIS_Hand_v4_technical_demo.mp4`. The final video is 1,111 frames at 15 FPS, 1280×720, approximately 74 seconds. Captions distinguish learned, scripted, fixed-base and failed/integrity-aborted attempts.

The public code/results archive excludes personal videos and weights. `submission/private_artifact_manifest.json` proposes representative actual recordings, checkpoint and private demonstration files with hashes for explicit review. **Publishing those private artifacts, pushing GitHub or uploading the submission requires user approval. Nothing has been published or sent.** The personal sample may include faces, background details or original audio and must be reviewed before release.

## Final interpretation

Implemented/tested/visually verified: full-body model loading, fixed-base two-arm motion, both-hand articulation, an explicit personal-geometry adapter, and retained learned LEAP replay. Physically exercised but unsuccessful: cylinder grasp recovery and bimanual lift/transfer. Experimental: precision embodiment transfer and assistance effects. Not implemented: dynamic balance, locomotion, learned left-hand coordination, autonomous robot imitation policy or LLM planning.

Freeze the preserved learned-hand submission and the clearly labeled fixed-base extension rather than representing this prototype as solved humanoid manipulation.
