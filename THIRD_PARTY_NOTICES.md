# Third-party notices

## Assembled G1 with Shadow Hands

Source: https://github.com/darth-alexus/humanoid_shadow_g1/tree/249acdb565b4040fbfff7d5265629d61ae82b5d0 . The composite notice is preserved verbatim in `licenses/g1-shadow-COMPOSITE.txt` and the fetched assets. Unitree-derived content retains its BSD-style conditions; Shadow-derived content and the integration carry Apache-2.0 terms. Cite Alexander Neuhauser, “Difficulty Estimation and Robotic Piano Performance via Reinforcement Learning,” Johannes Kepler University Linz, master's thesis, 2025, as requested for the integration, and retain original Unitree, Shadow and Menagerie attribution.

The upstream asset files are not edited. Runtime model composition uses absolute local mesh paths, the source's already-fixed pelvis, Earth gravity, explicit bounded G1 PD/force parameters, display lighting and optional free-object/table geometry. These changes are described in `humanoid_control.py` and `HUMANOID_V4_REPORT.md`. The source example's low gravity is not presented as Earth-gravity balance. The model is downloaded through a pinned checksum procedure instead of silently bundling new assets.

The alternative luckyrobots G1 challenge and HumanoidBench were inspected as references. No alternative policy, code or model assets were copied into the runtime. Research snapshots remain local; their reported capabilities are not our measured performance.

## handrobot differentiable LEAP FK

`learning/leap_fk.py` is adapted from `src/handrobot/dexhand/fk.py` at https://github.com/guptabhishekumar/handrobot/tree/ae31ea4fd3778b18b30f54d1b5d614ab8686b8b5 . The full upstream MIT notice is preserved in `licenses/handrobot-MIT.txt`. Changes replace the package-specific asset import/default XML location with this project's pinned Menagerie model. The kinematic computation is otherwise retained and was validated against MuJoCo at random joint angles. The rest of handrobot was not installed or copied into our runtime. Our training/splitting/evaluation pipeline is separate from its personal-training workflow.

## MuJoCo Menagerie / LEAP Hand

The working simulation loads the right LEAP Hand model from Google DeepMind's MuJoCo Menagerie, revision `c1a4eeb85694ae1dffe33ff1797d4e528928a133`.

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/c1a4eeb85694ae1dffe33ff1797d4e528928a133/leap_hand

Model copyright: 2023 Ananye Agarwal. MIT permission notice is preserved verbatim in `licenses/leap-hand-MIT.txt` and the downloaded model folder. The upstream README and changelog are preserved. Physics, meshes, actuator parameters, and model XML are unchanged; our code sets only the render-buffer resolution and camera, supplies bounded synthetic actuator targets, steps MuJoCo, and records measurements.

Model attribution: Kenneth Shaw, Ananye Agarwal, Deepak Pathak, “LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning,” Robotics: Science and Systems, 2023. See upstream README for model conversion provenance.

## somehand

Source: https://github.com/BotRunner64/somehand/tree/c4525805a82d1d6991bbed3ef74923f5d86d23b6

Apache License 2.0 is preserved verbatim in `licenses/somehand-APACHE-2.0.txt` and the source checkout. Source files were not modified. somehand's core API was used for the unsuccessful Dex5 integration trial; the working LEAP simulation has no runtime dependency on somehand.

The external asset repository https://huggingface.co/12e21/somehand-assets declares Apache-2.0 in its model card at revision `94cedd86c275bd4c7b92d61b5e5568c2c57f6483`. A snapshot is preserved in `licenses/somehand-assets-model-card.md`. The downloaded MJCF archive contains no per-model license/readme/notice files. Its Dex5 assets remain local and excluded from version control. Do not infer that the package's code license independently grants rights to every vendor mesh. This baseline redistributes neither Dex5 meshes nor third-party recordings.

## dexhand-sim

https://github.com/jason3535/dexhand-sim was consulted for model/reproducibility research. No source code, recordings, figures, or results were copied from it; no repository reuse license was found during inspection.

## Python dependencies

Dependencies retain their license metadata in the virtual environment. The resolved wheel URLs, SHA-256 hashes, and metadata are recorded in `reports/pip_install.json`; versions and hashes are pinned in `requirements.lock`. MuJoCo, MediaPipe, OpenCV, NumPy, and the other installed libraries are upstream work, not original JARVIS-Hand contributions.

## MediaPipe hand-landmarker asset

The Phase 2 app uses Google's versioned Hand Landmarker asset:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`. The asset is downloaded locally, excluded from version control, and used via the Apache-2.0 MediaPipe Tasks package with its existing notices. Official documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python . No training or third-party demonstration trajectories are used.

Original contributions: isolated setup, pinned asset retrieval and integrity checks, standalone bounded actuator exercise, reset diagnostics, palm-relative joint estimation, calibrated ROM mapping, filtered tracking-loss handling, live visualization, telemetry and validation tests. The underlying tracker, physics engine and robot model remain upstream work. Thumb retargeting is approximate; no grasp-performance claim is made.
