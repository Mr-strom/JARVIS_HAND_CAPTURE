# Release licensing and attribution review

The published repository includes the root MIT License for original JARVIS-Hand code, copyright 2026 Mr-strom. Personal artifacts were separately approved for public inclusion in this repository. Public access does not, by itself, grant a blanket downstream reuse license for personal videos, derived landmarks, calibration, or the trained checkpoint.

## Original project license

The owner selected and applied **MIT for original JARVIS-Hand source code**. See the root [LICENSE](../LICENSE) and [OSI MIT text](https://opensource.org/license/mit). This does not relicense third-party code, robot assets, personal videos, calibration, landmarks or trained weights.

The adapted `learning/leap_fk.py` also inherits upstream MIT terms; its Abhishek Kumar Gupta notice remains in `licenses/handrobot-MIT.txt`. The root MIT does not alter upstream obligations. No blanket dataset/media/checkpoint reuse license is selected here. Public inclusion permission and permission for downstream reuse are separate decisions.

## Reused code and robot assets

| Component | Pinned source / role | Terms and retained attribution | Candidate disposition |
|---|---|---|---|
| handrobot FK | `guptabhishekumar/handrobot` `ae31ea4fd3778b18b30f54d1b5d614ab8686b8b5`; adapted `src/handrobot/dexhand/fk.py` only | MIT, copyright 2026 Abhishek Kumar Gupta; full `licenses/handrobot-MIT.txt`; local path changes described in source/notices | Adapted FK source + full notice included; no full upstream training stack |
| LEAP Hand | Menagerie `c1a4eeb85694ae1dffe33ff1797d4e528928a133`, `leap_hand/` | MIT, copyright 2023 Ananye Agarwal; `licenses/leap-hand-MIT.txt`; cite Shaw, Agarwal, Pathak, RSS2023 | Assets not embedded; pinned downloader/integrity manifest and notices included |
| G1 + two Shadow Hands | `darth-alexus/humanoid_shadow_g1` `249acdb565b4040fbfff7d5265629d61ae82b5d0`; archive SHA256 `671300ff916197a4942ab010047198f47cd139ffae43b93502a3a44c1bd078ca` | Unitree BSD-style 3-clause conditions; Shadow Apache-2.0, copyright 2022 Shadow Robot Company; integration Apache-2.0. Retain `licenses/g1-shadow-COMPOSITE.txt`, Menagerie attribution and Neuhauser2025 JKU thesis citation. No endorsement or trademark rights implied | Assets not embedded; pinned downloader and composite notice included. Runtime Earth gravity, fixed pelvis, bounded actuator parameters documented |
| somehand | `BotRunner64/somehand` `c4525805a82d1d6991bbed3ef74923f5d86d23b6`; historical unsuccessful Dex5 trial | Apache-2.0 full notice `licenses/somehand-APACHE-2.0.txt`; original package not modified | Optional historical caller/source retrieval scripts only; not needed by submission runtime; no package binaries bundled |
| somehand Dex5 asset pack | Hugging Face `12e21/somehand-assets` `94cedd86c275bd4c7b92d61b5e5568c2c57f6483` | Model card declares Apache-2.0; no per-vendor mesh rights verified inside archive. Model-card snapshot retained | **Excluded**: no Dex5 meshes or asset archive redistributed |
| MediaPipe tracker code | Installed MediaPipe0.10.32 via locked wheel | Apache-2.0 package metadata; wheel retains upstream notices | Wheel not bundled |
| MediaPipe tracker weights | Google float16 HandLandmarker version1, SHA256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | Official Hand Tracking model card states Apache-2.0. Track this model-specific evidence rather than deriving weight rights solely from the package license. The `.task` contains palm detector and landmark detector TFLites, no separate license member | **Download only**, checksum pinned; weights not bundled |

MediaPipe model evidence: [official task documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), [official Hand Tracking model card, licensing on page2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf). The card describes the model family; it is not a cryptographic attestation of the exact task bundle. Keeping the bundle download-only avoids redistributing it in this candidate. No rights are claimed over its external pretraining data.

## Installed software and tooling

`reports/release_dependency_inventory.json` inventories **40 installed non-bootstrap distributions**, exact versions, license fields/classifiers and checksums of their installed notice files. These are upstream metadata, not a blanket relicensing opinion. The source candidate includes requirements/locks rather than vendoring Python wheels or native libraries.

| License family / upstream metadata | Packages |
|---|---|
| Apache-2.0 | absl-py2.5.1, etils1.14.0, flatbuffers25.12.19, mediapipe0.10.32, mujoco3.11.0, opencv-contrib-python4.12.0.88, tzdata2026.6; somehand0.3.0 established by its source notice |
| MIT / MIT-0 / MIT-CMU | cffi2.1.1 (MIT-0), filelock4.2.4, fonttools4.68.0, glfw2.11.0, iniconfig2.3.1, pillow13.0.0 (MIT-CMU), pluggy1.7.0, pyparsing3.3.3, pytest8.4.2, PyYAML6.0.3, sounddevice0.5.6, zipp4.1.2 |
| BSD-style | colorama0.4.6, contourpy1.3.3, cycler0.12.1, fsspec2026.9.0, Jinja2 3.1.6, kiwisolver1.5.1, MarkupSafe3.0.3, mpmath1.3.0, networkx3.6.1, numpy2.2.6, pycparser3.12, Pygments2.21.0, PyOpenGL3.1.10, python-dateutil4.0.0, scipy1.15.3, sympy1.13.1, torch2.5.1+cpu |
| PSF-style / Matplotlib terms | matplotlib3.11.2 has its own Matplotlib/PSF-style license and bundled notices; typing_extensions4.16.0 PSF-2.0 |
| Alternative licensing | packaging26.3: Apache-2.0 OR BSD-2-Clause |

Native libraries and fonts bundled within upstream wheels can have additional notices (e.g. numerical libraries, OpenCV components, Torch components, GLFW, Matplotlib fonts); those are retained by the wheels and referenced by the inventory. The candidate ships no wheels, runtime DLLs, font files, Python interpreter or GPU drivers. A future binary/standalone distribution needs a separate bundled-notice review.

Python3.11.9 and pip are environment/setup tools, not distributed project code. FFmpeg/ffprobe are local audit tools, not bundled; video encoding here uses installed OpenCV. No FFmpeg executable or other third-party binary is put in the release candidate. No cloud/paid inference service is used.

## Data and research-only references

The MLP is initialized/trained on the operator's own 34 recordings with geometry-derived robot targets. No FreiHAND, InterHand2.6M, HO-3D, DexYCB or third-party demonstration trajectories were downloaded or used to train this MLP. The Google tracker is pretrained upstream; that pretraining is distinct from our personal dataset. Do not claim that no upstream pretrained model is used.

The candidate proposes personal **derived** media, landmarks, calibration and checkpoint; all require owner consent. Raw originals and private research snapshots are excluded. Source hashes and relative clip timestamps support provenance without redistributing source filenames or absolute paths. They can still link artifacts to an original held elsewhere.

Research-only projects: `dexsuite/dex-retargeting`, `qqsq12321/AnyDexRetarget`, `personalrobotics/mj_manipulator`, `jason3535/dexhand-sim`, `luckyrobots/g1-manipulation-challenge` and `carlosferrazza/humanoid-bench`. No code, policy, recordings or model assets from these references entered the working runtime. dexhand-sim and luckyrobots lacked a verified reusable root license during inspection; nothing from them is redistributed. Pinned research details and compatibility limits remain in `RESEARCH_REUSE.md` / `HUMANOID_V4_REPORT.md`.

## Notice checks and residual limits

Existing full MIT/Apache/composite notices are retained verbatim. Model download procedures preserve asset licenses and readmes. The current code uses bounded runtime composition; source mesh/XML files are unchanged. Descriptions in early Phase1 notices about the original actuator-only scene are historical; later wrist/task/controller changes are documented in v4/final reports. The underlying engines, models, tracker and FK adaptation are not claimed as original work.

This audit inventories observed sources and license evidence. It does not establish ownership of undisclosed material or replace the owner's release decision. No new license was installed or applied.
