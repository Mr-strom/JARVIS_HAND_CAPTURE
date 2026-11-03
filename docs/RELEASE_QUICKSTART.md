> Final candidate update: see [integration report](../FINAL_INTEGRATION_REPORT.md). The default v5 commands below are preserved; optional v6 commands and tests are in the root README. The demo has been replaced and needs new approval.

# JARVIS-Hand v5 release-candidate quickstart

**Local review candidate only. Personal artifacts, original-code licensing and publication are awaiting owner approval.** No approval is implied by possession of this ZIP. No new robot or learning features were added during the release audit.

## Installation in a fresh extracted directory

Requires Windows11, CPython3.11.9 x64 and an OpenGL driver. Run these commands from the extracted candidate root. Keep the existing working project/environment separate.

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -c "import sys; assert sys.version_info[:3] == (3,11,9), sys.version"
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes --only-binary=:all: -r requirements.lock
& '.\.venv\Scripts\python.exe' -m pip install --require-hashes -r requirements-learning.lock
& '.\.venv\Scripts\python.exe' scripts\fetch_leap.py
& '.\.venv\Scripts\python.exe' scripts\fetch_g1_shadow.py
& '.\.venv\Scripts\python.exe' scripts\fetch_hand_landmarker.py
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pytest -q -rs tests
```

Before personal artifacts are restored, the suite should pass 23 tests and skip two: trained-checkpoint validation and the three-pathway test requiring checkpoint/calibration. This does not mean the private data were approved for distribution.

The candidate includes pending-approval artifacts under `release_artifacts/`, a directory that is not ignored by Git. Restore their checksummed bytes into the frozen runtime's expected private paths for local evaluation:

```powershell
& '.\.venv\Scripts\python.exe' scripts\restore_release_artifacts.py
& '.\.venv\Scripts\python.exe' -m pytest -q -rs tests
& '.\.venv\Scripts\python.exe' scripts\release_leap_metrics_smoke.py
& '.\.venv\Scripts\python.exe' scripts\release_sample_smoke.py
& '.\.venv\Scripts\python.exe' cross_embodiment.py --dataset private\final\heldout_landmarks
& '.\.venv\Scripts\python.exe' humanoid_control.py --viewer --seconds 30
```

The restoration script refuses to replace an existing different calibration, checkpoint or data file. Twenty-five tests should pass after restoration. The RGB sample smoke executes fresh MediaPipe detection → existing personal MLP → LEAP actuators and physics; it is not a new accuracy or latency benchmark. The held-out LEAP checker recomputes ROM/MLP predictions and all three geometry metrics using cached optimizer reference commands. It does not rerun optimizer training. The Shadow evaluation uses all five original held-out landmark recordings.

For the original webcam mirror, run `live_mirror.py`; C collects calibration and Q exits. `neural_live.py` consumes the restored checkpoint. The original ROM webcam behavior was visually verified; the neural live mode had a limited smoke test, while neural pose following was established by recorded replay. No stronger live neural claim is introduced here.

## Demonstration and sample links within the candidate

- [Final 54-second demonstration](../release_artifacts/JARVIS_Hand_final_demo.mp4)
- [Silent hand-masked personal sample](../release_artifacts/personal_hand_sample.mp4)
- [Sample landmark JSON](../release_artifacts/personal_hand_landmarks.json)
- [Evaluation figure](../reports/final_evaluation.png)
- [Detailed measured results](../FINAL_SUBMISSION_REPORT.md)

These are relative file links, not published URLs. No GitHub repository, hosted demo or submission upload is claimed. When approved, the owner can use those repository-relative paths for the release page. Keep raw recordings and the entire original `private/` tree out of Git.

## Reproducibility boundary

This proposed candidate supports inference, sample-to-physics execution and the original held-out comparisons. It contains only held-out personal references, not the 24 training recordings or five validation recordings. Exact full retraining still requires the complete private dataset and original collection pipeline. It cannot establish new-person generalization, robust bimanual learning, free-base balance or successful object grasping. Model assets and Python wheels are installed separately through pinned sources, not bundled.

## Publication checkpoint

Review `RELEASE_APPROVAL_REQUIRED.json` and `docs/RELEASE_LICENSE_REVIEW.md`. No root LICENSE has been added. MIT is recommended for original code only, awaiting owner confirmation of scope and copyright name; upstream notices remain in force. Separate approval is needed for personal RGB, landmarks, calibration, checkpoint and derived reference trajectories. No media/data/checkpoint license has been applied automatically. Do not push or upload this candidate before that decision.
