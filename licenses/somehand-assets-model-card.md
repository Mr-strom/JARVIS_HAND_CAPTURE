---
license: apache-2.0
tags:
- dex-hand
- teleoperation
- retargeting
- mujoco
- assets
---

# somehand-assets

External asset repository for `somehand`.

## Contents

This model repo stores runtime assets and sample data that are intentionally kept out of the code repository:

- `archives/mjcf_assets.tar.gz`: packaged MJCF hand assets and meshes under `assets/mjcf/`
- `models/hand_landmarker.task`: MediaPipe hand landmarker model under `assets/models/`
- `archives/reference_assets.tar.gz`: reference BVH assets, currently including `assets/ref_with_toe.bvh`
- `archives/sample_recordings.tar.gz`: sample replay recordings and short rendered examples under `recordings/`

## Suggested download commands

### ModelScope

```bash
export SOMEHAND_MODELSCOPE_REPO_ID=BingqianWu/somehand-assets
python scripts/setup/download_assets.py --only mjcf mediapipe examples
```

### Hugging Face

```bash
export SOMEHAND_HF_REPO_ID=12e21/somehand-assets
python scripts/setup/download_assets.py --source huggingface --only mjcf mediapipe examples
```

## Notes

- This repo is a **model repo** so asset updates are versioned as commits.
- The main `somehand` code repository should not store these large binaries directly.
- Sample recordings are provided for replay, visualization, and regression-style checks.
