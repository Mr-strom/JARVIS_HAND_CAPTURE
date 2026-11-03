# Adaptive Precision Retargeting Engine — v6 experiment

This is an **optional experiment**, alongside the untouched v5 default and frozen RC1. Personal data, weights, calibration, new video and publication remain unapproved. No license is applied.

Start with [measured report](INNOVATION_V6_REPORT.md), [ablation](HYBRID_ABLATION.md) and [merge decision](MERGE_DECISION.md). The bounded optimizer is more accurate than hybrid; hybrid trades some accuracy for lower average CPU work. The asynchronous camera path improved unattended display cadence but worsened software frame age and did not establish a live latency improvement. The final attended check subsequently tracked both methods; the operator reported no obvious improvement. See the final integration report.

## Existing environment — run from project root

```powershell
# Original commands preserved
& '.\.venv\Scripts\python.exe' simulate.py --viewer --seconds 30
& '.\.venv\Scripts\python.exe' live_mirror.py
& '.\.venv\Scripts\python.exe' neural_live.py
& '.\.venv\Scripts\python.exe' humanoid_control.py --viewer --seconds 30

# Explicit release selector: default is the unchanged v5 neural application
& '.\.venv\Scripts\python.exe' -m v6.launch --controller neural --seconds 30

# New optional modes; still needs the personal checkpoint/calibration
& '.\.venv\Scripts\python.exe' -m v6.live --controller hybrid --seconds 30
& '.\.venv\Scripts\python.exe' -m v6.live --controller optimizer --seconds 30
& '.\.venv\Scripts\python.exe' -m v6.live --controller neural --seconds 30
& '.\.venv\Scripts\python.exe' -m pytest -q tests v6/test_adaptive.py
```

The new app uses the unmirrored camera and reflects tracker-labeled Left world landmarks, consistently with personal-video ingestion. Handedness/mirroring is still a tracker estimate. Only the hand's four modeled digits control LEAP; wrist is fixed. Q/Esc closes it. Camera imagery is not recorded by the live app; numerical telemetry stays in ignored `private/v6/`.

Optional paths:

```powershell
& '.\.venv\Scripts\python.exe' -m v6.live --controller hybrid --checkpoint '<local-model-directory>\personal_mlp.pt' --calibration '<local-data-directory>\calibration.json'
```

## Reproduce evaluation with the private cached dataset

No dependency installation or retraining is needed in the verified environment. Fresh setup uses the existing hashed `requirements.lock`, then `requirements-learning.lock`, and the original pinned asset download scripts. Python3.11.9 is required. Do not create a new environment over the working one.

```powershell
& '.\.venv\Scripts\python.exe' -m v6.profile_camera --seconds 8
& '.\.venv\Scripts\python.exe' -m v6.profile_original --seconds 16 --label before
& '.\.venv\Scripts\python.exe' -m v6.live --seconds 16 --controller hybrid --label after
& '.\.venv\Scripts\python.exe' -m v6.evaluate --partition validation --dropouts
& '.\.venv\Scripts\python.exe' -m v6.evaluate --partition test --dropouts
& '.\.venv\Scripts\python.exe' -m v6.profile_controller
& '.\.venv\Scripts\python.exe' -m v6.summarize
& '.\.venv\Scripts\python.exe' -m v6.demo
```

Evaluation accepts `--checkpoint`, `--calibration` and `--data-root`. Data-root is a project-style root containing paths listed in its existing dataset audit; the audit itself currently comes from this project's preserved `reports/dataset_audit.json`. All-source raw-demo reproduction requires the private original recording33 and its provenance entry. These commands are not claimed reproducible from the small release sample alone. No personal artifacts have been approved as public inputs. Do not overwrite initial run reports when comparing experiments: use a separate extracted experiment workspace.

Inputs: private processed NPZs with `canonical`, `features`, `world`, `timestamps`, `frame_indices`, `valid`, `gesture`; original audit's recording-level split; preserved checkpoint and calibration. Source hash/cached extraction provenance is unchanged. Future-public redistribution of those inputs requires approval. The v6 code-only review bundle intentionally excludes them.

Aggregate outputs stay under `v6/`; per-frame CSVs are Git-ignored and private trajectory NPZs/video stay under `private/v6/`. `ADAPTIVE_RETARGETING_RESULTS.csv` is recording-level; `ADAPTIVE_RETARGETING_SUMMARY.csv` combines results with coverage. `LATENCY_PROFILE.json` separates mapping, full-controller, capture, inference, rendering and software-frame-age measurements. Whole-recording grouping is preserved; test recordings are reused, not a pristine prospective test set.

## Private demonstration

`private/v6/adaptive_precision_before_after.mp4` shows the same personal source timestamps and freshly stepped neural-v5-filter versus adaptive physics, followed by labeled injected tracking loss. It is recorded replay, not footage of the subsequent attended live check. It includes hand appearance/room fragments and has not been cleared for release; no public URL exists. The primary aggregate plot is [before/after evaluation](before_after_evaluation.png).

## Safety and limitations

Only the main application thread steps actual MuJoCo data. Vision/solver data are separate. Confidence is a geometry/continuity proxy, not per-landmark certainty. Late solver results fall back to the MLP. Four bounded evaluations and cooperative8ms timeout do not establish hard real-time guarantees. Safety limits requested positions, command velocity and command acceleration, not every physical servo derivative. Existing PhysicsGuard still fails on numerical warnings/unexpected resets.

The code-only review ZIP can be rebuilt with `python -m v6.build_review_bundle`; it is local review material, not publication permission. Frozen v5 RC1 is never rebuilt or overwritten by v6. Existing upstream notices, copyright terms and undecided original-code licensing remain in force. Do not push this branch or any private files without explicit approval.
