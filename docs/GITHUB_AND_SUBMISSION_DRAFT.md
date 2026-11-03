# Publication copy — draft, not submitted

## GitHub repository description

CPU personalized hand retargeting from 34 personal videos, evaluated with MuJoCo LEAP and fixed-base G1/Shadow simulations. Includes held-out comparisons, real demonstrations and honest unsuccessful grasp diagnostics.

## Suggested title

JARVIS-Hand: Personalized Hand Retargeting and Evaluated Cross-Embodiment Transfer

## Submission-form summary

JARVIS-Hand turns the operator's own recordings into real robot-hand motion using CPU vision, personalized neural retargeting and MuJoCo physics on a Windows laptop. We collected 34 videos and retained 3,672 valid hand observations. Whole recordings were split into 24 training, five validation and five test clips.

A 28,816-parameter MLP learns bounded LEAP commands distilled from geometry-based optimization through validated forward kinematics. On held-out recordings, fingertip geometry-target error decreased from 61.32 mm with ROM mapping to 32.96 mm with the personal network; the optimizer remained more accurate at 27.83 mm. These are estimated correspondence metrics, not external motion-capture ground truth.

We also tested transfer to Shadow Hands on a G1 model. The pelvis is fixed, arm behavior is programmed, and the left hand is scripted. Across identical held-out clips, neural transfer did not improve Shadow alignment: normalized physical fingertip error was 0.944 palm widths, compared with 0.918 for direct ROM and 0.895 for geometry from ROM-controlled LEAP. This is an experimental adaptation, not a general-purpose humanoid learning policy.

Actual contact-aware manipulation attempts achieved 0/5 valid two-second lift-and-holds. Failed or integrity-rejected trials are disclosed. The submission provides source, pinned dependencies and model attribution, a genuine personal-data sample, held-out evaluation artifacts, tests and real simulation demonstrations, subject to the owner's artifact-release approval. No autonomous VLA policy, dynamic balance or successful grasp is claimed.

## Installation and demo references

Use [release quickstart](RELEASE_QUICKSTART.md) for exact installation and evaluation commands. See the [48.6-second final video](../release_artifacts/JARVIS_Hand_final_demo.mp4), [personal sample](../release_artifacts/personal_hand_sample.mp4) and [results](../FINAL_SUBMISSION_REPORT.md). These links resolve inside the candidate; external GitHub/demo URLs do not exist yet. Insert actual published URLs only after approval and successful upload.

## What is learned versus programmed

- Learned: personal human-landmark/feature → right LEAP joint-command MLP.
- Analytical: FK, ROM normalization and experimental Shadow mapping.
- Programmed: arm IK trajectories and left-hand coordination.
- Simulator feedback: object state, contacts and forces in failed task experiments.
- Not achieved: reliable grasps, learned left-hand coordination, autonomous object policy or locomotion.
