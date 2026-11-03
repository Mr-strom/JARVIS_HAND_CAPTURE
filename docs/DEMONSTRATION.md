# Demonstration

## Public demonstration

[Watch JARVIS-Hand on Google Drive](https://drive.google.com/file/d/19wrV4-QulbFKnXYRngHCNEGr0R_2hMjI/view?usp=sharing).

The supplied link was opened in a signed-out browser session. The file preview loaded and playback started without an account sign-in, so it is accessible to a signed-out viewer at the time of this check. The video is named `Video Project 5.mp4`, 162.53 seconds, 1920×1080 at 30 FPS with AAC audio.

The video shows the guided hand session, MediaPipe landmarks, the LEAP physics output, the ROM/neural comparison, Shadow/G1 material, v6 comparison, and result cards. Its initial guided-session footage includes the operator's face and room. This README's still screenshot is taken later from the same demonstration where the human pane is already masked to the hand against black; that crop does not anonymize the full video. The supplied full-video link is intentionally included per the owner's request.

## Reproducing the principal demonstrations

After installation and restoring the approved sample, run the trained LEAP replay:

```powershell
& '.\.venv\Scripts\python.exe' scripts\replay_approved_sample.py --viewer
```

Run the fixed-pelvis G1/Shadow actuator demonstration:

```powershell
& '.\.venv\Scripts\python.exe' humanoid_control.py --viewer --seconds 30
```

Run live neural LEAP control with the connected camera:

```powershell
& '.\.venv\Scripts\python.exe' neural_live.py --hand any --seconds 30
```

The sample replay and live mirroring are distinct: the first replays approved personal sample landmarks; the live command uses current camera input. Neither establishes a successful object grasp. The G1 demo is fixed-base with scripted arm targets and left-hand coordination.

## Recording / privacy

The public demonstration is owner-provided. Raw training videos are not part of the repository. The linked demo itself includes identifiable operator footage in some sections; anyone reusing or forwarding it should consider that privacy exposure. No further video upload occurred during this documentation update.
