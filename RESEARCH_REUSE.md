# Pinned source research and reuse decisions

Source snapshots and metadata were inspected locally; repository claims are not our benchmark results. No package below was installed wholesale during this stage.

## BotRunner64/somehand

Revision: `c4525805a82d1d6991bbed3ef74923f5d86d23b6`.

Apache-2.0; Python>=3.10; MuJoCo/NumPy/SciPy/PyYAML core. Prior Dex5 assets had unstable raw stepping; core API remains available but stock terminal controls have a Windows termios issue. Preserved as prior research, not used for new LEAP training.

Source: https://github.com/BotRunner64/somehand/tree/c4525805a82d1d6991bbed3ef74923f5d86d23b6

## dexsuite/dex-retargeting

Revision: `3f56141bc8bd2760d5e452e382937269554ebb21`.

MIT; Python>=3.7,<3.13; dependencies include pin>=3.3.1, nlopt>=2.8, pytransform3d, lxml. LEAP-specific YAML/URDF conventions require a joint-name adapter. Examples optionally need SAPIEN. Not installed to avoid a parallel robot model and dependency stack; our teacher operates directly in the verified MJCF actuator order.

Source: https://github.com/dexsuite/dex-retargeting/tree/3f56141bc8bd2760d5e452e382937269554ebb21

## google-deepmind/mujoco_menagerie

Revision: `98a70c569c4c1b6da467a20012f766ee1f8dc521`.

Per-model licenses; LEAP MIT, MuJoCo>=3.1.3. Runtime remains pinned to c1a4eeb85694ae1dffe33ff1797d4e528928a133, despite the newer revision inspected here. Joint names and actuator order are checked against that exact model; no replacement assets introduced.

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521

## guptabhishekumar/handrobot

Revision: `ae31ea4fd3778b18b30f54d1b5d614ab8686b8b5`.

MIT; Python >=3.12,<3.13; full stack includes mink, torch/torchvision, MediaPipe0.10.35, optional LeRobot. Source contains Panda/SO-101 assets, IK/teleoperation, policy training and neural LEAP FK/retargeting implementations. We reused only isolated FK under MIT, adjusted local model paths and verified numerical equivalence. Full ACT/arm workflow was not executed on this laptop. Its original personal split is sequential within a recording, so we implemented grouped held-out clips independently.

Source: https://github.com/guptabhishekumar/handrobot/tree/ae31ea4fd3778b18b30f54d1b5d614ab8686b8b5

## personalrobotics/mj_manipulator

Revision: `478dbc80e3cdc89d8029f776db885a2b10069f7b`.

MIT; Python>=3.11; broad planning dependencies toppra, py_trees, mink>=1.1, sscbirrt, ssik, external mj-environment/prl-assets. Source describes generic arm planning/Cartesian control. Not installed: no validated LEAP-arm assembly exists here; adding the stack would not solve the current grasp-contact failures.

Source: https://github.com/personalrobotics/mj_manipulator/tree/478dbc80e3cdc89d8029f776db885a2b10069f7b

## qqsq12321/AnyDexRetarget

Revision: `9e49c03123bd9ebd99e4e1a04e893a1caa7bd9b3`.

MIT; Python>=3.10; package dependencies numpy/nlopt/pyyaml, fuller runtime also pin. Source has analytical/key-vector/DexPilot optimizers and robot-specific configurations. Not installed: our exact MuJoCo FK already provides validated optimization without new robot naming/transmission conventions or Pinocchio/NLopt binaries.

Source: https://github.com/qqsq12321/AnyDexRetarget/tree/9e49c03123bd9ebd99e4e1a04e893a1caa7bd9b3
