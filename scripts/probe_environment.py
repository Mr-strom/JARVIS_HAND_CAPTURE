from pathlib import Path
import importlib.metadata as meta
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['SOMEHAND_HOME'] = str(ROOT)


def main():
    report = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'base_prefix': sys.base_prefix, 'packages': {}, 'errors': []}
    for package in ['mujoco', 'numpy', 'mediapipe', 'opencv-contrib-python', 'scipy', 'PyYAML', 'pytest', 'somehand']:
        report['packages'][package] = meta.version(package)
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python.vision import HandLandmarker
        import mujoco
        import numpy as np
        from somehand.api import RetargetingEngine, load_retargeting_config, resolve_config_path
        config = load_retargeting_config(str(resolve_config_path('right/dex5_right.yaml')))
        engine = RetargetingEngine(config)
        report['core_api'] = engine.describe()
        m = engine.hand_model.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        report['raw_model'] = {'nq': m.nq, 'nu': m.nu, 'timestep': m.opt.timestep,
                               'initial_contacts': d.ncon, 'initial_max_penetration_m': max([0.0] + [-float(c.dist) for c in d.contact]),
                               'joints': engine.hand_model.get_joint_names()}
        target = np.clip(np.zeros(m.nu), m.actuator_ctrlrange[:,0], m.actuator_ctrlrange[:,1])
        d.ctrl[:] = target
        for _ in range(1000):
            mujoco.mj_step(m,d)
        report['raw_model'].update({'time_after_1000_steps': d.time, 'finite': bool(np.isfinite(d.qpos).all()),
                                     'max_abs_qvel': float(np.max(np.abs(d.qvel))),
                                     'warnings': [int(x.number) for x in d.warning]})
    except Exception:
        report['errors'].append(traceback.format_exc())
    (ROOT / 'reports/environment_probe.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps(report,indent=2))
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
