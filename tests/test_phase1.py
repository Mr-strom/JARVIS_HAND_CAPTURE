import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulate import load_hand, run, targets


def test_actuator_mapping_and_target_limits():
    model = load_hand()
    assert model.nu == model.nq == 16
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
    assert len(np.unique(model.actuator_trnid[:,0])) == 16
    for t in np.linspace(0, 12, 1001):
        control = targets(model,t)
        assert np.isfinite(control).all()
        assert np.all(control >= model.actuator_ctrlrange[:,0])
        assert np.all(control <= model.actuator_ctrlrange[:,1])


def test_physics_repeatability_and_motion():
    first = run(seconds=6, output='repeat_a')
    second = run(seconds=6, output='repeat_b')
    assert first['passed'] and second['passed']
    with np.load(ROOT/'reports/repeat_a_trajectory.npz') as a, np.load(ROOT/'reports/repeat_b_trajectory.npz') as b:
        delta = {key: float(np.max(np.abs(a[key]-b[key]))) for key in a.files}
    (ROOT/'reports/repeatability.json').write_text(json.dumps({'same_machine_same_environment': True,'maximum_absolute_differences':delta},indent=2),encoding='utf-8')
    assert all(value == 0 for value in delta.values())


def test_saved_render_contains_motion():
    import cv2
    video = cv2.VideoCapture(str(ROOT/'reports/leap_demo.mp4'))
    assert video.isOpened()
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()
    assert len(frames) >= 180
    assert frames[0].shape == (720,960,3)
    difference = float(np.mean(np.abs(frames[15].astype(float)-frames[90].astype(float))))
    assert difference > 1.0
    (ROOT/'reports/video_validation.json').write_text(json.dumps({'decoded_frames':len(frames),'resolution':[960,720],'open_curled_mean_pixel_difference':difference},indent=2),encoding='utf-8')
