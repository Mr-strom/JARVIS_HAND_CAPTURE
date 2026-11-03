"""Phase 1: synthetic actuator exercise, not a human demonstration or grasp test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ['SOMEHAND_HOME'] = str(ROOT)


def load_hand():
    from somehand.api import RetargetingEngine, load_retargeting_config, resolve_config_path
    config = load_retargeting_config(str(resolve_config_path('right/dex5_right.yaml')))
    engine = RetargetingEngine(config)
    model = engine.hand_model.model
    # Explicit simulation parameters; upstream source and XML remain unchanged.
    model.opt.timestep = 0.001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.dof_damping[:] = 0.05
    model.dof_armature[:] = 0.0001
    model.actuator_gainprm[:, 0] = 3.0
    model.actuator_biasprm[:, 1] = -3.0
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    return engine, model


def targets(model, sim_time):
    """Gentle periodic curl/spread inside each actuator's actual ctrlrange."""
    low, high = model.actuator_ctrlrange.T
    curl = 0.5 - 0.5 * np.cos(2 * np.pi * sim_time / 6.0)
    target = low + (high - low) * (0.08 + 0.44 * curl)
    symmetric = (low < 0) & (high > 0)
    target[symmetric] = 0.20 * np.minimum(-low[symmetric], high[symmetric]) * np.sin(2 * np.pi * sim_time / 6.0)
    return np.clip(target, low, high)


def camera_for(model):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = model.stat.center
    camera.distance = model.stat.extent * 1.5
    camera.azimuth = 135
    camera.elevation = -20
    return camera


def run(seconds=12.0, render=False, viewer=False, output='phase1', realtime=True):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    engine, model = load_hand()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_contacts = int(data.ncon)
    initial_penetration = max([0.0] + [-float(c.dist) for c in data.contact])
    qindices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    jids = model.actuator_trnid[:, 0]
    limits = model.jnt_range[jids]
    qmin = data.qpos[qindices].copy()
    qmax = qmin.copy()
    max_velocity = 0.0
    max_penetration = initial_penetration
    max_contacts = initial_contacts
    control_violations = 0
    joint_violation_steps = 0
    max_joint_violation = 0.0
    state = []
    frames = []
    frame_times = []
    render_seconds = 0.0
    renderer = None
    window = None
    camera = camera_for(model)
    fps = 30
    output_dir = ROOT / 'reports'
    output_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    try:
        if render:
            renderer = mujoco.Renderer(model, height=720, width=960)
        if viewer:
            import mujoco.viewer as mjviewer
            window = mjviewer.launch_passive(model, data)
            window.cam.lookat[:] = camera.lookat
            window.cam.distance = camera.distance
            window.cam.azimuth = camera.azimuth
            window.cam.elevation = camera.elevation
        next_frame = 0.0
        steps = round(seconds / model.opt.timestep)
        for step in range(steps):
            target = targets(model, step * model.opt.timestep)
            control_violations += int(np.count_nonzero((target < model.actuator_ctrlrange[:,0]) | (target > model.actuator_ctrlrange[:,1])))
            data.ctrl[:] = target
            # After initialization, only actuator controls are assigned, never qpos.
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                raise RuntimeError(f'Nonfinite state at step {step}')
            expected_time = (step + 1) * model.opt.timestep
            if abs(data.time - expected_time) > 1e-6:
                raise RuntimeError(f'Unexpected simulation reset at step {step}')
            q = data.qpos[qindices]
            qmin = np.minimum(qmin, q)
            qmax = np.maximum(qmax, q)
            max_velocity = max(max_velocity, float(np.max(np.abs(data.qvel))))
            excursion = float(np.max(np.maximum(limits[:,0]-q, q-limits[:,1])))
            max_joint_violation = max(max_joint_violation, excursion)
            joint_violation_steps += int(excursion > 1e-4)
            max_contacts = max(max_contacts, data.ncon)
            max_penetration = max(max_penetration, max([0.0] + [-float(c.dist) for c in data.contact]))
            if data.time + 1e-9 >= next_frame:
                state.append(np.concatenate(([data.time], data.ctrl.copy(), data.qpos.copy())))
                if renderer:
                    tick = time.perf_counter()
                    renderer.update_scene(data, camera=camera)
                    frames.append(renderer.render().copy())
                    frame_times.append(float(data.time))
                    render_seconds += time.perf_counter() - tick
                if window:
                    if not window.is_running():
                        break
                    window.sync()
                    if realtime:
                        delay = data.time - (time.perf_counter()-start)
                        if delay > 0:
                            time.sleep(delay)
                next_frame += 1/fps
        elapsed = time.perf_counter() - start
    finally:
        if renderer:
            renderer.close()
        if window:
            window.close()
    warning_counts = {mujoco.mjtWarning(i).name: int(w.number) for i,w in enumerate(data.warning)}
    spans = qmax-qmin
    metrics = {'model': engine.describe(), 'synthetic_actuator_demo': True,
               'simulation_seconds': float(data.time), 'wall_seconds': elapsed,
               'physics_steps': step+1, 'timestep': model.opt.timestep,
               'realtime_factor_including_rendering': float(data.time/elapsed),
               'finite_state': True, 'control_limit_violations': control_violations,
               'joint_limit_violation_steps_tolerance_1e_4_rad': joint_violation_steps,
               'max_joint_limit_excursion_rad': max_joint_violation,
               'max_abs_velocity_rad_s': max_velocity,
               'initial_contacts': initial_contacts, 'max_contacts': max_contacts,
               'initial_max_penetration_m': initial_penetration, 'max_penetration_m': max_penetration,
               'warnings': warning_counts, 'actuators_moving_over_0_05_rad': int(np.sum(spans > 0.05)),
               'joint_motion_ranges_rad': dict(zip(engine.hand_model.get_joint_names(), spans.tolist())),
               'rendered_frames': len(frames), 'render_seconds': render_seconds,
               'viewer_requested': viewer,
               'settings': {'integrator': 'implicitfast','damping':0.05,'armature':0.0001,'position_kp':3.0}}
    if frames:
        import cv2
        from PIL import Image
        writer = cv2.VideoWriter(str(output_dir / f'{output}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (960,720))
        if not writer.isOpened():
            raise RuntimeError('MP4 writer failed')
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        open_index = min(range(len(frame_times)), key=lambda i: abs(frame_times[i]-0.5))
        closed_index = min(range(len(frame_times)), key=lambda i: abs(frame_times[i]-3.0))
        Image.fromarray(frames[open_index]).save(output_dir / f'{output}_open.png')
        Image.fromarray(frames[closed_index]).save(output_dir / f'{output}_curled.png')
        preview = [Image.fromarray(frame).resize((480,360)) for frame in frames[::3]]
        preview[0].save(output_dir / f'{output}.gif', save_all=True, append_images=preview[1:], duration=100, loop=0)
        metrics['open_curled_pixel_mean_abs_difference'] = float(np.mean(np.abs(frames[open_index].astype(float)-frames[closed_index].astype(float))))
    np.savez_compressed(output_dir / f'{output}_trajectory.npz', samples=np.asarray(state), final_qpos=data.qpos.copy(), final_qvel=data.qvel.copy())
    metrics['passed'] = bool(control_violations == 0 and not any(warning_counts.values()) and np.all(spans > 0.05) and max_joint_violation < 0.01 and data.time >= seconds-0.002)
    (output_dir / f'{output}_metrics.json').write_text(json.dumps(metrics,indent=2), encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--render', action='store_true', help='Save MP4, GIF, snapshots and measured state')
    parser.add_argument('--viewer', action='store_true', help='Open interactive MuJoCo window; close it to stop')
    parser.add_argument('--output', default='phase1')
    args = parser.parse_args()
    if not args.output.replace('_','').replace('-','').isalnum():
        parser.error('output must be a simple filename stem')
    result = run(args.seconds, args.render, args.viewer, args.output)
    raise SystemExit(0 if result['passed'] else 1)
