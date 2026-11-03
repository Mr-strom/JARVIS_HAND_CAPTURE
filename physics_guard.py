"""Separate physics failures from explicitly observed viewer reset boundaries."""
import json
import mujoco
import numpy as np


class PhysicsFault(RuntimeError):
    pass


class PhysicsGuard:
    def __init__(self, model, data):
        self.model = model
        self.last_time = float(data.time)
        self.dt = float(model.opt.timestep)
        self.steps = 0
        self.resets = []

    @staticmethod
    def warnings(data):
        return {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning)}

    def step(self, data):
        before = float(data.time)
        if abs(before - self.last_time) > 1e-8:
            raise PhysicsFault(f'Unacknowledged time change before step {self.steps}: {self.last_time} -> {before}')
        if self.model.opt.timestep != self.dt:
            raise PhysicsFault('Physics timestep changed during the run')
        mujoco.mj_step(self.model, data)
        warning_counts = self.warnings(data)
        after = float(data.time)
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or not np.isfinite(data.qacc).all() or any(warning_counts.values())
                or abs(after - (before + self.dt)) > 1e-8):
            raise PhysicsFault(json.dumps({'step': self.steps, 'before': before, 'after': after,
                                          'expected': before+self.dt, 'warnings': warning_counts}))
        self.last_time = after
        self.steps += 1

    def accept_viewer_sync(self, data, before_time):
        """Only call immediately around passive viewer.sync(), with no stepping."""
        after = float(data.time)
        if abs(after-before_time) <= 1e-8:
            return
        if (after > before_time or abs(after) > 1e-8 or any(self.warnings(data).values())
                or not np.isfinite(data.qpos).all()):
            raise PhysicsFault(f'Unexpected viewer state change: {before_time} -> {after}')
        self.resets.append({'source':'passive_viewer_sync', 'at_total_step':self.steps,
                            'before':before_time, 'after':after})
        self.last_time = after
