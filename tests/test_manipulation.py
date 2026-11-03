import numpy as np
import mujoco
from manipulation import scene,observe
from physics_guard import PhysicsGuard

def test_object_is_free_and_wrist_is_force_bounded():
    m=scene();d=mujoco.MjData(m)
    assert m.jnt_type[m.joint('object_free').id]==mujoco.mjtJoint.mjJNT_FREE
    assert m.neq==0  # no attachment equality
    for name in ['base_x','base_y','base_z']:
        a=m.actuator(name).id
        np.testing.assert_equal(m.actuator_forcerange[a],[-30,30])
    g=PhysicsGuard(m,d)
    for _ in range(200):g.step(d)
    assert np.isfinite(observe(m,d)['position']).all()
