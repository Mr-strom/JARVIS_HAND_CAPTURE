import mujoco,numpy as np,pytest
from humanoid_control import load_model,BimanualController
from embodiment_adapter import ShadowGeometryAdapter
from physics_guard import PhysicsGuard

def test_fixed_base_earth_gravity_and_disjoint_control_groups():
    m=load_model();c=BimanualController(m)
    np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81])
    assert not np.any(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    groups=[set(c.arms[s]) for s in c.arms]+[set(c.hands[s]) for s in c.hands]
    assert [len(g) for g in groups]==[5,5,20,20]
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not a&b

def test_ik_transform_and_finite_control():
    m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=BimanualController(m)
    for side,prefix in [('left','lh'),('right','rh')]:
        target=d.xpos[m.body(prefix+'_palm').id]+np.array([.03,0,.02])
        q,residual=c.ik(d,side,target);assert residual<.001
        scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos;scratch.qpos[c.qadr[side]]=q;mujoco.mj_forward(m,scratch)
        np.testing.assert_allclose(scratch.xpos[m.body(prefix+'_palm').id],target,atol=.001)
    with pytest.raises(ValueError):c.step(d,np.full(m.nu,np.nan))
    guard=PhysicsGuard(m,d)
    for _ in range(200):c.step(d,np.zeros(m.nu));guard.step(d)
    assert np.all(d.ctrl>=m.actuator_ctrlrange[:,0]) and np.all(d.ctrl<=m.actuator_ctrlrange[:,1])

def test_adapter_uses_geometry_and_side_specific_named_actuators():
    m=load_model();pose=np.array([.5,0,.6,.7]*3+[.4,.5,.6,.7])
    left=ShadowGeometryAdapter(m,'left');right=ShadowGeometryAdapter(m,'right')
    assert right.representation(pose).shape==(4,4)
    l=left.commands(pose);r=right.commands(pose)
    assert set(l).isdisjoint(r)
    for a,v in l.items():assert m.actuator(a).name.startswith('lh_') and m.actuator_ctrlrange[a,0]<=v<=m.actuator_ctrlrange[a,1]
    for a,v in r.items():assert m.actuator(a).name.startswith('rh_') and m.actuator_ctrlrange[a,0]<=v<=m.actuator_ctrlrange[a,1]
    with pytest.raises(ValueError):right.commands(np.zeros(20))

def test_task_object_free_and_not_attached():
    m=load_model(True)
    assert m.jnt_type[m.joint('task_object_free').id]==mujoco.mjtJoint.mjJNT_FREE
    assert m.neq==0
