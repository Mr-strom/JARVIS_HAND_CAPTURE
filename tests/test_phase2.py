import json
import numpy as np
import mujoco
import pytest
from physics_guard import PhysicsGuard, PhysicsFault
from retargeting import Calibration, Retargeter, NAMES, joint_features
from simulate import load_hand


def controller():
    m=load_hand()
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in m.actuator_trnid[:,0]]
    return Retargeter(names,m.actuator_ctrlrange)


def hand_points(curl=0.):
    p=np.zeros((21,3))
    p[1]=[.035,.018,0];p[2]=[.06,.035,.005];p[3]=[.075,.055,.005];p[4]=[.085,.07,.005]
    for start,x in [(5,.03),(9,.0),(13,-.025),(17,-.05)]:
        p[start]=[x,.065,0]
        for k,length in enumerate([.035,.025,.02],1):
            angle=curl*k
            p[start+k]=p[start+k-1]+np.array([0,length*np.cos(angle),length*np.sin(angle)])
    return p


def test_geometry_flexion_and_wrist_invariance():
    open_features=joint_features(hand_points())
    curl_features=joint_features(hand_points(.55))
    for name in ['if_mcp','if_pip','if_dip','mf_pip','rf_pip']:
        i=NAMES.index(name);assert curl_features[i]>open_features[i]+.25
    theta=.75
    rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    moved=hand_points(.55)@rotation.T*2+np.array([3,4,5])
    np.testing.assert_allclose(joint_features(moved),curl_features,atol=1e-7)


def test_little_finger_has_no_independent_actuator():
    p=hand_points(.2);before=joint_features(p)
    p[18:21]+=np.array([.03,-.02,.04])
    np.testing.assert_allclose(joint_features(p),before)
    assert len(NAMES)==16


def test_joint_identity_and_splay_direction():
    c=controller();f=c.calibration.low.copy()
    baseline=c.map_features(f)
    f[NAMES.index('if_pip')]=c.calibration.high[NAMES.index('if_pip')]
    difference=c.map_features(f)-baseline
    assert np.flatnonzero(abs(difference)>1e-8).tolist()==[c.names.index('if_pip')]
    f=np.zeros(16);f[NAMES.index('if_rot')]=.4
    assert c.map_features(f)[c.names.index('if_rot')]<0


def test_tracking_loss_hold_then_neutral_and_reacquisition():
    c=controller();closed=c.calibration.high.copy()
    values=[]
    for i in range(100):values.append(c.update(closed,i/30))
    last_time=99/30;previous=c.value.copy()
    c.update(None,last_time+.1)
    assert c.state=='HOLD'
    for t in np.arange(last_time+.4,last_time+3,.033):values.append(c.update(None,float(t)))
    assert np.max(np.abs(c.value-c.neutral))<.01
    assert np.max(np.abs(previous-c.neutral))>.5
    before=c.value.copy();after=c.update(closed,float(t)+.033)
    assert np.max(abs(after-before))<=c.max_speed*.033+1e-8


def test_bounds_filter_and_calibration_roundtrip(tmp_path):
    c=controller();rng=np.random.default_rng(42);previous=c.value.copy()
    for i in range(1000):
        value=c.update(rng.uniform(-8,8,16),i/60)
        assert np.all(value>=c.limits[:,0]) and np.all(value<=c.limits[:,1])
        assert np.max(abs(value-previous))<=c.max_speed/60+1e-9
        previous=value
    calibration=Calibration();samples=np.tile(np.linspace(0,1.1,60)[:,None],(1,16))
    samples[:,0]=.1  # insufficient ROM: preserve defaults for this channel
    calibration.fit(samples)
    assert not calibration.personalized[0] and calibration.personalized[1]
    calibration.save(tmp_path/'calibration.json')
    restored=Calibration.load(tmp_path/'calibration.json')
    np.testing.assert_equal(restored.low,calibration.low)
    with pytest.raises(ValueError):joint_features(np.zeros((21,3)))
    with pytest.raises(ValueError):c.map_features(np.full(16,np.nan))


def test_explicit_viewer_reset_is_acknowledged_without_weakening_guard():
    m=load_hand();d=mujoco.MjData(m);g=PhysicsGuard(m,d)
    for _ in range(13534):g.step(d)
    before=float(d.time);mujoco.mj_resetData(m,d)
    g.accept_viewer_sync(d,before);g.step(d)
    assert g.steps==13535 and len(g.resets)==1 and d.time==m.opt.timestep
    mujoco.mj_resetData(m,d)
    with pytest.raises(PhysicsFault,match='Unacknowledged'):g.step(d)


def test_actual_numerical_failure_still_stops():
    m=load_hand();d=mujoco.MjData(m);g=PhysicsGuard(m,d)
    g.step(d);d.qvel[0]=1e15
    with pytest.raises(PhysicsFault):g.step(d)
    assert any(g.warnings(d).values())


def test_zero_lag_bin_is_not_reported_as_zero_latency():
    from live_mirror import response_lag
    t=np.arange(0,6,.1)
    signal=np.tile(np.sin(t*2)[:,None],(1,16))
    samples=np.column_stack([t,signal,signal,signal])
    result=response_lag(samples.tolist())
    assert not result['available']
    assert 'zero latency is NOT established' in result['reason']
