import json
import mujoco
import numpy as np
import pytest
from personal_data import canonical
from cross_embodiment import human_geometry, geometry_metrics, ShadowGeometry, Pathways, METHODS
from humanoid_control import load_model
from final_grasp_recovery import valid_hold
from simulate import ROOT
from test_phase2 import hand_points


def test_normalized_geometry_is_rigid_and_scale_invariant():
    p=hand_points(.4);a=.7
    rotation=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    h=human_geometry(canonical(p));moved=human_geometry(canonical(p@rotation.T*3+[2,4,5]))
    np.testing.assert_allclose(h,moved,atol=1e-10)
    result=geometry_metrics(h,moved)
    assert result['fingertip_error_palm_widths']<1e-10
    assert result['pinch_error_palm_widths']<1e-10
    assert result['finger_direction_cosine']==pytest.approx(1.)


def test_shadow_curl_sign_and_coupled_tendon_reference():
    m=load_model();g=ShadowGeometry(m);cmd=np.clip(np.zeros(m.nu),m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    opened=g.kinematic(cmd).copy();cmd[m.actuator('rh_A_FFJ3').id]=.5;cmd[m.actuator('rh_A_FFJ0').id]=.8
    closed=g.kinematic(cmd)
    assert closed[0,3,2]>opened[0,3,2]+.1 # +human curl z matches +Shadow flexion
    for name in ['rh_FFJ1','rh_FFJ2']:
        assert g.scratch.qpos[m.jnt_qposadr[m.joint(name).id]]==pytest.approx(.4)
    # Index/little MCPs have a small forward offset as well as lateral separation.
    assert g.width==pytest.approx(.04658156824324398)


def test_pathways_use_same_heldout_input_and_stay_in_named_limits():
    if not (ROOT/'private/calibration.json').exists() or not (ROOT/'private/learning/personal_mlp.pt').exists():
        pytest.skip('Personal calibration/checkpoint distribution requires owner approval')
    m=load_model();paths=Pathways(m);p=hand_points(.45)
    for method in METHODS:
        cmd=paths.command(method,p)
        assert cmd.shape==(m.nu,) and np.isfinite(cmd).all()
        assert np.all(cmd>=m.actuator_ctrlrange[:,0]) and np.all(cmd<=m.actuator_ctrlrange[:,1])
        assert all(cmd[a]==paths.neutral[a] for a in range(m.nu) if not m.actuator(a).name.startswith('rh_'))
        with pytest.raises(ValueError):paths.command(method,np.full((21,3),np.nan))


def test_stable_hold_thresholds_are_not_relaxed():
    assert valid_hold(.020,{'if','th'},False,.149)
    assert not valid_hold(.0199,{'if','th'},False,.01)
    assert not valid_hold(.04,{'if'},False,.01)
    assert not valid_hold(.04,{'if','th'},True,.01)
    assert not valid_hold(.04,{'if','th'},False,.15)
