import json
import numpy as np
import torch
from personal_data import canonical
from train_personal import verify_fk,Network
from simulate import ROOT

def test_fk_matches_exact_mujoco_model():
    verify_fk()

def test_dataset_groups_do_not_leak():
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text())
    assigned={};hashes={}
    for entry in audit['split']:
        assert assigned.setdefault(entry['group'],entry['partition'])==entry['partition']
        assert hashes.setdefault(entry['sha256'],entry['partition'])==entry['partition']
    assert set(assigned.values())=={'train','validation','test'}

def test_normalization_removes_scale_translation():
    p=np.random.default_rng(42).normal(size=(21,3))
    np.testing.assert_allclose(canonical(p.copy()),canonical(p*2+3),atol=1e-10)

def test_trained_predictions_are_finite_and_bounded():
    path=ROOT/'private/learning/personal_mlp.pt'
    if not path.exists():
        import pytest
        pytest.skip('Training not yet complete')
    checkpoint=torch.load(path,weights_only=True);net=Network();net.load_state_dict(checkpoint['state'])
    with torch.no_grad():q=net(torch.randn(20,79))
    assert torch.isfinite(q).all() and (q>=0).all() and (q<=1).all()

def test_human_curl_target_has_same_sign_as_robot_curl():
    from train_personal import targets
    from test_phase2 import hand_points
    from personal_data import canonical
    fk=verify_fk()
    p=np.array([canonical(hand_points(.5))])
    target=targets(fk,p).numpy()[0]
    assert np.all(target[[3,7,11],2]<target[[0,4,8],2])
