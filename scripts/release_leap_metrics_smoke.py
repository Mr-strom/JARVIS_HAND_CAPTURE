"""Recompute frozen held-out LEAP geometry metrics; no training or new model."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from train_personal import Network
from learning.leap_fk import LeapFK
from retargeting import Calibration,Retargeter
from simulate import ROOT,load_hand

def main():
    z=np.load(ROOT/'private/release_audit/leap_heldout_reference.npz',allow_pickle=False)
    ck=torch.load(ROOT/'private/learning/personal_mlp.pt',map_location='cpu',weights_only=True)
    net=Network();net.load_state_dict(ck['state']);net.eval();torch.set_num_threads(2)
    x=z['x'];m=load_hand();names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
    rom=Retargeter(names,m.actuator_ctrlrange,Calibration.load(ROOT/'private/calibration.json'))
    baseline=np.array([rom.map_features(row[63:]) for row in x])
    # Match the frozen training evaluator's float32 normalizer arithmetic exactly.
    # The runtime adapter is untouched; its float64 intermediates can differ by ~1e-6 rad.
    with torch.no_grad():neural=ck['low']+(np.array(ck['high'])-ck['low'])*net(torch.tensor((x-np.array(ck['mean'],dtype=np.float32))/np.array(ck['std'],dtype=np.float32),dtype=torch.float32)).numpy()
    assert np.max(abs(baseline-z['baseline']))<1e-6
    assert np.max(abs(neural-z['neural']))<1e-6
    expected=json.loads((ROOT/'reports/personal_model_results.json').read_text());fk=LeapFK();result={}
    for method,q in [('baseline',baseline),('optimizer',z['optimizer']),('neural',neural)]:
        with torch.no_grad():points=fk(torch.tensor(q,dtype=torch.float32)).numpy()
        tip=points[:,3::4];goal=z['goal'][:,3::4]
        metrics={'held_out_fingertip_error_mm':float(np.linalg.norm(tip-goal,axis=-1).mean()*1000),
            'held_out_pinch_error_mm':float(np.abs(np.linalg.norm(tip[:,:3]-tip[:,3:4],axis=-1)-np.linalg.norm(goal[:,:3]-goal[:,3:4],axis=-1)).mean()*1000)}
        for key,value in metrics.items():assert abs(value-expected[method][key])<.001,(method,key,value)
        result[method]=metrics
    report={'frames':len(x),'training_performed':False,'optimizer_reference':'cached frozen optimization predictions, not recomputed teacher training','baseline_and_mlp_predictions_recomputed':True,'tolerance_mm':.001,'results':result}
    (ROOT/'reports/release_leap_metrics_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
