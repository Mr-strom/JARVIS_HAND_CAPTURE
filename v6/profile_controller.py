"""Warmed CPU world-landmarks -> bounded command timing, no camera/physics."""
import json,time
import numpy as np
from v6.adaptive import AdaptiveEngine,MotionSafety
from learning.runtime import NeuralRetargeter,neural_features
from retargeting import Calibration
from simulate import ROOT,load_hand

def main():
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text());clips=[e for e in audit['split'] if e['partition']=='test'];m=load_hand();names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]];results={}
    for method in ['rom','neural','optimizer','hybrid','neural_v5']:
        ms=[];forward=[];attempts=0;timeouts=0
        for entry in clips:
            z=np.load(ROOT/entry['cache']);t=z['timestamps']-z['timestamps'][0]
            engine=AdaptiveEngine('neural' if method=='neural_v5' else method);safety=MotionSafety(m.actuator_ctrlrange);old=NeuralRetargeter(names,m.actuator_ctrlrange,Calibration.load(ROOT/'private/calibration.json')) if method=='neural_v5' else None
            valid=np.flatnonzero(z['valid'])[0];x=neural_features(z['world'][valid]);q=engine.personal.predict(x);goal=engine.geometry.targets(x[:63].reshape(21,3))
            for _ in range(10):engine.personal.predict(x);engine.geometry.solve(goal,q,8,4)
            for i,stamp in enumerate(t):
                tick=time.perf_counter();world=z['world'][i] if z['valid'][i] else None
                if old is not None:command=old.update(neural_features(world) if world is not None else None,float(stamp))
                else:
                    decision=engine.observe_world(world,float(stamp));command=safety.update(decision,float(stamp));info=decision.info.get('solver',{})
                    if info:attempts+=1;timeouts+=info['timeout']
                    if 'neural_forward_ms' in decision.info:forward.append(decision.info['neural_forward_ms'])
                elapsed=(time.perf_counter()-tick)*1000
                if z['valid'][i]:ms.append(elapsed)
        results[method]={'world_to_safe_command_mean_ms':float(np.mean(ms)),'world_to_safe_command_p50_ms':float(np.percentile(ms,50)),'world_to_safe_command_p95_ms':float(np.percentile(ms,95)),'samples':len(ms),'solver_attempts':attempts,'solver_timeouts':int(timeouts),'neural_forward_mean_ms':float(np.mean(forward)) if forward else None}
    (ROOT/'v6/full_controller_profile.json').write_text(json.dumps({'scope':'Actual paired wall timer from already extracted world21x3 through feature extraction, quality, neural/geometry selection and safety.update. No MediaPipe/camera/render/physics. Warmed10 calls per recording/method;921 original valid observations. Safety evaluated once per observation for CPU timing only, not used for accuracy/physics metrics.','results':results},indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
