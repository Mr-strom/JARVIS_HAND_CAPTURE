"""Replay explicitly supplied personal sample; missing artifacts are errors."""
from pathlib import Path
import sys,json,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco,cv2
from simulate import ROOT,load_hand,camera_for
from retargeting import Calibration
from learning.runtime import NeuralRetargeter
from physics_guard import PhysicsGuard

def main():
    p=argparse.ArgumentParser();p.add_argument('--viewer',action='store_true');a=p.parse_args()
    rows=json.loads((ROOT/'release_artifacts/personal_hand_landmarks.json').read_text())['rows']
    m=load_hand();d=mujoco.MjData(m);mujoco.mj_forward(m,d);g=PhysicsGuard(m,d)
    c=NeuralRetargeter([m.joint(int(j)).name for j in m.actuator_trnid[:,0]],m.actuator_ctrlrange,Calibration.load(ROOT/'private/calibration.json'));q=np.zeros(16);states=[];start=time.perf_counter();origin=rows[0]['source_timestamp_s']
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        for row in rows:
            t=row['source_timestamp_s']-origin
            while d.time+1e-10<t:d.ctrl[:]=q;g.step(d)
            x=np.r_[np.array(row['canonical']).ravel(),row['features']] if row['valid_cached'] else None;q=c.update(x,t);states.append(d.qpos.copy())
            if a.viewer:
                renderer.update_scene(d,camera=camera_for(m));cv2.imshow('Approved personal sample -> trained LEAP',cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1)&255 in [27,ord('q')]:break
                time.sleep(max(0,t-(time.perf_counter()-start)))
    cv2.destroyAllWindows();assert max(np.ptp(states,axis=0))>.05 and not sum(g.warnings(d).values())
    print(json.dumps({'sample_rows':len(states),'sim_seconds':d.time,'warnings':g.warnings(d),'source':'personally recorded cached landmarks; original timestamps; no synthetic substitute'},indent=2))
if __name__=='__main__':main()
