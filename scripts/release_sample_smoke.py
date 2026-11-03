"""Audit-only: decode proposed RGB sample -> MediaPipe -> existing MLP -> LEAP.

This smoke test proves execution/movement, not geometry accuracy equivalence
between the masked video and the original uncropped evaluation recordings.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,mujoco,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from simulate import ROOT,load_hand,camera_for
from retargeting import Calibration
from learning.runtime import NeuralRetargeter,neural_features
from physics_guard import PhysicsGuard

def main():
    m=load_hand();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
    control=NeuralRetargeter(names,m.actuator_ctrlrange,Calibration.load(ROOT/'private/calibration.json'));guard=PhysicsGuard(m,d)
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_hands=2)
    cap=cv2.VideoCapture(str(ROOT/'private/final/personal_hand_sample.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise RuntimeError('Cannot decode proposed sample')
    n=0;detected=0;spans=[];controls=[];limit_violations=0;maxexc=0.;snapshots=[]
    with vision.HandLandmarker.create_from_options(options) as detector,mujoco.Renderer(m,height=480,width=640) as renderer:
        while True:
            ok,frame=cap.read()
            if not ok:break
            t=n/fps;result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),round(t*1000))
            features=None
            if result.hand_world_landmarks:
                i=max(range(len(result.handedness)),key=lambda k:(result.handedness[k][0].category_name=='Right',result.handedness[k][0].score))
                world=np.array([[a.x,a.y,a.z] for a in result.hand_world_landmarks[i]])
                if result.handedness[i][0].category_name=='Left':world[:,0]*=-1
                features=neural_features(world);detected+=1
            command=control.update(features,t)
            while d.time+1e-10<(n+1)/fps:
                d.ctrl[:]=command;guard.step(d)
                limit_violations+=int(np.count_nonzero((d.ctrl<m.actuator_ctrlrange[:,0])|(d.ctrl>m.actuator_ctrlrange[:,1])))
                maxexc=max(maxexc,float(np.max(np.maximum(m.jnt_range[:,0]-d.qpos,d.qpos-m.jnt_range[:,1]))))
            spans.append(d.qpos.copy());controls.append(command.copy())
            if n in [0,45,79]:
                renderer.update_scene(d,camera=camera_for(m));view=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                cv2.putText(view,f'Actual sample -> MediaPipe -> MLP -> LEAP, frame {n}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);snapshots.append(view)
            n+=1
    cap.release();motion=np.ptp(np.array(spans),axis=0);command_motion=np.ptp(np.array(controls),axis=0)
    assert n==80 and detected>10 and max(motion)>.05 and max(command_motion)>.05
    assert limit_violations==0 and sum(guard.warnings(d).values())==0
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    cv2.imwrite(str(out/'release_sample_smoke_frames.jpg'),np.hstack(snapshots))
    report={'frames_decoded':n,'frames_detected':detected,'tracking_coverage_scope':'redacted/zoomed sample smoke only, not original dataset coverage','sim_seconds':float(d.time),
        'max_joint_motion_rad':float(max(motion)),'max_command_motion_rad':float(max(command_motion)),
        'actuators_with_joint_motion_over_0_01rad':int(np.count_nonzero(motion>.01)),
        'command_limit_violations':limit_violations,'max_raw_joint_excursion_rad':max(0.,maxexc),'warnings':guard.warnings(d),
        'synthetic_targets_used':False,'source':'proposed personal RGB sample, freshly detected landmarks and preserved neural checkpoint',
        'scope':'recorded-video inference/control smoke; no webcam, latency or geometry-accuracy claim'}
    (out/'release_sample_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
