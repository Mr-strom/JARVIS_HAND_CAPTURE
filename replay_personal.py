"""Replay actual held-out personal motion with baseline/neural physics and source video."""
import argparse,json,time
import cv2,numpy as np,torch,mujoco
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from simulate import ROOT,load_hand,camera_for
from physics_guard import PhysicsGuard
from train_personal import Network
from live_mirror import CONNECTIONS

def main(render=True):
    predictions=np.load(ROOT/'private/learning/predictions.npz');audit=json.loads((ROOT/'reports/dataset_audit.json').read_text())
    reports=[];held=[x for x in audit['split'] if x['partition']=='test'];chosen=max(held,key=lambda x:x['valid_frames'])['index']
    renderer=None;writer=None;detector=None
    if render:
        m=load_hand();renderer=mujoco.Renderer(m,height=480,width=640)
        writer=cv2.VideoWriter(str(ROOT/'private/learning/personal_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(1920,520))
        if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
        detector=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_hands=2))
    for clip in held:
        mask=predictions['clip_ids']==clip['index'];t=predictions['timestamps'][mask];frames=predictions['frame_indices'][mask]
        models=[load_hand(),load_hand()];datas=[mujoco.MjData(m) for m in models];guards=[PhysicsGuard(m,d) for m,d in zip(models,datas)];last=[np.zeros(16),np.zeros(16)];maxexc=[0.,0.];motion=[[],[]];fault=None
        arrays=[predictions['baseline'][mask],predictions['neural'][mask]]
        cap=cv2.VideoCapture(clip['source']) if render and clip['index']==chosen else None
        decoded=-1
        try:
            for n,stamp in enumerate(t):
                views=[]
                for k,(m,d,g) in enumerate(zip(models,datas,guards)):
                    while d.time+1e-9<stamp:
                        age=d.time-(t[n-1] if n else 0)
                        request=arrays[k][n-1] if n and age<.3 else np.zeros(16)
                        last[k]+=np.clip(request-last[k],-2.5*m.opt.timestep,2.5*m.opt.timestep)
                        d.ctrl[:]=np.clip(last[k],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);g.step(d)
                        maxexc[k]=max(maxexc[k],float(np.max(np.maximum(m.jnt_range[:,0]-d.qpos,d.qpos-m.jnt_range[:,1]))))
                    motion[k].append(d.qpos.copy())
                    if cap:
                        renderer.update_scene(d,camera=camera_for(m));views.append(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
                if cap:
                    ok=True;frame=None
                    while decoded<int(frames[n]):
                        ok,frame=cap.read();decoded+=1
                        if not ok:break
                    if not ok:raise RuntimeError('Could not decode original comparison frame')
                    frame=cv2.resize(frame,(640,480));rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),round(stamp*1000)+n)
                    for hand in result.hand_landmarks:
                        xy=[(round(p.x*640),round(p.y*480)) for p in hand]
                        for a,b in CONNECTIONS:cv2.line(frame,xy[a],xy[b],(40,220,60),2)
                    canvas=np.hstack([frame,*views]);footer=np.zeros((40,1920,3),np.uint8)
                    for x,label in [(10,'PERSONAL held-out video + MediaPipe'),(650,'Phase 2 ROM baseline'),(1290,'PERSONAL trained MLP -> MuJoCo')]:cv2.putText(footer,label,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                    combined=np.vstack([canvas,footer]);writer.write(combined)
                    if n==len(t)//2:cv2.imwrite(str(ROOT/'private/learning/comparison.png'),combined)
        except Exception as e:fault=str(e)
        finally:
            if cap:cap.release()
        reports.append({'clip':clip['source'],'partition':'test','frames':len(t),'errors':fault,'models':[{'method':name,'steps':g.steps,'warnings':g.warnings(d),'max_joint_limit_excursion_rad':max(0.,ex),'joint_motion_range_rad':np.ptp(np.array(mot),axis=0).tolist() if mot else []} for name,g,d,ex,mot in zip(['baseline','neural'],guards,datas,maxexc,motion)]})
        print('physics replay',clip['index'],'error',fault,flush=True)
    if writer:writer.release()
    if renderer:renderer.close()
    if detector:detector.close()
    (ROOT/'reports/personal_replay_results.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-render',action='store_true');a=p.parse_args();main(not a.no_render)
