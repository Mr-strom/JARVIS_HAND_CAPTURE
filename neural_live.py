# Additive neural variant of Phase 2; original live_mirror.py preserved.
"""Live webcam -> MediaPipe CPU -> bounded LEAP actuator control. No grasping."""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import threading
import time
import traceback

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mujoco
import numpy as np

from physics_guard import PhysicsGuard
from retargeting import Calibration
from learning.runtime import NeuralRetargeter as Retargeter, neural_features as joint_features
from simulate import ROOT, load_hand, camera_for

CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
             (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


class Camera:
    def __init__(self,index):
        self.index=index; self.lock=threading.Lock(); self.stop=threading.Event(); self.ready=threading.Event()
        self.frame=None; self.seq=0; self.timestamp=0.;self.first_time=None;self.error=None;self.backend=None
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
        if not self.ready.wait(12):
            self.stop.set();raise RuntimeError('Camera startup timed out')
        if self.error:
            raise RuntimeError(self.error)

    def _run(self):
        cap=None
        try:
            for backend in (cv2.CAP_DSHOW,cv2.CAP_MSMF):
                cap=cv2.VideoCapture(self.index,backend)
                if cap.isOpened():
                    self.backend=cap.getBackendName();break
                cap.release()
            if not cap or not cap.isOpened():
                raise RuntimeError(f'Cannot open webcam {self.index}')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_FPS,30)
            self.ready.set()
            failures=0
            while not self.stop.is_set():
                success,frame=cap.read();stamp=time.perf_counter()
                if not success:
                    failures+=1
                    if failures>=10:raise RuntimeError('Camera stopped delivering frames')
                    time.sleep(.02);continue
                failures=0
                with self.lock:
                    self.frame=frame;self.timestamp=stamp;self.seq+=1
                    if self.first_time is None:self.first_time=stamp
        except Exception as exc:
            self.error=str(exc);self.ready.set()
        finally:
            if cap is not None:cap.release()

    def latest(self):
        with self.lock:return self.seq,self.timestamp,self.frame

    def close(self):
        self.stop.set();self.thread.join(timeout=3)


def stats(values):
    if not values:return {'mean':None,'p50':None,'p95':None}
    return {'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95))}


def response_lag(samples):
    """Estimate raw-target -> measured-joint lag, only with sufficient real motion."""
    if len(samples)<60:return {'available':False,'reason':'insufficient tracked samples'}
    a=np.asarray(samples);times=a[:,0]
    if times[-1]-times[0]<3:return {'available':False,'reason':'tracked duration below 3 seconds'}
    grid=np.arange(times[0],times[-1],1/30)
    candidates=[]
    for j in range(16):
        x=np.interp(grid,times,a[:,1+j]);y=np.interp(grid,times,a[:,33+j])
        if np.ptp(x)<.25 or np.ptp(y)<.15:continue
        x=np.diff(x);y=np.diff(y)
        correlations=[]
        for lag in range(16):
            xx=x[:len(x)-lag] if lag else x
            yy=y[lag:]
            if len(xx)<30 or np.std(xx)<1e-5 or np.std(yy)<1e-5:correlations.append(-1.);continue
            correlations.append(float(np.corrcoef(xx,yy)[0,1]))
        best=int(np.argmax(correlations))
        if correlations[best]>.5:candidates.append({'joint_index':j,'lag_ms':best*1000/30,'correlation':correlations[best]})
    if not candidates:return {'available':False,'reason':'insufficient correlated movement'}
    resolution=max(1000/30,float(np.median(np.diff(times)))*1000)
    median_lag=float(np.median([x['lag_ms'] for x in candidates]))
    # Interpolating a slow webcam stream cannot create finer timing information.
    # A best fit at lag zero is an unresolved bin, not proof of zero latency.
    if median_lag < resolution:
        return {'available':False,'reason':'best-fit lag is below native sample resolution; zero latency is NOT established',
                'native_sample_resolution_ms':resolution,'diagnostic_channels':candidates}
    return {'available':True,'median_lag_ms':median_lag,
            'resolution_ms':resolution,'channels':candidates,'scope':'raw landmark-derived target to measured robot joint; camera/display latency excluded'}


def run(args):
    calibration_path=ROOT/'private/neural_preview_calibration.json'
    calibration=Calibration.load(calibration_path) if calibration_path.exists() else Calibration()
    model=load_hand();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    joint_ids=model.actuator_trnid[:,0];qindices=model.jnt_qposadr[joint_ids]
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in joint_ids]
    controller=Retargeter(names,model.actuator_ctrlrange,calibration)
    guard=PhysicsGuard(model,data)
    renderer=None;camera=None;detector=None;error=None
    process_ms=[];pipeline_ms=[];telemetry=[];update_times=[];detected=0;processed=0;updates=0;clipped_frames=0;stale_frames=0
    loss_transitions=0;states={};joint_violation_steps=0;max_excursion=0.;max_contacts=0;max_penetration=0.
    calibration_samples=[];calibration_end=None;message='PERSONAL trained MLP. Q quits; C only checks feature ROM, not model training'
    evidence_low=None;evidence_high=None;low_score=float('inf');high_score=-float('inf')
    start=time.perf_counter();last_physics=start;accumulator=0.;dropped_time=0.;last_seq=0;last_timestamp=-1;latest_observed='none'
    last_preview=None;recent=deque(maxlen=60);report={};last_status=start
    title='JARVIS-Hand | PERSONAL MLP + LEAP | C calibrate, Q quit'
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    def summarize():
        elapsed=max(time.perf_counter()-start,1e-9)
        capture_duration=(camera.timestamp-camera.first_time) if camera and camera.first_time else 0
        return {'elapsed_seconds':elapsed,'camera_backend':camera.backend if camera else None,
            'webcam_frames':camera.seq if camera else 0,'webcam_fps':(camera.seq-1)/capture_duration if capture_duration>0 else None,
            'processed_frames':processed,'detected_frames':detected,'detection_ratio':detected/max(processed,1),
            'mediapipe_processing_ms':stats(process_ms),'robot_update_fps':updates/elapsed,'display_fps':updates/elapsed,
            'physics_steps':guard.steps,'physics_steps_per_wall_second':guard.steps/elapsed,
            'simulation_seconds':guard.steps*guard.dt,'dropped_catchup_seconds':dropped_time,
            'capture_read_complete_to_render_ms':stats(pipeline_ms),'control_limit_violations':clipped_frames,
            'joint_limit_violation_steps_tolerance_1e_4_rad':joint_violation_steps,'max_joint_limit_excursion_rad':max_excursion,
            'simulation_warnings':guard.warnings(data),'max_contacts':max_contacts,'max_penetration_m':max_penetration,
            'state_counts':states,'tracking_loss_transitions':loss_transitions,'last_detected_label':latest_observed,
            'hand_selection':args.hand,'personalized_channels':sum(calibration.personalized),'calibration_samples':calibration.samples,
            'stale_frames_rejected':stale_frames,
            'source':'live webcam -> personal trained MLP; no synthetic targets','cpu_inference':True,'error':error,
            'joint_names':names,'model_physics':'unchanged pinned Menagerie LEAP'}
    try:
        detector=vision.HandLandmarker.create_from_options(options)
        renderer=mujoco.Renderer(model,height=480,width=640)
        camera=Camera(args.camera)
        cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1280,560)
        start=time.perf_counter();last_physics=start
        while not args.seconds or time.perf_counter()-start<args.seconds:
            loop_begin=time.perf_counter()
            seq,captured,frame=camera.latest()
            features=None;landmarks=None;fresh=seq!=last_seq and frame is not None
            if camera.error:raise RuntimeError(camera.error)
            if fresh:
                last_seq=seq
                # MediaPipe and the preview both see the same selfie-mirrored image.
                preview=cv2.flip(frame,1)
                rgb=cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)
                timestamp=max(last_timestamp+1,round(captured*1000));last_timestamp=timestamp
                tick=time.perf_counter()
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
                process_ms.append((time.perf_counter()-tick)*1000);processed+=1
                choices=[]
                for i,hand in enumerate(result.handedness):
                    label=hand[0].category_name;score=hand[0].score
                    latest_observed=label
                    if args.hand=='any' or label.lower()==args.hand:choices.append((score,i))
                if choices:
                    _,index=max(choices)
                    landmarks=result.hand_landmarks[index]
                    world=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[index]])
                    try:features=joint_features(world)
                    except ValueError:features=None
                    if features is not None and time.perf_counter()-captured>.25:
                        stale_frames+=1;features=None
                    if features is not None:detected+=1
                if landmarks:
                    xy=[(int(p.x*preview.shape[1]),int(p.y*preview.shape[0])) for p in landmarks]
                    for a,b in CONNECTIONS:cv2.line(preview,xy[a],xy[b],(70,220,90),2)
                    for point in xy:cv2.circle(preview,point,3,(30,90,255),-1)
                last_preview=cv2.resize(preview,(640,480))
            if last_preview is None:
                cv2.waitKey(1);time.sleep(.005);continue
            now=time.perf_counter()
            if calibration_end is not None:
                if features is not None:calibration_samples.append(features.copy())
                message=f'CALIBRATE {max(0,calibration_end-now):.1f}s: open, curl, spread; thumb across palm'
                if now>=calibration_end:
                    try:
                        calibration.fit(np.asarray(calibration_samples)[:,-16:]);calibration.save(calibration_path)
                        message=f'Calibrated {sum(calibration.personalized)}/16 channels. C: repeat, Q: quit'
                    except ValueError as exc:message=str(exc)
                    calibration_end=None
            previous_state=controller.state
            control=controller.update(features,now)
            if previous_state in ('TRACKING','HOLD') and controller.state=='RETURNING':loss_transitions+=1
            states[controller.state]=states.get(controller.state,0)+1
            clipped_frames+=int(np.count_nonzero((control<model.actuator_ctrlrange[:,0])|(control>model.actuator_ctrlrange[:,1])))
            accumulator+=now-last_physics;last_physics=now
            if accumulator>.10:dropped_time+=accumulator-.10;accumulator=.10
            data.ctrl[:]=control
            while accumulator>=guard.dt:
                guard.step(data);accumulator-=guard.dt
                excursion=max(0.,float(np.max(np.maximum(model.jnt_range[joint_ids,0]-data.qpos[qindices],data.qpos[qindices]-model.jnt_range[joint_ids,1]))))
                max_excursion=max(max_excursion,excursion);joint_violation_steps+=int(excursion>1e-4)
                max_contacts=max(max_contacts,int(data.ncon))
                max_penetration=max(max_penetration,max([0.] + [-float(c.dist) for c in data.contact]))
                if excursion>.02:raise RuntimeError(f'Joint limit excursion {excursion:.6f} rad exceeds safety threshold')
            renderer.update_scene(data,camera=camera_for(model))
            robot=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            if fresh and features is not None:
                telemetry.append(np.concatenate(([now-start],controller.request.copy(),control.copy(),data.qpos[qindices].copy())))
            rendered=time.perf_counter()
            if fresh:pipeline_ms.append((rendered-captured)*1000)
            updates+=1;recent.append(rendered);update_times.append(rendered)
            fps=(len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 else 0.
            combined=np.hstack((last_preview,robot))
            footer=np.zeros((80,1280,3),dtype=np.uint8)
            lines=[f'{controller.state} | detected {detected}/{processed} | update {fps:.1f} FPS | MediaPipe {np.mean(process_ms[-30:]) if process_ms else 0:.1f} ms',
                   message, 'Right LEAP: thumb + index + middle + ring. Fixed wrist. No grasping. Q / Esc: quit']
            for i,line in enumerate(lines):cv2.putText(footer,line,(12,21+i*25),cv2.FONT_HERSHEY_SIMPLEX,.52,(210,235,230),1,cv2.LINE_AA)
            combined=np.vstack((combined,footer))
            cv2.imshow(title,combined)
            if args.evidence and features is not None:
                score=float(np.mean(controller.request))
                if score<low_score:low_score=score;evidence_low=combined.copy()
                if score>high_score:high_score=score;evidence_high=combined.copy()
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27):break
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==ord('c'):calibration_samples=[];calibration_end=time.perf_counter()+8
            if rendered-last_status>2:
                (ROOT/f'reports/{args.output}_status.json').write_text(json.dumps(summarize(),indent=2),encoding='utf-8')
                last_status=rendered
            remaining=1/30-(time.perf_counter()-loop_begin)
            if remaining>0:time.sleep(remaining)
    except Exception:
        error=traceback.format_exc()
        print(error,flush=True)
    finally:
        report=summarize()
        report['motion_response_lag']=response_lag(telemetry)
        report['latency_limitations']='Software pipeline starts when OpenCV read completes. Exposure, camera buffering, scanout and display latency are unmeasured. Filter and actuator lag are estimated separately from real tracked motion.'
        report['tracked_joint_motion_range_rad']=(np.ptp(np.asarray(telemetry)[:,33:],axis=0).tolist() if telemetry else None)
        report['tracked_raw_target_range_rad']=(np.ptp(np.asarray(telemetry)[:,1:17],axis=0).tolist() if telemetry else None)
        private=ROOT/'private';private.mkdir(exist_ok=True)
        if telemetry:np.savez_compressed(private/f'{args.output}_telemetry.npz',samples=np.asarray(telemetry))
        if args.evidence and evidence_low is not None:
            cv2.imwrite(str(private/f'{args.output}_open.png'),evidence_low)
            cv2.imwrite(str(private/f'{args.output}_curled.png'),evidence_high)
        (ROOT/f'reports/{args.output}_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if camera:camera.close()
        if renderer:renderer.close()
        if detector:detector.close()
        cv2.destroyAllWindows()
        print(json.dumps(report,indent=2),flush=True)
    return 1 if error else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--hand',choices=['right','left','any'],default='right')
    parser.add_argument('--seconds',type=float,default=0,help='0 runs until Q/Esc')
    parser.add_argument('--output',default='live_run')
    parser.add_argument('--evidence',action='store_true',help='Save two local private diagnostic screenshots')
    args=parser.parse_args()
    if not args.output.replace('_','').replace('-','').isalnum():parser.error('output must be a simple filename stem')
    if not np.isfinite(args.seconds) or args.seconds<0:parser.error('seconds must be finite and nonnegative')
    raise SystemExit(run(args))
