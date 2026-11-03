"""Latest-only camera/vision worker and main-thread MuJoCo adaptive control."""
import argparse,threading,time,json,collections
from pathlib import Path
import cv2,mujoco,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from v6.capture import LatestCamera
from v6.adaptive import AdaptiveEngine,MotionSafety
from physics_guard import PhysicsGuard
from simulate import ROOT,load_hand,camera_for
from live_mirror import CONNECTIONS

def stats(x):return {'mean':float(np.mean(x)),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95))} if x else None

class VisionWorker:
    def __init__(self,camera,engine):
        self.camera=camera;self.engine=engine;self.stop=threading.Event();self.lock=threading.Lock();self.packet=None;self.error=None;self.processed=0;self.detected=0;self.accepted=0;self.skipped=0;self.timings=collections.defaultdict(list);self.modes=collections.Counter();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        opts=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_hands=2)
        last=0;last_ms=-1
        try:
            with vision.HandLandmarker.create_from_options(opts) as detector:
                while not self.stop.is_set():
                    seq,captured,frame=self.camera.latest()
                    if seq==last or frame is None:self.stop.wait(.001);continue
                    self.skipped+=max(0,seq-last-1);last=seq;start=time.perf_counter();self.timings['capture_to_processing_ms'].append((start-captured)*1000)
                    xy=[];world=None;label='none';ms=max(last_ms+1,round(captured*1000));last_ms=ms
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);tick=time.perf_counter();result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ms);self.timings['mediapipe_ms'].append((time.perf_counter()-tick)*1000);self.processed+=1
                    if result.hand_world_landmarks:
                        i=max(range(len(result.handedness)),key=lambda k:(result.handedness[k][0].category_name=='Right',result.handedness[k][0].score));label=result.handedness[i][0].category_name
                        world=np.array([[a.x,a.y,a.z] for a in result.hand_world_landmarks[i]])
                        if label=='Left':world[:,0]*=-1 # same canonicalization as personal-data ingestion
                        xy=[(int(p.x*frame.shape[1]),int(p.y*frame.shape[0])) for p in result.hand_landmarks[i]];self.detected+=1
                    tick=time.perf_counter();decision=self.engine.observe_world(world,captured,time.perf_counter());self.timings['retarget_world_ms'].append((time.perf_counter()-tick)*1000);self.modes[decision.mode]+=1
                    finished=time.perf_counter();self.timings['capture_to_decision_ms'].append((finished-captured)*1000)
                    if decision.target is not None:self.accepted+=1
                    packet=(seq,captured,frame,tuple(xy),label,decision,finished)
                    with self.lock:self.packet=packet
        except Exception as e:self.error=repr(e)
    def latest(self):
        with self.lock:return self.packet
    def close(self):
        self.stop.set();self.thread.join(5)
        if self.thread.is_alive():raise RuntimeError('Vision/optimizer worker did not finish')

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=30);p.add_argument('--controller',choices=['neural','hybrid','optimizer'],default='neural');p.add_argument('--label',default='after');p.add_argument('--checkpoint',type=Path);p.add_argument('--calibration',type=Path);a=p.parse_args()
    m=load_hand();d=mujoco.MjData(m);mujoco.mj_forward(m,d);g=PhysicsGuard(m,d);engine=AdaptiveEngine(a.controller,a.checkpoint,a.calibration);control=MotionSafety(m.actuator_ctrlrange)
    renderer=mujoco.Renderer(m,height=480,width=640);cam=None;worker=None;timings=collections.defaultdict(list);fps_times=[];latest_seq=0;first_renders=[];stale=0;maxexc=0.;limit_count=0;dropped=0.;seen_modes=collections.Counter();error=None;trace=[];title='JARVIS-Hand v6 | Adaptive Precision | Q quit';start=time.perf_counter();last=start;acc=0.;last_frame=None;decision=None
    try:
        cam=LatestCamera();worker=VisionWorker(cam,engine);cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1280,560);start=time.perf_counter();last=start
        while time.perf_counter()-start<a.seconds:
            loop=time.perf_counter();packet=worker.latest();fresh=False
            if worker.error:raise RuntimeError(worker.error)
            if cam.error:raise RuntimeError(cam.error)
            if packet is not None:
                seq,captured,frame,xy,label,candidate,finished=packet
                fresh=seq!=latest_seq
                if fresh:
                    latest_seq=seq;decision=candidate;copy_tick=time.perf_counter();last_frame=frame.copy();timings['frame_copy_ms'].append((time.perf_counter()-copy_tick)*1000)
                    for i,j in CONNECTIONS:
                        if xy:cv2.line(last_frame,xy[i],xy[j],(60,220,90),2)
                    if time.perf_counter()-captured>.25:decision=None;stale+=1
            now=time.perf_counter();tick=time.perf_counter();target=control.update(decision,now);timings['control_filter_ms'].append((time.perf_counter()-tick)*1000);seen_modes[decision.mode if decision is not None and decision.target is not None and control.state=='TRACKING' else 'RECOVERY']+=1
            acc+=now-last;last=now
            if acc>.1:dropped+=acc-.1;acc=.1
            d.ctrl[:]=target
            while acc>=g.dt:
                tick=time.perf_counter();g.step(d);timings['physics_step_ms'].append((time.perf_counter()-tick)*1000);acc-=g.dt
                exc=np.maximum(m.jnt_range[:,0]-d.qpos,d.qpos-m.jnt_range[:,1]);maxexc=max(maxexc,float(exc.max()));limit_count+=int(np.count_nonzero(exc>1e-4))
                if maxexc>.02:raise RuntimeError('Simulation joint excursion safety failure')
            tick=time.perf_counter();renderer.update_scene(d,camera=camera_for(m));robot=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);timings['render_ms'].append((time.perf_counter()-tick)*1000)
            rendered=time.perf_counter();fps_times.append(rendered)
            if packet is not None:
                timings['displayed_frame_age_ms'].append((rendered-captured)*1000)
                if fresh:timings['first_render_pipeline_ms'].append((rendered-captured)*1000);first_renders.append(captured)
            preview=cv2.resize(last_frame,(640,480)) if last_frame is not None else np.zeros((480,640,3),np.uint8);footer=np.zeros((80,1280,3),np.uint8)
            mode=decision.mode if decision is not None and decision.target is not None else 'RECOVERY';reason=decision.reason if decision is not None else 'waiting/stale'
            for i,line in enumerate([f'{mode} / {control.state} | {reason}',f'Unique camera frames {cam.seq} | processed {worker.processed} | accepted {worker.accepted} | Q quit','Fixed wrist, no grasp. Camera unmirrored; Left labels reflected as in personal dataset.']):cv2.putText(footer,line,(10,22+i*24),cv2.FONT_HERSHEY_SIMPLEX,.5,(225,235,235),1)
            cv2.imshow(title,np.vstack([np.hstack([preview,robot]),footer]));trace.append(np.r_[now-start,target,d.qpos])
            if cv2.waitKey(1)&255 in (27,ord('q')):break
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            remaining=1/30-(time.perf_counter()-loop)
            if remaining>0:time.sleep(remaining)
    except Exception as e:error=repr(e)
    finally:
        end=time.perf_counter()
        if worker:worker.close()
        if cam:cam.close()
        renderer.close();cv2.destroyAllWindows()
    elapsed=end-start;times=cam.unique_times if cam else [];report={'error':error,'elapsed_seconds':elapsed,'unique_camera_frames':len(times),'unique_camera_fps':(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0,'camera_settings':cam.settings if cam else {},'robot_display_fps':len(fps_times)/elapsed,'unique_processed_frames':worker.processed if worker else 0,'unique_detected_frame_fps':worker.detected/elapsed if worker else 0,'unique_accepted_frame_fps':worker.accepted/elapsed if worker else 0,'skipped_latest_buffer_frames':worker.skipped if worker else 0,'stale_display_results_rejected':stale,'controller':a.controller,'mode_display_counts':dict(seen_modes),'worker_modes':dict(worker.modes) if worker else {},'timings':{k:stats(v) for k,v in timings.items()},'worker_timings':{k:stats(v) for k,v in worker.timings.items()} if worker else {},'sim_seconds':float(d.time),'dropped_catchup_seconds':dropped,'max_raw_joint_excursion_rad':max(0.,maxexc),'joint_limit_samples_over_1e_4':limit_count,'warnings':g.warnings(d),'latency_scope':'read completion -> first/displayed robot render; camera exposure/buffering/display scanout/filter settling excluded. Unique frames counted once, no interpolation.','tracking_quality':'geometry/continuity proxies, not landmark confidence'}
    (ROOT/f'v6/live_{a.label}_profile.json').write_text(json.dumps(report,indent=2));private=ROOT/'private/v6';private.mkdir(exist_ok=True,parents=True);np.savez_compressed(private/f'live_{a.label}_telemetry.npz',samples=trace);print(json.dumps(report,indent=2))
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
