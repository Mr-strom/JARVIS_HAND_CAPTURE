"""Real personal input and freshly stepped paired LEAP physics, original timing.

All resulting imagery stays private. No interpolation or unrelated footage is
counted as captured input.30fps export repeats the latest original observation.
"""
import json,time
import cv2,mujoco,numpy as np
from v6.adaptive import AdaptiveEngine,MotionSafety
from v6.evaluate import angle_metrics
from v6.geometry import Geometry
from retargeting import Retargeter
from physics_guard import PhysicsGuard
from simulate import ROOT,load_hand,camera_for
from v6.privacy_display import VideoHandMask

FPS=30;SIZE=(1920,620)

def main():
    private=ROOT/'private/v6';private.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(private/'adaptive_precision_before_after.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),FPS,SIZE)
    if not writer.isOpened():raise RuntimeError('Cannot encode demonstration')
    manifest={'private':True,'published':False,'fps':FPS,'resolution':SIZE,'segments':[],'input':'original personal recording33, cached original landmarks/features; new privacy pass only for displayed hand mask/landmark drawing','timing':'original source timestamps, no sped-up playback; latest sampled human image repeated between observations; actual500Hz physics, ~29.4Hz bounded control,30fps export'}
    def card(title,lines,seconds):
        f=np.full((SIZE[1],SIZE[0],3),(23,20,17),np.uint8);cv2.putText(f,title,(45,80),cv2.FONT_HERSHEY_SIMPLEX,1.1,(90,220,235),2,cv2.LINE_AA)
        for i,line in enumerate(lines):cv2.putText(f,line,(45,155+60*i),cv2.FONT_HERSHEY_SIMPLEX,.8,(235,235,235),1,cv2.LINE_AA)
        for _ in range(round(seconds*FPS)):writer.write(f)
    card('JARVIS-Hand v6 | Adaptive Precision Retargeting Engine',[
        'Same personal MLP; bounded geometry refinement only when selected.',
        'LEFT: real personal recording + landmarks. CENTER: neural + v5 filter. RIGHT: adaptive engine.',
        'All robot views below are newly stepped MuJoCo physics, with matching source timestamps.',
        'Private demonstration. Geometry targets are estimates, not motion-capture ground truth.'],4)
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text());entry=next(e for e in audit['split'] if e['index']==33);z=np.load(ROOT/entry['cache']);times=z['timestamps']-z['timestamps'][0]
    measure=Geometry();total_frames=4*FPS
    def replay(dropout,intervals):
        nonlocal total_frames
        masker=VideoHandMask();m=load_hand();states={k:mujoco.MjData(m) for k in ['neural_v5','hybrid']};engines={k:AdaptiveEngine('neural' if k=='neural_v5' else 'hybrid') for k in states};names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
        filters={'neural_v5':Retargeter(names,m.actuator_ctrlrange,engines['neural_v5'].personal.rom.calibration),'hybrid':MotionSafety(m.actuator_ctrlrange)};guards={k:PhysicsGuard(m,d) for k,d in states.items()}
        for d in states.values():mujoco.mj_forward(m,d)
        for e in engines.values():
            valid=np.flatnonzero(z['valid'])[0];x=np.r_[z['canonical'][valid].ravel(),z['features'][valid]];q=e.personal.predict(x);goal=e.geometry.targets(z['canonical'][valid])
            for _ in range(5):e.geometry.solve(goal,q,8,4)
        decisions={k:None for k in states};index=0;current=-1;lastctrl=-1e9;nextrender=0.;end=max(b for a,b,label in intervals);cap=cv2.VideoCapture(entry['source']);decoded=-1;latest_image=None;source_cache_index=None;goal=None;errors={k:[] for k in states};timeouts=0;solver_attempts=0;counts={label:0 for a,b,label in intervals}
        with mujoco.Renderer(m,height=480,width=640) as renderer:
            while states['hybrid'].time<end:
                t=float(states['hybrid'].time)
                while index<len(times) and times[index]<=t+1e-9:
                    current=index;hidden=dropout and 2<=times[index]%4<2.45;features=np.r_[z['canonical'][index].ravel(),z['features'][index]] if z['valid'][index] and not hidden else None
                    for k,e in engines.items():
                        dec=e.observe(features,float(times[index]));decisions[k]=dec
                        info=dec.info.get('solver',{})
                        if k=='hybrid' and info:solver_attempts+=1;timeouts+=int(info['timeout'])
                    goal=measure.targets(z['canonical'][index]) if z['valid'][index] else None;index+=1
                if t-lastctrl>=1/30-1e-9:
                    for k,f in filters.items():
                        dec=decisions[k]
                        if k=='neural_v5':
                            if dec is not None and dec.target is not None and t-dec.timestamp<=.25:f.request=dec.target.copy();f.last_seen=dec.timestamp
                            command=f.update(None,t)
                        else:command=f.update(dec,t)
                        states[k].ctrl[:]=command
                    lastctrl=t
                for k,d in states.items():guards[k].step(d)
                if t+1e-9>=nextrender:
                    nextrender+=1/FPS;selected=next(((a,b,label) for a,b,label in intervals if a<=t<b),None)
                    if selected is None or current<0:continue
                    if source_cache_index!=current:
                        wanted=int(z['frame_indices'][current])
                        while decoded<wanted:
                            ok,frame=cap.read();decoded+=1
                            if not ok:raise RuntimeError('Original personal video could not decode')
                        latest_image=masker.mask(frame,float(z['timestamps'][current]));source_cache_index=current
                    views=[]
                    for k,d in states.items():
                        renderer.update_scene(d,camera=camera_for(m));f=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);dec=decisions[k];mode=dec.mode if dec is not None else 'RECOVERY';mode=mode if filters[k].state=='TRACKING' else filters[k].state
                        error=angle_metrics(measure,d.qpos,goal)[0] if goal is not None else None
                        if error is not None:errors[k].append(error)
                        cv2.putText(f,f'{k} | {mode}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                        text=f'Physical fingertip proxy {error:.1f} mm' if error is not None else 'No reliable target geometry'
                        cv2.putText(f,text,(12,458),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);views.append(f)
                    source=latest_image.copy();hidden=dropout and 2<=t%4<2.45
                    cv2.putText(source,f'Personal source t={z["timestamps"][current]:.3f}s',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    if hidden:cv2.putText(source,'SIMULATED INPUT DROPOUT',(12,455),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,170,255),2)
                    footer=np.zeros((140,1920,3),np.uint8);dec=decisions['hybrid'];reason=dec.reason if dec else 'waiting'
                    lines=[selected[2]+' | Original timing; sampled source frames held between observations.',
                        f'Adaptive selection: {reason}. Command speed <=2.5 rad/s; acceleration <=30 rad/s^2.',
                        'Held-out accepted inputs: raw tip MLP31.11mm / hybrid27.28mm / always-optimizer22.06mm.',
                        'Mean physical replay: v5-filtered neural49.93mm / hybrid50.10mm. Improvement is a tradeoff, not universal.']
                    for line_i,line in enumerate(lines):cv2.putText(footer,line,(15,25+30*line_i),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,235,235),1,cv2.LINE_AA)
                    writer.write(np.vstack([np.hstack([source,*views]),footer]));total_frames+=1;counts[selected[2]]+=1
                    if counts[selected[2]]==80:cv2.imwrite(str(private/('demo_'+selected[2].replace(' ','_')+'.jpg')),np.vstack([np.hstack([source,*views]),footer]))
        cap.release();masker.close();manifest['segments'].append({'dropouts':dropout,'intervals':intervals,'frames':counts,'privacy_mask_source_observations':masker.total,'privacy_mask_detections':masker.detected,'warnings':{k:g.warnings(states[k]) for k,g in guards.items()},'solver_attempts':solver_attempts,'solver_timeouts':timeouts,'rendered_interval_mean_tip_error_mm':{k:float(np.mean(v)) for k,v in errors.items()}})
    try:
        replay(False,[(4.,9.,'Precision pinch'),(44.,49.,'Faster movement')])
        card('Temporary tracking failure | deterministic experiment',[
            'Same personal recording; inputs withheld for450ms at2s and6s.',
            'Both controllers hold briefly and return safely. Reacquisition is rate-limited.',
            'This is injected tracking loss, not a claim that the camera failed.'],3);total_frames+=3*FPS
        replay(True,[(0.,8.,'Dropout and recovery')])
        card('Measured result | keep the verified v5 default',[
            'Hybrid improves raw geometry using about half the always-optimizer mean mapping time.',
            'Peak command acceleration falls147 ->30 rad/s^2; physical mean error is nearly unchanged.',
            'Prediction was rejected. Live attended hand-following remains unverified in this sprint.',
            'No new grasp or Shadow accuracy claim. No files published; private footage needs approval.'],4);total_frames+=4*FPS
    finally:writer.release()
    manifest['frames']=total_frames;manifest['seconds']=total_frames/FPS;(ROOT/'v6/DEMO_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
