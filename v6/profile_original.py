"""Instrument the unchanged original neural live application at runtime only."""
import argparse,json,time,hashlib
from types import SimpleNamespace
import numpy as np,cv2,mujoco
import neural_live as app
from simulate import ROOT

def stat(x):return {'mean':float(np.mean(x)),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95))} if x else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=16);p.add_argument('--label',default='before');a=p.parse_args()
    times={k:[] for k in ['read_ms','render_ms','controller_ms','physics_step_ms','consume_frame_age_ms','hash_ms']};hashes=set();read_t=[]
    OriginalCapture=cv2.VideoCapture
    class Capture:
        def __init__(self,*args,**kwargs):self.cap=OriginalCapture(*args,**kwargs)
        def __getattr__(self,k):return getattr(self.cap,k)
        def read(self):
            tick=time.perf_counter();ok,f=self.cap.read();stamp=time.perf_counter();times['read_ms'].append((stamp-tick)*1000)
            if ok:
                h=time.perf_counter();digest=hashlib.sha256(f.tobytes()).digest();times['hash_ms'].append((time.perf_counter()-h)*1000)
                if digest not in hashes:hashes.add(digest);read_t.append(stamp)
            return ok,f
    cv2.VideoCapture=Capture
    original_latest=app.Camera.latest;last_seq=[None]
    def latest(self):
        seq,stamp,f=original_latest(self)
        if seq!=last_seq[0] and f is not None:times['consume_frame_age_ms'].append((time.perf_counter()-stamp)*1000);last_seq[0]=seq
        return seq,stamp,f
    app.Camera.latest=latest
    OriginalRenderer=mujoco.Renderer
    class Renderer:
        def __init__(self,*args,**kwargs):self.renderer=OriginalRenderer(*args,**kwargs)
        def __getattr__(self,k):return getattr(self.renderer,k)
        def render(self,*args,**kwargs):
            tick=time.perf_counter();f=self.renderer.render(*args,**kwargs);times['render_ms'].append((time.perf_counter()-tick)*1000);return f
    mujoco.Renderer=Renderer
    original_update=app.Retargeter.update
    def update(self,*args,**kwargs):
        tick=time.perf_counter();q=original_update(self,*args,**kwargs);times['controller_ms'].append((time.perf_counter()-tick)*1000);return q
    app.Retargeter.update=update
    original_step=app.PhysicsGuard.step
    def step(self,*args,**kwargs):
        tick=time.perf_counter();q=original_step(self,*args,**kwargs);times['physics_step_ms'].append((time.perf_counter()-tick)*1000);return q
    app.PhysicsGuard.step=step
    label='v6_original_'+a.label
    result=app.run(SimpleNamespace(camera=0,hand='right',seconds=a.seconds,output=label,evidence=False))
    report=json.loads((ROOT/f'reports/{label}_metrics.json').read_text());report['instrumentation']={k:stat(v) for k,v in times.items()};report['unique_camera_frames']=len(hashes);report['unique_camera_fps']=(len(read_t)-1)/(read_t[-1]-read_t[0]) if len(read_t)>1 else 0
    report['scope']='Original neural_live.py code/targeting unchanged; runtime wrappers time capture, render, control, physics and consumption. Hash instrumentation adds measured CPU cost. No camera imagery saved.'
    (ROOT/f'v6/original_{a.label}_profile.json').write_text(json.dumps(report,indent=2));raise SystemExit(result)

if __name__=='__main__':main()
