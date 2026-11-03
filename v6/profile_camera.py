"""Capture-only measurements; no private imagery saved and no persistent settings."""
import time,json,argparse
from pathlib import Path
import cv2,numpy as np
from v6.capture import LatestCamera
from simulate import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=8);p.add_argument('--exposure-test',action='store_true');a=p.parse_args()
    configs=[(cv2.CAP_DSHOW,640,480,False,None),(cv2.CAP_DSHOW,320,240,False,None),(cv2.CAP_MSMF,640,480,False,None),(cv2.CAP_DSHOW,640,480,True,None)]
    if a.exposure_test:configs.append((cv2.CAP_DSHOW,640,480,False,-5))
    rows=[]
    for backend,w,h,mjpg,exposure in configs:
        camera=None;start=time.perf_counter()
        try:
            camera=LatestCamera(backend=backend,width=w,height=h,mjpg=mjpg,exposure=exposure);start=time.perf_counter()
            while time.perf_counter()-start<a.seconds:
                if camera.error:raise RuntimeError(camera.error)
                time.sleep(.02)
            elapsed=time.perf_counter()-start;camera.close()
            times=camera.unique_times;rows.append({'settings':camera.settings,'unique_frames':len(times),'unique_fps':(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0,'return_fps':camera.total/elapsed,'duplicate_returns':camera.duplicates,'mean_read_ms':float(np.mean(camera.reads)),'p95_read_ms':float(np.percentile(camera.reads,95)),'error':None})
        except Exception as e:
            if camera:camera.close()
            rows.append({'requested':[backend,w,h,mjpg,exposure],'error':str(e)})
        print(json.dumps(rows[-1]),flush=True)
    (ROOT/'v6/camera_capture_profile.json').write_text(json.dumps({'runs':rows,'scope':'exact full-frame hash uniqueness; capture-read-completion timestamps, not exposure timestamps; no recording saved; no inference/render','lighting':'not controlled or measured; sequential same-session tests'},indent=2))

if __name__=='__main__':main()
