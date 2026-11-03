"""Measure delivered frame rate and read blocking without inference or saved imagery."""
import sys,time,json,hashlib
from pathlib import Path
import cv2,numpy as np
root=Path(__file__).resolve().parents[1];results=[]
for backend,width,height in [(cv2.CAP_DSHOW,640,480),(cv2.CAP_DSHOW,320,240),(cv2.CAP_MSMF,640,480)]:
    cap=cv2.VideoCapture(0,backend)
    if not cap.isOpened():results.append({'backend':backend,'opened':False});continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height);cap.set(cv2.CAP_PROP_FPS,30)
    ticks=[];hashes=set();start=time.perf_counter();frames=0;shape=None
    while time.perf_counter()-start<8:
        t=time.perf_counter();ok,frame=cap.read();ticks.append((time.perf_counter()-t)*1000)
        if not ok:break
        frames+=1;shape=frame.shape;hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
    elapsed=time.perf_counter()-start;cap.release()
    results.append({'backend':backend,'requested_resolution':[width,height],'actual_shape':shape,'frames':frames,'elapsed_seconds':elapsed,'fps':frames/elapsed,'mean_read_ms':float(np.mean(ticks)),'p95_read_ms':float(np.percentile(ticks,95)),'distinct_frame_hashes':len(hashes),'scope':'capture only, no inference/render; no exposure settings changed'})
    print(results[-1],flush=True)
(root/'reports/camera_benchmark.json').write_text(json.dumps(results,indent=2))
