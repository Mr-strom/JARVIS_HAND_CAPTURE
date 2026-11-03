"""Timestamped latest-only camera producer. No simulator state is shared."""
import threading,time,hashlib
import cv2

class LatestCamera:
    def __init__(self,index=0,backend=cv2.CAP_DSHOW,width=640,height=480,mjpg=False,exposure=None):
        self.args=(index,backend,width,height,mjpg,exposure);self.lock=threading.Lock();self.stop=threading.Event();self.ready=threading.Event()
        self.frame=None;self.seq=0;self.timestamp=0.;self.error=None;self.reads=[];self.unique_times=[];self.total=0;self.duplicates=0;self.settings={};self.seen=set()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        if not self.ready.wait(12):self.stop.set();raise RuntimeError('Camera startup timeout')
        if self.error:raise RuntimeError(self.error)
    def _run(self):
        index,backend,width,height,mjpg,exposure=self.args;cap=None;original=None
        try:
            cap=cv2.VideoCapture(index,backend)
            if not cap.isOpened():raise RuntimeError('Camera could not open')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height);cap.set(cv2.CAP_PROP_FPS,30)
            if mjpg:cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            if exposure is not None:
                original=(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),cap.get(cv2.CAP_PROP_EXPOSURE))
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,.25);cap.set(cv2.CAP_PROP_EXPOSURE,exposure)
            self.settings={'backend':cap.getBackendName(),'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT),'reported_fps':cap.get(cv2.CAP_PROP_FPS),'exposure':cap.get(cv2.CAP_PROP_EXPOSURE),'auto_exposure':cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),'requested_mjpg':mjpg,'requested_exposure':exposure}
            self.ready.set();failures=0
            while not self.stop.is_set():
                before=time.perf_counter();ok,frame=cap.read();after=time.perf_counter()
                if not ok:
                    failures+=1
                    if failures>=10:raise RuntimeError('Camera read failed ten times')
                    continue
                failures=0;self.total+=1;self.reads.append((after-before)*1000)
                digest=hashlib.sha256(frame.tobytes()).digest()
                if digest in self.seen:self.duplicates+=1;continue
                self.seen.add(digest);self.unique_times.append(after)
                with self.lock:self.frame=frame;self.seq+=1;self.timestamp=after
        except Exception as e:self.error=str(e);self.ready.set()
        finally:
            if cap is not None:
                if original is not None:
                    cap.set(cv2.CAP_PROP_EXPOSURE,original[1]);cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,original[0]);self.settings['exposure_restore_attempted']=True
                cap.release()
    def latest(self):
        with self.lock:return self.seq,self.timestamp,self.frame
    def close(self):
        self.stop.set();self.thread.join(3)
        if self.thread.is_alive():raise RuntimeError('Camera worker did not stop')

