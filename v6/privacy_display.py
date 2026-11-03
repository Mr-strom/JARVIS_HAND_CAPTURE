"""Private presentation-only VIDEO landmark mask, preserving source aspect ratio."""
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from simulate import ROOT
from live_mirror import CONNECTIONS

class VideoHandMask:
    def __init__(self):
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_hands=2)
        self.detector=vision.HandLandmarker.create_from_options(options);self.last=-1;self.total=0;self.detected=0
    def mask(self,frame,timestamp):
        h,w=frame.shape[:2];frame=cv2.resize(frame,(640,round(h*640/w)));h,w=frame.shape[:2]
        stamp=max(self.last+1,round(timestamp*1000));self.last=stamp
        result=self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),stamp);self.total+=1;canvas=np.zeros((480,640,3),np.uint8)
        if not result.hand_landmarks:return canvas
        i=max(range(len(result.handedness)),key=lambda k:(result.handedness[k][0].category_name=='Right',result.handedness[k][0].score));xy=np.rint(np.array([[p.x,p.y] for p in result.hand_landmarks[i]])*[w,h]).astype('int32')
        mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(xy),255);mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
        image=np.zeros_like(frame);image[mask>0]=frame[mask>0]
        for a,b in CONNECTIONS:cv2.line(image,tuple(xy[a]),tuple(xy[b]),(60,220,90),1)
        for point in xy:cv2.circle(image,tuple(point),2,(255,220,60),-1)
        x,y,bw,bh=cv2.boundingRect(mask);crop=image[y:y+bh,x:x+bw];scale=min(600/max(1,bw),430/max(1,bh));crop=cv2.resize(crop,(max(1,round(bw*scale)),max(1,round(bh*scale))))
        oy=(480-crop.shape[0])//2;ox=(640-crop.shape[1])//2;canvas[oy:oy+crop.shape[0],ox:ox+crop.shape[1]]=crop;self.detected+=1;return canvas
    def close(self):self.detector.close()
