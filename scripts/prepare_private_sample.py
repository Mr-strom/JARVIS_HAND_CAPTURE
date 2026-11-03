"""Local-only minimal personal sample and masked presentation panes; no upload.

All RGB outside the detected hand convex hull is replaced by black. This is a
privacy aid, not an anonymity guarantee; manually review before publication.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from simulate import ROOT

OUT=ROOT/'private/final';OUT.mkdir(parents=True,exist_ok=True)
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


class Masker:
    def __init__(self):
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.IMAGE,num_hands=2)
        self.detector=vision.HandLandmarker.create_from_options(options)
    def close(self):self.detector.close()
    def mask(self,frame,landmarks=False):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);r=self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb));out=np.zeros_like(frame)
        if not r.hand_landmarks:return out,None
        i=max(range(len(r.handedness)),key=lambda j:(r.handedness[j][0].category_name=='Right',r.handedness[j][0].score))
        p=np.array([[a.x,a.y,a.z] for a in r.hand_landmarks[i]])
        xy=np.rint(p[:,:2]*[frame.shape[1],frame.shape[0]]).astype('int32')
        mask=np.zeros(frame.shape[:2],np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(xy),255)
        mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
        out[mask>0]=frame[mask>0]
        if landmarks:
            for a,b in EDGES:cv2.line(out,tuple(xy[a]),tuple(xy[b]),(70,230,80),1,cv2.LINE_AA)
            for a in xy:cv2.circle(out,tuple(a),2,(255,220,40),-1)
        x,y,w,h=cv2.boundingRect(mask);crop=out[y:y+h,x:x+w]
        scale=min(frame.shape[1]*.8/max(w,1),frame.shape[0]*.8/max(h,1))
        crop=cv2.resize(crop,(max(1,round(w*scale)),max(1,round(h*scale))))
        out=np.zeros_like(frame);oy=(out.shape[0]-crop.shape[0])//2;ox=(out.shape[1]-crop.shape[1])//2
        out[oy:oy+crop.shape[0],ox:ox+crop.shape[1]]=crop
        return out,p


def sanitize_pane(masker,source,target,pane,raw_entry=None):
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(3));h=int(cap.get(4))
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot encode masked clip')
    count=0;detections=0;raw=None;decoded=-1
    if raw_entry is not None:
        raw=cv2.VideoCapture(raw_entry['source']);z=np.load(ROOT/raw_entry['cache']);raw_indices=z['frame_indices'][z['valid'].astype(bool)]
    while True:
        ok,f=cap.read()
        if not ok:break
        x,y,pw,ph=pane;input_frame=f[y:y+ph,x:x+pw]
        if raw is not None:
            while decoded<int(raw_indices[count]):
                ok,source_frame=raw.read();decoded+=1
                if not ok:raise RuntimeError('Original comparison source decode failed')
            input_frame=cv2.resize(source_frame,(pw,ph))
        image,p=masker.mask(input_frame,True);f[y:y+ph,x:x+pw]=image
        detections+=int(p is not None);writer.write(f);count+=1
    cap.release();writer.release()
    if raw is not None:raw.release()
    return {'path':str(target.relative_to(ROOT)),'frames':count,'hand_mask_detections':detections,'audio':'none: video-only encoder','mask':'RGB outside detected hand hull replaced by black; absent tracking black pane'}


def main():
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text());entry=max([e for e in audit['split'] if e['partition']=='test'],key=lambda e:e['valid_frames'])
    z=np.load(ROOT/entry['cache']);rows=np.where((z['timestamps']>=5)&(z['timestamps']<13))[0]
    masker=Masker();report=[];cap=cv2.VideoCapture(entry['source']);writer=cv2.VideoWriter(str(OUT/'personal_hand_sample.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(640,480));export=[];decoded=-1;sheets=[]
    try:
        for n,row in enumerate(rows):
            while decoded<int(z['frame_indices'][row]):
                ok,f=cap.read();decoded+=1
                if not ok:raise RuntimeError('Sample decode failed')
            frame=cv2.resize(f,(640,480));masked,p=masker.mask(frame,False);writer.write(masked)
            export.append({'source_frame':int(z['frame_indices'][row]),'source_timestamp_s':float(z['timestamps'][row]),'valid_cached':bool(z['valid'][row]),
                'canonical':z['canonical'][row].tolist() if z['valid'][row] else None,'features':z['features'][row].tolist() if z['valid'][row] else None,
                'image_landmarks_redaction_pass':None if p is None else p.tolist()})
            if n%8==0:sheets.append(cv2.resize(masked,(240,180)))
        cap.release();writer.release()
        meta={'schema':1,'personally_recorded':True,'partition':'test','recording_id':entry['index'],'source_sha256':entry['sha256'],
            'original_timestamp_range_s':[5,13],'nominal_export_fps':12,'warning':'Cached landmarks keep original timestamps; video nominal12fps, hand hull masked and zoomed. Image landmarks refer to original resized640x480 frame before zoom. This held-out sample cannot reproduce full training.',
            'coordinate_system':'canonical x=index/thumb-side,y=palm-forward,z=cross(x,y), wrist origin, index/little MCP span=1; reflection per original ingestion',
            'rows':export}
        (OUT/'personal_hand_landmarks.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
        while len(sheets)%4:sheets.append(np.zeros((180,240,3),np.uint8))
        cv2.imwrite(str(OUT/'sample_privacy_review.jpg'),np.vstack([np.hstack(sheets[i:i+4]) for i in range(0,len(sheets),4)]))
        report.append({'path':'private/final/personal_hand_sample.mp4','frames':len(export),'audio':'none','original_face_and_room':'masked outside hand hull; manual review required','raw_original_preserved':True})
        report.append(sanitize_pane(masker,ROOT/'private/learning/personal_comparison.mp4',OUT/'leap_replay_masked.mp4',(0,0,640,480),entry))
        report.append(sanitize_pane(masker,OUT/'cross_embodiment.mp4',OUT/'cross_embodiment_masked.mp4',(0,0,640,480)))
    finally:
        masker.close();cap.release();writer.release()
    (ROOT/'reports/privacy_preparation.json').write_text(json.dumps({'published':False,'approval_required':True,'raw_review':'Two proposed original videos visibly contain face, body and private room. Original audio not transcribed or cleared. Do not publish originals.',
        'screening_limits':'Visual contact sheets and hand-hull masking reduce disclosure; no guarantee of full anonymization. Videos written without audio. Personal geometry/checkpoint remain personal derived artifacts.',
        'artifacts':report},indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
