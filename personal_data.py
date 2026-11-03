"""Non-destructive personal-video discovery and cached CPU hand extraction."""
from pathlib import Path
import hashlib,json,time,argparse
import cv2,numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from retargeting import joint_features
from simulate import ROOT

EXT={'.mp4','.mov','.avi','.mkv','.webm','.m4v','.wmv'}
SCHEMA=2
def canonical(p):
    x=p[5]-p[17];scale=np.linalg.norm(x);x=x/scale
    y=p[9]-p[0];y-=x*np.dot(x,y);y/=np.linalg.norm(y)
    z=np.cross(x,y)
    return (p-p[0])@np.stack([x,y,z],axis=1)/scale

def classify(p,f):
    pinch=np.linalg.norm(p[4]-p[8])
    bends=np.mean(f[[0,2,3,4,6,7,8,10,11]])
    if pinch<.25:return 'pinch'
    if bends<.25:return 'open'
    if bends>.9:return 'curled'
    if np.isfinite(bends):return 'mixed_flexion'
    return 'unknown'

def ingest(source):
    source=Path(source).resolve();cache=ROOT/'data/personal/processed';cache.mkdir(parents=True,exist_ok=True)
    videos=sorted(p for p in source.rglob('*') if p.suffix.lower() in EXT)
    entries=[];seen={}
    for path in videos:
        digest=hashlib.sha256(path.read_bytes()).hexdigest();out=cache/(digest+'.npz')
        meta=cache/(digest+'.json')
        if digest in seen:
            entry=dict(seen[digest]);entry['source']=str(path);entry['duplicate_of']=seen[digest]['source'];entries.append(entry);continue
        if out.exists() and meta.exists() and json.loads(meta.read_text()).get('schema')==SCHEMA:
            entry=json.loads(meta.read_text());entry['source']=str(path);entries.append(entry);seen[digest]=entry;print('cached',path.name,flush=True);continue
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0:fps=30.
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'assets/models/hand_landmarker.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_hands=2)
        times=[];indices=[];world=[];norm=[];features=[];valid=[];labels=[];gestures=[];hashes=[];wrist=[];orient=[]
        last_ms=-1;frame_i=0;start=time.perf_counter();errors=0
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok,frame=cap.read()
                if not ok:break
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if timestamp<=0 and frame_i:timestamp=frame_i/fps
                # Nominal 12Hz via integer frame stride; effective rates vary (e.g. 10-15Hz).
                if frame_i%max(1,round(fps/12)):
                    frame_i+=1;continue
                gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(9,8))
                hashes.append(np.packbits(gray[:,1:]>gray[:,:-1]).tobytes().hex())
                small=cv2.resize(frame,(640,round(frame.shape[0]*640/frame.shape[1])))
                rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
                ms=max(last_ms+1,round(timestamp*1000));last_ms=ms
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ms)
                times.append(timestamp);indices.append(frame_i);frame_i+=1
                choice=None
                if result.handedness:
                    # Prefer detected right; reflect detected left world x for canonicalization.
                    choice=max(range(len(result.handedness)),key=lambda i:(result.handedness[i][0].category_name=='Right',result.handedness[i][0].score))
                p=np.full((21,3),np.nan);q=p.copy();f=np.full(16,np.nan);label='none';g='unknown';image_wrist=np.full(3,np.nan);basis=np.full(9,np.nan)
                if choice is not None:
                    label=result.handedness[choice][0].category_name
                    p=np.array([[a.x,a.y,a.z] for a in result.hand_world_landmarks[choice]])
                    image_wrist=np.array([result.hand_landmarks[choice][0].x,result.hand_landmarks[choice][0].y,result.hand_landmarks[choice][0].z])
                    if label=='Left':p[:,0]*=-1
                    try:
                        q=canonical(p);f=joint_features(p);g=classify(q,f)
                        basis=np.r_[p[5]-p[17],p[9]-p[0],np.cross(p[5]-p[17],p[9]-p[0])]
                    except ValueError:errors+=1
                good=bool(np.isfinite(f).all() and np.isfinite(q).all())
                world.append(p);norm.append(q);features.append(f);valid.append(good);labels.append(label);gestures.append(g);wrist.append(image_wrist);orient.append(basis)
        cap.release();t=np.array(times);q=np.array(norm);v=np.full_like(q,np.nan);a=v.copy()
        for i in range(1,len(t)):
            if valid[i] and valid[i-1] and t[i]>t[i-1]:v[i]=(q[i]-q[i-1])/(t[i]-t[i-1])
        for i in range(2,len(t)):
            if np.isfinite(v[i:i+1]).all() and np.isfinite(v[i-1:i]).all():a[i]=(v[i]-v[i-1])/(t[i]-t[i-1])
        np.savez_compressed(out,timestamps=t,frame_indices=indices,world=world,canonical=q,features=features,valid=valid,handedness=labels,gesture=gestures,dhash=hashes,velocity=v,acceleration=a,wrist_image=wrist,palm_orientation=orient)
        entry={'schema':SCHEMA,'sha256':digest,'source':str(path),'cache':str(out.relative_to(ROOT)),'fps':fps,'decoded_frames':frame_i,'reported_frames':count,'sampled_frames':len(valid),'valid_frames':sum(valid),'coverage':sum(valid)/max(1,len(valid)),'seconds':count/fps,'wall_seconds':time.perf_counter()-start,'geometry_errors':errors,'handedness_counts':{x:labels.count(x) for x in set(labels)},'gesture_counts':{x:gestures.count(x) for x in set(gestures)},'orientation':'decoded orientation; mirror state cannot be inferred reliably without external reference; detected-left reflection applied','finger_labels':'geometry only; folder and filenames are not labels'}
        meta.write_text(json.dumps(entry,indent=2));entries.append(entry);seen[digest]=entry
        print(path.name,entry['valid_frames'],'/',entry['sampled_frames'],flush=True)
    manifest={'source_root':str(source),'schema':SCHEMA,'recordings':entries,'originals_modified':False}
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(ROOT/'data/personal/raw'));args=p.parse_args();ingest(args.source)
