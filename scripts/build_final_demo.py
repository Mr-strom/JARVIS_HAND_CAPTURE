"""Compose real captured outputs; no new physics results or publication action."""
from pathlib import Path
import cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[1];FPS=15;SIZE=(1280,720)

def main():
    out=ROOT/'private/final/JARVIS_Hand_final_demo.mp4';writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),FPS,SIZE)
    if not writer.isOpened():raise RuntimeError('Cannot encode final video')
    segments=[];total=0
    def card(title,lines,seconds):
        nonlocal total
        f=np.full((720,1280,3),(27,23,18),np.uint8)
        cv2.putText(f,title,(40,95),cv2.FONT_HERSHEY_SIMPLEX,.95,(90,225,230),2,cv2.LINE_AA)
        for i,line in enumerate(lines):cv2.putText(f,line,(40,185+60*i),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
        count=round(seconds*FPS)
        for _ in range(count):writer.write(f)
        total+=count
    def clip(path,label,start=0,seconds=8):
        nonlocal total
        cap=cv2.VideoCapture(str(ROOT/path));fps=cap.get(cv2.CAP_PROP_FPS);n=0;written=0
        if not cap.isOpened():raise RuntimeError(path)
        while True:
            ok,f=cap.read()
            if not ok:break
            t=n/max(fps,1);n+=1
            if t<start:continue
            if t>=start+seconds:break
            expected=round((t-start+1/max(fps,1))*FPS)
            if written>=expected:continue
            scale=min(1280/f.shape[1],650/f.shape[0]);image=cv2.resize(f,(round(f.shape[1]*scale),round(f.shape[0]*scale)))
            canvas=np.zeros((720,1280,3),np.uint8);x=(1280-image.shape[1])//2;y=(650-image.shape[0])//2;canvas[y:y+image.shape[0],x:x+image.shape[1]]=image
            cv2.putText(canvas,label,(15,692),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
            while written<expected:writer.write(canvas);written+=1
        cap.release();segments.append({'source':path,'start_s':start,'requested_seconds':seconds,'written_frames':written,'caption':label});total+=written
    card('JARVIS-Hand: personal learning, measured transfer',[
        '34 personal videos | 28,816-parameter CPU MLP | real MuJoCo execution',
        'Learned: right LEAP retargeting. Analytical: Shadow adapter.',
        'Scripted: arm and left-hand coordination. Pelvis fixed.',
        'No successful physical grasp is claimed.'],4)
    clip('private/final/leap_replay_masked.mp4','A | Same personal frames: hand + landmarks / ROM / trained MLP. Mask misses are black; nominal export timing.',5,12)
    card('A | Personal LEAP learning results',[
        'Personal video -> landmarks -> bounded FK teacher -> CPU MLP.',
        'Original training log: epoch 77; train loss .013664 / validation .039865.',
        'Fingertip proxy: ROM 61.32 mm / optimizer 27.83 mm / MLP 32.96 mm.',
        'Five held-out clips / 921 observations. Whole-recording splits.',
        'Optimizer remains more accurate. Geometry proxies, not external truth.'],4)
    clip('private/g1_personal_transfer.mp4','B | Full fixed-base G1: right personal MLP via geometry adapter; arms and left hand scripted. Pose montage.',0,2.6)
    clip('private/final/cross_embodiment_masked.mp4','B | Same held-out personal frames and actual Shadow physics: direct ROM / ROM via LEAP / MLP via LEAP.',0,10)
    card('B | Transfer did not improve fingertip alignment',[
        'Normalized physical tip error: direct ROM .918 / geometry ROM .895 / MLP .944.',
        'Units: each hand own palm width. Five recording means, no test fitting.',
        'Zero numerical warnings; small raw soft-limit excursions were counted.',
        'Lower actuator tracking error does not imply better human geometry.'],4)
    clip('private/final/grasp_recovery.mp4','C | Free 35 g cylinder: actual contacts/forces, thumb preposition and bounded closure. Failed attempt.',1.8,7)
    card('C | Honest task outcome and submission boundary',[
        'Final bounded LEAP recovery: 0/5 valid two-second lift-and-holds.',
        'Archived humanoid trials: 0/5 single and 0/5 assisted; deep contacts rejected.',
        'No weld, object teleport or disabled collision shortcut.',
        'Source, reproducible metrics and limitations accompany this real footage.',
        'Personal data, weights and this silent masked video need release approval.'],5)
    writer.release();report={'path':str(out.relative_to(ROOT)),'frames':total,'fps':FPS,'seconds':total/FPS,'size':SIZE,'audio':False,'segments':segments,'published':False,'private_approval_required':True}
    (ROOT/'reports/final_demo_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
