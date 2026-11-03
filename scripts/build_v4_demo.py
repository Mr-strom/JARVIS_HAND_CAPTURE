"""Compose genuine local experiment footage with explicit scope/failure captions."""
import cv2,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FPS=15;size=(1280,720)
out=ROOT/'private/JARVIS_Hand_v4_technical_demo.mp4'
writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),FPS,size)
if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
def card(title,lines,seconds=4):
    image=np.zeros((720,1280,3),np.uint8);image[:]=[24,20,16]
    cv2.putText(image,title,(45,95),cv2.FONT_HERSHEY_SIMPLEX,1.,(100,220,230),2,cv2.LINE_AA)
    for i,line in enumerate(lines):cv2.putText(image,line,(45,170+i*48),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
    for _ in range(round(seconds*FPS)):writer.write(image)
def clip(path,label):
    cap=cv2.VideoCapture(str(ROOT/path));fps=cap.get(cv2.CAP_PROP_FPS);count=0;written=0
    if not cap.isOpened():raise RuntimeError(f'Missing experiment video: {path}')
    while True:
        ok,frame=cap.read()
        if not ok:break
        count+=1
        expected=round(count/max(fps,1)*FPS)
        if expected<=written:continue
        scale=min(1280/frame.shape[1],650/frame.shape[0]);image=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
        canvas=np.zeros((720,1280,3),np.uint8);y=(650-image.shape[0])//2;x=(1280-image.shape[1])//2;canvas[y:y+image.shape[0],x:x+image.shape[1]]=image
        cv2.putText(canvas,label,(20,690),cv2.FONT_HERSHEY_SIMPLEX,.52,(235,235,235),1,cv2.LINE_AA)
        while written<expected:writer.write(canvas);written+=1
    cap.release()
card('JARVIS-Hand: personal learning + humanoid experiments',[
'34 personal recordings; 3,672 usable detections; whole-recording holdouts.',
'Trained CPU MLP: 28,816 parameters. No locomotion or bimanual policy training.',
'Real MuJoCo physics. No object attachments or hidden grasp welds.',
'Research goal: contact-adaptive assistance. Task success is not established.'])
clip('private/learning/personal_comparison.mp4','Held-out personal samples: source + landmarks | ROM | trained MLP. Presentation sampling, not original playback timing.')
card('Held-out geometry results',[
'Fingertip target error: ROM 61.32mm / optimizer 27.83mm / neural 32.96mm.',
'Pinch-distance error: ROM 41.12mm / optimizer 8.93mm / neural 16.68mm.',
'Targets are monocular geometry proxies, not external motion-capture truth.',
'The optimizer remains more accurate. Original trained system is preserved.'])
clip('private/g1_both_arms_hands.mp4','Fixed-pelvis G1 at Earth gravity: both arms and both hands physically actuated. Scripted IK and finger commands.')
clip('private/g1_personal_transfer.mp4','Experimental right-hand transfer through LEAP FK geometry and Shadow ROM. Left/arms scripted; no precision-transfer claim.')
clip('private/g1_assisted_task.mp4','Attempted real-contact tabletop task. No successful grasp. Unsafe deep contact causes an explicit abort.')
card('Outcomes and limits',[
'G1: 60 seconds / 30,000 steps without numerical warnings; pelvis fixed.',
'LEAP grasp recovery: 0/5 valid holds. G1 lift/hold: 0/5 single, 0/5 assisted.',
'Deep-contact attempts rejected. Brief table support is not a successful grasp.',
'No learned left-hand coordination, locomotion, autonomous policy or LLM.',
'Personal recordings, weights and this video require approval before publishing.'],6)
writer.release();cap=cv2.VideoCapture(str(out));report={'path':str(out.relative_to(ROOT)),'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':cap.get(cv2.CAP_PROP_FPS),'resolution':[1280,720],'private':True,'published':False};cap.release()
(ROOT/'reports/v4_demo_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
