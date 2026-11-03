"""Grouped personal-data learning, geometry-derived teacher and held-out comparison."""
import copy,json,time,csv,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import mujoco
from simulate import ROOT,load_hand
from retargeting import Calibration,Retargeter
from learning.leap_fk import LeapFK,KEYPOINT_BODIES,KEYPOINT_NAMES,TIP_OFFSETS

torch.set_num_threads(2)
HUMAN=[5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4]
class Network(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(79,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,16))
    def forward(self,x):return torch.sigmoid(self.net(x))

def verify_fk():
    fk=LeapFK();m=load_hand();d=mujoco.MjData(m);rng=np.random.default_rng(1);worst=0.
    for _ in range(20):
        q=rng.uniform(m.jnt_range[:,0],m.jnt_range[:,1]);d.qpos[:]=q;mujoco.mj_forward(m,d)
        palm=m.body('palm').id;rot=d.xmat[palm].reshape(3,3);points=[]
        for body,name in zip(KEYPOINT_BODIES,KEYPOINT_NAMES):
            i=m.body(body).id;pos=d.xpos[i].copy()
            if name.endswith('_tip'):pos+=d.xmat[i].reshape(3,3)@np.array(TIP_OFFSETS[body])
            points.append((pos-d.xpos[palm])@rot)
        got=fk(torch.tensor(q[None],dtype=torch.float32)).detach().numpy()[0]
        worst=max(worst,float(np.max(abs(got-np.array(points)))))
    if worst>1e-6:raise RuntimeError(f'FK mismatch: {worst}')
    (ROOT/'reports/fk_validation.json').write_text(json.dumps({'poses':20,'max_absolute_error_m':worst},indent=2))
    return fk

def targets(fk,p):
    human=torch.tensor(p[:,HUMAN],dtype=torch.float32)
    # Verified against positive LEAP curl: human +z curl maps to robot -z.
    # Human thumb-side +x maps to robot +y; forward +y maps to robot +x.
    human=torch.stack([human[:,:,1],human[:,:,0],-human[:,:,2]],dim=-1)
    zero=fk(torch.zeros(1,16))[0];result=[]
    for start in range(0,16,4):
        h=human[:,start:start+4];r=zero[start:start+4]
        hl=torch.linalg.norm(h[:,1:]-h[:,:-1],dim=-1).sum(-1).clamp(min=.01)
        rl=torch.linalg.norm(r[1:]-r[:-1],dim=-1).sum()
        result.append(r[0]+(h-h[:,:1])*(rl/hl)[:,None,None])
    return torch.cat(result,dim=1)

def objective(points,goal):
    tips=points[:,3::4];target=goal[:,3::4]
    tip=((tips-target)**2).mean((1,2))
    pinch=torch.linalg.norm(tips[:,:3]-tips[:,3:4],dim=-1)
    intended=torch.linalg.norm(target[:,:3]-target[:,3:4],dim=-1)
    return tip+.25*((pinch-intended)**2).mean(1)

def prepare():
    manifest=json.loads((ROOT/'data/personal/processed/manifest.json').read_text());clips=[];seen=set()
    for entry in manifest['recordings']:
        if entry['sha256'] in seen:continue
        seen.add(entry['sha256']);z=np.load(ROOT/entry['cache']);ok=z['valid'].astype(bool)
        if ok.sum()<20:continue
        clips.append((entry,{k:z[k][ok] for k in ['canonical','features','timestamps','frame_indices','gesture']},z['dhash']))
    # Conservative whole-recording near-duplicate grouping: duration and pose-sequence similarity.
    parent=list(range(len(clips)));pairs=[]
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(clips)):
        for j in range(i):
            ei,zi,hi=clips[i];ej,zj,hj=clips[j]
            if abs(ei['seconds']-ej['seconds'])>max(.5,.08*ei['seconds']):continue
            a=zi['canonical'][np.linspace(0,len(zi['canonical'])-1,12).astype(int)]
            b=zj['canonical'][np.linspace(0,len(zj['canonical'])-1,12).astype(int)]
            distance=float(np.nanmean(abs(a-b)))
            if distance<.06:parent[find(i)]=find(j);pairs.append([i,j,distance])
    groups=sorted(set(find(i) for i in parent));rng=np.random.default_rng(42);rng.shuffle(groups)
    if len(groups)<5:raise RuntimeError('Insufficient independent recording groups for honest train/validation/test split')
    ntest=max(1,round(len(groups)*.15));nval=max(1,round(len(groups)*.15))
    partitions={g:('test' if n<ntest else 'validation' if n<ntest+nval else 'train') for n,g in enumerate(groups)}
    split={i:partitions[find(i)] for i in range(len(clips))}
    audit={'recordings_discovered':len(manifest['recordings']),'usable_unique_recordings':len(clips),'near_duplicate_pairs':pairs,'group_count':len(groups),'split':[{**e,'index':i,'group':find(i),'partition':split[i]} for i,(e,z,h) in enumerate(clips)],'criterion':'whole recordings; suspected near-duplicates share partition; seed42; filenames not gesture labels','session_limitation':'recordings may share capture sessions/background across partitions; this tests held-out clips, not held-out people or all sessions'}
    (ROOT/'reports/dataset_audit.json').write_text(json.dumps(audit,indent=2));return clips,split

def main():
    torch.manual_seed(42);rng=np.random.default_rng(42);fk=verify_fk();clips,split=prepare();model=load_hand()
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(i)) for i in model.actuator_trnid[:,0]]
    cal=Calibration.load(ROOT/'private/calibration.json');baseline=Retargeter(names,model.actuator_ctrlrange,cal)
    low=np.maximum(model.actuator_ctrlrange[:,0],np.array([0,-.35,0,0]*3+[0,0,0,0]))
    high=np.minimum(model.actuator_ctrlrange[:,1],np.array([1.25,.35,1.5,1.15]*3+[1,1.05,1.1,1.2]))
    features=[];poses=[];base=[];clipids=[];times=[];indices=[]
    for i,(e,z,h) in enumerate(clips):
        features.append(np.c_[z['canonical'].reshape(-1,63),z['features']]);poses.append(z['canonical']);base.extend(baseline.map_features(f) for f in z['features']);clipids.extend([i]*len(z['features']));times.extend(z['timestamps']);indices.extend(z['frame_indices'])
    x=np.concatenate(features).astype('float32');pose=np.concatenate(poses);base=np.array(base,dtype='float32');ids=np.array(clipids);goal=targets(fk,pose)
    train=np.array([split[i]=='train' for i in ids]);val=np.array([split[i]=='validation' for i in ids]);test=np.array([split[i]=='test' for i in ids])
    out=ROOT/'private/learning';out.mkdir(exist_ok=True)
    cache=out/'teacher.npz';signature=hashlib.sha256(b'geometry-v2-y-x-minus-z'+x.tobytes()+base.tobytes()).hexdigest()
    if cache.exists() and str(np.load(cache)['signature'])==signature:teacher=np.load(cache)['q']
    else:
        result=[];tick=time.perf_counter()
        lo=torch.tensor(low,dtype=torch.float32);span=torch.tensor(high-low,dtype=torch.float32)
        for start in range(0,len(x),256):
            initial=torch.tensor(base[start:start+256]);g=goal[start:start+256]
            raw=torch.logit(((initial-lo)/span).clamp(.001,.999)).requires_grad_();optim=torch.optim.Adam([raw],lr=.12)
            best=initial.clone();bestloss=objective(fk(initial),g).detach()
            for step in range(100):
                q=lo+span*raw.sigmoid();err=objective(fk(q),g)
                better=err.detach()<bestloss;best[better]=q.detach()[better];bestloss=torch.minimum(bestloss,err.detach())
                loss=err.mean();optim.zero_grad();loss.backward();optim.step()
            result.append(best.detach().numpy());print('teacher',start,'/',len(x),flush=True)
        teacher=np.concatenate(result);np.savez_compressed(cache,q=teacher,signature=signature)
        (ROOT/'reports/teacher_timing.json').write_text(json.dumps({'seconds':time.perf_counter()-tick,'frames':len(x),'iterations':100},indent=2))
    mean=x[train].mean(0);std=x[train].std(0).clip(.05);xx=torch.tensor((x-mean)/std);yy=torch.tensor((teacher-low)/(high-low),dtype=torch.float32)
    net=Network();optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4);history=[];bestloss=float('inf');best=None;patience=0
    for epoch in range(400):
        net.train();order=rng.permutation(np.where(train)[0]);losses=[]
        for batch in np.array_split(order,max(1,len(order)//128)):
            pred=net(xx[batch]);loss=((pred-yy[batch])**2).mean();optim.zero_grad();loss.backward();optim.step();losses.append(loss.item())
        net.eval()
        with torch.no_grad():vl=((net(xx[val])-yy[val])**2).mean().item()
        history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_loss':vl})
        if vl<bestloss-1e-6:bestloss=vl;best=copy.deepcopy(net.state_dict());patience=0
        else:patience+=1
        if epoch%20==0:print('epoch',epoch,'validation',vl,flush=True)
        if patience>=35:break
    net.load_state_dict(best);net.eval();torch.save({'state':best,'mean':mean.tolist(),'std':std.tolist(),'low':low.tolist(),'high':high.tolist(),'names':names,'dataset_signature':signature,'architecture':'79-128-128-16 sigmoid','teacher':'bounded batched FK optimization; not human robot labels'},out/'personal_mlp.pt')
    with torch.no_grad():neural=low+(high-low)*net(xx).numpy()
    np.savez_compressed(out/'predictions.npz',x=x,goal=goal.numpy(),baseline=base,optimizer=teacher,neural=neural,clip_ids=ids,timestamps=times,frame_indices=indices,held_out=test)
    metrics={}
    for method,q in [('baseline',base),('optimizer',teacher),('neural',neural)]:
        with torch.no_grad():points=fk(torch.tensor(q[test],dtype=torch.float32)).numpy()
        g=goal.numpy()[test];tips=points[:,3::4];gt=g[:,3::4]
        smooth=[]
        for i in sorted(set(ids[test])):
            mask=(ids==i);dt=np.diff(np.array(times)[mask]);dq=np.diff(q[mask],axis=0);smooth.extend(np.linalg.norm(dq,axis=1)/dt)
        metrics[method]={'held_out_fingertip_error_mm':float(np.linalg.norm(tips-gt,axis=-1).mean()*1000),'held_out_pinch_error_mm':float(np.abs(np.linalg.norm(tips[:,:3]-tips[:,3:4],axis=-1)-np.linalg.norm(gt[:,:3]-gt[:,3:4],axis=-1)).mean()*1000),'joint_limit_violations':int(np.sum((q[test]<model.actuator_ctrlrange[:,0])|(q[test]>model.actuator_ctrlrange[:,1]))),'mean_joint_vector_speed_rad_s':float(np.mean(smooth))}
    tick=time.perf_counter()
    with torch.no_grad():
        for _ in range(1000):net(xx[:1])
    metrics['neural']['inference_ms_single_cpu']=(time.perf_counter()-tick)
    metrics['training']={'frames':len(x),'train':int(train.sum()),'validation':int(val.sum()),'test':int(test.sum()),'epochs':len(history),'best_validation_loss':bestloss,'parameters':sum(p.numel() for p in net.parameters()),'seed':42,'objective_warning':'errors use proportion-adjusted human geometry targets, not measured robot ground truth','selection':'held-out recordings never used for scaler, gradients or early stopping'}
    (ROOT/'reports/personal_model_results.json').write_text(json.dumps(metrics,indent=2));(ROOT/'reports/training_history.json').write_text(json.dumps(history,indent=2))
    with (ROOT/'reports/training_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
