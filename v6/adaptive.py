"""Personal neural warm-start, bounded precision solve and confidence-aware recovery.

Quality values are geometry/continuity proxies, not per-landmark confidences.
Every Geometry owns scratch MjData; actual physics is owned by its caller only.
"""
from dataclasses import dataclass
import time
from pathlib import Path
import numpy as np,torch
from train_personal import Network
from retargeting import Calibration,Retargeter,NAMES
from learning.runtime import neural_features
from simulate import ROOT,load_hand
from v6.geometry import Geometry

@dataclass(frozen=True)
class Decision:
    target:object
    mode:str
    reason:str
    timestamp:float
    info:dict

class PersonalModel:
    def __init__(self,checkpoint=None,calibration=None):
        torch.set_num_threads(2)
        ck=torch.load(checkpoint or ROOT/'private/learning/personal_mlp.pt',map_location='cpu',weights_only=True)
        self.net=Network();self.net.load_state_dict(ck['state']);self.net.eval()
        self.mean=np.array(ck['mean']);self.std=np.array(ck['std']);self.low=np.array(ck['low']);self.high=np.array(ck['high'])
        m=load_hand();names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
        if tuple(ck['names'])!=tuple(names):raise ValueError('Checkpoint actuator ordering mismatch')
        cal=Calibration.load(calibration or ROOT/'private/calibration.json');self.rom=Retargeter(names,m.actuator_ctrlrange,cal)
    def predict(self,x):
        with torch.inference_mode():q=self.net(torch.tensor((x-self.mean)/self.std,dtype=torch.float32)[None])[0].numpy()
        return np.clip(self.low+(self.high-self.low)*q,self.low,self.high)

class AdaptiveEngine:
    def __init__(self,method='hybrid',checkpoint=None,calibration=None,budget_ms=8.,max_nfev=4,pinch_enter=.55):
        if method not in ('rom','neural','optimizer','hybrid'):raise ValueError(method)
        self.method=method;self.personal=PersonalModel(checkpoint,calibration);self.geometry=Geometry();self.budget_ms=budget_ms;self.max_nfev=max_nfev;self.pinch_enter=pinch_enter
        self.previous=None;self.previous_t=None;self.bone_reference=None;self.selected='NEURAL';self.last_switch=-1e9;self.switches=0
    def observe_world(self,world,timestamp,now=None):
        if world is None:return Decision(None,'RECOVERY','missing_detection',timestamp,{})
        try:x=neural_features(world)
        except (ValueError,FloatingPointError):return Decision(None,'RECOVERY','invalid_geometry',timestamp,{})
        return self.observe(x,timestamp,now)
    def observe(self,x,timestamp,now=None):
        tick=time.perf_counter();now=timestamp if now is None else now
        if x is None:return Decision(None,'RECOVERY','missing_detection',timestamp,{})
        x=np.asarray(x)
        if x.shape!=(79,) or not np.isfinite(x).all():return Decision(None,'RECOVERY','nonfinite_or_invalid_shape',timestamp,{})
        if not np.isfinite(timestamp) or now-timestamp>.25 or now<timestamp-.01:return Decision(None,'RECOVERY','stale_or_future',timestamp,{})
        if self.previous_t is not None and timestamp<=self.previous_t:return Decision(None,'RECOVERY','nonmonotonic_observation',timestamp,{})
        p=x[:63].reshape(21,3);chains=p[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]];bones=np.linalg.norm(np.diff(chains,axis=1),axis=-1)
        if np.min(bones)<.025 or np.max(bones)>2 or np.max(abs(p))>6:return Decision(None,'RECOVERY','implausible_bone_geometry',timestamp,{})
        if self.bone_reference is not None and np.median(abs(bones-self.bone_reference)/np.maximum(self.bone_reference,.05))>.55:return Decision(None,'RECOVERY','bone_length_discontinuity',timestamp,{})
        speed=0. if self.previous is None else float(np.linalg.norm(p[[4,8,12,16]]-self.previous[[4,8,12,16]],axis=-1).mean()/max(timestamp-self.previous_t,.001))
        self.previous=p.copy();self.previous_t=timestamp;self.bone_reference=bones.copy() if self.bone_reference is None else .9*self.bone_reference+.1*bones
        neural_start=time.perf_counter();q=self.personal.rom.map_features(x[63:]) if self.method=='rom' else self.personal.predict(x);neural_ms=0. if self.method=='rom' else (time.perf_counter()-neural_start)*1000
        if self.method in ('rom','neural'):
            if self.method=='rom':q=self.personal.rom.map_features(x[63:])
            return Decision(q.copy(),'ROM' if self.method=='rom' else 'NEURAL','comparison_baseline',timestamp,{'neural_forward_ms':neural_ms,'speed_palm_widths_s':speed,'solver':{},'controller_ms':(time.perf_counter()-tick)*1000})
        goal=self.geometry.targets(p);tips=self.geometry.points_jac(q,False)[0];gt=goal[3::4]
        error=float(np.linalg.norm(tips-gt,axis=-1).mean());pinch_error=float(np.abs(np.linalg.norm(tips[:3]-tips[3],axis=-1)-np.linalg.norm(gt[:3]-gt[3],axis=-1)).mean())
        pinch=float(np.linalg.norm(p[4]-p[8]));wanted='NEURAL';reason='ordinary_motion';solver={}
        if self.method=='rom':q=self.personal.rom.map_features(x[63:]);wanted='ROM';reason='comparison_baseline'
        elif self.method=='optimizer':wanted='PRECISION';reason='always_optimize_comparator'
        elif self.method=='hybrid':
            enter=(pinch<self.pinch_enter or error>.040 or pinch_error>.020) and speed<4.
            leave=(pinch>self.pinch_enter+.20 and error<.030 and pinch_error<.014) or speed>6.
            wanted=self.selected
            if wanted=='PRECISION' and leave and timestamp-self.last_switch>=.25:wanted='NEURAL'
            elif wanted!='PRECISION' and enter and timestamp-self.last_switch>=.25:wanted='PRECISION'
            reason='pinch_or_geometry_error_slow_motion' if wanted=='PRECISION' else 'ordinary_or_fast_motion'
        if wanted=='PRECISION':
            fitted,solver=self.geometry.solve(goal,q,self.budget_ms,self.max_nfev)
            if solver['timeout']:wanted='NEURAL';reason='solver_deadline_fallback'
            elif solver['final_cost']<=solver['initial_cost']+1e-12:q=fitted
            else:wanted='NEURAL';reason='no_safe_improvement'
        if wanted!=self.selected:self.last_switch=timestamp;self.switches+=1
        self.selected=wanted
        info={'neural_forward_ms':neural_ms,'solver':solver,'speed_palm_widths_s':speed,'human_pinch_palm_widths':pinch,'neural_tip_error_m':error,'neural_pinch_error_m':pinch_error,'quality':'finite, nondegenerate, plausible and temporally consistent bone-length proxies; not landmark confidence','controller_ms':(time.perf_counter()-tick)*1000}
        return Decision(np.clip(q,self.geometry.low,self.geometry.high).copy(),wanted,reason,timestamp,info)

class MotionSafety:
    """Position bounds + velocity/acceleration limits, with bounded loss fallback.

    Discrete braking margins ensure a bounded-acceleration stop before a limit.
    Changes of modes only alter requested positions, never reset velocity/state.
    """
    def __init__(self,limits,tau=.045,max_speed=2.5,max_acceleration=30.):
        self.limits=np.asarray(limits).copy();self.neutral=np.clip(np.zeros(len(limits)),self.limits[:,0],self.limits[:,1]);self.value=self.neutral.copy();self.velocity=np.zeros(len(limits));self.request=self.value.copy();self.seen=None;self.last=None;self.state='NEUTRAL';self.tau=tau;self.max_speed=max_speed;self.max_acceleration=max_acceleration
    def update(self,decision,now):
        if self.last is not None and now<self.last:raise ValueError('Nonmonotonic control clock')
        elapsed=0. if self.last is None else min(now-self.last,.1);self.last=now
        if decision is not None and decision.target is not None and 0<=now-decision.timestamp<=.25:
            q=np.asarray(decision.target)
            if q.shape!=self.value.shape or not np.isfinite(q).all():raise ValueError('Invalid actuator request')
            self.request=np.clip(q,self.limits[:,0],self.limits[:,1]);self.seen=decision.timestamp
        age=np.inf if self.seen is None else now-self.seen
        desired=self.request if age<=.3 else self.neutral
        self.state='TRACKING' if age<=.3 else ('RETURNING' if np.max(abs(self.value-self.neutral))>.01 else 'NEUTRAL')
        if elapsed<=0:return self.value.copy()
        steps=max(1,int(np.ceil(elapsed/.005)));dt=elapsed/steps;a=self.max_acceleration
        for _ in range(steps):
            desired_velocity=np.clip((desired-self.value)/(self.tau if age<=.3 else .2),-self.max_speed,self.max_speed)
            lower_distance=np.maximum(0,self.value-self.limits[:,0]);upper_distance=np.maximum(0,self.limits[:,1]-self.value)
            upper_safe=np.maximum(0,np.sqrt(2*a*upper_distance)-a*dt);lower_safe=np.maximum(0,np.sqrt(2*a*lower_distance)-a*dt)
            lo=np.maximum.reduce([np.full_like(self.value,-self.max_speed),self.velocity-a*dt,-lower_safe])
            hi=np.minimum.reduce([np.full_like(self.value,self.max_speed),self.velocity+a*dt,upper_safe])
            if np.any(lo>hi+1e-8):raise RuntimeError('No acceleration-safe command within actuator limits')
            self.velocity=np.clip(desired_velocity,lo,hi);self.value+=self.velocity*dt
            if np.any(self.value<self.limits[:,0]-1e-9) or np.any(self.value>self.limits[:,1]+1e-9):raise RuntimeError('Command boundary safety failed')
        return self.value.copy()
