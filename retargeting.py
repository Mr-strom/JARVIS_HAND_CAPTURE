"""CPU landmark geometry and bounded LEAP retargeting; no synthetic trajectories."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

FINGERS = {'if': (5,6,7,8), 'mf': (9,10,11,12), 'rf': (13,14,15,16)}
NAMES = tuple(f'{f}_{j}' for f in FINGERS for j in ('mcp','rot','pip','dip')) + ('th_cmc','th_axl','th_mcp','th_ipl')


def unit(v):
    length = float(np.linalg.norm(v))
    if length < 1e-7:
        raise ValueError('Degenerate landmark geometry')
    return v / length


def bend(a, b, c):
    return float(np.arccos(np.clip(np.dot(unit(b-a),unit(c-b)), -1,1)))


def joint_features(points):
    p = np.asarray(points, dtype=float)
    if p.shape != (21,3) or not np.isfinite(p).all():
        raise ValueError('Expected 21 finite 3D landmarks')
    # Palm-relative basis removes rigid wrist rotation. Little MCP is an orientation anchor only.
    x = unit(p[5]-p[17])  # towards index/thumb side
    y = p[9]-p[0]
    y = unit(y-np.dot(y,x)*x)
    normal = unit(np.cross(x,y))
    output = {}
    for finger,(mcp,pip,dip,tip) in FINGERS.items():
        metacarpal = unit(p[mcp]-p[0])
        proximal = unit(p[pip]-p[mcp])
        forward = unit(metacarpal-np.dot(metacarpal,normal)*normal)
        lateral = unit(np.cross(forward,normal))
        if np.dot(lateral,x)<0:
            lateral = -lateral
        output[f'{finger}_mcp'] = float(np.arctan2(abs(np.dot(proximal,normal)),np.dot(proximal,forward)))
        output[f'{finger}_rot'] = float(np.arctan2(np.dot(proximal,lateral),np.dot(proximal,forward)))
        output[f'{finger}_pip'] = bend(p[mcp],p[pip],p[dip])
        output[f'{finger}_dip'] = bend(p[pip],p[dip],p[tip])
    thumb_base = unit(p[2]-p[1])
    planar = thumb_base-np.dot(thumb_base,normal)*normal
    spread = np.arccos(np.clip(np.dot(unit(planar),y),-1,1))
    output['th_cmc'] = float(np.clip(1.5-spread,0,1.5))
    # Opposition proxy: tip approaching ring MCP, normalized by palm width.
    output['th_axl'] = float(np.clip(1.8-np.linalg.norm(p[4]-p[13])/np.linalg.norm(p[5]-p[17]),0,1.8))
    output['th_mcp'] = bend(p[1],p[2],p[3])
    output['th_ipl'] = bend(p[2],p[3],p[4])
    return np.array([output[name] for name in NAMES])


class Calibration:
    def __init__(self):
        self.low = np.zeros(16)
        self.high = np.array([1.4,.4,1.7,1.35]*3 + [1.0,1.35,1.3,1.4])
        for i,name in enumerate(NAMES):
            if name.endswith('_rot'):
                self.low[i] = -.4
        self.personalized = [False]*16
        self.samples = 0

    def fit(self, samples):
        values = np.asarray(samples)
        if values.ndim != 2 or values.shape[0]<30 or values.shape[1]!=16 or not np.isfinite(values).all():
            raise ValueError('Calibration needs at least 30 valid tracked frames')
        lo,hi = np.percentile(values,[5,95],axis=0)
        for i,name in enumerate(NAMES):
            threshold = .12 if name.endswith('_rot') else .20
            if hi[i]-lo[i] >= threshold:
                self.low[i],self.high[i] = lo[i],hi[i]
                self.personalized[i] = True
        self.samples = len(values)

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps({'schema':1,'names':NAMES,'low':self.low.tolist(),'high':self.high.tolist(),
                                        'personalized':self.personalized,'samples':self.samples},indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        raw=json.loads(Path(path).read_text(encoding='utf-8'))
        if raw.get('schema')!=1 or tuple(raw.get('names',[]))!=NAMES:
            raise ValueError('Incompatible calibration file')
        result=cls()
        low,high=np.array(raw['low']),np.array(raw['high'])
        if low.shape!=(16,) or high.shape!=(16,) or not np.isfinite([low,high]).all() or np.any(high-low<.05):
            raise ValueError('Invalid calibration ranges')
        result.low,result.high=low,high
        result.personalized=raw['personalized'];result.samples=raw['samples']
        return result


class Retargeter:
    def __init__(self, actuator_names, ctrlrange, calibration=None):
        self.names=tuple(actuator_names)
        if set(self.names)!=set(NAMES) or len(self.names)!=16:
            raise ValueError('Expected all 16 named LEAP joints exactly once')
        self.order=np.array([NAMES.index(name) for name in self.names])
        self.limits=np.asarray(ctrlrange).copy()
        self.calibration=calibration or Calibration()
        self.neutral=np.clip(np.zeros(16),self.limits[:,0],self.limits[:,1])
        self.value=self.neutral.copy()
        self.request=self.neutral.copy()
        self.last_seen=None
        self.last_update=None
        self.state='NEUTRAL'
        self.tau=.06
        self.max_speed=2.5

    def map_features(self,features):
        values=np.asarray(features,dtype=float)
        if values.shape!=(16,) or not np.isfinite(values).all():
            raise ValueError('Expected 16 finite joint features')
        c=self.calibration
        normalized=np.clip((values-c.low)/(c.high-c.low),0,1)
        result=np.empty(16)
        caps={'mcp':1.25,'pip':1.5,'dip':1.15,'th_cmc':1.0,'th_axl':1.05,'th_mcp':1.1,'th_ipl':1.2}
        for i,name in enumerate(NAMES):
            if name.endswith('_rot'):
                # Positive LEAP rotation moves away from thumb: opposite to our palm x axis.
                result[i]=-.35*(2*normalized[i]-1)
            else:
                result[i]=caps.get(name,caps.get(name.split('_')[-1],1.0))*normalized[i]
        return np.clip(result[self.order],self.limits[:,0],self.limits[:,1])

    def update(self,features,now):
        if self.last_update is not None and now<self.last_update:
            raise ValueError('Nonmonotonic controller timestamp')
        dt=0.0 if self.last_update is None else min(now-self.last_update,.1)
        self.last_update=now
        if features is not None:
            self.request=self.map_features(features)
            self.last_seen=now
        age=float('inf') if self.last_seen is None else now-self.last_seen
        if age <= .3:
            self.state='TRACKING' if features is not None else 'HOLD'
            desired=self.request
            tau=self.tau
        else:
            self.state='RETURNING' if np.max(np.abs(self.value-self.neutral))>.01 else 'NEUTRAL'
            desired=self.neutral
            tau=.2
        alpha=1-np.exp(-dt/tau)
        delta=np.clip(alpha*(desired-self.value),-self.max_speed*dt,self.max_speed*dt)
        self.value=np.clip(self.value+delta,self.limits[:,0],self.limits[:,1])
        return self.value.copy()
