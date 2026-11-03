"""LEAP FK geometry -> normalized Shadow ROM, never LEAP array copying."""
import numpy as np,torch
from learning.leap_fk import LeapFK

def _angle(a,b):
    return np.arccos(np.clip(np.dot(a,b)/max(1e-9,np.linalg.norm(a)*np.linalg.norm(b)),-1,1))

class ShadowGeometryAdapter:
    def __init__(self,model,side='right'):
        self.m=model;self.side=side;self.prefix='rh_' if side=='right' else 'lh_';self.fk=LeapFK()

    def representation(self,leap_q):
        q=np.asarray(leap_q)
        if q.shape!=(16,) or not np.isfinite(q).all():raise ValueError('Expected finite learned LEAP pose')
        with torch.no_grad():points=self.fk(torch.tensor(q[None],dtype=torch.float32)).numpy()[0]
        features=[]
        for k in range(4):
            chain=points[k*4:k*4+4];v=np.diff(chain,axis=0)
            features.append([_angle(v[0],np.array([1,0,0]))/(np.pi/2),_angle(v[0],v[1])/(np.pi/2),_angle(v[1],v[2])/(np.pi/2),np.arctan2(v[0,1],max(1e-6,v[0,0]))])
        return np.clip(np.asarray(features),[-0.,0.,0.,-.7],[1.5,1.5,1.5,.7])

    def commands(self,leap_q):
        f=self.representation(leap_q);out={}
        for k,digit in enumerate(['FF','MF','RF']):
            out[self.prefix+'A_'+digit+'J3']=1.2*f[k,0]
            out[self.prefix+'A_'+digit+'J0']=1.6*(f[k,1]+f[k,2])/2
            out[self.prefix+'A_'+digit+'J4']=-.4*f[k,3]
        out[self.prefix+'A_THJ4']=.8*f[3,0]
        out[self.prefix+'A_THJ1']=.8*(f[3,1]+f[3,2])/2
        out[self.prefix+'A_THJ5']=.25
        # Little finger and support-hand behavior are not learned from missing supervision.
        result={self.m.actuator(name).id:float(np.clip(value,*self.m.actuator_ctrlrange[self.m.actuator(name).id])) for name,value in out.items()}
        return result
