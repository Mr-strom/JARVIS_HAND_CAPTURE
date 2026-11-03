"""Same LEAP fingertip/pinch objective, evaluated with analytical MuJoCo Jacobians."""
import time
import mujoco,numpy as np
from scipy.optimize import least_squares
from simulate import load_hand
from learning.leap_fk import TIP_OFFSETS

class SolverDeadline(RuntimeError):pass

class Geometry:
    def __init__(self):
        self.m=load_hand();self.d=mujoco.MjData(self.m)
        self.bodies=[self.m.body(b).id for b in ['if_ds','mf_ds','rf_ds','th_ds']]
        self.offsets=[np.array(TIP_OFFSETS[b]) for b in ['if_ds','mf_ds','rf_ds','th_ds']]
        self.palm=self.m.body('palm').id
        self.low=np.maximum(self.m.actuator_ctrlrange[:,0],np.array([0,-.35,0,0]*3+[0,0,0,0]))
        self.high=np.minimum(self.m.actuator_ctrlrange[:,1],np.array([1.25,.35,1.5,1.15]*3+[1,1.05,1.1,1.2]))
        mujoco.mj_forward(self.m,self.d)
        from learning.leap_fk import LeapFK
        import torch
        with torch.no_grad():self.open_points=LeapFK()(torch.zeros(1,16)).numpy()[0]

    def targets(self,canonical):
        p=np.asarray(canonical)
        if p.shape!=(21,3) or not np.isfinite(p).all():raise ValueError('Expected finite canonical21x3')
        human=p[[5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4]][:,[1,0,2]]*np.array([1,1,-1])
        out=[]
        for start in range(0,16,4):
            h=human[start:start+4];r=self.open_points[start:start+4]
            scale=np.linalg.norm(np.diff(r,axis=0),axis=1).sum()/max(.01,np.linalg.norm(np.diff(h,axis=0),axis=1).sum())
            out.extend(r[0]+(h-h[:1])*scale)
        return np.array(out)

    def points_jac(self,q,need_jac=True):
        self.d.qpos[:]=q;mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)
        rot=self.d.xmat[self.palm].reshape(3,3);origin=self.d.xpos[self.palm]
        tips=[];jac=[]
        for b,offset in zip(self.bodies,self.offsets):
            point=self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@offset
            tips.append((point-origin)@rot)
            if need_jac:
                jp=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,jp,None,point,b);jac.append(rot.T@jp)
        return np.array(tips),np.array(jac)

    def residual_jac(self,q,goal,need_jac=True):
        tips,jac=self.points_jac(q,need_jac);gt=np.asarray(goal)[3::4] if np.asarray(goal).shape==(16,3) else np.asarray(goal)
        delta=tips[:3]-tips[3];distance=np.linalg.norm(delta,axis=-1)
        intended=np.linalg.norm(gt[:3]-gt[3],axis=-1)
        residual=np.r_[(tips-gt).ravel(),distance-intended]
        # Dividing all15 residuals by sqrt12 gives exactly the archived objective:
        # mean12 fingertip squared error + .25*mean3 pinch squared error.
        if not need_jac:return residual,None
        pinch=np.einsum('ij,ijk->ik',delta/np.maximum(distance[:,None],1e-9),jac[:3]-jac[3])
        return residual,np.vstack([jac.reshape(12,16),pinch])

    def solve(self,goal,initial,budget_ms=8.,max_nfev=8):
        start=time.perf_counter();deadline=start+budget_ms/1000;best=np.clip(initial,self.low,self.high).copy();initial_cost=float(np.sum(self.residual_jac(best,goal,False)[0]**2));bestcost=initial_cost;evals=0;timeout=False
        def fun(q):
            nonlocal best,bestcost,evals
            if time.perf_counter()>deadline:raise SolverDeadline()
            residual,_=self.residual_jac(q,goal,False);cost=float(residual@residual);evals+=1
            if cost<bestcost:best=q.copy();bestcost=cost
            return residual
        def jac(q):
            if time.perf_counter()>deadline:raise SolverDeadline()
            return self.residual_jac(q,goal)[1]
        try:least_squares(fun,np.clip(initial,self.low+1e-8,self.high-1e-8),jac=jac,bounds=(self.low,self.high),max_nfev=max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-6)
        except SolverDeadline:timeout=True
        elapsed=(time.perf_counter()-start)*1000
        if elapsed>budget_ms:timeout=True
        return best,{'elapsed_ms':elapsed,'timeout':timeout,'evaluations':evals,'initial_cost':initial_cost,'final_cost':bestcost,'improved':bestcost<initial_cost}

def metrics(tips,goal):
    gt=goal[3::4];pinch=np.abs(np.linalg.norm(tips[:3]-tips[3],axis=-1)-np.linalg.norm(gt[:3]-gt[3],axis=-1)).mean()*1000
    return float(np.linalg.norm(tips-gt,axis=-1).mean()*1000),float(pinch)
