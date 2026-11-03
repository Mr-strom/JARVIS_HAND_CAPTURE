"""Fixed-pelvis G1/Shadow control with named groups and bounded position actuators."""
import argparse,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from simulate import ROOT
from physics_guard import PhysicsGuard

ASSETS=ROOT/'assets/g1_shadow'
ARM_SUFFIX=('shoulder_pitch_joint','shoulder_roll_joint','shoulder_yaw_joint','elbow_joint','wrist_roll_joint')

def resource_usage():
    import ctypes as ct
    from ctypes import wintypes as wt
    class Counters(ct.Structure):
        _fields_=[('cb',wt.DWORD),('PageFaultCount',wt.DWORD)]+[(name,ct.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    memory=Counters();memory.cb=ct.sizeof(memory)
    kernel=ct.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wt.HANDLE
    get=ct.WinDLL('psapi').GetProcessMemoryInfo;get.argtypes=[wt.HANDLE,ct.POINTER(Counters),wt.DWORD]
    if not get(kernel.GetCurrentProcess(),ct.byref(memory),memory.cb):return {}
    return {'working_set_MiB':memory.WorkingSetSize/2**20,'peak_working_set_MiB':memory.PeakWorkingSetSize/2**20}

def load_model(object_scene=False):
    root=ET.parse(ASSETS/'shadow_g1.xml').getroot();root.find('compiler').set('meshdir',str(ASSETS/'assets'))
    root.find('option').set('gravity','0 0 -9.81')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'headlight',ambient='.45 .45 .45',diffuse='.7 .7 .7',specular='.15 .15 .15')
    w=root.find('worldbody');ET.SubElement(w,'light',pos='1 -1 3',dir='-1 1 -2',directional='true')
    ET.SubElement(w,'geom',name='laboratory_floor',type='plane',size='3 3 .05',rgba='.15 .2 .25 1')
    if object_scene:
        ET.SubElement(w,'geom',name='task_table',type='box',pos='.5 0 .785',size='.2 .4 .025',rgba='.25 .3 .4 1')
        b=ET.SubElement(w,'body',name='task_object',pos='.35 0 .86');ET.SubElement(b,'freejoint',name='task_object_free')
        ET.SubElement(b,'geom',name='task_cylinder',type='cylinder',size='.025 .05',mass='.08',friction='.8 .01 .001',rgba='1 .2 .05 1')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    for a in range(m.nu):
        if m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT:
            j=int(m.actuator_trnid[a,0]);m.actuator_ctrlrange[a]=m.jnt_range[j];m.actuator_ctrllimited[a]=1
            if not m.actuator(a).name.startswith(('rh_','lh_')):
                # The upstream unclassified G1 position actuators default to kp=1.
                # Explicit bounded PD provides gravity-loaded joint tracking.
                m.actuator_gainprm[a,0]=100.;m.actuator_biasprm[a,1]=-100.;m.actuator_biasprm[a,2]=-6.
                bound=25. if 'shoulder' in m.actuator(a).name or 'elbow' in m.actuator(a).name or 'wrist' in m.actuator(a).name else 80.
                m.actuator_forcerange[a]=[-bound,bound];m.actuator_forcelimited[a]=1
    return m

class BimanualController:
    def __init__(self,m):
        self.m=m
        self.arms={s:np.array([m.actuator(s+'_'+suffix).id for suffix in ARM_SUFFIX]) for s in ('left','right')}
        self.hands={s:np.array([a for a in range(m.nu) if m.actuator(a).name.startswith('lh_' if s=='left' else 'rh_')]) for s in ('left','right')}
        self.qadr={s:np.array([m.jnt_qposadr[m.actuator_trnid[a,0]] for a in ids]) for s,ids in self.arms.items()}
        self.dof={s:np.array([m.jnt_dofadr[m.actuator_trnid[a,0]] for a in ids]) for s,ids in self.arms.items()}
        self.command=np.clip(np.zeros(m.nu),m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])

    def ik(self,data,side,position):
        scratch=mujoco.MjData(self.m);scratch.qpos[:]=data.qpos
        ids=self.qadr[side];act=self.arms[side];body=self.m.body('lh_palm' if side=='left' else 'rh_palm').id
        def error(q):
            scratch.qpos[ids]=q;mujoco.mj_forward(self.m,scratch);return scratch.xpos[body]-position
        def jac(q):
            error(q);jp=np.zeros((3,self.m.nv));jr=jp.copy();mujoco.mj_jacBody(self.m,scratch,jp,jr,body);return jp[:,self.dof[side]]
        bounds=self.m.actuator_ctrlrange[act]
        fit=least_squares(error,np.clip(data.qpos[ids],bounds[:,0]+1e-8,bounds[:,1]-1e-8),jac=jac,bounds=(bounds[:,0],bounds[:,1]),max_nfev=60)
        return fit.x,float(np.linalg.norm(error(fit.x)))

    def step(self,data,desired):
        if desired.shape!=(self.m.nu,) or not np.isfinite(desired).all():raise ValueError('Invalid control target')
        target=np.clip(desired,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
        self.command+=np.clip(target-self.command,-1.5*self.m.opt.timestep,1.5*self.m.opt.timestep)
        # Feed-forward gravity compensation through bounded position actuators, not external forces.
        mujoco.mj_rne(self.m,data,0,data.qfrc_bias)
        applied=self.command.copy()
        for a in range(self.m.nu):
            if self.m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and not self.m.actuator(a).name.startswith(('rh_','lh_')):
                j=int(self.m.actuator_trnid[a,0]);dof=int(self.m.jnt_dofadr[j]);applied[a]+=data.qfrc_bias[dof]/self.m.actuator_gainprm[a,0]
        data.ctrl[:]=np.clip(applied,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])

def camera(m,close=False):
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[.05,0,.95 if close else .7];cam.distance=1.25 if close else 2.5;cam.azimuth=135;cam.elevation=-15;return cam

def demonstrate(seconds=12,render=False,viewer=False):
    (ROOT/'private').mkdir(exist_ok=True)
    m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=BimanualController(m);g=PhysicsGuard(m,d)
    neutral=c.command.copy();samples=[];warnings={};peak_pen=0.;max_exc=0.;lat=[];frames=0;renderer=None;writer=None;window=None
    palm0={s:d.xpos[m.body('lh_palm' if s=='left' else 'rh_palm').id].copy() for s in c.arms}
    reach={s:c.ik(d,s,palm0[s]+np.array([.06,0,.04])) for s in c.arms}
    start=time.perf_counter();cpu_start=time.process_time();tracking_errors={'left':[],'right':[]};scratch=mujoco.MjData(m)
    if render:
        import cv2
        renderer=mujoco.Renderer(m,height=720,width=960);writer=cv2.VideoWriter(str(ROOT/'private/g1_both_arms_hands.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,720))
        if not writer.isOpened():raise RuntimeError('Could not open humanoid video encoder')
    if viewer:
        from mujoco import viewer as mjviewer
        window=mjviewer.launch_passive(m,d);window.cam.lookat[:]=camera(m).lookat;window.cam.distance=2.5;window.cam.azimuth=135
    qmin=d.qpos.copy();qmax=qmin.copy()
    try:
        for i in range(round(seconds/m.opt.timestep)):
            t=i*m.opt.timestep;desired=neutral.copy()
            for side,shift in [('left',0),('right',np.pi)]:
                alpha=.5-.5*np.cos(2*np.pi*t/6+shift)
                desired[c.arms[side]]=alpha*reach[side][0]
                for a in c.hands[side]:
                    name=m.actuator(a).name
                    if any(k in name for k in ['FFJ3','MFJ3','RFJ3','LFJ3']):desired[a]=.7*alpha
                    elif name.endswith('J0'):desired[a]=1.1*alpha
                    elif name.endswith('THJ1'):desired[a]=.5*alpha
            tick=time.perf_counter();c.step(d,desired);g.step(d);lat.append((time.perf_counter()-tick)*1000)
            qmin=np.minimum(qmin,d.qpos);qmax=np.maximum(qmax,d.qpos)
            limited=np.where(m.jnt_limited)[0];adr=m.jnt_qposadr[limited]
            max_exc=max(max_exc,float(np.max(np.maximum(m.jnt_range[limited,0]-d.qpos[adr],d.qpos[adr]-m.jnt_range[limited,1]))))
            peak_pen=max(peak_pen,max([0.]+[-float(x.dist) for x in d.contact]))
            if i%17==0:
                scratch.qpos[:]=d.qpos
                for side in c.arms:scratch.qpos[c.qadr[side]]=desired[c.arms[side]]
                mujoco.mj_forward(m,scratch)
                for side in c.arms:
                    body=m.body('lh_palm' if side=='left' else 'rh_palm').id;tracking_errors[side].append(float(np.linalg.norm(d.xpos[body]-scratch.xpos[body])))
                samples.append(d.qpos.copy())
                if renderer:
                    import cv2
                    renderer.update_scene(d,camera=camera(m));frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);cv2.putText(frame,'Fixed pelvis / Earth gravity / scripted IK + finger control',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);writer.write(frame);frames+=1
                    if i//17==30:cv2.imwrite(str(ROOT/'private/g1_arms_hands.png'),frame)
                if window:
                    if not window.is_running():break
                    before=float(d.time);window.sync();g.accept_viewer_sync(d,before)
                    remaining=g.steps*g.dt-(time.perf_counter()-start)
                    if remaining>0:time.sleep(remaining)
    finally:
        if renderer:renderer.close();writer.release()
        if window:window.close()
    elapsed=time.perf_counter()-start
    report={'model':'pinned G1 plus Shadow Hands','fixed_pelvis':True,'gravity':m.opt.gravity.tolist(),'njnt':m.njnt,'nu':m.nu,'hands_actuators':{s:len(ids) for s,ids in c.hands.items()},'arms_actuators':{s:len(ids) for s,ids in c.arms.items()},'ik_residual_m':{s:r[1] for s,r in reach.items()},'simulated_seconds':g.steps*g.dt,'wall_seconds':elapsed,'real_time_factor':g.steps*g.dt/elapsed,'mean_control_and_step_ms':float(np.mean(lat)),'warnings':g.warnings(d),'max_joint_limit_excursion_rad':max(0.,max_exc),'max_penetration_m':peak_pen,'rendered_frames':frames,'joint_motion_range_rad':{m.joint(j).name:float(qmax[m.jnt_qposadr[j]]-qmin[m.jnt_qposadr[j]]) for j in range(m.njnt)},'balance_or_locomotion_validated':False,'learning_used_in_this_demo':False,'runtime_controller':'explicit bounded PD plus gravity feedforward; upstream mesh and joint geometry retained'}
    report['resources']=resource_usage();report['resources']['process_cpu_percent_one_core_scale']=100*(time.process_time()-cpu_start)/elapsed
    report['palm_tracking_error_m']={s:{'median':float(np.median(v)),'p95':float(np.percentile(v,95))} for s,v in tracking_errors.items()}
    (ROOT/'reports/g1_control_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=12);p.add_argument('--render',action='store_true');p.add_argument('--viewer',action='store_true');a=p.parse_args();demonstrate(a.seconds,a.render,a.viewer)
