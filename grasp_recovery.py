"""Small geometry/opposition correction; original benchmark and success test retained."""
import argparse,json
import numpy as np,mujoco
from scipy.optimize import least_squares
from manipulation import scene,observe,trial as original_trial
from learning.leap_fk import KEYPOINT_BODIES,TIP_OFFSETS
from physics_guard import PhysicsGuard
from simulate import ROOT,camera_for

def solve_grip(m,d,center):
    aids=np.arange(16);qadr=m.jnt_qposadr[m.actuator_trnid[aids,0]];scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos
    # Opposed surface targets, with a 6mm fingertip-pad clearance. Physical parameters unchanged.
    radial=.028
    offsets=np.array([[radial/np.sqrt(2),radial/np.sqrt(2),0],[radial/np.sqrt(2),-radial/np.sqrt(2),.003],[0,-radial,.003],[-radial/np.sqrt(2),-radial/np.sqrt(2),0]])
    targets=center+offsets
    bodies=['if_ds','mf_ds','rf_ds','th_ds']
    def fun(q):
        scratch.qpos[qadr]=q;mujoco.mj_forward(m,scratch);points=[]
        for body in bodies:
            i=m.body(body).id;points.append(scratch.xpos[i]+scratch.xmat[i].reshape(3,3)@np.array(TIP_OFFSETS[body]))
        return np.r_[(np.array(points)-targets).ravel(),.002*(q-.4)]
    low=m.actuator_ctrlrange[aids,0]+.005;high=m.actuator_ctrlrange[aids,1]-.005
    initial=np.clip(np.array([.6,0,1.,.7]*3+[.7,1.,.6,.7]),low,high)
    result=least_squares(fun,initial,bounds=(low,high),max_nfev=120)
    return result.x,float(np.linalg.norm(fun(result.x)[:12]))

def run(offset,render=False):
    m=scene(offset);d=mujoco.MjData(m);mujoco.mj_forward(m,d);g=PhysicsGuard(m,d);goal=np.zeros(m.nu);command=goal.copy();grip=None;reference=None;phase='SETTLE';phase_t=0.;ready=0.;hold=0.;maxhold=0.;maxlift=0.;success=False;error=None;events=[];maxexc=0.;peak_force=0.;fiterror=None;renderer=None;video=None
    if render:
        import cv2
        renderer=mujoco.Renderer(m,height=480,width=640);video=cv2.VideoWriter(str(ROOT/'private/grasp_recovery_attempt.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
    zact=m.actuator('base_z').id
    def change(new):
        nonlocal phase,phase_t
        events.append({'from':phase,'to':new,'time':d.time});phase=new;phase_t=float(d.time)
    try:
        for i in range(7500):
            o=observe(m,d);normals=[];gid=m.geom('cylinder').id
            for ct in d.contact:
                if gid in (ct.geom1,ct.geom2):normals.append(ct.frame[:3]*(-1 if ct.geom1==gid else 1))
            opp=min([1.]+[float(np.dot(a,b)) for n,a in enumerate(normals) for b in normals[n+1:]])
            valid=len(o['digits'])>=2 and opp<-.3
            ready=ready+m.opt.timestep if valid else 0.
            peak_force=max(peak_force,o['normal_force_N'])
            if reference is not None:
                h=o['position'][2]-reference;maxlift=max(maxlift,h)
                good=h>=.02 and len(o['digits'])>=2 and not o['table_contact'] and np.linalg.norm(o['velocity'])<.15
                hold=hold+m.opt.timestep if good else 0.;maxhold=max(maxhold,hold)
                if hold>=2:success=True
            if phase=='SETTLE' and d.time>.8 and np.linalg.norm(o['velocity'])<.08:
                reference=float(o['position'][2]);grip,fiterror=solve_grip(m,d,o['position']);change('OPPOSE_CLOSE')
            elif phase=='OPPOSE_CLOSE':
                goal[:16]=grip
                if ready>.12:change('LIFT')
                elif d.time-phase_t>4:change('RECOVER_NO_OPPOSITION')
            elif phase=='LIFT':
                s=min(1.,(d.time-phase_t)/3.);goal[zact]=.07*(.5-.5*np.cos(np.pi*s))
                if s>=1:change('HOLD')
            elif phase=='HOLD':
                if success:change('RELEASE')
                elif d.time-phase_t>3:change('RECOVER_HOLD_FAILED')
            elif phase.startswith('RECOVER') or phase=='RELEASE':goal[:16]*=(1-m.opt.timestep);goal[zact]=max(0.,goal[zact]-.02*m.opt.timestep)
            command+=np.clip(goal-command,-2.5*m.opt.timestep,2.5*m.opt.timestep);d.ctrl[:]=np.clip(command,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);g.step(d)
            ids=m.actuator_trnid[:,0];q=d.qpos[m.jnt_qposadr[ids]];maxexc=max(maxexc,float(np.max(np.maximum(m.jnt_range[ids,0]-q,q-m.jnt_range[ids,1]))))
            if renderer and i%17==0:
                import cv2
                renderer.update_scene(d,camera=camera_for(m));frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);cv2.putText(frame,phase,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);video.write(frame)
    except Exception as exc:error=str(exc)
    finally:
        if renderer:renderer.close();video.release()
    return {'controller':'opposition_geometry_and_smooth_lift','offset_m':offset,'success':success,'max_lift_m':float(maxlift),'max_valid_hold_s':maxhold,'peak_normal_force_N':peak_force,'grip_surface_fit_residual_m':fiterror,'max_joint_limit_excursion_rad':max(0.,maxexc),'warnings':g.warnings(d),'error':error,'events':events,'mass_friction_geometry_unchanged':True,'criterion_unchanged':True,'algorithm':'kinematic surface targets and opposing-contact dwell before lift, not a learned grasp policy'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');a=p.parse_args();results=[]
    for offset in [-.008,-.004,0,.004,.008]:
        r=run(offset,a.render and offset==0);results.append(r);print(offset,r['success'],r['max_valid_hold_s'],r['error'],flush=True)
    (ROOT/'reports/grasp_recovery_results.json').write_text(json.dumps(results,indent=2))
