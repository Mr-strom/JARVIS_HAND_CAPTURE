"""One bounded recovery configuration; original v4 benchmarks remain unchanged.

Only actuator-side limits/damping and control logic change. No collision,
inertial, friction, object or success-criterion changes are made.
"""
import argparse, json, time
import cv2, mujoco, numpy as np
from manipulation import scene, observe
from physics_guard import PhysicsGuard
from simulate import ROOT, camera_for


def valid_hold(height, digits, table, speed):
    return height >= .02 and len(digits) >= 2 and not table and speed < .15


def contact_feedback(m, d):
    gid=m.geom('cylinder').id;contacts=[];force={f:0. for f in ('if','mf','rf','th')}
    for i,ct in enumerate(d.contact):
        if gid not in (ct.geom1,ct.geom2):continue
        other=ct.geom2 if ct.geom1==gid else ct.geom1
        body=m.body(int(m.geom_bodyid[other])).name or ''
        if body[:2] not in force:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        if f[0]<.01:continue
        normal=ct.frame[:3]*(-1 if ct.geom1==gid else 1)
        force[body[:2]]+=float(f[0]);contacts.append((body[:2],normal))
    opposition=min([1.]+[float(a@b) for digit,a in contacts for other,b in contacts if digit!=other])
    return force,opposition


def run(offset=0., render=False):
    m=scene(offset)
    # Motor torque ceilings are explicit; no unlimited-force position closure.
    m.actuator_forcelimited[:16]=1;m.actuator_forcerange[:16]=[-.4,.4]
    m.actuator_biasprm[:16,2]=-.08  # same kp=3; more velocity damping
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);guard=PhysicsGuard(m,d)
    commands=np.zeros(m.nu);goals=commands.copy();phase='SETTLE';phase_t=0.;reference=None
    ready=0.;hold=0.;maxhold=0.;maxlift=0.;success=False;error=None;events=[];trace=[]
    maxpen=0.;maxexc=0.;peakforce=0.;tracking=[];opposed=0;observations=0
    qa=m.jnt_qposadr[m.actuator_trnid[:16,0]];jointids=m.actuator_trnid[:,0];allqa=m.jnt_qposadr[jointids]
    closure=np.zeros(4);grip=np.array([[.85,0,1.2,.9]]*3+[[.8,.9,.85,.9]])
    bx=m.actuator('base_x').id;bz=m.actuator('base_z').id
    writer=None;renderer=None;start=time.perf_counter()
    if render:
        renderer=mujoco.Renderer(m,height=480,width=640)
        writer=cv2.VideoWriter(str(ROOT/'private/final/grasp_recovery.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
        if not writer.isOpened():raise RuntimeError('Cannot encode grasp evidence')
    def change(p):
        nonlocal phase,phase_t
        events.append({'time_s':float(d.time),'from':phase,'to':p});phase=p;phase_t=float(d.time)
    try:
        for step in range(round(14/m.opt.timestep)):
            o=observe(m,d);forces,opp=contact_feedback(m,d);age=d.time-phase_t
            peakforce=max(peakforce,sum(forces.values()))
            opposed+=int(opp<-.5);observations+=1
            ready=ready+m.opt.timestep if opp<-.5 and forces['th']>.03 else 0.
            if reference is not None:
                h=float(o['position'][2]-reference);maxlift=max(maxlift,h)
                held=valid_hold(h,o['digits'],o['table_contact'],np.linalg.norm(o['velocity']))
                hold=hold+m.opt.timestep if held else 0.;maxhold=max(maxhold,hold)
                if hold>=2:success=True
            if phase=='SETTLE' and d.time>.8 and np.linalg.norm(o['velocity'])<.08:
                reference=float(o['position'][2]);goals[bx]=np.clip(o['position'][0]-.04,-.04,.04);change('ALIGN_OPPOSITION')
            elif phase=='ALIGN_OPPOSITION':
                goals[13]=.65 # thumb axial positioning through physical actuator
                if age>1.:change('COMPLIANT_CLOSE')
            elif phase=='COMPLIANT_CLOSE':
                for k,f in enumerate(('if','mf','rf','th')):
                    rate=.22 if forces[f]<.8 else (-.12 if forces[f]>4 else 0.)
                    closure[k]=np.clip(closure[k]+rate*m.opt.timestep,0,1)
                    goals[k*4:k*4+4]=grip[k]*closure[k]
                goals[13]=max(goals[13],.65)
                if ready>=.2:change('SLOW_LIFT')
                elif age>5.:change('RECOVER_NO_OPPOSITION')
            elif phase=='SLOW_LIFT':
                goals[bz]=min(.07,goals[bz]+.015*m.opt.timestep)
                if opp>-.2 and age>.5:change('RECOVER_CONTACT_LOSS')
                elif goals[bz]>=.07:change('HOLD')
            elif phase=='HOLD':
                if success:change('RELEASE')
                elif age>3:change('RECOVER_HOLD_FAILED')
            elif phase.startswith('RECOVER') or phase=='RELEASE':
                goals[:16]*=max(0.,1-m.opt.timestep);goals[bz]=max(0.,goals[bz]-.015*m.opt.timestep)
            commands+=np.clip(goals-commands,-.8*m.opt.timestep,.8*m.opt.timestep)
            # Small tracking correction only in free motion, never integrate into contacts.
            applied=commands.copy();corr=np.clip(commands[:16]-d.qpos[qa],-.08,.08)
            for k,f in enumerate(('if','mf','rf','th')):
                if forces[f]<.05:applied[k*4:k*4+4]+=.3*corr[k*4:k*4+4]
            d.ctrl[:]=np.clip(applied,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);guard.step(d)
            tracking.append(float(np.sqrt(np.mean((commands[:16]-d.qpos[qa])**2))))
            maxexc=max(maxexc,float(np.max(np.maximum(m.jnt_range[jointids,0]-d.qpos[allqa],d.qpos[allqa]-m.jnt_range[jointids,1]))))
            maxpen=max(maxpen,max([0.]+[-float(ct.dist) for ct in d.contact]))
            if step%17==0:
                trace.append({'time_s':float(d.time),'phase':phase,'height_m':float(o['position'][2]),'digit_force_N':forces,'opposition_cosine':opp,'tracking_rmse_rad':tracking[-1],'max_joint_applied_target_error_rad':float(np.max(abs(d.ctrl[:16]-d.qpos[qa]))),'max_penetration_m':maxpen})
                if renderer:
                    cam=camera_for(m);cam.lookat[:]=[.035,-.01,.11];cam.distance=.45
                    renderer.update_scene(d,camera=cam);f=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                    cv2.putText(f,phase,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                    cv2.putText(f,f'Opposition {opp:.2f} | force {sum(forces.values()):.2f} N | valid hold {hold:.2f}s',(10,460),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);writer.write(f)
            if maxpen>.005:raise RuntimeError(f'Integrity abort: penetration {maxpen:.6f} m > 0.005 m')
            if maxexc>.02:raise RuntimeError(f'Integrity abort: soft joint excursion {maxexc:.6f} rad > 0.02 rad')
    except RuntimeError as exc:error=str(exc);success=False
    finally:
        if renderer:renderer.close();writer.release()
    return {'offset_m':offset,'success':success,'max_lift_m':maxlift,'max_valid_hold_s':maxhold,'peak_digit_contact_force_N':peakforce,
        'max_penetration_m':maxpen,'max_joint_limit_excursion_rad':max(0.,maxexc),'mean_joint_tracking_rmse_rad':float(np.mean(tracking)),
        'opposing_contact_fraction':opposed/max(1,observations),'error':error,'warnings':guard.warnings(d),'sim_seconds':float(d.time),'wall_seconds':time.perf_counter()-start,'events':events,'trace':trace}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');a=p.parse_args();(ROOT/'private/final').mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();results=[]
    for offset in [-.008,-.004,0,.004,.008]:
        result=run(offset,a.render and offset==0);results.append(result);print(offset,result['success'],result['max_valid_hold_s'],result['error'],flush=True)
        if time.perf_counter()-start>20*60:break
    report={'configuration_count':1,'time_box_minutes':20,'elapsed_seconds':time.perf_counter()-start,'criterion':'Original unchanged: lift >= .02m, >=2 digits, no table, speed < .15m/s continuously 2s',
        'physics':'Original 35g radius22mm height50mm cylinder and friction/geometry/collision unchanged. Finger motor torque cap .4Nm, same kp3, kd .08; wrist30N. Seeded pregrasp, not tabletop pickup.',
        'control':'thumb preposition, physical x alignment, force-bounded per-digit closure, opposed normal dwell .2s, 15mm/s lift, contact-loss recovery, 5mm/.02rad integrity gates',
        'comparison':'Archived v4 baseline/recovery remain in original reports; no retuning after these five offsets','trials':results}
    (ROOT/'reports/final_grasp_recovery.json').write_text(json.dumps(report,indent=2))
