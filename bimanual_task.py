"""Matched single/assisted tabletop attempts; success requires an actual unassisted hold."""
import argparse,json,time
import cv2,mujoco,numpy as np
from humanoid_control import load_model,BimanualController,camera
from embodiment_adapter import ShadowGeometryAdapter
from physics_guard import PhysicsGuard
from simulate import ROOT

def observation(m,d):
    gid=m.geom('task_cylinder').id;oid=m.body('task_object').id;contacts={'left':[],'right':[]};table=False
    for i,ct in enumerate(d.contact):
        if gid not in (ct.geom1,ct.geom2):continue
        other=ct.geom2 if ct.geom1==gid else ct.geom1
        if other==m.geom('task_table').id:table=True
        name=m.body(int(m.geom_bodyid[other])).name or ''
        side='left' if name.startswith('lh_') else 'right' if name.startswith('rh_') else None
        if side:
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
            normal=ct.frame.reshape(3,3)[0]*(-1 if ct.geom1==gid else 1)
            contacts[side].append({'body':name,'position':ct.pos.tolist(),'normal':normal.tolist(),'normal_N':float(force[0]),'tangent_N':float(np.linalg.norm(force[1:3]))})
    return {'position':d.xpos[oid].copy(),'velocity':d.cvel[oid,3:].copy(),'contacts':contacts,'table':table}

def run_trial(assisted,offset=0,render=False,duration=16):
    m=load_model(True);d=mujoco.MjData(m);obj_joint=m.joint('task_object_free').id;obj_adr=m.jnt_qposadr[obj_joint]
    d.qpos[obj_adr]+=offset  # initial condition only, before physics begins
    mujoco.mj_forward(m,d);c=BimanualController(m);g=PhysicsGuard(m,d);adapter=ShadowGeometryAdapter(m)
    z=np.load(ROOT/'private/learning/predictions.npz');rows=np.where(~z['held_out'])[0];idx=rows[np.argmax(np.mean(z['neural'][rows],axis=1))];pose=adapter.commands(z['neural'][idx])
    desired=c.command.copy();neutral=desired.copy();targets={'left':np.array([.28,.033,.86]),'right':np.array([.28,-.033,.86])}
    ik={side:c.ik(d,side,p) for side,p in targets.items()};liftik=c.ik(d,'right',targets['right']+np.array([0,0,.06]));leftout=c.ik(d,'left',targets['left']+np.array([0,.075,0]))
    phase='SETTLE';phase_t=0.;closure=0.;handover=0.;lift=0.;hold=0.;maxhold=0.;stabilized=0.;maxstabilized=0.;grip_ready=0.;maxlift=0.;success=False;error=None;events=[];trace=[];maxexc=0.;maxpen=0.;max_arm_error=0.;lat=[];slip_duration=0.;slip_events=0;renderer=None;video=None
    if render:renderer=mujoco.Renderer(m,height=720,width=960);video=cv2.VideoWriter(str(ROOT/'private/g1_assisted_task.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,720))
    def change(next_phase,obs):
        nonlocal phase,phase_t
        events.append({'time':float(d.time),'from':phase,'to':next_phase,'object_position':obs['position'].tolist(),'left_contacts':len(obs['contacts']['left']),'right_contacts':len(obs['contacts']['right'])});phase=next_phase;phase_t=float(d.time)
    start=time.perf_counter();right_stable=0.;max_right_stable=0.
    try:
        for step in range(round(duration/m.opt.timestep)):
            tick=time.perf_counter();obs=observation(m,d);age=d.time-phase_t
            left=obs['contacts']['left'];right=obs['contacts']['right'];digits={x['body'][3:5] for x in right if x['body'][3:5] in ['ff','mf','rf','lf','th'] and x['normal_N']>.005}
            good=len(digits)>=2 and sum(x['normal_N'] for x in right)>.02
            grip_ready=grip_ready+m.opt.timestep if good else 0.
            height=float(obs['position'][2]-.86);maxlift=max(maxlift,height)
            held=height>=.02 and good and not left and not obs['table'] and np.linalg.norm(obs['velocity'])<.15
            hold=hold+m.opt.timestep if held else 0.;maxhold=max(maxhold,hold)
            if hold>=2:success=True
            cooperative=bool(left and right and np.linalg.norm(obs['velocity'])<.05 and np.linalg.norm(obs['position'][:2]-np.array([.35+offset,0]))<.015)
            stabilized=stabilized+m.opt.timestep if cooperative else 0.;maxstabilized=max(maxstabilized,stabilized)
            upright=d.xmat[m.body('task_object').id].reshape(3,3)[2,2]>.95
            shared_stable=bool(right and upright and np.linalg.norm(obs['velocity'])<.05 and np.linalg.norm(obs['position'][:2]-np.array([.35+offset,0]))<.015)
            right_stable=right_stable+m.opt.timestep if shared_stable else 0.;max_right_stable=max(max_right_stable,right_stable)
            if phase in ('LIFT','HOLD') and not good:
                slip_duration+=m.opt.timestep
                if slip_duration-m.opt.timestep<.1<=slip_duration:slip_events+=1
            else:slip_duration=0.
            if phase=='SETTLE' and d.time>.5 and np.linalg.norm(obs['velocity'])<.04:change('LEFT_SUPPORT' if assisted else 'RIGHT_APPROACH',obs)
            elif phase=='LEFT_SUPPORT':
                approach=min(1.,age/2.5)
                desired[c.arms['left']]=(1-approach)*neutral[c.arms['left']]+approach*ik['left'][0]
                for a in c.hands['left']:
                    if m.actuator(a).name.endswith(('J3','J0')):desired[a]=.25
                if left and age>.3:change('RIGHT_APPROACH',obs)
                elif age>3:change('RECOVER_NO_LEFT_CONTACT',obs)
            elif phase=='RIGHT_APPROACH':
                approach=min(1.,age/2.5)
                desired[c.arms['right']]=(1-approach)*neutral[c.arms['right']]+approach*ik['right'][0]
                p=d.xpos[m.body('rh_palm').id]
                if (right and np.linalg.norm(p-obs['position'])<.15) or np.linalg.norm(p-targets['right'])<.018:
                    ik['right']=(d.qpos[c.qadr['right']].copy(),ik['right'][1]);targets['right']=p.copy()
                    liftik=c.ik(d,'right',p+np.array([0,0,.06]));desired[c.arms['right']]=ik['right'][0]
                    change('CLOSE',obs)
                elif age>4:change('RECOVER_REACH_ERROR',obs)
            elif phase=='CLOSE':
                closure=min(1.,closure+.45*m.opt.timestep)
                for a,v in pose.items():desired[a]=v*closure
                if grip_ready>.12:change('HANDOVER' if assisted else 'LIFT',obs)
                elif age>3:change('RECOVER_NO_GRIP',obs)
            elif phase=='HANDOVER':
                if good and np.linalg.norm(obs['velocity'])<.12:handover=min(1.,handover+m.opt.timestep)
                else:handover=max(0.,handover-2*m.opt.timestep)
                desired[c.arms['left']]=(1-handover)*ik['left'][0]+handover*leftout[0]
                for a in c.hands['left']:desired[a]*=max(0.,1-2*m.opt.timestep)
                if handover>=1 and not left:change('LIFT',obs)
                elif age>3:change('RECOVER_SUPPORT_NOT_RELEASED',obs)
            elif phase=='LIFT':
                lift=min(1.,lift+.4*m.opt.timestep);desired[c.arms['right']]=(1-lift)*ik['right'][0]+lift*liftik[0]
                if lift>=1:change('HOLD',obs)
                if slip_duration>.5:change('RECOVER_GRIP_LOSS',obs)
            elif phase=='HOLD':
                if success:change('RELEASE',obs)
                elif age>3:change('RECOVER_HOLD_FAILED',obs)
            elif phase.startswith('RECOVER') or phase=='RELEASE':
                desired+=(neutral-desired)*min(1.,m.opt.timestep)
            c.step(d,desired);g.step(d);lat.append((time.perf_counter()-tick)*1000)
            joints=np.where(m.jnt_limited)[0];q=d.qpos[m.jnt_qposadr[joints]];maxexc=max(maxexc,float(np.max(np.maximum(m.jnt_range[joints,0]-q,q-m.jnt_range[joints,1]))))
            maxpen=max(maxpen,max([0.]+[-float(ct.dist) for ct in d.contact]))
            if maxpen>.005:
                raise RuntimeError(f'Physical integrity abort: contact penetration {maxpen:.6f} m exceeds 5mm task threshold')
            if maxexc>.02:
                raise RuntimeError(f'Physical integrity abort: joint excursion {maxexc:.6f} rad exceeds 0.02rad threshold')
            if step%25==0:
                trace.append({'time':float(d.time),'phase':phase,'position':obs['position'].tolist(),'speed':float(np.linalg.norm(obs['velocity'])),'left_force_N':sum(x['normal_N'] for x in left),'right_force_N':sum(x['normal_N'] for x in right),'right_digits':len(digits),'hold':hold,'cooperative_stable':stabilized})
            if renderer and step%17==0:
                renderer.update_scene(d,camera=camera(m));frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);cv2.putText(frame,'Fixed pelvis / learned right grasp shape + scripted support',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);cv2.putText(frame,phase,(12,52),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);video.write(frame)
                if step==3400:cv2.imwrite(str(ROOT/'private/g1_task.png'),frame)
    except Exception as exc:error=str(exc)
    finally:
        if renderer:renderer.close();video.release()
    result={'assisted':assisted,'offset_m':offset,'success':success,'max_lift_m':maxlift,'max_valid_unassisted_hold_s':maxhold,'max_cooperative_table_stabilization_s':maxstabilized,'debounced_grip_loss_events':slip_events,'final_phase':phase,'events':events,'ik_residuals_m':{s:v[1] for s,v in ik.items()},'lift_ik_residual_m':liftik[1],'actual_final_palm_error_m':{side:float(np.linalg.norm(d.xpos[m.body('lh_palm' if side=='left' else 'rh_palm').id]-targets[side])) for side in targets},'warnings':g.warnings(d),'error':error,'max_joint_limit_excursion_rad':max(0.,maxexc),'max_penetration_m':maxpen,'mean_control_step_ms':float(np.mean(lat)),'real_time_factor':g.steps*g.dt/(time.perf_counter()-start),'trace':trace,'criterion':'object >=2cm above table rest, 2+ right digits, no left or table contact, speed<.15m/s for2s','learned':'fixed right posture selected from personal MLP predictions; analytical Shadow adapter','scripted':'arm IK, left support, state machine; no learned bimanual policy'}
    result['max_shared_right_contact_stabilization_s']=max_right_stable
    result['physics_integrity_valid']=error is None and maxpen<=.005 and maxexc<=.02
    result['shared_stabilization_success']=max_right_stable>=.5 and result['physics_integrity_valid']
    result['success']=success and result['physics_integrity_valid']
    result['shared_stabilization_criterion']='upright cylinder, <=15mm planar displacement, speed<.05m/s and actual right-hand contact for .5s; left assistance allowed but not required; table support allowed'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');p.add_argument('--trials',type=int,default=5);a=p.parse_args();results=[]
    for assisted in (False,True):
        for offset in np.linspace(-.008,.008,a.trials):
            r=run_trial(assisted,float(offset),a.render and assisted and abs(offset)<1e-6);results.append(r);print('assisted',assisted,'offset',offset,'success',r['success'],'phase',r['final_phase'],'stable',r['max_cooperative_table_stabilization_s'],'error',r['error'],flush=True)
            (ROOT/'reports/bimanual_task_results.json').write_text(json.dumps(results,indent=2))
