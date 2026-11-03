"""Seeded pregrasp-cylinder benchmark. No object attachments or object teleporting."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json,argparse
import numpy as np,mujoco
from simulate import ROOT,camera_for
from physics_guard import PhysicsGuard

def scene(offset=0):
    folder=ROOT/'assets/menagerie/leap_hand';root=ET.parse(folder/'right_hand.xml').getroot()
    root.find('compiler').set('meshdir',str(folder/'assets'))
    world=root.find('worldbody');palm=world.find('body');world.remove(palm)
    base=ET.SubElement(world,'body',name='controlled_wrist')
    ET.SubElement(base,'inertial',pos='0 0 0',mass='.1',diaginertia='.001 .001 .001')
    for name,axis in [('base_x','1 0 0'),('base_y','0 1 0'),('base_z','0 0 1')]:ET.SubElement(base,'joint',name=name,type='slide',axis=axis,range='-.12 .12',damping='8',armature='.02')
    base.append(palm);act=root.find('actuator')
    for name in ['base_x','base_y','base_z']:ET.SubElement(act,'position',name=name,joint=name,kp='800',kv='40',ctrlrange='-.12 .12',forcerange='-30 30')
    ET.SubElement(world,'light',pos='0 0 1',dir='0 0 -1',directional='true')
    ET.SubElement(world,'geom',name='table',type='box',pos='0 0 -.025',size='.35 .35 .025',rgba='.2 .3 .4 1')
    obj=ET.SubElement(world,'body',name='object',pos=f'{.04+offset} -.01 .19')
    ET.SubElement(obj,'freejoint',name='object_free')
    ET.SubElement(obj,'geom',name='cylinder',type='cylinder',size='.022 .025',mass='.035',rgba='1 .15 .05 1',friction='1 .01 .001')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    # This benchmark adds a force-bounded XYZ wrist; finger XML/gains are preserved.
    return model

def observe(m,d):
    oid=m.body('object').id;gid=m.geom('cylinder').id;digits=set();forces=[];table_contact=False
    for i,c in enumerate(d.contact):
        if gid not in (c.geom1,c.geom2):continue
        other=c.geom2 if c.geom1==gid else c.geom1
        body=int(m.geom_bodyid[other]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
        if name.startswith(('if_','mf_','rf_','th_')):digits.add(name[:2])
        if other==m.geom('table').id:table_contact=True
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);forces.append(float(force[0]))
    return {'position':d.xpos[oid].copy(),'velocity':d.cvel[oid,3:].copy(),'digits':digits,'normal_force_N':sum(forces),'table_contact':table_contact}

def trial(improved,offset=0,render=False):
    m=scene(offset);d=mujoco.MjData(m);mujoco.mj_forward(m,d);guard=PhysicsGuard(m,d)
    finger_ids=[i for i in range(m.nu) if not (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) or '').startswith('base_')]
    base_ids=[m.actuator(name).id for name in ['base_x','base_y','base_z']]
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(m.actuator_trnid[i,0])) for i in finger_ids]
    goal=np.zeros(m.nu);command=goal.copy();phase='SETTLE';phase_start=0.;events=[];reference=None;closure=0.;hold=0.;maxhold=0.;maxlift=0.;success=False;slips=0;previous_contact=False;error=None
    renderer=mujoco.Renderer(m,height=480,width=640) if render else None;video=None
    if render:
        import cv2
        video=cv2.VideoWriter(str(ROOT/'private/manipulation_trial.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
    def transition(next_phase,obs):
        nonlocal phase,phase_start
        events.append({'time':float(d.time),'from':phase,'to':next_phase,'digits':sorted(obs['digits']),'normal_force_N':obs['normal_force_N']});phase=next_phase;phase_start=float(d.time)
    joint_ids=m.actuator_trnid[:,0];qindices=m.jnt_qposadr[joint_ids];max_joint_excursion=0.;max_wrist_force=0.
    try:
        for step in range(6000):
            obs=observe(m,d);t=float(d.time);age=t-phase_start
            if reference is None and t>.8:reference=float(obs['position'][2])
            if reference is not None:
                lift=float(obs['position'][2]-reference);maxlift=max(maxlift,lift)
                held=lift>=.02 and len(obs['digits'])>=2 and not obs['table_contact'] and np.linalg.norm(obs['velocity'])<.15
                hold=hold+m.opt.timestep if held else 0.;maxhold=max(maxhold,hold)
                if hold>=2:success=True
            contact=len(obs['digits'])>=2
            if previous_contact and not contact and phase in ('LIFT','HOLD'):slips+=1
            previous_contact=contact
            if phase=='SETTLE' and reference is not None and np.linalg.norm(obs['velocity'])<.08:transition('ALIGN',obs)
            elif phase=='ALIGN':
                # Ground-truth alignment around the seeded palm pregrasp; no claimed tabletop reach.
                goal[base_ids[0]]=np.clip(obs['position'][0]-.04,-.04,.04)
                if abs(d.qpos[m.jnt_qposadr[m.joint('base_x').id]]-goal[base_ids[0]])<.004:transition('CLOSE',obs)
            elif phase=='CLOSE':
                closure=min(1.,closure+m.opt.timestep*.6)
                if contact:transition('CONTACT_CHECK',obs)
                elif closure>=1 and age>2:transition('RECOVER',obs)
            elif phase=='CONTACT_CHECK':
                if contact and obs['normal_force_N']>.02:transition('LIFT',obs)
                elif age>1:transition('CLOSE',obs)
            elif phase=='LIFT':
                goal[base_ids[2]]=min(.07,goal[base_ids[2]]+.025*m.opt.timestep)
                if improved:
                    # Bound correction using measured force rather than closing harder indefinitely.
                    correction=-.25 if obs['normal_force_N']>8 else (.12 if not contact else 0.)
                    closure=float(np.clip(closure+correction*m.opt.timestep,.3,1.2))
                if goal[base_ids[2]]>=.069:transition('HOLD',obs)
                if age>4 and not contact:transition('RECOVER',obs)
            elif phase=='HOLD':
                if improved:
                    correction=-.25 if obs['normal_force_N']>8 else (.12 if not contact else 0.)
                    closure=float(np.clip(closure+correction*m.opt.timestep,.3,1.2))
                if success:transition('RELEASE',obs)
                elif age>3:transition('RECOVER',obs)
            elif phase in ('RELEASE','RECOVER'):
                closure=max(0,closure-m.opt.timestep);goal[base_ids[2]]=max(0,goal[base_ids[2]]-.025*m.opt.timestep)
            for i,name in zip(finger_ids,names):
                value=0 if name.endswith('_rot') else closure*(.9 if name.startswith('th_') else 1.25)
                goal[i]=np.clip(value,*m.actuator_ctrlrange[i])
            command+=np.clip(goal-command,-2.5*m.opt.timestep,2.5*m.opt.timestep)
            d.ctrl[:]=np.clip(command,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);guard.step(d)
            max_joint_excursion=max(max_joint_excursion,float(np.max(np.maximum(m.jnt_range[joint_ids,0]-d.qpos[qindices],d.qpos[qindices]-m.jnt_range[joint_ids,1]))))
            max_wrist_force=max(max_wrist_force,float(np.max(np.abs(d.actuator_force[base_ids]))))
            if renderer and step%17==0:
                import cv2
                renderer.update_scene(d,camera=camera_for(m));frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);cv2.putText(frame,phase,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);video.write(frame)
    except Exception as exc:error=str(exc)
    finally:
        if renderer:renderer.close()
        if video:video.release()
    return {'controller':'contact_correction' if improved else 'baseline','offset_m':offset,'success':success,'max_lift_m':maxlift,'max_valid_hold_s':maxhold,'contact_loss_edges_unfiltered':slips,'max_joint_limit_excursion_rad':max(0.,max_joint_excursion),'max_wrist_actuator_force_N':max_wrist_force,'final_phase':phase,'error':error,'warnings':guard.warnings(d),'events':events,'criterion':'>=2cm above settled pregrasp reference, >=2 distinct digits contacting, no table contact, speed<0.15m/s continuously for2s','initialization':'35g cylinder radius22mm height50mm seeded above palm and allowed to settle; not tabletop pick-and-place','object_teleporting_after_initialization':False,'wrist':'three force-limited physical slide joints, 30N per axis'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');a=parser.parse_args();results=[]
    for improved in (False,True):
        for offset in [-.008,-.004,0,.004,.008]:
            result=trial(improved,offset,render=a.render and improved and offset==0);results.append(result);print(result['controller'],offset,result['success'],result['error'],flush=True)
    (ROOT/'reports/manipulation_results.json').write_text(json.dumps(results,indent=2))
