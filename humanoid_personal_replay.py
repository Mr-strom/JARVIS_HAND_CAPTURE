"""Personal learned right-hand transfer demo; left arm/hand assistance is scripted."""
import json,time
import cv2,mujoco,numpy as np
from humanoid_control import load_model,BimanualController,camera
from physics_guard import PhysicsGuard
from embodiment_adapter import ShadowGeometryAdapter
from simulate import ROOT

def main():
    m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=BimanualController(m);g=PhysicsGuard(m,d);adapter=ShadowGeometryAdapter(m)
    z=np.load(ROOT/'private/learning/predictions.npz');choices=np.where(z['held_out'])[0];selected=choices[np.linspace(0,len(choices)-1,180).astype(int)]
    neutral=c.command.copy();qrows=[];commands=[];lengths=[];frames=[];start=time.perf_counter();max_exc=0.;max_pen=0.
    initial={side:d.xpos[m.body('lh_palm' if side=='left' else 'rh_palm').id].copy() for side in c.arms};goals={side:c.ik(d,side,p+np.array([.04,0,.03]))[0] for side,p in initial.items()}
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        video=cv2.VideoWriter(str(ROOT/'private/g1_personal_transfer.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,720))
        for n,index in enumerate(selected):
            desired=neutral.copy();alpha=.5-.5*np.cos(n/30*2*np.pi/6)
            for side in goals:desired[c.arms[side]]=goals[side]*alpha
            for a,value in adapter.commands(z['neural'][index]).items():desired[a]=value
            for a in c.hands['left']:
                if m.actuator(a).name.endswith(('J3','J0')):desired[a]=.5*alpha
            for _ in range(17):
                c.step(d,desired);g.step(d)
                j=np.where(m.jnt_limited)[0];adr=m.jnt_qposadr[j];max_exc=max(max_exc,float(np.max(np.maximum(m.jnt_range[j,0]-d.qpos[adr],d.qpos[adr]-m.jnt_range[j,1]))))
                max_pen=max(max_pen,max([0.]+[-float(x.dist) for x in d.contact]))
            qrows.append(d.qpos.copy());commands.append(desired.copy());lengths.append(d.actuator_length.copy());renderer.update_scene(d,camera=camera(m,close=n>=80));frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            cv2.putText(frame,'RIGHT: personal MLP -> LEAP FK geometry -> Shadow ROM',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
            cv2.putText(frame,'LEFT + ARMS: scripted IK. Fixed pelvis. No manipulation claim.',(15,55),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            video.write(frame)
            if n in [30,90,150]:cv2.imwrite(str(ROOT/f'private/g1_transfer_{n}.png'),frame)
        video.release()
    span=np.ptp(np.array(qrows),axis=0);report={'source':'held-out personal neural predictions, 180 uniformly sampled poses (montage, not original timing)','learned_component':'right LEAP MLP; Shadow adapter is analytical FK geometry-to-ROM and has not been trained','left_hand_supervision':'none; scripted','arms':'scripted Cartesian IK','fixed_pelvis':True,'gravity':m.opt.gravity.tolist(),'sim_seconds':g.steps*g.dt,'wall_seconds':time.perf_counter()-start,'warnings':g.warnings(d),'max_joint_limit_excursion_rad':max(0.,max_exc),'max_penetration_m':max_pen,'joint_motion_range_rad':{m.joint(j).name:float(span[m.jnt_qposadr[j]]) for j in range(m.njnt)},'precision_transfer_accuracy_validated':False,'object_manipulation':False}
    hand=c.hands['right'];actual=np.asarray(lengths)[:,hand];target=np.asarray(commands)[:,hand]
    report['right_hand_command_tracking_rmse_rad_or_tendon_radians']=float(np.sqrt(np.mean((actual-target)**2)))
    report['tracking_metric_scope']='joint angles or tendon angle sums vs requested Shadow commands; not human fingertip accuracy'
    np.savez_compressed(ROOT/'private/g1_transfer_trajectory.npz',qpos=qrows,desired=commands,actuator_length=lengths,source_rows=selected)
    (ROOT/'reports/g1_personal_transfer_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
