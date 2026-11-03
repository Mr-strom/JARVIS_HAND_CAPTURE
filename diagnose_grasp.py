"""Instrument unchanged original grasp benchmark before proposing a correction."""
import json
import numpy as np,mujoco
import manipulation
from simulate import ROOT

def diagnose():
    records=[];original=manipulation.observe
    def observed(m,d):
        result=original(m,d)
        if round(d.time/m.opt.timestep)%10:return result
        gid=m.geom('cylinder').id;contacts=[]
        for i,c in enumerate(d.contact):
            if gid not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==gid else c.geom1;body=int(m.geom_bodyid[other]);name=m.body(body).name
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);normal=c.frame.reshape(3,3)[0].copy()*(-1 if c.geom1==gid else 1)
            contacts.append({'body':name,'position_m':c.pos.tolist(),'normal_on_object':normal.tolist(),'normal_N':float(force[0]),'tangential_N':float(np.linalg.norm(force[1:3])),'friction_mu':float(c.friction[0]),'friction_utilization_proxy':float(np.linalg.norm(force[1:3])/max(1e-8,c.friction[0]*force[0]))})
        fingers=[c for c in contacts if (c['body'] or '').startswith(('if_','mf_','rf_','th_'))]
        opposition=min([1.]+[float(np.dot(a['normal_on_object'],b['normal_on_object'])) for i,a in enumerate(fingers) for b in fingers[i+1:] if a['body'][:2]!=b['body'][:2]])
        aids=[a for a in range(m.nu) if not m.actuator(a).name.startswith('base_')];jids=m.actuator_trnid[aids,0];adr=m.jnt_qposadr[jids]
        records.append({'time':d.time,'position':result['position'].tolist(),'velocity':result['velocity'].tolist(),'digits':sorted(result['digits']),'contacts':contacts,'opposition_min_cos':opposition,'max_joint_target_error_rad':float(np.max(abs(d.qpos[adr]-d.ctrl[aids]))),'wrist_z':float(d.qpos[m.jnt_qposadr[m.joint('base_z').id]]),'object_height':float(result['position'][2])})
        return result
    manipulation.observe=observed
    try:result=manipulation.trial(False,0,False)
    finally:manipulation.observe=original
    active=[r for r in records if 2.4<r['time']<8]
    summary={'original_trial':result,'sample_count':len(records),'multi_digit_fraction_during_lift_window':float(np.mean([len(r['digits'])>=2 for r in active])),'opposing_contacts_fraction_during_lift_window':float(np.mean([r['opposition_min_cos']<-.5 for r in active])),'median_max_target_error_rad':float(np.median([r['max_joint_target_error_rad'] for r in active])),'max_friction_utilization_proxy':max([0.]+[c['friction_utilization_proxy'] for r in active for c in r['contacts'] if c['normal_N']>.01]),'interpretation':'Inspect detailed normals, forces and tracking error; contact-count flicker alone is not verified slip.'}
    (ROOT/'reports/grasp_diagnosis.json').write_text(json.dumps(summary,indent=2));(ROOT/'private/grasp_contact_trace.json').write_text(json.dumps(records,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':diagnose()
