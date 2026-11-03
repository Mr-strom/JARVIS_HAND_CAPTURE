"""Measure individual arm/hand actuator response under fixed-base Earth-gravity physics."""
from pathlib import Path
import sys,json
import mujoco,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from humanoid_control import load_model,BimanualController
from physics_guard import PhysicsGuard
m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=BimanualController(m);g=PhysicsGuard(m,d);zero=c.command.copy();results=[]
selected=list(c.arms['left'])+list(c.arms['right'])+list(c.hands['left'])+list(c.hands['right'])
for a in selected:
    for _ in range(250):c.step(d,zero);g.step(d)
    before=float(d.actuator_length[a]);desired=zero.copy();lo,hi=m.actuator_ctrlrange[a]
    desired[a]=min(hi,.25) if hi>.1 else max(lo,-.2)
    for _ in range(350):c.step(d,desired);g.step(d)
    response=float(d.actuator_length[a])-before
    results.append({'actuator':m.actuator(a).name,'target_delta':float(desired[a]-zero[a]),'measured_length_delta_rad_or_tendon_rad':response,'measured_response_over_5mrad':abs(response)>.005})
(root/'reports/g1_actuator_sweep.json').write_text(json.dumps({'results':results,'count':len(results),'responses_over_5mrad':sum(x['measured_response_over_5mrad'] for x in results),'warnings':g.warnings(d),'scope':'individual physical response, not a precision tracking or isolated-motion guarantee'},indent=2))
print('Responded',sum(x['measured_response_over_5mrad'] for x in results),'/',len(results));print([x for x in results if not x['measured_response_over_5mrad']])
