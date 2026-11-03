"""Recording-group validation/test evaluation; private traces never enter v5 outputs."""
import argparse,csv,json,time,collections
from pathlib import Path
import numpy as np,mujoco
from v6.adaptive import AdaptiveEngine,MotionSafety,Decision
from v6.geometry import Geometry,metrics
from retargeting import Retargeter
from simulate import ROOT,load_hand
from physics_guard import PhysicsGuard

LABELS={'rom':'rom','neural':'neural','optimizer':'optimizer','hybrid':'hybrid','neural_v5_filter':'neural'}

def stats(x):
    return {'mean':float(np.mean(x)),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95))} if len(x) else {'mean':None,'p50':None,'p95':None}

def angle_metrics(geom,q,goal):
    tips,_=geom.points_jac(q,False);tip,pinch=metrics(tips,goal);m,d=geom.m,geom.d;rot=d.xmat[geom.palm].reshape(3,3)
    base=np.array([(d.xpos[m.body(name).id]-d.xpos[geom.palm])@rot for name in ['if_bs','mf_bs','rf_bs','th_mp']])
    a=tips-base;b=goal[3::4]-goal[::4]
    cosine=np.sum(a*b,axis=1)/np.maximum(1e-9,np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    return [tip,pinch,float(cosine.mean())]

def raw_accel(values,times):
    q=np.asarray(values);t=np.asarray(times);dt=np.diff(t)
    good=(dt>.001)&(dt<.2)
    if len(q)<3:return None
    v=np.diff(q,axis=0)/np.maximum(dt[:,None],1e-9);acc=np.diff(v,axis=0)/np.maximum(((dt[:-1]+dt[1:])/2)[:,None],1e-9)
    ok=good[:-1]&good[1:]
    return float(np.linalg.norm(acc[ok],axis=1).mean()) if ok.any() else None

def run_clip(entry,partition,dropout=False,data_root=ROOT,checkpoint=None,calibration=None):
    z=np.load(Path(data_root)/entry['cache']);timestamps=z['timestamps']-z['timestamps'][0];valid=z['valid'].astype(bool)
    x=np.c_[z['canonical'].reshape(-1,63),z['features']];m=load_hand();measure=Geometry();names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
    engines={k:AdaptiveEngine(v,checkpoint,calibration) for k,v in LABELS.items()}
    filters={k:MotionSafety(m.actuator_ctrlrange) for k in LABELS};legacy=Retargeter(names,m.actuator_ctrlrange,engines['neural'].personal.rom.calibration)
    filters['neural_v5_filter']=legacy
    states={k:mujoco.MjData(m) for k in LABELS};guards={k:PhysicsGuard(m,d) for k,d in states.items()}
    for d in states.values():mujoco.mj_forward(m,d)
    for e in engines.values():
        indices=np.where(valid)[0]
        if len(indices):
            xx=x[indices[0]];goal=e.geometry.targets(xx[:63].reshape(21,3));q=e.personal.predict(xx)
            for _ in range(3):e.personal.predict(xx);e.geometry.solve(goal,q,e.budget_ms,e.max_nfev)
    decisions={k:None for k in LABELS};rows={k:[] for k in LABELS};raw={k:[] for k in LABELS};rawt={k:[] for k in LABELS}
    latency={k:[] for k in LABELS};filter_ms={k:[] for k in LABELS};mode_counts={k:collections.Counter() for k in LABELS};timeouts=collections.Counter();attempts=collections.Counter();rejected=collections.Counter();limits=collections.Counter();maxexc=collections.Counter();physical_tracking={k:[] for k in LABELS}
    controls={k:[] for k in LABELS};velocities={k:[] for k in LABELS};trace={k:[] for k in LABELS};lastq={k:states[k].qpos.copy() for k in LABELS};max_control_jump=collections.Counter();lastctrl={k:np.zeros(16) for k in LABELS}
    lastcontrol=-1e9;controltimes=[];filterstates={k:collections.Counter() for k in LABELS};steps=0;physics_ms={k:[] for k in LABELS}
    def advance(target):
        nonlocal lastcontrol,steps
        while states['rom'].time+1e-10<target:
            now=float(states['rom'].time)
            if now-lastcontrol>=1/30-1e-9:
                controltimes.append(now)
                for method in LABELS:
                    f=filters[method];tick=time.perf_counter();dec=decisions[method]
                    if method=='neural_v5_filter':
                        if dec is not None and dec.target is not None and now-dec.timestamp<=.25:legacy.request=dec.target.copy();legacy.last_seen=dec.timestamp
                        command=legacy.update(None,now)
                    else:command=f.update(dec,now)
                    filter_ms[method].append((time.perf_counter()-tick)*1000);states[method].ctrl[:]=command
                    max_control_jump[method]=max(max_control_jump[method],float(np.max(abs(command-lastctrl[method]))));lastctrl[method]=command.copy()
                    controls[method].append(command.copy());velocities[method].append(f.velocity.copy() if hasattr(f,'velocity') else np.zeros(16));filterstates[method][f.state]+=1
                lastcontrol=now
            for method,d in states.items():
                tick=time.perf_counter();guards[method].step(d);physics_ms[method].append((time.perf_counter()-tick)*1000)
                exc=np.maximum(m.jnt_range[:,0]-d.qpos,d.qpos-m.jnt_range[:,1]);maxexc[method]=max(maxexc[method],float(exc.max()));limits[method]+=int(np.count_nonzero(exc>1e-4))
                physical_tracking[method].append(float(np.sqrt(np.mean((d.qpos-d.ctrl)**2))))
            steps+=1
    for n,t in enumerate(timestamps):
        advance(float(t));hidden=dropout and (t%4)>=2 and (t%4)<2.45
        for method,e in engines.items():
            tick=time.perf_counter();decision=e.observe(x[n] if valid[n] and not hidden else None,float(t));latency[method].append((time.perf_counter()-tick)*1000);decisions[method]=decision
            mode_counts[method][decision.mode]+=1;info=decision.info.get('solver',{})
            if info:attempts[method]+=1;timeouts[method]+=int(info['timeout'])
            if valid[n] and decision.target is None:rejected[method]+=1
            if valid[n]:
                goal=measure.targets(z['canonical'][n]);actual=angle_metrics(measure,states[method].qpos,goal)
                rawmetrics=angle_metrics(measure,decision.target,goal) if decision.target is not None else [np.nan]*3
                row={'recording_id':entry['index'],'partition':partition,'scenario':'dropout' if dropout else 'clean','method':method,'source_time_s':float(z['timestamps'][n]),'mode':decision.mode,'reason':decision.reason,'category':str(z['gesture'][n]) if 'gesture' in z.files else 'unknown','speed':decision.info.get('speed_palm_widths_s',np.nan),
                    'raw_fingertip_mm':rawmetrics[0],'raw_pinch_mm':rawmetrics[1],'raw_direction_cosine':rawmetrics[2],'physical_fingertip_mm':actual[0],'physical_pinch_mm':actual[1],'physical_direction_cosine':actual[2],'mapping_ms':latency[method][-1]}
                rows[method].append(row)
                if decision.target is not None:raw[method].append(decision.target.copy());rawt[method].append(t)
                trace[method].append({'time':float(t),'source_frame':int(z['frame_indices'][n]),'qpos':states[method].qpos.copy(),'command':states[method].ctrl.copy(),'target':None if decision.target is None else decision.target.copy(),'mode':decision.mode,'tip_error_mm':actual[0]})
        if n%200==0:print(partition,entry['index'],'dropout',dropout,'frame',n,'/',len(timestamps),flush=True)
    summaries=[]
    for method,e in engines.items():
        r=rows[method];q=np.array(controls[method]);dt=np.diff(controltimes)
        v=np.diff(q,axis=0)/dt[:,None] if len(q)>1 else np.zeros((1,16));a=np.diff(v,axis=0)/((dt[:-1]+dt[1:])/2)[:,None] if len(v)>1 else np.zeros((1,16))
        summaries.append({'recording_id':entry['index'],'partition':partition,'scenario':'dropout' if dropout else 'clean','method':method,'samples':len(r),'accepted_raw_samples':len(raw[method]),
            **{key:float(np.nanmean([row[key] for row in r])) for key in ['raw_fingertip_mm','raw_pinch_mm','raw_direction_cosine','physical_fingertip_mm','physical_pinch_mm','physical_direction_cosine']},
            'raw_command_acceleration_norm_rad_s2':raw_accel(raw[method],rawt[method]),'control_acceleration_norm_rad_s2':float(np.linalg.norm(a,axis=1).mean()),'max_control_velocity_rad_s':float(np.max(abs(v))),
            'max_discrete_control_acceleration_rad_s2':float(np.max(abs(a))),'max_command_jump_rad':max_control_jump[method],'mapping_ms_mean':stats(latency[method])['mean'],'mapping_ms_p50':stats(latency[method])['p50'],'mapping_ms_p95':stats(latency[method])['p95'],
            'control_filter_ms_mean':stats(filter_ms[method])['mean'],'control_filter_ms_p95':stats(filter_ms[method])['p95'],'physics_step_ms_mean':stats(physics_ms[method])['mean'],
            'joint_tracking_rmse_rad':float(np.sqrt(np.mean(np.array(physical_tracking[method])**2))),'joint_limit_violation_samples':limits[method],'max_raw_joint_excursion_rad':max(0.,maxexc[method]),'command_limit_violations':int(np.count_nonzero((q<m.actuator_ctrlrange[:,0])|(q>m.actuator_ctrlrange[:,1]))),
            'solver_attempts':attempts[method],'solver_timeouts':timeouts[method],'mode_switches':e.switches,'switches_per_second':e.switches/max(.001,float(timestamps[-1])),'rejected_or_hidden_valid_samples':rejected[method],
            'mode_counts':dict(mode_counts[method]),'filter_state_counts':dict(filterstates[method]),'warnings':guards[method].warnings(states[method]),'sim_seconds':float(states[method].time)})
    return summaries,[row for r in rows.values() for row in r],trace

def main():
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['validation','test'],default='validation');p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--checkpoint',type=Path);p.add_argument('--calibration',type=Path);p.add_argument('--dropouts',action='store_true');a=p.parse_args()
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text());clips=[e for e in audit['split'] if e['partition']==a.partition];summaries=[];allrows=[];private=ROOT/'private/v6';private.mkdir(parents=True,exist_ok=True)
    for entry in clips:
        for dropout in ([False,True] if a.dropouts else [False]):
            summary,rows,traces=run_clip(entry,a.partition,dropout,a.data_root,a.checkpoint,a.calibration);summaries.extend(summary);allrows.extend(rows)
            for method,trace in traces.items():
                np.savez_compressed(private/f'{a.partition}_{entry["index"]}_{method}_{"dropout" if dropout else "clean"}.npz',times=[x['time'] for x in trace],source_frames=[x['source_frame'] for x in trace],qpos=[x['qpos'] for x in trace],commands=[x['command'] for x in trace],modes=[x['mode'] for x in trace],tip_error_mm=[x['tip_error_mm'] for x in trace])
    out=ROOT/'v6';(out/f'{a.partition}_results.json').write_text(json.dumps({'protocol':'all primary methods common v6 motion safety; neural_v5_filter isolates preserved filter. Causal previous commands stepped to each source timestamp; physics500Hz/control~29.4Hz; original invalid observations retained. Geometry is proportion-adjusted LEAP proxy, not ground truth.','threshold_selection':'validation only; reused final test clips already informed earlier project design, not pristine prospective holdout','solver':'same fingertip+.25pinch objective; analytical Jacobian, max4 evaluations/8ms cooperative deadline selected from validation, personalized neural initial pose; late results discarded for neural fallback','latency_scope':'precomputed79 features -> validated raw request; control filter/physics separately timed. Excludes image extraction/camera/display.','results':summaries},indent=2))
    with (out/f'{a.partition}_frames.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
    with (out/('ADAPTIVE_RETARGETING_RESULTS.csv' if a.partition=='test' else 'VALIDATION_RESULTS.csv')).open('w',newline='') as f:
        rows=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in summaries];w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print('COMPLETE',a.partition,len(summaries),flush=True)

if __name__=='__main__':main()
