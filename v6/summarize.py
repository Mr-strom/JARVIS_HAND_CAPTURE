"""Summarize actual v6 runs, category coverage, dropout recovery and latency."""
from pathlib import Path
import json,csv,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate import ROOT

def main():
    out=ROOT/'v6';test=json.loads((out/'test_results.json').read_text());rows=test['results'];summary=[]
    for scenario in ['clean','dropout']:
        for method in ['rom','neural','optimizer','hybrid','neural_v5_filter']:
            rr=[x for x in rows if x['scenario']==scenario and x['method']==method];n=sum(x['samples'] for x in rr);accepted=sum(x['accepted_raw_samples'] for x in rr)
            result={'scenario':scenario,'method':method,'recordings':len(rr),'physical_samples':n,'accepted_raw_samples':accepted,'accepted_raw_coverage':accepted/max(1,n)}
            for key in ['raw_fingertip_mm','raw_pinch_mm','raw_direction_cosine']:
                result[key]=sum(x[key]*x['accepted_raw_samples'] for x in rr if np.isfinite(x[key]))/max(1,accepted)
            for key in ['physical_fingertip_mm','physical_pinch_mm','physical_direction_cosine','joint_tracking_rmse_rad','mapping_ms_mean','mapping_ms_p50','mapping_ms_p95','control_filter_ms_mean','control_filter_ms_p95','physics_step_ms_mean','control_acceleration_norm_rad_s2']:
                result[key]=sum(x[key]*x['samples'] for x in rr)/max(1,n)
            for key in ['joint_limit_violation_samples','command_limit_violations','solver_attempts','solver_timeouts','mode_switches']:result[key]=sum(x[key] for x in rr)
            for key in ['max_command_jump_rad','max_control_velocity_rad_s','max_discrete_control_acceleration_rad_s2','max_raw_joint_excursion_rad']:result[key]=max(x[key] for x in rr)
            result['warning_count']=sum(sum(x['warnings'].values()) for x in rr);result['solver_timeout_rate']=result['solver_timeouts']/max(1,result['solver_attempts']);result['mode_counts']=dict(sum([collections.Counter(x['mode_counts']) for x in rr],collections.Counter()))
            result['switches_per_second']=result['mode_switches']/sum(x['sim_seconds'] for x in rr);summary.append(result)
    (out/'summary.json').write_text(json.dumps({'aggregation':'raw geometry weighted by accepted observations, physical geometry by all original valid observations; latency summaries weighted recording statistics, see frame CSV for pooled quantiles','results':summary},indent=2))
    with (out/'ADAPTIVE_RETARGETING_SUMMARY.csv').open('w',newline='') as f:
        rr=[{k:v for k,v in x.items() if k!='mode_counts'} for x in summary];w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
    # Compute pooled, genuinely measured mapping percentiles rather than averaging percentiles.
    with (out/'test_frames.csv').open() as f:frame_rows=list(csv.DictReader(f))
    timing={}
    for method in ['rom','neural','optimizer','hybrid']:
        values=np.array([float(x['mapping_ms']) for x in frame_rows if x['method']==method and x['scenario']=='clean'])
        timing[method]={'observed_mapping_mean_ms':float(values.mean()),'observed_mapping_p50_ms':float(np.percentile(values,50)),'observed_mapping_p95_ms':float(np.percentile(values,95)),'samples':len(values)}
    camera=json.loads((out/'camera_capture_profile.json').read_text());before=json.loads((out/'original_before_profile.json').read_text());after=json.loads((out/'live_after_profile.json').read_text())
    latency={'scope':'Offline mapping79 precomputed features -> decision (includes quality/MLP/target/solve as used); control filtering and physics separately measured. Online read-completion -> render excludes exposure/buffering/scanout/settling.',
        'offline_pooled_mapping':timing,'offline_full_controller_scope':'Per-observation mapping and per-control-tick safety are different frequencies; not added as a fabricated p95. Detailed costs in experiment CSV and online worker timings.',
        'capture_only':camera,'original_online':before,'worker_online':after,'attended_hand_verification':False,'online_limitation':'Both actual camera checks detected zero hands. They measure unattended capture/detection/render throughput only; no live retargeting performance or latency improvement established.',
        'historical_camera_comparison':'Current exact-hash capture was about30fps versus earlier7-10fps, without our exposure changes. Lighting/driver/session were not controlled across dates; not attributed to v6.'}
    if (out/'full_controller_profile.json').exists():latency['full_controller_measured_profile']=json.loads((out/'full_controller_profile.json').read_text())
    (out/'LATENCY_PROFILE.json').write_text(json.dumps(latency,indent=2))
    # Predeclared geometry categories; fast threshold is validation75th percentile.
    with (out/'validation_frames.csv').open() as f:val=list(csv.DictReader(f))
    speeds=[float(x['speed']) for x in val if x['method']=='neural' and x['scenario']=='clean' and np.isfinite(float(x['speed']))]
    threshold=float(np.percentile(speeds,75));audit=json.loads((ROOT/'reports/dataset_audit.json').read_text());finger_lookup={}
    for entry in audit['split']:
        if entry['partition']!='test':continue
        z=np.load(ROOT/entry['cache'])
        for n,t in enumerate(z['timestamps']):
            f=z['features'][n];b=np.array([np.mean(f[[0,2,3]]),np.mean(f[[4,6,7]]),np.mean(f[[8,10,11]])])
            finger_lookup[(entry['index'],round(float(t),6))]=bool(np.isfinite(b).all() and np.count_nonzero(b>.7)==1 and np.count_nonzero(b<.35)>=2)
    category_rows=[]
    for category in ['open_close','precision_pinch','fast','individual_finger_proxy']:
        for method in ['rom','neural','optimizer','hybrid']:
            selected=[]
            for row in frame_rows:
                if row['method']!=method or row['scenario']!='clean':continue
                qualifies=(row['category'] in ('open','curled')) if category=='open_close' else (row['category']=='pinch') if category=='precision_pinch' else (float(row['speed'])>threshold) if category=='fast' else finger_lookup.get((int(row['recording_id']),round(float(row['source_time_s']),6)),False)
                if qualifies:selected.append(row)
            clips=len(set(x['recording_id'] for x in selected));enough=len(selected)>=20 and clips>=2
            category_rows.append({'category':category,'method':method,'frames':len(selected),'recordings':clips,'enough_for_descriptive_comparison':enough,
                'raw_fingertip_mm':float(np.nanmean([float(x['raw_fingertip_mm']) for x in selected])) if selected else None,
                'physical_fingertip_mm':float(np.mean([float(x['physical_fingertip_mm']) for x in selected])) if selected else None})
    (out/'category_results.json').write_text(json.dumps({'fast_threshold_palm_widths_s':threshold,'threshold_source':'validation75th percentile','minimum':'20 frames and2 recordings; geometry heuristics, not manual gesture labels; categories can overlap','results':category_rows},indent=2))
    recovery=[]
    for entry in audit['split']:
        if entry['partition']!='test':continue
        for method in ['neural_v5_filter','neural','hybrid']:
            clean=np.load(ROOT/f'private/v6/test_{entry["index"]}_{method}_clean.npz');lost=np.load(ROOT/f'private/v6/test_{entry["index"]}_{method}_dropout.npz');t=lost['times'];error=np.max(abs(lost['commands']-clean['commands']),axis=1)
            for end in np.arange(2.45,float(t[-1]),4):
                eligible=np.where((t>=end)&(t<end+3.55))[0];delay=None
                for idx in eligible[:-1]:
                    if error[idx]<.1 and error[idx+1]<.1:delay=float(t[idx]-end);break
                recovery.append({'recording_id':entry['index'],'method':method,'dropout_end_s':float(end),'recovery_delay_s':delay,'criterion':'within.1rad maximum of clean command for2 cached observations; censored if not achieved before next gap/clip end'})
    (out/'dropout_recovery.json').write_text(json.dumps(recovery,indent=2))
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');primary=[x for x in summary if x['scenario']=='clean' and x['method']!='neural_v5_filter'];labels=[x['method'] for x in primary]
    for ax,key,title in [(axs[0,0],'raw_fingertip_mm','Accepted raw fingertip proxy error (mm)'),(axs[0,1],'physical_fingertip_mm','Physical replay error, all valid inputs (mm)')]:
        ax.bar(labels,[x[key] for x in primary],color=['#748aa6','#d4a23c','#469c85','#866ec7']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    for x in primary:axs[1,0].scatter(timing[x['method']]['observed_mapping_p50_ms'],x['raw_fingertip_mm'],label=x['method'],s=70)
    axs[1,0].set(xlabel='Measured pooled mapping p50 (ms)',ylabel='Raw fingertip proxy error (mm)',title='Accuracy / CPU computation tradeoff');axs[1,0].legend()
    smooth=[x for x in summary if x['scenario']=='clean' and x['method'] in ('neural_v5_filter','neural','hybrid')];axs[1,1].bar([x['method'] for x in smooth],[x['max_discrete_control_acceleration_rad_s2'] for x in smooth]);axs[1,1].set(title='Maximum command acceleration',ylabel='rad/s squared');axs[1,1].tick_params(axis='x',labelsize=8)
    fig.suptitle('Held-out personal recordings: measured v6 tradeoffs; geometry proxies, not grasp success');fig.savefig(out/'before_after_evaluation.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
