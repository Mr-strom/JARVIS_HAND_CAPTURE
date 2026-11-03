"""Aggregate executed experiments and make a publication-ready static figure."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    cross=json.loads((ROOT/'reports/cross_embodiment_results.json').read_text())
    grasp=json.loads((ROOT/'reports/final_grasp_recovery.json').read_text());trials=grasp['trials']
    summary={'configuration_count':1,'trials':len(trials),'successful_2_second_holds':sum(t['success'] for t in trials),
        'max_lift_m':max(t['max_lift_m'] for t in trials),'max_valid_hold_s':max(t['max_valid_hold_s'] for t in trials),
        'max_penetration_m':max(t['max_penetration_m'] for t in trials),'max_joint_limit_excursion_rad':max(t['max_joint_limit_excursion_rad'] for t in trials),
        'numerical_warnings':sum(sum(t['warnings'].values()) for t in trials),'integrity_aborts':sum(t['error'] is not None for t in trials),
        'mean_tracking_rmse_rad':float(np.mean([t['mean_joint_tracking_rmse_rad'] for t in trials])),
        'mean_opposing_contact_fraction':float(np.mean([t['opposing_contact_fraction'] for t in trials])),
        'peak_digit_contact_force_N_including_seed_impact':max(t['peak_digit_contact_force_N'] for t in trials),
        'interpretation':'0 successful holds; lower command-tracking error is not grasp success. Old median max error and new mean RMSE are different statistics, not a matched improvement claim.'}
    (ROOT/'reports/final_grasp_summary.json').write_text(json.dumps(summary,indent=2))
    rows=[{k:v for k,v in t.items() if k not in ('trace','events','warnings')} for t in trials]
    with (ROOT/'reports/final_grasp.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    methods=list(cross['summary']);labels=['Direct ROM','ROM via LEAP','MLP via LEAP'];colors=['#7289a5','#2a9d8f','#e9a23b']
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for ax,key,title in zip(axes.flat[:3],['fingertip_error_palm_widths','finger_direction_cosine','pinch_error_palm_widths'],['Fingertip error (palm widths), lower is better','Finger direction cosine, higher is better','Pinch error (palm widths), lower is better']):
        means=[cross['summary'][m][key] for m in methods]
        sd=[np.std([r[key] for r in cross['per_recording'] if r['method']==m],ddof=1) for m in methods]
        ax.bar(labels,means,yerr=sd,color=colors,capsize=4)
        for i,v in enumerate(means):ax.text(i,v+.02,f'{v:.3f}',ha='center',fontsize=10)
        ax.set_title(title,fontsize=11);ax.set_ylim(0,max(1.,max(np.array(means)+sd)*1.2));ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];ax.plot([t['offset_m']*1000 for t in trials],[t['max_valid_hold_s'] for t in trials],'o-',label='Measured valid hold')
    ax.axhline(2,color='#be3838',ls='--',label='Required hold');ax.set(xlabel='Initial object x offset (mm)',ylabel='Continuous valid hold (s)',title='Bounded grasp recovery: 0/5 successes',ylim=(-.1,2.4));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Personal movement transfer: five held-out recordings\nShadow dynamics; bars are recording means ± recording SD, not confidence intervals',fontsize=13)
    fig.savefig(ROOT/'reports/final_evaluation.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
